from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .crop import crop_and_save_table_block, crop_table_image
from .engine import MODEL_NAME, MODEL_VERSION, _load_table_engine, run_table_engine
from .normalize import build_table_analysis


def analyze_table_blocks(
    page: Dict[str, Any],
    page_image_path: Optional[str] = None,
    lang: str = "korean",
    engine=None,
) -> List[Dict[str, Any]]:
    """Take a full page result (as produced by the layout pipeline: `{"page_id":
    ..., "blocks": [...]}`) and return a list of complete, schema-conformant
    `semantic_analyses` records — one per `table`-type block — ready to drop
    into the API response as-is.

    This mirrors feature/formula-analysis's `analyze_formula_blocks(page,
    page_image_path)` interface exactly, so the integration owner can call
    all three analyzers the same way instead of writing different glue code
    per module. Each record includes `detection` (from the block's own
    `score`/`detector`), `context` (previous/next block id by list-index
    adjacency in `page["blocks"]`, same convention as formula), and
    `crop_path` (saved via crop_and_save_table_block), in addition to the
    `analysis` result itself.
    """
    page_id = page.get("page_id")
    blocks = page.get("blocks", [])

    return [
        _analyze_single_table_block(page_id, block, blocks, index, page_image_path, lang, engine)
        for index, block in enumerate(blocks)
        if block.get("type") == "table"
    ]


def _analyze_single_table_block(
    page_id: Optional[int],
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    page_image_path: Optional[str],
    lang: str,
    engine,
) -> Dict[str, Any]:
    bbox = block.get("bbox")
    crop_path = crop_and_save_table_block(page_image_path, block, page_id)

    if page_image_path and bbox:
        output = analyze_table_block(page_image_path, bbox, lang=lang, engine=engine)
    else:
        model = {"name": MODEL_NAME, "version": MODEL_VERSION}
        output = {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": ["페이지 이미지 또는 bbox가 없어 표 구조 인식을 실행하지 못했습니다."],
        }

    previous_block_id = get_neighbor_block_id(blocks, block_index - 1)
    next_block_id = get_neighbor_block_id(blocks, block_index + 1)

    return {
        "schema_version": "1.0.0",
        "page_id": page_id,
        "block_id": block.get("block_id"),
        "type": "table",
        "bbox": bbox,
        "crop_path": crop_path,
        "detection": {
            "model": {"name": block.get("detector", "model-a"), "version": None},
            "confidence": block.get("score"),
        },
        "analysis": output["analysis"],
        "context": {
            "previous_block_id": previous_block_id,
            "next_block_id": next_block_id,
            "caption_block_id": None,
            "nearby_block_ids": [
                block_id for block_id in [previous_block_id, next_block_id] if block_id is not None
            ],
        },
        "warnings": output["warnings"],
    }


def get_neighbor_block_id(blocks: List[Dict[str, Any]], index: int) -> Optional[str]:
    """Return the block_id at `index` in `blocks`, or None if out of range.
    Same convention as feature/formula-analysis's `get_neighbor_block_id`."""
    if index < 0 or index >= len(blocks):
        return None
    return blocks[index].get("block_id")


def analyze_table_block(
    image_path: str | Path,
    bbox: List[int],
    lang: str = "korean",
    engine=None,
) -> Dict:
    """Crop `bbox` out of the page image, run table-structure recognition,
    and return `{"analysis": {...}, "warnings": [...]}` matching the
    `analysis`/`warnings` portions of schemas/block_analysis.schema.json for
    a `table`-type block.

    Lower-level primitive used internally by `analyze_table_blocks` (and
    still usable directly/in tests). It never raises: any failure (bad crop,
    engine error, unparseable output) degrades to `analysis.status ==
    "failed"` with `result: None` and a `warnings` entry explaining why, so
    one broken table can't take down a whole page's processing.
    """
    model = {"name": MODEL_NAME, "version": MODEL_VERSION}

    crop = crop_table_image(image_path, bbox)
    if crop is None:
        return {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": ["표 영역 crop에 실패했습니다 (bbox가 이미지 범위를 벗어났거나 비어 있음)."],
        }

    try:
        active_engine = engine or _load_table_engine(lang=lang)
        raw_result = run_table_engine(active_engine, crop)
    except Exception as exc:  # noqa: BLE001 - never let one block kill the page
        return {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": [f"표 구조 인식 중 오류가 발생했습니다: {exc}"],
        }

    return build_table_analysis(raw_result, model_name=MODEL_NAME, model_version=MODEL_VERSION)
