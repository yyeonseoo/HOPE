from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from .crop import crop_and_save_table_block, crop_table_image
from .description import generate_table_description
from .engine import MODEL_NAME, MODEL_VERSION, _load_table_engine, run_table_engine
from .normalize import build_table_analysis

# Layout blocks the layout model classified as something other than
# "table" but which are worth re-checking with the table-structure engine
# anyway, in case the layout pipeline misclassified an actual table (a
# boxed grid of numbers is an easy figure/formula mix-up upstream). See
# `_try_reclassify_as_table` for the confidence bar used to keep vs. drop
# these candidates.
RECLASSIFIABLE_TYPES = {"figure", "formula"}
_MIN_RECLASSIFIED_ROWS = 2
_MIN_RECLASSIFIED_COLUMNS = 2


def analyze_table_blocks(
    page: Dict[str, Any],
    page_image_path: Optional[str] = None,
    lang: str = "korean",
    engine=None,
) -> List[Dict[str, Any]]:
    """Take a full page result (as produced by the layout pipeline: `{"page_id":
    ..., "blocks": [...]}`) and return a list of complete, schema-conformant
    `semantic_analyses` records — one per `table`-type block, plus any
    `figure`/`formula` block that re-checks as a real table (see
    RECLASSIFIABLE_TYPES) — ready to drop into the API response as-is.

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

    results = []
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type == "table":
            results.append(
                _analyze_single_table_block(page_id, block, blocks, index, page_image_path, lang, engine)
            )
        elif block_type in RECLASSIFIABLE_TYPES:
            reclassified = _try_reclassify_as_table(
                page_id, block, blocks, index, page_image_path, lang, engine
            )
            if reclassified is not None:
                results.append(reclassified)

    return results


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

    if page_image_path and bbox:
        output = analyze_table_block(page_image_path, bbox, lang=lang, engine=engine)
    else:
        model = {"name": MODEL_NAME, "version": MODEL_VERSION}
        output = {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": ["페이지 이미지 또는 bbox가 없어 표 구조 인식을 실행하지 못했습니다."],
        }

    return _assemble_table_record(page_id, block, blocks, block_index, bbox, page_image_path, output)


def _try_reclassify_as_table(
    page_id: Optional[int],
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    page_image_path: Optional[str],
    lang: str,
    engine,
) -> Optional[Dict[str, Any]]:
    """Re-run table-structure recognition on a block the layout pipeline
    labeled `figure` or `formula`. Returns a full schema-conformant record
    (with `type` forced to `"table"`) if the result looks like a genuine
    table, or None if the caller should leave the block under its original
    type -- e.g. it really was a figure/formula, or the crop/engine failed.

    This is a table-branch-only prototype (see OWNERSHIP.md): it does not
    touch the shared layout postprocessing in page_pipeline.py, so
    reclassified blocks only ever show up here, in `semantic_analyses` --
    `page["blocks"]` itself is untouched and still says `figure`/`formula`.
    """
    bbox = block.get("bbox")
    if not page_image_path or not bbox:
        return None

    output = analyze_table_block(page_image_path, bbox, lang=lang, engine=engine)

    if not _looks_like_real_table(output["analysis"]):
        return None

    original_type = block.get("type")
    output = {
        **output,
        "warnings": output["warnings"]
        + [
            f"레이아웃 모델이 이 블록을 '{original_type}'(으)로 분류했지만, "
            "표 구조 인식 결과 실제 표로 보여 table 후보로 재분류했습니다."
        ],
    }

    return _assemble_table_record(page_id, block, blocks, block_index, bbox, page_image_path, output)


def _looks_like_real_table(analysis: Dict[str, Any]) -> bool:
    """Confidence bar for reclassifying a non-table block as a table: a
    non-failed status plus at least a 2x2 grid with some recognized cell
    text. A 1x1 or all-blank grid is far more likely to be table-engine
    noise on a genuine figure/formula image than a real missed table, so
    it's rejected rather than reclassified.
    """
    if analysis.get("status") == "failed":
        return False
    result = analysis.get("result")
    if not result:
        return False
    if result.get("row_count", 0) < _MIN_RECLASSIFIED_ROWS:
        return False
    if result.get("column_count", 0) < _MIN_RECLASSIFIED_COLUMNS:
        return False
    return any(cell.get("text") for cell in result.get("cells", []))


def _assemble_table_record(
    page_id: Optional[int],
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    bbox: Optional[List[int]],
    page_image_path: Optional[str],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    crop_path = crop_and_save_table_block(page_image_path, block, page_id)
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
        "description": generate_table_description(output["analysis"]),
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

    return build_table_analysis(
        raw_result, model_name=MODEL_NAME, model_version=MODEL_VERSION, table_crop=crop
    )
