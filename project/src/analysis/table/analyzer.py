from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from .crop import crop_table_image
from .engine import MODEL_NAME, MODEL_VERSION, _load_table_engine, run_table_engine
from .normalize import build_table_analysis


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

    This is the module's public entrypoint (exported via __init__.py). It
    never raises: any failure (bad crop, engine error, unparseable output)
    degrades to `analysis.status == "failed"` with `result: None` and a
    `warnings` entry explaining why, so one broken table can't take down a
    whole page's processing.

    Deliberately does NOT build `detection`/`context`/`page_id`/`block_id` —
    those belong to the full block_analysis record, which the integration
    owner assembles around this function's output (see OWNERSHIP.md).
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
