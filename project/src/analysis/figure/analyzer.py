from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

from .classifier import metadata_figure_type
from .crop import crop_and_save_figure_block
from .engine import FigureUnderstandingEngine, run_figure_engine
from .normalize import build_figure_analysis


def analyze_figure_blocks(
    page: Mapping[str, Any],
    page_image_path: str | Path | None = None,
    engine: FigureUnderstandingEngine | None = None,
    output_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Analyze every figure block using the same interface as other modules."""
    page_id = page.get("page_id")
    blocks = page.get("blocks", [])
    if not isinstance(blocks, list):
        return []

    return [
        analyze_figure_block(page_id, block, blocks, index, page_image_path, engine, output_dir)
        for index, block in enumerate(blocks)
        if isinstance(block, Mapping) and block.get("type") == "figure"
    ]


def analyze_figure_block(
    page_id: int | None,
    block: Mapping[str, Any],
    blocks: list[Mapping[str, Any]],
    block_index: int,
    page_image_path: str | Path | None = None,
    engine: FigureUnderstandingEngine | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    bbox = block.get("bbox")
    crop_path = crop_and_save_figure_block(page_image_path, block, page_id, output_dir)
    previous_id = _neighbor_id(blocks, block_index - 1)
    next_id = _neighbor_id(blocks, block_index + 1)
    caption_id = _adjacent_caption_id(blocks, block_index)

    if crop_path is None:
        normalized = {
            "analysis": {
                "status": "failed",
                "model": {"name": "figure-analysis-unconfigured", "version": None},
                "confidence": None,
                "result": None,
            },
            "warnings": ["Page image or a valid figure bbox was not available."],
        }
    else:
        raw = run_figure_engine(engine, crop_path)
        if engine is None:
            explicit_type = metadata_figure_type(block)
            if explicit_type != "unknown":
                raw["figure_type"] = explicit_type
                raw["warnings"] = []
        normalized = build_figure_analysis(raw)

    nearby_ids = list(dict.fromkeys(item for item in [previous_id, next_id, caption_id] if item is not None))
    return {
        "schema_version": "1.0.0",
        "page_id": page_id,
        "block_id": block.get("block_id"),
        "type": "figure",
        "bbox": bbox,
        "crop_path": crop_path,
        "detection": {
            "model": {"name": str(block.get("detector") or "layout detector"), "version": None},
            "confidence": _safe_confidence(block.get("score")),
        },
        "analysis": normalized["analysis"],
        "context": {
            "previous_block_id": previous_id,
            "next_block_id": next_id,
            "caption_block_id": caption_id,
            "nearby_block_ids": nearby_ids,
        },
        "warnings": normalized["warnings"],
    }


def _neighbor_id(blocks: list[Mapping[str, Any]], index: int) -> Optional[str]:
    if index < 0 or index >= len(blocks):
        return None
    return blocks[index].get("block_id")


def _adjacent_caption_id(blocks: list[Mapping[str, Any]], index: int) -> Optional[str]:
    for candidate_index in (index - 1, index + 1):
        if 0 <= candidate_index < len(blocks) and blocks[candidate_index].get("type") == "caption":
            return blocks[candidate_index].get("block_id")
    return None


def _safe_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return min(1.0, max(0.0, float(value)))
