from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .classifier import metadata_figure_type
from .crop import crop_and_save_figure_block
from .engine import FigureUnderstandingEngine, run_figure_engine
from .normalize import build_figure_analysis


def analyze_figure_blocks(
    page: Mapping[str, Any],
    page_image_path: str | Path | None = None,
    engine: FigureUnderstandingEngine | None = None,
    output_dir: str | Path | None = None,
    ocr_lines: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Analyze every figure block using the same interface as other modules."""
    page_id = page.get("page_id")
    blocks = page.get("blocks", [])
    if not isinstance(blocks, list):
        return []

    return [
        analyze_figure_block(page_id, block, blocks, index, page_image_path, engine, output_dir, ocr_lines)
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
    ocr_lines: Sequence[Mapping[str, Any]] | None = None,
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
        evidence = _figure_text_evidence(ocr_lines, bbox)
        raw = run_figure_engine(engine, crop_path, evidence=evidence)
        if engine is None:
            explicit_type = metadata_figure_type(block)
            if explicit_type != "unknown":
                raw["figure_type"] = explicit_type
                raw["warnings"] = []
        normalized = build_figure_analysis(raw)

    nearby_ids = list(dict.fromkeys(item for item in [previous_id, next_id, caption_id] if item is not None))
    record = {
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
    if "description" in normalized:
        record["description"] = normalized["description"]
    return record


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


def _figure_text_evidence(
    ocr_lines: Sequence[Mapping[str, Any]] | None,
    figure_bbox: Any,
) -> list[str]:
    """Collect reasonably reliable text whose center lies inside a figure."""
    if not ocr_lines or not isinstance(figure_bbox, (list, tuple)) or len(figure_bbox) != 4:
        return []
    x1, y1, x2, y2 = figure_bbox
    evidence: list[str] = []
    for line in ocr_lines:
        bbox = line.get("bbox")
        text = str(line.get("text") or "").strip()
        score = line.get("score", 1.0)
        if not text or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        minimum_score = 0.8 if line.get("source") == "pdf_text" else 0.9
        if isinstance(score, (int, float)) and not isinstance(score, bool) and score < minimum_score:
            continue
        lx1, ly1, lx2, ly2 = bbox
        center_x, center_y = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2 and text not in evidence:
            evidence.append(text)
    return evidence
