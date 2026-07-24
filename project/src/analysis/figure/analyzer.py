from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .classifier import metadata_figure_type
from .context import build_figure_context
from .context_builder import FigureContext, FigureContextBuilder
from .crop import crop_and_save_figure_block
from .engine import FigureUnderstandingEngine, run_figure_engine
from .generator import FigureDescriptionGenerator
from .graph_visual import GraphVisualCue, analyze_graph_visual
from .grounding import TopicGroundingScorer, compute_grounding_scores
from .normalize import build_figure_analysis
from .postprocess import build_context_aware_figure_record, split_additive_fields
from .prompt_builder import FigurePromptBuilder
from .type_signals import GRAPH_FIGURE_TYPES, TypeSignals, extract_type_signals

# Nearest `window_size` paragraphs before/after the figure, not just the
# single closest one -- richer context for the educational prompt (see
# FigureContextBuilder.build's `window_size` parameter).
_CONTEXT_WINDOW_SIZE = 2

_context_builder = FigureContextBuilder()
_prompt_builder = FigurePromptBuilder()
_description_generator = FigureDescriptionGenerator()


def analyze_figure_blocks(
    page: Mapping[str, Any],
    page_image_path: str | Path | None = None,
    engine: FigureUnderstandingEngine | None = None,
    output_dir: str | Path | None = None,
    ocr_lines: Sequence[Mapping[str, Any]] | None = None,
    semantic_analyses: Sequence[Mapping[str, Any]] | None = None,
    pdf_path: str | Path | None = None,
    source_dpi: int | None = None,
) -> list[dict[str, Any]]:
    """Analyze every figure block using the same interface as other modules."""
    page_id = page.get("page_id")
    blocks = page.get("blocks", [])
    if not isinstance(blocks, list):
        return []

    return [
        analyze_figure_block(
            page_id, block, blocks, index, page_image_path, engine, output_dir, ocr_lines,
            semantic_analyses, pdf_path, source_dpi
        )
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
    semantic_analyses: Sequence[Mapping[str, Any]] | None = None,
    pdf_path: str | Path | None = None,
    source_dpi: int | None = None,
) -> dict[str, Any]:
    bbox = block.get("bbox")
    crop_path = crop_and_save_figure_block(
        page_image_path,
        block,
        page_id,
        output_dir,
        pdf_path=pdf_path,
        source_dpi=source_dpi,
    )
    previous_id = _neighbor_id(blocks, block_index - 1)
    next_id = _neighbor_id(blocks, block_index + 1)
    caption_id = _adjacent_caption_id(blocks, block_index)
    selected_context = build_figure_context(blocks, block_index, semantic_analyses)
    figure_context = _context_builder.build(
        blocks, block, page_id=page_id, ocr_lines=ocr_lines, window_size=_CONTEXT_WINDOW_SIZE
    )

    additive_fields: dict[str, Any] = {}
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
    elif _supports_context_aware_pipeline(engine):
        try:
            normalized = _run_context_aware_pipeline(engine, crop_path, figure_context, ocr_lines, bbox)
            additive_fields = split_additive_fields(normalized)
        except Exception as exc:  # One bad figure must not fail the whole page.
            evidence = _figure_text_evidence(ocr_lines, bbox)
            raw = run_figure_engine(engine, crop_path, evidence=evidence, context=selected_context)
            raw.setdefault("warnings", []).append(
                f"Context-aware description generation failed and fell back to the legacy engine path: {exc}"
            )
            normalized = build_figure_analysis(raw)
    else:
        evidence = _figure_text_evidence(ocr_lines, bbox)
        raw = run_figure_engine(engine, crop_path, evidence=evidence, context=selected_context)
        if engine is None:
            explicit_type = metadata_figure_type(block)
            if explicit_type != "unknown":
                raw["figure_type"] = explicit_type
                raw["warnings"] = []
        normalized = build_figure_analysis(raw)

    nearby_ids = list(dict.fromkeys(
        item
        for item in [
            previous_id,
            next_id,
            caption_id,
            *(context_item["block_id"] for context_item in selected_context),
        ]
        if item is not None
    ))
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
    record.update(additive_fields)
    return record


def _supports_context_aware_pipeline(engine: FigureUnderstandingEngine | None) -> bool:
    """True when `engine` exposes the sub-components (an OpenCLIP-style
    classifier plus a captioner with `caption_with_prompt`) the context-aware
    pipeline needs. Engines that don't -- including the minimal test doubles
    in tests/figure/test_analyzer.py -- keep using the legacy
    `engine.analyze(...)` path, so their behavior is unchanged."""
    return (
        engine is not None
        and hasattr(engine, "classifier")
        and hasattr(engine, "captioner")
        and _description_generator.supports(getattr(engine, "captioner"))
    )


def _run_context_aware_pipeline(
    engine: FigureUnderstandingEngine,
    crop_path: str,
    figure_context: FigureContext,
    ocr_lines: Sequence[Mapping[str, Any]] | None,
    bbox: Any,
) -> dict[str, Any]:
    """Classify -> extract type-specific signals -> build an educational
    prompt -> generate -> score grounding -> assemble the schema-conformant
    + additive record. This is the orchestration the analyzer performs for
    engines that support it; the heavy lifting lives in context_builder.py,
    type_signals.py, prompt_builder.py, generator.py, grounding.py, and
    postprocess.py."""
    prediction = engine.classifier.classify(crop_path)
    figure_type = prediction.route
    evidence = _figure_text_evidence(ocr_lines, bbox)
    type_signals = extract_type_signals(figure_type, evidence, _graph_visual_cue(figure_type, crop_path))
    prompt_result = _prompt_builder.build(figure_type, figure_context, type_signals)
    description = _description_generator.generate(
        engine.captioner, crop_path, prompt_result.prompt, evidence=evidence
    )
    grounding_scorer = getattr(engine, "grounding_scorer", None) or TopicGroundingScorer()
    grounding_scores = compute_grounding_scores(description.text, figure_context, grounding_scorer)
    classifier_model = {
        "name": str(getattr(engine.classifier, "model_name", engine.classifier.__class__.__name__)),
        "version": getattr(engine.classifier, "model_version", None),
    }
    return build_context_aware_figure_record(
        figure_type=figure_type,
        classifier_model=classifier_model,
        classifier_confidence=prediction.confidence,
        description=description,
        grounding_scores=grounding_scores,
        figure_context=figure_context,
        prompt_trace=prompt_result.trace,
        type_signals=type_signals,
        evidence=evidence,
    )


def _graph_visual_cue(figure_type: str, crop_path: str) -> GraphVisualCue | None:
    """Cheap OpenCV trend check, only run for graph-family figure types --
    see type_signals.py's `trend` field. Never raises: a bad/unreadable crop
    just means no trend signal, not a failed analysis."""
    if figure_type not in GRAPH_FIGURE_TYPES:
        return None
    try:
        from PIL import Image

        with Image.open(crop_path) as source:
            return analyze_graph_visual(source)
    except Exception:
        return None


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
) -> list[dict[str, Any]]:
    """Collect reasonably reliable text whose center lies inside a figure."""
    if not ocr_lines or not isinstance(figure_bbox, (list, tuple)) or len(figure_bbox) != 4:
        return []
    x1, y1, x2, y2 = figure_bbox
    evidence: list[dict[str, Any]] = []
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
        if x1 <= center_x <= x2 and y1 <= center_y <= y2:
            if any(item["text"] == text and item["bbox"] == bbox for item in evidence):
                continue
            evidence.append({
                "id": f"t{len(evidence) + 1}",
                "text": text,
                "bbox": [lx1 - x1, ly1 - y1, lx2 - x1, ly2 - y1],
                "relative_bbox": [
                    (lx1 - x1) / max(1, x2 - x1),
                    (ly1 - y1) / max(1, y2 - y1),
                    (lx2 - x1) / max(1, x2 - x1),
                    (ly2 - y1) / max(1, y2 - y1),
                ],
                "score": float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
                "source": str(line.get("source") or "ocr"),
            })
    return evidence
