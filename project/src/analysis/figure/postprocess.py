from __future__ import annotations

from typing import Any, Mapping, Sequence

from .context_builder import FigureContext
from .generator import GeneratedDescription
from .grounding import GroundingScores
from .normalize import build_figure_analysis
from .prompt_builder import PromptTrace
from .summary import derive_summary
from .type_signals import TypeSignals
from .warnings import derive_warning_codes

# Additive keys layered onto the schema-conformant record produced by
# `build_figure_analysis`. Kept out of `analysis`/`context` (whose shapes are
# shared with the formula/table analyzers and locked by
# schemas/block_analysis.schema.json) so existing consumers of those two
# fields are unaffected; see schema changes for the (optional) top-level
# properties these populate.
#
# This set -- description/summary/confidence/grounding/warnings/
# warning_codes/context_used/context_source -- is deliberately the same
# shape a future formula/table "context-aware" pipeline would produce, so
# page_reliability.py can eventually read all three analyzers the same way
# (see request item 11); Figure is just the first to populate it.
_ADDITIVE_RECORD_KEYS = (
    "figure_type",
    "education_context",
    "grounding",
    "confidence",
    "summary",
    "context_source",
    "warning_codes",
    "context_used",
    "prompt_trace",
    "type_signals",
)


def build_context_aware_figure_record(
    *,
    figure_type: str,
    classifier_model: Mapping[str, Any],
    classifier_confidence: float | None,
    description: GeneratedDescription,
    grounding_scores: GroundingScores,
    figure_context: FigureContext,
    prompt_trace: PromptTrace | None = None,
    type_signals: TypeSignals | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the schema-conformant `analysis`/`description`/`warnings`
    fields (via the same `build_figure_analysis` normalizer the legacy engine
    path uses) plus the additive fields the context-aware pipeline adds:
    `figure_type`, `education_context`, `grounding`, `confidence`, `summary`,
    `context_source`, `warning_codes`, `context_used`, `prompt_trace`,
    `type_signals`.
    """
    context_used = figure_context.has_any_text()
    raw = {
        "model": dict(classifier_model),
        "confidence": classifier_confidence,
        "figure_type": figure_type,
        "description_text": description.text,
        "description_model": {"name": description.model_name, "version": description.model_version},
        "description_confidence": description.confidence,
        "generation_time_seconds": description.generation_time_seconds,
        "context_used": context_used,
        "warnings": list(description.warnings),
        # A description-only engine never attempts axes/series -- see
        # build_figure_analysis's own handling of this flag.
        "description_only": True,
    }
    normalized = build_figure_analysis(raw)
    normalized["figure_type"] = figure_type
    normalized["education_context"] = figure_context.to_dict()
    normalized["context_source"] = figure_context.context_source.to_dict()
    normalized["grounding"] = grounding_scores.to_dict()
    normalized["confidence"] = _top_level_confidence(normalized, grounding_scores)
    normalized["summary"] = derive_summary(description.text)
    normalized["context_used"] = context_used
    normalized["warning_codes"] = derive_warning_codes(
        figure_context=figure_context,
        description=description,
        grounding_scores=grounding_scores,
        classifier_confidence=classifier_confidence,
        evidence=evidence,
    )
    if prompt_trace is not None:
        normalized["prompt_trace"] = prompt_trace.to_dict()
    if type_signals is not None and type_signals.has_any():
        normalized["type_signals"] = type_signals.to_dict()
    return normalized


def _top_level_confidence(
    normalized: Mapping[str, Any], grounding_scores: GroundingScores
) -> float | None:
    description = normalized.get("description")
    if isinstance(description, Mapping) and isinstance(description.get("confidence"), (int, float)):
        return description["confidence"]
    return grounding_scores.overall_score


def split_additive_fields(normalized: dict[str, Any]) -> dict[str, Any]:
    """Pop the additive (non-schema-locked) keys out of a normalized record so
    callers can merge the remainder into a schema-conformant analysis record
    and re-attach these at the top level of the final semantic-analysis
    record."""
    return {key: normalized.pop(key) for key in _ADDITIVE_RECORD_KEYS if key in normalized}
