from __future__ import annotations

from typing import Any, Mapping, Sequence

from .context_builder import FigureContext
from .generator import GeneratedDescription
from .grounding import LOW_TOPIC_SIMILARITY_THRESHOLD, GroundingScores

# A small, provider-agnostic warning-code taxonomy. Intentionally free of any
# figure-specific dependency (plain strings in, plain strings out) so it can
# move to a shared location (e.g. src/analysis/shared/warnings.py) once the
# formula and table analyzers adopt the same structure, without callers
# needing to change. Until then it lives here because this is the first
# analyzer to use it -- see analyzer.py orchestration item 11 in the request
# this was built for.


class WarningCode:
    NO_CAPTION = "no_caption"
    LOW_OCR_CONFIDENCE = "low_ocr_confidence"
    CONTEXT_MISSING = "context_missing"
    GROUNDING_MISMATCH = "grounding_mismatch"
    GENERATION_FAILED = "generation_failed"
    LOW_IMAGE_QUALITY = "low_image_quality"
    TYPE_UNCERTAIN = "figure_type_uncertain"


ALL_WARNING_CODES = frozenset(
    value for name, value in vars(WarningCode).items() if not name.startswith("_")
)

# Below this, OCR evidence that survived `_figure_text_evidence`'s own
# score filter (0.8/0.9 minimum) is still borderline enough to flag.
_LOW_OCR_SCORE_THRESHOLD = 0.9
# Below this, the type-route classifier's own top score is low enough that
# the chosen figure_type itself, not just the description, should be
# treated with caution.
_LOW_TYPE_CONFIDENCE_THRESHOLD = 0.5


def derive_warning_codes(
    *,
    figure_context: FigureContext,
    description: GeneratedDescription,
    grounding_scores: GroundingScores,
    classifier_confidence: float | None = None,
    evidence: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Classify *why* a figure result might be untrustworthy, from signals
    already computed elsewhere in the pipeline. This does not replace the
    free-text `warnings` list (which keeps whatever specific messages
    generation/grounding produced) -- it's a small fixed vocabulary meant for
    programmatic use (e.g. page_reliability.py) where matching against
    free-text messages would be brittle.

    `LOW_IMAGE_QUALITY` is declared in the taxonomy but never returned here:
    no image-quality detector exists yet, and guessing one from context is
    exactly the kind of fabrication this pipeline avoids elsewhere.
    """
    codes: list[str] = []
    if not figure_context.caption:
        codes.append(WarningCode.NO_CAPTION)
    if not figure_context.has_any_text():
        codes.append(WarningCode.CONTEXT_MISSING)
    if evidence and any(
        isinstance(item.get("score"), (int, float)) and item["score"] < _LOW_OCR_SCORE_THRESHOLD
        for item in evidence
    ):
        codes.append(WarningCode.LOW_OCR_CONFIDENCE)
    if grounding_scores.overall_score is not None and grounding_scores.overall_score < LOW_TOPIC_SIMILARITY_THRESHOLD:
        codes.append(WarningCode.GROUNDING_MISMATCH)
    if not description.text.strip():
        codes.append(WarningCode.GENERATION_FAILED)
    if classifier_confidence is not None and classifier_confidence < _LOW_TYPE_CONFIDENCE_THRESHOLD:
        codes.append(WarningCode.TYPE_UNCERTAIN)
    return codes
