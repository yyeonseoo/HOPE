from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .captioners import ChatGPTCaptioner
from .context import augment_caption_with_context
from .grounding import TopicGroundingScorer, find_topic_mismatch_warning
from .openclip_classifier import FigureRouteClassifier


class HuggingFaceFigureCaptionEngine:
    """Route a crop with OpenCLIP and invoke a replaceable route captioner."""

    model_name = "openclip-caption-router"
    model_version = "1.0"

    def __init__(
        self,
        classifier: FigureRouteClassifier,
        captioner: ChatGPTCaptioner,
        grounding_scorer: TopicGroundingScorer | None = None,
    ) -> None:
        self.classifier = classifier
        self.captioner = captioner
        self.grounding_scorer = grounding_scorer or TopicGroundingScorer()

    def analyze(
        self,
        image_path: str | Path,
        evidence: Sequence[Any] | None = None,
        context: Sequence[Any] | None = None,
    ) -> dict[str, Any]:
        prediction = self.classifier.classify(image_path)
        parameters = inspect.signature(self.captioner.caption).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {}
        handles_context_inline = bool(getattr(self.captioner, "handles_context_inline", False))
        if "evidence" in parameters or accepts_kwargs:
            kwargs["evidence"] = evidence
        if "context" in parameters or accepts_kwargs:
            kwargs["context"] = context if handles_context_inline else None
        caption = self.captioner.caption(image_path, prediction.route, **kwargs)
        fuse = getattr(self.captioner, "fuse_with_context", None)
        if handles_context_inline:
            pass
        elif context and callable(fuse):
            caption = fuse(caption, prediction.route, context)
        else:
            augmented_text, used_context_ids = augment_caption_with_context(
                caption.text, prediction.route, context
            )
            if augmented_text != caption.text or used_context_ids:
                caption = replace(
                    caption,
                    text=augmented_text,
                    context_block_ids=used_context_ids,
                )
        warnings = list(caption.warnings)
        if caption.confidence is None:
            warnings.append("Caption confidence was unavailable from the generation model.")
        _, mismatch_warnings = find_topic_mismatch_warning(caption.text, context, self.grounding_scorer)
        warnings.extend(mismatch_warnings)
        return {
            "model": {
                "name": getattr(self.classifier, "model_name", self.classifier.__class__.__name__),
                "version": getattr(self.classifier, "model_version", None),
            },
            "confidence": prediction.confidence,
            "figure_type": prediction.route,
            "description_text": caption.text,
            "description_model": {"name": caption.model_name, "version": caption.model_version},
            "description_confidence": caption.confidence,
            "generation_time_seconds": caption.generation_time_seconds,
            "context_used": bool(caption.context_block_ids),
            "context_block_ids": list(caption.context_block_ids),
            "warnings": warnings,
            "description_only": True,
        }
def create_openai_figure_engine(
    *,
    device: str = "auto",
    model: str | None = None,
    api_key: str | None = None,
) -> HuggingFaceFigureCaptionEngine:
    """Build the OpenCLIP + ChatGPT (vision) pipeline lazily.

    Figure-type routing (OpenCLIP) and topic grounding stay local; only
    caption generation is delegated to the OpenAI API. `api_key` defaults to
    the `OPENAI_API_KEY` environment variable when omitted, matching the
    OpenAI SDK's own default. `model` defaults to the `HOPE_FIGURE_GPT_MODEL`
    environment variable (falling back to `gpt-5`) when omitted -- see
    `ChatGPTCaptioner.__init__`.
    """
    from .openclip_classifier import OpenCLIPFigureTypeClassifier

    return HuggingFaceFigureCaptionEngine(
        classifier=OpenCLIPFigureTypeClassifier(device=device),
        captioner=ChatGPTCaptioner(model=model, api_key=api_key),
        grounding_scorer=TopicGroundingScorer(device=device),
    )
