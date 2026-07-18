from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .captioners import Qwen3VLCaptioner
from .context import augment_caption_with_context
from .openclip_classifier import FigureRouteClassifier


class HuggingFaceFigureCaptionEngine:
    """Route a crop with OpenCLIP and invoke a replaceable route captioner."""

    model_name = "openclip-caption-router"
    model_version = "1.0"

    def __init__(
        self,
        classifier: FigureRouteClassifier,
        captioner: Qwen3VLCaptioner,
    ) -> None:
        self.classifier = classifier
        self.captioner = captioner

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
def create_huggingface_figure_engine(
    *,
    device: str = "auto",
    model_id: str = Qwen3VLCaptioner.model_name,
) -> HuggingFaceFigureCaptionEngine:
    """Build the five-way OpenCLIP + Qwen3-VL pipeline lazily."""
    from .openclip_classifier import OpenCLIPFigureTypeClassifier

    return HuggingFaceFigureCaptionEngine(
        classifier=OpenCLIPFigureTypeClassifier(device=device),
        captioner=Qwen3VLCaptioner(model_id=model_id, device=device),
    )
