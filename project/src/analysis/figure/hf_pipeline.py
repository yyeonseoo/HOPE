from __future__ import annotations

from pathlib import Path
from typing import Any

from .captioners import Qwen3VLCaptioner
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

    def analyze(self, image_path: str | Path) -> dict[str, Any]:
        prediction = self.classifier.classify(image_path)
        caption = self.captioner.caption(image_path, prediction.route)
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
            "warnings": warnings,
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
