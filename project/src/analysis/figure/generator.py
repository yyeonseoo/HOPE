from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class GeneratedDescription:
    """Result of running one context-aware prompt through a captioner."""

    text: str
    confidence: float | None
    generation_time_seconds: float | None
    model_name: str
    model_version: str | None
    warnings: list[str] = field(default_factory=list)


class FigureDescriptionGenerator:
    """Run a prebuilt prompt through a captioner's generic instruction-following
    entrypoint (``caption_with_prompt``), keeping the model call itself out of
    the analyzer/prompt-builder layers."""

    def supports(self, captioner: Any) -> bool:
        return callable(getattr(captioner, "caption_with_prompt", None))

    def generate(
        self,
        captioner: Any,
        image_path: str | Path,
        prompt: str,
        *,
        evidence: Sequence[Any] | None = None,
    ) -> GeneratedDescription:
        if not self.supports(captioner):
            return GeneratedDescription(
                text="",
                confidence=None,
                generation_time_seconds=None,
                model_name=str(getattr(captioner, "model_name", "figure-generator-unconfigured")),
                model_version=getattr(captioner, "model_version", None),
                warnings=["Captioner does not support context-aware prompt generation (caption_with_prompt)."],
            )
        output = captioner.caption_with_prompt(image_path, prompt, evidence=evidence)
        return GeneratedDescription(
            text=output.text,
            confidence=output.confidence,
            generation_time_seconds=output.generation_time_seconds,
            model_name=output.model_name,
            model_version=output.model_version,
            warnings=list(output.warnings),
        )
