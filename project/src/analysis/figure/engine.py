from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


DEFAULT_MODEL = {"name": "figure-analysis-unconfigured", "version": None}


@runtime_checkable
class FigureUnderstandingEngine(Protocol):
    """Adapter contract for a chart or figure understanding model."""

    model_name: str
    model_version: str | None

    def analyze(self, image_path: str | Path) -> Mapping[str, Any]:
        """Return raw semantic fields for one cropped figure."""


def run_figure_engine(
    engine: FigureUnderstandingEngine | None,
    crop_path: str | Path,
    evidence: Sequence[Any] | None = None,
    context: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run an injected model while keeping failures local to one block."""
    if engine is None:
        return {
            "model": DEFAULT_MODEL,
            "confidence": None,
            "figure_type": "unknown",
            "warnings": ["Figure understanding model is not configured."],
        }

    model = {
        "name": str(getattr(engine, "model_name", engine.__class__.__name__)),
        "version": getattr(engine, "model_version", None),
    }
    try:
        parameters = inspect.signature(engine.analyze).parameters
        accepts_kwargs = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        kwargs: dict[str, Any] = {}
        if "evidence" in parameters or accepts_kwargs:
            kwargs["evidence"] = evidence
        if "context" in parameters or accepts_kwargs:
            kwargs["context"] = context
        raw = engine.analyze(crop_path, **kwargs)
    except Exception as exc:  # One bad figure must not fail the whole page.
        return {"failed": True, "model": model, "warnings": [f"Figure analysis failed: {exc}"]}

    if not isinstance(raw, Mapping):
        return {
            "failed": True,
            "model": model,
            "warnings": ["Figure model returned a non-object result."],
        }

    output = dict(raw)
    output.setdefault("model", model)
    return output
