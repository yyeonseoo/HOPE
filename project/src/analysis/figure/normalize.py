from __future__ import annotations

from typing import Any, Mapping

from .classifier import normalize_figure_type
from .engine import DEFAULT_MODEL


CHART_TYPES = {"line_chart", "bar_chart", "pie_chart", "scatter_plot", "graph"}


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return min(1.0, max(0.0, float(value)))


def normalize_axis(value: Any) -> dict[str, str | None] | None:
    if not isinstance(value, Mapping):
        return None
    label = _nullable_text(value.get("label"))
    unit = _nullable_text(value.get("unit"))
    return {"label": label, "unit": unit} if label is not None or unit is not None else None


def normalize_series(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    normalized = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        points = []
        raw_points = item.get("points")
        if isinstance(raw_points, list):
            for point in raw_points:
                if not isinstance(point, Mapping) or "x" not in point or "y" not in point:
                    continue
                x, y = point.get("x"), point.get("y")
                if not isinstance(x, (str, int, float)) and x is not None:
                    x = str(x)
                if not isinstance(y, (str, int, float)) and y is not None:
                    y = str(y)
                points.append({"x": x, "y": y})
        normalized.append({"name": _nullable_text(item.get("name")), "points": points})
    return normalized


def normalize_model(value: Any) -> dict[str, str | None]:
    if not isinstance(value, Mapping) or not _nullable_text(value.get("name")):
        return dict(DEFAULT_MODEL)
    return {"name": _nullable_text(value.get("name")), "version": _nullable_text(value.get("version"))}


def build_figure_analysis(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize model output into the shared figure analysis contract."""
    model = normalize_model(raw.get("model"))
    warnings = [str(item) for item in raw.get("warnings", []) if str(item).strip()] if isinstance(raw.get("warnings"), list) else []
    if raw.get("failed"):
        return {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": warnings or ["Figure analysis failed without an error message."],
        }

    figure_type = normalize_figure_type(raw.get("figure_type"))
    result = {
        "kind": "figure",
        "figure_type": figure_type,
        "title": _nullable_text(raw.get("title")),
        "x_axis": normalize_axis(raw.get("x_axis")),
        "y_axis": normalize_axis(raw.get("y_axis")),
        "series": normalize_series(raw.get("series")),
    }

    status = "success"
    if figure_type == "unknown":
        status = "partial"
        warnings.append("Figure type was not recognized.")
    elif figure_type in CHART_TYPES | {"other"} and not (
        result["title"] or result["x_axis"] or result["y_axis"] or result["series"]
    ):
        status = "partial"
        warnings.append("Figure output did not contain a title, axes, or data series.")

    normalized = {
        "analysis": {
            "status": status,
            "model": model,
            "confidence": _confidence(raw.get("confidence")),
            "result": result,
        },
        "warnings": list(dict.fromkeys(warnings)),
    }
    description = _normalize_generated_description(raw)
    if description is not None:
        normalized["description"] = description
    return normalized


def _normalize_generated_description(raw: Mapping[str, Any]) -> dict[str, Any] | None:
    text = _nullable_text(raw.get("description_text"))
    model = raw.get("description_model")
    has_description_fields = text is not None or isinstance(model, Mapping) or "generation_time_seconds" in raw
    if not has_description_fields:
        return None

    confidence = _confidence(raw.get("description_confidence"))
    elapsed = raw.get("generation_time_seconds")
    generation_time = (
        max(0.0, float(elapsed))
        if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool)
        else None
    )
    return {
        "status": "success" if text else "failed",
        "model": normalize_model(model) if isinstance(model, Mapping) else None,
        "confidence": confidence,
        "generation_time_seconds": generation_time,
        "short_text": text,
        "long_text": text,
        "transcription_notes": None,
        "context_used": False,
        "review_status": "unreviewed" if text else "needs_review",
    }
