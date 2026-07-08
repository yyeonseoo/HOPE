from __future__ import annotations

from typing import Any, Mapping


FIGURE_TYPES = {
    "line_chart",
    "bar_chart",
    "pie_chart",
    "scatter_plot",
    "diagram",
    "illustration",
    "photo",
    "other",
    "unknown",
}

ALIASES = {
    "line": "line_chart",
    "line_graph": "line_chart",
    "bar": "bar_chart",
    "bar_graph": "bar_chart",
    "pie": "pie_chart",
    "scatter": "scatter_plot",
    "scatter_chart": "scatter_plot",
    "image": "illustration",
    "graph": "other",
    "chart": "other",
}


def normalize_figure_type(value: Any) -> str:
    """Map model labels to the finite schema vocabulary without guessing."""
    if value is None:
        return "unknown"
    label = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    label = ALIASES.get(label, label)
    return label if label in FIGURE_TYPES else "unknown"


def metadata_figure_type(block: Mapping[str, Any]) -> str:
    """Read an explicit upstream subtype when present; do not infer from pixels."""
    context = block.get("context") if isinstance(block.get("context"), Mapping) else {}
    for candidate in (block.get("figure_type"), block.get("subtype"), context.get("figure_type")):
        normalized = normalize_figure_type(candidate)
        if normalized != "unknown":
            return normalized
    return "unknown"
