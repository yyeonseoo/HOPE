from __future__ import annotations

import re
from typing import Any, Mapping

from .pdf_vector import infer_axis_labels, summarize_path_trend


def classify_figure_route(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Conservatively route a vector PDF figure to graph or image."""
    words = evidence.get("words") if isinstance(evidence.get("words"), list) else []
    paths = evidence.get("paths") if isinstance(evidence.get("paths"), list) else []
    x_label, y_label = infer_axis_labels(words)

    usable_paths = []
    usable_path_indices = []
    for path_index, path in enumerate(paths):
        if not isinstance(path, Mapping):
            continue
        points = path.get("points") if isinstance(path.get("points"), list) else []
        trends = summarize_path_trend(points)
        if 2 <= len(points) <= 16 and 1 <= len(trends) <= 4:
            usable_paths.append(path)
            usable_path_indices.append(path_index)

    reasons = []
    if not _valid_axis_label(x_label):
        reasons.append("x_axis_label_missing_or_invalid")
    if not _valid_axis_label(y_label):
        reasons.append("y_axis_label_missing_or_invalid")
    if not 1 <= len(usable_paths) <= 6 or len(paths) > 6:
        reasons.append("data_path_count_out_of_range")

    if reasons:
        return {
            "route_type": "image",
            "confidence": None,
            "reasons": reasons,
            "x_axis_label": x_label,
            "y_axis_label": y_label,
            "usable_path_count": len(usable_paths),
            "usable_path_indices": usable_path_indices,
        }
    return {
        "route_type": "graph",
        "confidence": None,
        "reasons": [],
        "x_axis_label": x_label,
        "y_axis_label": y_label,
        "usable_path_count": len(usable_paths),
        "usable_path_indices": usable_path_indices,
    }


def _valid_axis_label(value: Any) -> bool:
    text = str(value or "").strip().replace("\u200c", "")
    if not 1 <= len(text) <= 20 or text.isdigit():
        return False
    if any(character in text for character in ";![]{}"):
        return False
    return bool(re.search(r"[A-Za-z가-힣]", text))
