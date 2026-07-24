from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .captioners import _trusted_axis_labels
from .graph_visual import GraphVisualCue

GRAPH_FIGURE_TYPES = {"graph", "line_chart", "bar_chart", "pie_chart", "scatter_plot"}
_DIAGRAM_TYPES = {"diagram", "mathematical_diagram"}
_MAX_LABEL_ITEMS = 6
_MAX_LEGEND_ITEMS = 4
_PURE_NUMBER_PATTERN = re.compile(r"^[+-]?\d+(?:\.\d+)?$")


@dataclass(frozen=True)
class TypeSignals:
    """Figure-type-specific structured signals extracted mechanically from
    OCR text (position + content) and, for graphs, a cheap OpenCV trend
    check. Every field is either something actually detected or left at its
    empty default -- there is no fallback model for "relation" or
    "interaction" extraction, so those are only ever populated when a future
    detector is added, never guessed from the current signals.
    """

    x_axis: str | None = None
    y_axis: str | None = None
    legend: tuple[str, ...] = ()
    trend: str | None = None
    components: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    interactions: tuple[str, ...] = ()
    scene: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "x_axis": self.x_axis,
            "y_axis": self.y_axis,
            "legend": list(self.legend),
            "trend": self.trend,
            "components": list(self.components),
            "relations": list(self.relations),
            "objects": list(self.objects),
            "interactions": list(self.interactions),
            "scene": self.scene,
        }

    def has_any(self) -> bool:
        return any(
            [
                self.x_axis,
                self.y_axis,
                self.legend,
                self.trend,
                self.components,
                self.relations,
                self.objects,
                self.interactions,
                self.scene,
            ]
        )


def extract_type_signals(
    figure_type: str,
    evidence: Sequence[Mapping[str, Any]] | None,
    visual_cue: GraphVisualCue | None = None,
) -> TypeSignals:
    """Extract whatever figure-type-specific signal is mechanically
    determinable from OCR `evidence` (same shape as
    analyzer.py's `_figure_text_evidence`: text + relative_bbox) and, for
    graphs, a `GraphVisualCue`. `evidence`/`visual_cue` absence just yields
    fewer populated fields, never a fabricated one.
    """
    if figure_type in GRAPH_FIGURE_TYPES:
        return _extract_graph_signals(evidence, visual_cue)
    if figure_type in _DIAGRAM_TYPES:
        return TypeSignals(components=_label_candidates(evidence, _MAX_LABEL_ITEMS))
    if figure_type == "illustration":
        return TypeSignals(objects=_label_candidates(evidence, _MAX_LABEL_ITEMS))
    if figure_type == "photo":
        # Photos rarely carry in-image OCR text; when they do, it's more
        # likely an incidental label than a scene description, so only
        # `objects` (not `scene`, which no detector exists for) is filled.
        return TypeSignals(objects=_label_candidates(evidence, _MAX_LEGEND_ITEMS))
    return TypeSignals()


def _extract_graph_signals(
    evidence: Sequence[Mapping[str, Any]] | None,
    visual_cue: GraphVisualCue | None,
) -> TypeSignals:
    x_axis, y_axis = _trusted_axis_labels(evidence)
    used = {label for label in (x_axis, y_axis) if label}
    legend = tuple(_label_candidates(evidence, _MAX_LEGEND_ITEMS, exclude=used))
    trend = None
    if visual_cue is not None and visual_cue.state == "plotted":
        trend = visual_cue.trend
    return TypeSignals(x_axis=x_axis, y_axis=y_axis, legend=legend, trend=trend)


def _label_candidates(
    evidence: Sequence[Mapping[str, Any]] | None,
    limit: int,
    *,
    exclude: set[str] | None = None,
) -> list[str]:
    """Best-effort short text labels from OCR evidence: excludes bare numbers
    (axis ticks, not labels) and anything already claimed by another field.
    This is a coarse filter, not a legend/label detector -- it can include
    stray non-label text, which is why it stays capped and low-priority in
    the prompt rather than presented as a verified fact.
    """
    if not evidence:
        return []
    exclude = exclude or set()
    candidates: list[str] = []
    for item in evidence:
        text = str(item.get("text") or "").strip()
        if not text or text in exclude or text in candidates:
            continue
        if _PURE_NUMBER_PATTERN.match(text.replace(" ", "")):
            continue
        candidates.append(text)
        if len(candidates) >= limit:
            break
    return candidates
