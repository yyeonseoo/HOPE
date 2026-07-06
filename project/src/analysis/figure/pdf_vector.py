from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import fitz

from .description import build_context_free_description


AXIS_ORIGIN_LABELS = {"o", "0"}


def analyze_pdf_vector_figure(
    pdf_path: str | Path,
    page_number: int,
    bbox: Sequence[float],
    dpi: int,
    block_id: str,
    detection_score: float | None = None,
    detector: str = "layout detector",
) -> dict[str, Any]:
    """Analyze a born-digital PDF figure without OCR or raster heuristics."""
    evidence = extract_vector_evidence(pdf_path, page_number, bbox, dpi)
    x_label, y_label = infer_axis_labels(evidence["words"])
    series = infer_visual_series(evidence["paths"], evidence["words"])
    figure_type = "line_chart" if series and (x_label or y_label) else "diagram"

    analysis = {
        "status": "success" if series else "partial",
        "model": {"name": "pymupdf-vector-parser", "version": fitz.VersionBind},
        "confidence": None,
        "result": {
            "kind": "figure",
            "figure_type": figure_type,
            "title": None,
            "x_axis": {"label": x_label, "unit": None} if x_label else None,
            "y_axis": {"label": y_label, "unit": None} if y_label else None,
            "series": [{"name": item["name"], "points": []} for item in series],
        },
    }
    warnings = []
    if not series:
        warnings.append("PDF vector paths did not contain a readable data series.")
    if not x_label or not y_label:
        warnings.append("One or more axis labels were not identified from PDF text positions.")

    record = {
        "schema_version": "1.0.0",
        "page_id": page_number,
        "block_id": block_id,
        "type": "figure",
        "bbox": list(bbox),
        "crop_path": None,
        "detection": {
            "model": {"name": detector, "version": None},
            "confidence": _confidence(detection_score),
        },
        "analysis": analysis,
        "context": {
            "previous_block_id": None,
            "next_block_id": None,
            "caption_block_id": None,
            "nearby_block_ids": [],
        },
        "warnings": warnings,
    }
    record["description"] = build_vector_description(analysis, series)
    return {"record": record, "evidence": evidence}


def extract_vector_evidence(
    pdf_path: str | Path,
    page_number: int,
    bbox: Sequence[float],
    dpi: int,
) -> dict[str, Any]:
    if len(bbox) != 4 or dpi <= 0:
        raise ValueError("bbox must contain four values and dpi must be positive")

    scale = 72.0 / dpi
    clip = fitz.Rect(*(float(value) * scale for value in bbox))
    if clip.is_empty:
        raise ValueError("bbox is empty")

    document = fitz.open(str(pdf_path))
    try:
        if page_number < 1 or page_number > len(document):
            raise ValueError(f"page_number must be between 1 and {len(document)}")
        page = document[page_number - 1]
        words = []
        for word in page.get_text("words"):
            word_rect = fitz.Rect(word[:4])
            if word_rect.intersects(clip):
                center = fitz.Point((word_rect.x0 + word_rect.x1) / 2, (word_rect.y0 + word_rect.y1) / 2)
                words.append(
                    {
                        "text": str(word[4]).strip(),
                        "x": _unit((center.x - clip.x0) / clip.width),
                        "y": _unit((center.y - clip.y0) / clip.height),
                    }
                )

        paths = []
        for drawing in page.get_drawings():
            color = drawing.get("color")
            if not drawing["rect"].intersects(clip) or not _is_chromatic(color):
                continue
            points = _drawing_points(drawing.get("items", []), clip)
            if len(points) >= 2:
                paths.append(
                    {
                        "color": [round(float(channel), 4) for channel in color],
                        "points": points,
                    }
                )
        return {"clip_pdf": list(clip), "words": words, "paths": paths}
    finally:
        document.close()


def infer_axis_labels(words: list[Mapping[str, Any]]) -> tuple[str | None, str | None]:
    candidates = [item for item in words if _label_text(item.get("text"))]
    x_candidates = [item for item in candidates if item["y"] >= 0.72 and item["x"] >= 0.45]
    y_candidates = [item for item in candidates if item["x"] <= 0.28 and item["y"] <= 0.35]
    x_label = max(x_candidates, key=lambda item: item["x"], default=None)
    y_label = min(y_candidates, key=lambda item: item["y"], default=None)
    return (
        str(x_label["text"]) if x_label else None,
        str(y_label["text"]) if y_label else None,
    )


def infer_visual_series(
    paths: list[Mapping[str, Any]],
    words: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    labels = [item for item in words if _series_label(item.get("text"))]
    output = []
    for index, path in enumerate(paths, start=1):
        points = path.get("points") if isinstance(path.get("points"), list) else []
        if len(points) < 2:
            continue
        end = points[-1]
        nearest = min(
            labels,
            key=lambda item: math.dist((float(item["x"]), float(item["y"])), (end["x"], 1 - end["y"])),
            default=None,
        )
        name = str(nearest["text"]) if nearest and math.dist(
            (float(nearest["x"]), float(nearest["y"])), (end["x"], 1 - end["y"])
        ) <= 0.22 else f"계열 {index}"
        output.append(
            {
                "name": name,
                "trend": summarize_path_trend(points),
                "color": path.get("color"),
            }
        )
    return output


def summarize_path_trend(points: list[Mapping[str, float]], tolerance: float = 0.015) -> list[str]:
    trends = []
    for previous, current in zip(points, points[1:]):
        delta = float(current["y"]) - float(previous["y"])
        trend = "증가" if delta > tolerance else "감소" if delta < -tolerance else "일정"
        if not trends or trends[-1] != trend:
            trends.append(trend)
    return trends


def build_vector_description(
    analysis: Mapping[str, Any],
    series: list[Mapping[str, Any]],
) -> dict[str, Any]:
    base = build_context_free_description(analysis)
    if not series:
        return base

    phrases = []
    for item in series:
        trends = item.get("trend") or []
        if trends:
            past = {"증가": "증가한", "감소": "감소한", "일정": "일정하게 유지된"}
            final = {"증가": "증가한다", "감소": "감소한다", "일정": "일정하게 유지된다"}
            sequence = [past.get(trend, trend) for trend in trends[:-1]]
            sequence.append(final.get(trends[-1], trends[-1]))
            phrases.append(f"{item['name']}는 " + " 뒤 ".join(sequence) + ".")
    result = analysis.get("result", {})
    x_axis = result.get("x_axis") or {}
    y_axis = result.get("y_axis") or {}
    axes = []
    if y_axis.get("label"):
        axes.append(f"세로축은 {y_axis['label']}")
    if x_axis.get("label"):
        axes.append(f"가로축은 {x_axis['label']}")
    short_text = ", ".join(axes) + "을 나타낸 선그래프." if axes else "선그래프."
    base.update(
        {
            "status": "success",
            "short_text": short_text,
            "long_text": " ".join([short_text, *phrases]),
            "transcription_notes": "수치 눈금이 없는 경우 선의 변화 형태만 설명했습니다.",
        }
    )
    return base


def _drawing_points(items: list[Any], clip: fitz.Rect) -> list[dict[str, float]]:
    points = []
    for item in items:
        if not item or item[0] not in {"l", "c"}:
            continue
        endpoints = (item[1], item[2]) if item[0] == "l" else (item[1], item[4])
        for point in endpoints:
            if clip.contains(point):
                normalized = {
                    "x": round(_unit((point.x - clip.x0) / clip.width), 5),
                    "y": round(_unit(1 - (point.y - clip.y0) / clip.height), 5),
                }
                if not points or points[-1] != normalized:
                    points.append(normalized)
    return points


def _is_chromatic(color: Any) -> bool:
    if not isinstance(color, (list, tuple)) or len(color) < 3:
        return False
    return max(color[:3]) - min(color[:3]) >= 0.12


def _label_text(value: Any) -> bool:
    text = str(value or "").strip()
    return bool(text) and text.lower() not in AXIS_ORIGIN_LABELS and not text.isdigit()


def _series_label(value: Any) -> bool:
    text = str(value or "").strip()
    return 0 < len(text) <= 3 and text.lower() not in AXIS_ORIGIN_LABELS and not text.isdigit()


def _unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return _unit(float(value))
