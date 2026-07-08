from __future__ import annotations

from typing import Any, Mapping


def build_context_free_description(analysis: Mapping[str, Any]) -> dict[str, Any]:
    """Verbalize only facts present in a normalized figure result."""
    result = analysis.get("result")
    if analysis.get("status") == "failed" or not isinstance(result, Mapping):
        return _description("failed", None, None, "구조 분석 결과가 없어 설명을 생성하지 못했습니다.")

    figure_type = str(result.get("figure_type") or "unknown")
    title = _text(result.get("title"))
    x_axis = _axis_text(result.get("x_axis"))
    y_axis = _axis_text(result.get("y_axis"))
    series = result.get("series") if isinstance(result.get("series"), list) else []

    summary_parts = [title or _type_label(figure_type)]
    if x_axis:
        summary_parts.append(f"X축은 {x_axis}")
    if y_axis:
        summary_parts.append(f"Y축은 {y_axis}")
    if series:
        summary_parts.append(f"{len(series)}개 데이터 계열이 있다")
    short_text = ". ".join(summary_parts).rstrip(".") + "."

    details = []
    for index, item in enumerate(series, start=1):
        if not isinstance(item, Mapping):
            continue
        name = _text(item.get("name")) or f"계열 {index}"
        points = item.get("points") if isinstance(item.get("points"), list) else []
        point_texts = []
        for point in points:
            if isinstance(point, Mapping) and point.get("x") is not None and point.get("y") is not None:
                point_texts.append(f"{point['x']}에서 {point['y']}")
        if point_texts:
            details.append(f"{name}: " + ", ".join(point_texts) + ".")
        else:
            details.append(f"{name}: 추출된 데이터 점이 없다.")

    long_text = " ".join([short_text, *details]).strip()
    status = "success" if series and any(item.get("points") for item in series if isinstance(item, Mapping)) else "partial"
    note = None if status == "success" else "축 또는 데이터가 일부 누락되어 원본 그림 확인이 필요합니다."
    return _description(status, short_text, long_text, note)


def _description(status: str, short_text: str | None, long_text: str | None, note: str | None) -> dict[str, Any]:
    return {
        "status": status,
        "model": {"name": "structured-data-verbalizer", "version": "1.0"},
        "short_text": short_text,
        "long_text": long_text,
        "transcription_notes": note,
        "context_used": False,
        "review_status": "unreviewed",
    }


def _axis_text(axis: Any) -> str | None:
    if not isinstance(axis, Mapping):
        return None
    label, unit = _text(axis.get("label")), _text(axis.get("unit"))
    if label and unit:
        return f"{label}, 단위는 {unit}"
    return label or (f"단위 {unit}" if unit else None)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _type_label(figure_type: str) -> str:
    return {
        "line_chart": "선그래프",
        "bar_chart": "막대그래프",
        "pie_chart": "원그래프",
        "scatter_plot": "산점도",
        "diagram": "도식",
        "illustration": "삽화",
        "photo": "사진",
        "other": "그래프 또는 그림",
        "unknown": "유형을 확인하지 못한 그림",
    }.get(figure_type, "그림")
