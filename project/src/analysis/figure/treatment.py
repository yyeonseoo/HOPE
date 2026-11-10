"""Rule-based braille treatment triage for figure blocks.

Braille textbook guidelines decide *how* a visual is handled before anyone
writes a description: tactile graphic first, a transcriber's note when a
tactile graphic is impractical, and an explicit omission marker when the
visual adds nothing (see reports/HOPE 점역 지침 대조 평가.md, section 5.2).
This module only suggests a default; the reviewer's choice in the UI
(`braille_review.visual_treatment`) always wins.

ponytail: figure type + nearby question wording only. Caption/body
duplication checks and vector inspection are the upgrade path.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

_TACTILE_TYPES = {
    "graph", "line_chart", "bar_chart", "pie_chart", "scatter_plot",
    "mathematical_diagram", "diagram",
}
_PICTURE_TYPES = {"photo", "illustration"}
# Badges and icons measured 0.33% of the page in the audit pages; the
# smallest real graph was 1.6%. Calibration knob for other textbooks.
_ICON_AREA_RATIO = 0.005
_NUMBER = re.compile(r"\d")
# "오른쪽 그림은 … 대관람차이다": the body names what the picture shows.
_BODY_NAMES_PICTURE = re.compile(r"(?:오른쪽|왼쪽|위|아래|다음)\s*(?:그림|사진)은\s*[^.]{0,40}이다")
# Imperative endings used by Korean textbook questions ("구하시오", "그려 보자").
_QUESTION_PATTERN = re.compile(r"(?:하|쓰|그리|구하|나타내|말하|만드|찾)시오|(?:그려|구해|나타내어|이야기해) 보자")


def suggest_visual_treatment(
    figure_type: str | None,
    nearby_texts: Iterable[str] = (),
    area_ratio: float | None = None,
    figure_texts: Iterable[str] = (),
) -> dict[str, Any]:
    figure_type = figure_type or "unknown"
    near_question = any(_QUESTION_PATTERN.search(text or "") for text in nearby_texts)

    if area_ratio is not None and area_ratio < _ICON_AREA_RATIO:
        return _suggestion("omit", "페이지에 비해 아주 작은 그림(배지·아이콘)은 생략하고 표시를 남긴다.", ["BF2016-6.4.1"])
    if figure_type == "icon":
        return _suggestion("omit", "장식·아이콘은 생략하고 생략 표시를 남긴다.", ["BF2016-6.4.1"])
    if figure_type in _TACTILE_TYPES:
        if near_question:
            return _suggestion(
                "tactile_graphic",
                "문제 그림은 설명하면 답이 드러나므로 촉도로 제작한다.",
                ["TG2022-1.6", "TG2022-7.3"],
            )
        if not any(str(t or "").strip() for t in figure_texts):
            # A large decorative icon is easily classified as a diagram.
            return _suggestion(
                "tactile_graphic",
                "그래프·도식은 촉도 제작을 우선한다. 다만 그림 안에 글자·눈금이 없어 장식일 수도 있으니 확인한다.",
                ["TG2022-1.7"],
            )
        return _suggestion("tactile_graphic", "그래프·도식은 촉도 제작을 우선한다.", ["TG2022-1.7"])
    if figure_type in _PICTURE_TYPES and sum(bool(_NUMBER.search(t or "")) for t in figure_texts) >= 3:
        # e.g. a bar chart of yearly revenue classified as a photo.
        return _suggestion(
            "tactile_graphic",
            "그림 안에 수치가 여러 개 있어 그래프로 보고 촉도를 우선한다.",
            ["TG2022-1.7"],
        )
    if figure_type in _PICTURE_TYPES:
        # TG2022 §1.8 Q1: the body already says what the picture shows.
        if any(_BODY_NAMES_PICTURE.search(text or "") for text in nearby_texts):
            return _suggestion(
                "omit",
                "본문이 그림의 대상을 이미 설명하므로 생략하고 표시를 남긴다.",
                ["TG2022-1.8", "TG2022-5.4.2"],
            )
        return _suggestion(
            "transcriber_note",
            "사진·삽화는 한 문장 점역자 주 또는 생략 표시로 처리한다.",
            ["TG2022-5.4.2"],
        )
    if figure_type == "table":
        return _suggestion("transcriber_note", "표 형태 그림은 표로 재분류할지 검토한다.", [])
    return _suggestion("transcriber_note", "유형을 판별하지 못해 점역자 주로 두고 검수자가 판단한다.", [])


def _suggestion(treatment: str, reason: str, rationale: list[str]) -> dict[str, Any]:
    return {
        "suggested_visual_treatment": treatment,
        "treatment_reason": reason,
        "treatment_rationale": rationale,
        "suggested_by": "rule",
    }
