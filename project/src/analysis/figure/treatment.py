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
from typing import Any, Iterable, Mapping

from src.textbook_profile import load_profile

_PROFILE = load_profile()

_TACTILE_TYPES = {
    "graph", "line_chart", "bar_chart", "pie_chart", "scatter_plot",
    "mathematical_diagram", "diagram",
}
_PICTURE_TYPES = {"photo", "illustration"}
# Badges and icons measured 0.33% of the page in the audit pages; the
# smallest real graph was 1.6%. Calibration knob for other textbooks.
_ICON_AREA_RATIO = _PROFILE["icon_area_ratio"]
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
        return _suggestion("omit", "페이지에 비해 아주 작은 그림(배지·아이콘)은 생략하고 표시를 남긴다.", ["BF2016-6.4.1"], strong=True)
    if figure_type == "icon":
        return _suggestion("omit", "장식·아이콘은 생략하고 생략 표시를 남긴다.", ["BF2016-6.4.1"], strong=True)
    if figure_type in _TACTILE_TYPES:
        if near_question:
            return _suggestion(
                "tactile_graphic",
                "문제 그림은 설명하면 답이 드러나므로 촉도로 제작한다.",
                ["TG2022-1.6", "TG2022-7.3"],
                strong=True,
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


def _suggestion(treatment: str, reason: str, rationale: list[str], strong: bool = False) -> dict[str, Any]:
    suggestion = {
        "suggested_visual_treatment": treatment,
        "treatment_reason": reason,
        "treatment_rationale": rationale,
        "suggested_by": "rule",
    }
    if strong:
        suggestion["rule_is_strong"] = True
    return suggestion


# --- Guideline decision tree (TG2022 §1.8) ---------------------------------
# The guideline's own five questions, asked of a vision model and walked in
# the guideline's order. Answers and reasons are kept so the reviewer sees
# why a treatment was suggested, not only what.
DECISION_QUESTIONS = [
    ("q1_text_repeats", "본문이나 캡션이 이 그림의 정보를 이미 모두 말하고 있는가?"),
    ("q2_words_better", "촉각 그래픽보다 글로 설명하는 편이 더 이해하기 쉬운가? (사진, 장면, 단순한 대상 등)"),
    ("q3_needs_sight", "이 그림을 이해하려면 색·원근·명암처럼 눈으로만 알 수 있는 정보가 꼭 필요한가?"),
    ("q4_hard_to_touch", "만져서 구별하기 어려울 만큼 복잡하거나 요소가 작고 빽빽한가?"),
    ("q5_needed_for_task", "학생이 문제를 풀거나 본문을 이해하는 데 이 그림의 정보가 필요한가?"),
]


def decision_tree_prompt(nearby_texts: Iterable[str]) -> str:
    context = " / ".join(t.strip() for t in nearby_texts if t and t.strip())[:800]
    questions = "\n".join(f"- {key}: {text}" for key, text in DECISION_QUESTIONS)
    return (
        "당신은 점자교과서의 시각자료 처리 방식을 정하는 점역사입니다. 교과서 그림과 주변 본문을 보고 "
        "아래 질문에 답하세요. 그림 내용을 설명하거나 정답을 풀지 마세요.\n"
        f"주변 본문: {context or '(없음)'}\n"
        f"질문:\n{questions}\n"
        'JSON 하나로만 답하세요. 형식: {"q1_text_repeats": {"answer": "예" 또는 "아니오", "reason": "한 문장"}, ...} '
        "다섯 질문 모두 포함하세요."
    )


def treatment_from_answers(answers: Mapping[str, Mapping[str, Any]]) -> dict[str, Any] | None:
    """Walk TG2022 §1.8 in order. None when the answers are incomplete."""
    def yes(key: str) -> bool | None:
        value = str((answers.get(key) or {}).get("answer", "")).strip()
        return True if value.startswith("예") else False if value.startswith("아니") else None

    flags = {key: yes(key) for key, _ in DECISION_QUESTIONS}
    if any(flag is None for flag in flags.values()):
        return None
    if flags["q1_text_repeats"]:
        result = ("omit", "본문이나 캡션이 이미 그림의 정보를 말한다.", "TG2022-1.8-1")
    elif not flags["q5_needed_for_task"]:
        result = ("omit", "과제나 본문 이해에 필요하지 않다.", "TG2022-1.8-5")
    elif flags["q2_words_better"]:
        result = ("transcriber_note", "글로 설명하는 편이 더 이해하기 쉽다.", "TG2022-1.8-2")
    elif flags["q3_needs_sight"]:
        result = ("transcriber_note", "눈으로만 알 수 있는 정보가 핵심이라 촉각으로 옮기기 어렵다.", "TG2022-1.8-3")
    elif flags["q4_hard_to_touch"]:
        result = ("tactile_graphic", "복잡하므로 단순화한 촉도와 안내 주로 제작한다.", "TG2022-1.4")
    else:
        result = ("tactile_graphic", "필요한 정보이고 촉각으로 옮길 수 있어 촉도로 제작한다.", "TG2022-1.8")
    treatment, reason, rule_id = result
    return {"treatment": treatment, "reason": reason, "rationale": [rule_id]}


def parse_decision_answers(text: str) -> dict[str, Any] | None:
    import json

    start, end = (text or "").find("{"), (text or "").rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start:end + 1])
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def combine_with_decision_tree(rule: Mapping[str, Any], answers: Mapping[str, Any] | None) -> dict[str, Any]:
    """Hybrid: certain rules (tiny figure, icon, figure in a question) win;
    otherwise the guideline decision tree decides. Both are kept for review
    and for comparing rule-only / tree-only / hybrid offline."""
    combined = dict(rule)
    combined["rule_suggestion"] = rule["suggested_visual_treatment"]
    tree = treatment_from_answers(answers) if answers else None
    if tree is None:
        return combined
    combined["decision_tree"] = [
        {
            "id": key,
            "question": question,
            "answer": str((answers.get(key) or {}).get("answer", "")),
            "reason": str((answers.get(key) or {}).get("reason", "")),
        }
        for key, question in DECISION_QUESTIONS
    ]
    combined["tree_suggestion"] = tree["treatment"]
    if not rule.get("rule_is_strong"):
        combined.update(
            suggested_visual_treatment=tree["treatment"],
            treatment_reason=tree["reason"],
            treatment_rationale=tree["rationale"],
            suggested_by="decision_tree",
        )
    return combined
