from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .context_builder import FigureContext
from .type_signals import TypeSignals

_GOAL_INSTRUCTION = (
    "당신은 한국 경제수학 점자교과서의 점역자 주를 쓰는 점역사입니다. "
    "점역자 주는 그림에 '보이는 것'을 객관적으로 묘사합니다. 그림이 무엇을 설명하기 위해 쓰였는지, "
    "어떤 의미나 인과관계를 보여주는지 해석하지 말고, 축 이름·단위·눈금·점의 좌표·선의 모양처럼 "
    "이미지에서 확인되는 사실만 쓰세요. 본문이 이미 말한 내용은 반복하지 마세요. "
    "단, 이미지가 실제로는 작은 색상 도형, 구분선, 불릿 아이콘처럼 그 자체에 학습 내용이 담겨 있지 "
    "않은 단순 장식이라면, 주변 문맥과 억지로 연결해 주제를 지어내지 말고 단순 장식용 아이콘이라는 "
    "사실을 그대로 설명하세요."
)

_CAPTION_PRIORITY_INSTRUCTION = (
    "아래 문맥 중 '캡션'은 가장 신뢰도가 높은 정보입니다. 캡션이 있으면 설명의 가장 중요한 근거로 "
    "사용하세요. 캡션이 이미지에서 보이는 내용과 다르게 느껴지면 캡션 쪽을 따르세요."
)

_OCR_INSTRUCTION = (
    "'그림 내부 텍스트'에 축 이름, 범례, 라벨처럼 실제로 읽을 수 있는 문자가 있으면 설명에 정확히 "
    "반영하세요. 없으면 새로 만들지 마세요."
)

_HALLUCINATION_GUARD = (
    "다음 규칙을 반드시 지키세요.\n"
    "- 원문에 없는 내용을 추측하지 않는다.\n"
    "- 이미지와 Context에서 확인 가능한 내용만 설명한다.\n"
    "- 숫자를 임의로 생성하지 않는다.\n"
    "- 모르면 모른다고 한다.\n"
    "- 문제의 정답, 풀이 방법, 풀이 전략을 쓰지 않는다.\n"
    "- 본문과 그림에 없는 용어(예: 교과서에 나오지 않은 함수 이름)를 새로 쓰지 않는다.\n"
    "- 이미지가 실제로는 작은 색상 도형, 구분선, 불릿 아이콘처럼 정보가 없는 단순 장식이라면, "
    "주변 문맥을 끌어와 억지로 의미나 주제를 부여하지 말고 단순 장식용 아이콘이라는 사실 그대로 설명한다."
)

_OUTPUT_INSTRUCTION = (
    # NISE2021 p.19: 점역자 주 안에 한 문장으로 최대한 간결하게.
    "목록이나 마크다운, 제목 없이 한국어 한 문장으로만 답하세요. 그림 전체를 한 문장으로 간결하게 "
    "묘사하고, 꼭 필요한 값(라벨, 수치)만 그 문장 안에 넣으세요. "
    "문장은 교과서 본문처럼 '~다/~이다/~한다'로 끝나는 평서형으로 쓰고, '~습니다', '~해요'와 같은 "
    "존댓말 종결어미는 사용하지 마세요."
)

_GRAPH_OUTPUT_INSTRUCTION = (
    "목록이나 마크다운, 제목 없이 한국어 문단 하나로만 답하세요. 첫 문장은 축 이름과 단위를 포함해 "
    "그래프 전체를 한 문장으로 묘사하고, 이어서 눈금 값과 꺾이는 점의 좌표처럼 보이는 값을 적으세요. "
    "단일 그래프는 2~4문장으로 쓰고, 소그래프가 여러 개이면 모든 소그래프를 빠짐없이 묘사하기 위해 최대 "
    "6문장까지 사용할 수 있습니다. 문장은 교과서 본문처럼 '~다/~이다/~한다'로 끝나는 평서형으로 "
    "쓰고, '~습니다', '~해요'와 같은 존댓말 종결어미는 사용하지 마세요."
)

# Tactile graphics are drawn from data, not prose: ask for a spec list the
# tactile-graphic maker can check against the print page.
_TACTILE_SPEC_INSTRUCTION = (
    "이 그림은 촉각 그래픽으로 제작된다. 문장으로 묘사하지 말고, 촉각 그래픽 제작자가 쓸 항목만 "
    "'항목: 값' 형식으로 세미콜론(;)으로 구분해 한 줄로 적으세요. 항목은 가로축, 세로축(이름과 단위), "
    "눈금(보이는 값), 점(라벨과 좌표), 선(모양과 구간, 실선/점선), 기타 표기 순서로 쓰고, 보이지 않는 "
    "항목은 빼세요. 축 이름·눈금·라벨 값은 '그림 내부 텍스트'에 있는 값을 우선 그대로 쓰세요. "
    # TG2022 Unit 3 planning-sheet items the tactile-graphic maker decides on.
    "이어서 제작 계획 항목을 같은 형식으로 덧붙이세요: 생략할 요소(배경 색, 사진, 장식, 보조 격자 등), "
    "분할 필요(그래프가 여러 개이거나 복잡하면 나눌지 여부), 안내 주(학생에게 촉도의 구성을 알려 주는 "
    "한 문장, 정답이나 해석은 쓰지 않음)."
)

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "graph": (
        "그래프 유형 지침: 먼저 이미지 안에 번호나 구획으로 분리된 소그래프가 몇 개인지 확인하세요. "
        "소그래프가 여러 개이면 하나로 뭉뚱그리지 말고, 보이는 순서대로 모든 소그래프의 선이 "
        "증가·유지·감소하는 구간을 각각 빠짐없이 묘사하세요. 구간이 바뀌는 점은 "
        "눈금에서 읽히는 좌표로 적고, 그래프의 의미는 해석하지 마세요."
    ),
    "line_chart": (
        "그래프 유형 지침: 먼저 이미지 안에 번호나 구획으로 분리된 소그래프가 몇 개인지 확인하세요. "
        "소그래프가 여러 개이면 하나로 뭉뚱그리지 말고, 보이는 순서대로 모든 소그래프의 선이 "
        "증가·유지·감소하는 구간을 각각 빠짐없이 묘사하세요. 구간이 바뀌는 점은 "
        "눈금에서 읽히는 좌표로 적고, 그래프의 의미는 해석하지 마세요."
    ),
    "bar_chart": (
        "그래프 유형 지침: 각 막대의 항목 이름과 눈금에서 읽히는 값을 순서대로 적으세요."
    ),
    "pie_chart": (
        "그래프 유형 지침: 각 부분의 항목 이름과 표시된 비율을 순서대로 적으세요."
    ),
    "scatter_plot": (
        "그래프 유형 지침: 축 이름과 범위, 점의 개수와 라벨이 붙은 점의 좌표를 적으세요."
    ),
    "table": (
        "표 유형 지침: 행과 열의 머리글, 칸의 개수, 비어 있는 칸의 위치를 적으세요."
    ),
    "mathematical_diagram": (
        "수학 도식 유형 지침: 도형의 구성 요소와 배치, 표시된 기호나 문자를 보이는 그대로 적으세요."
    ),
    "diagram": (
        "도식 유형 지침: 구성 요소와 화살표가 잇는 순서를 보이는 그대로 적으세요."
    ),
    "illustration": (
        "삽화 유형 지침: 삽화에 보이는 대상과 장면을 한 문장으로 묘사하세요."
    ),
    "photo": (
        "사진 유형 지침: 사진에 보이는 대상과 장면을 한 문장으로 묘사하세요."
    ),
    "icon": (
        "아이콘 유형 지침: 아이콘이 나타내는 단순한 의미나 범주를 한 문장으로 간단히 설명하세요."
    ),
}

_DEFAULT_TYPE_INSTRUCTION = _TYPE_INSTRUCTIONS["illustration"]

# Role-hint instruction branches (see layout_detection.py's role_hint
# tagging). Matched loosely against the incoming role_hint string so future
# values (e.g. "생각열기", "활동하기", "확인하기") degrade to the concept-first
# default instead of raising, since the layout pipeline's role vocabulary is
# expected to grow independently of this module.
_ROLE_HINT_INSTRUCTIONS: dict[str, str] = {
    "example": "이 그림은 '예제' 영역에 속합니다. 풀이에 쓰이는 값이 보이면 그대로 적되 풀이 과정은 쓰지 마세요.",
    "solution": "이 그림은 '풀이/확인' 영역에 속합니다. 보이는 값만 적고 풀이 과정을 다시 설명하지 마세요.",
    "problem": (
        "이 그림은 '문제' 영역에 속합니다. 정답을 추론하거나 알려주지 말고, 문제를 이해하는 데 "
        "필요한 정보만 제공하세요."
    ),
}
_DEFAULT_ROLE_HINT_INSTRUCTION = ""
# Text after these markers (worked solutions, answers) must not leak into the
# description context.
_SOLUTION_START = re.compile(r"^\s*(?:풀이|답|사고 과정|정답)")

_CONTEXT_FIELD_LABELS: tuple[tuple[str, str], ...] = (
    ("caption", "캡션"),
)

_TYPE_SIGNAL_LABELS: tuple[tuple[str, str], ...] = (
    ("x_axis", "x축"),
    ("y_axis", "y축"),
    ("trend", "시각적으로 확인된 추세"),
    ("scene", "장면"),
)


@dataclass(frozen=True)
class PromptTrace:
    """Which context elements actually made it into the prompt -- computed
    from the same data the prompt sections are built from, so it can't drift
    from what the model actually saw. Used for debugging/explainability, not
    sent to the model itself."""

    use_title: bool = False
    use_caption: bool = False
    use_previous: bool = False
    use_next: bool = False
    use_context_window: bool = False
    use_role_hint: bool = False
    use_ocr: bool = False
    use_type_signals: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "use_title": self.use_title,
            "use_caption": self.use_caption,
            "use_previous": self.use_previous,
            "use_next": self.use_next,
            "use_context_window": self.use_context_window,
            "use_role_hint": self.use_role_hint,
            "use_ocr": self.use_ocr,
            "use_type_signals": self.use_type_signals,
        }


@dataclass(frozen=True)
class PromptResult:
    prompt: str
    trace: PromptTrace = field(default_factory=PromptTrace)


class FigurePromptBuilder:
    """Turn a figure's textbook context (and, optionally, figure-type-specific
    signals) into an instruction prompt that asks the captioning model for an
    educational, context-grounded description instead of a generic
    image-only one."""

    def build(
        self,
        figure_type: str,
        context: FigureContext,
        type_signals: TypeSignals | None = None,
        treatment: str | None = None,
    ) -> PromptResult:
        title_section, use_title = self._title_section(context)
        context_lines_section, trace_partial = self._context_lines_section(context)
        type_signal_section, use_type_signals = self._type_signal_section(type_signals)
        role_instruction, use_role_hint = self._role_instruction(context)

        if treatment == "tactile_graphic":
            output_instruction = _TACTILE_SPEC_INSTRUCTION
        elif figure_type in {"graph", "line_chart"}:
            output_instruction = _GRAPH_OUTPUT_INSTRUCTION
        else:
            output_instruction = _OUTPUT_INSTRUCTION
        sections = [
            _GOAL_INSTRUCTION,
            title_section,
            context_lines_section,
            type_signal_section,
            _CAPTION_PRIORITY_INSTRUCTION,
            _TYPE_INSTRUCTIONS.get(figure_type, _DEFAULT_TYPE_INSTRUCTION),
            role_instruction,
            _OCR_INSTRUCTION,
            _HALLUCINATION_GUARD,
            output_instruction,
        ]
        prompt = "\n\n".join(part for part in sections if part)
        trace = PromptTrace(
            use_title=use_title,
            use_caption=trace_partial["use_caption"],
            use_previous=trace_partial["use_previous"],
            use_next=trace_partial["use_next"],
            use_context_window=trace_partial["use_context_window"],
            use_role_hint=use_role_hint,
            use_ocr=bool(context.figure_ocr),
            use_type_signals=use_type_signals,
        )
        return PromptResult(prompt=prompt, trace=trace)

    def _title_section(self, context: FigureContext) -> tuple[str, bool]:
        lines = []
        chapter = context.chapter_title or context.page_title
        if chapter:
            lines.append(f"페이지/단원 제목: {chapter}")
        section = context.section_title
        subsection = context.subsection_title or context.nearest_section_title
        if section:
            lines.append(f"섹션 제목: {section}")
        if subsection and subsection != section:
            lines.append(f"소단원 제목: {subsection}")
        if not lines:
            return "", False
        return "\n".join(lines), True

    def _context_lines_section(self, context: FigureContext) -> tuple[str, dict[str, bool]]:
        lines = [
            f"{label}: {value}"
            for field_name, label in _CONTEXT_FIELD_LABELS
            if (value := getattr(context, field_name, None))
        ]
        use_caption = bool(context.caption)

        previous = context.previous_paragraphs or ((context.previous_paragraph,) if context.previous_paragraph else ())
        next_ = context.next_paragraphs or ((context.next_paragraph,) if context.next_paragraph else ())
        cut = next((i for i, text in enumerate(next_) if _SOLUTION_START.match(text or "")), None)
        if cut is not None:
            next_ = next_[:cut]
        if previous:
            label = "바로 앞 문단" if len(previous) == 1 else f"앞 문단(최근 {len(previous)}개)"
            lines.append(f"{label}: " + " / ".join(previous))
        if next_:
            label = "바로 뒤 문단" if len(next_) == 1 else f"뒤 문단(다음 {len(next_)}개)"
            lines.append(f"{label}: " + " / ".join(next_))
        if context.nearby_formula:
            lines.append(f"근처 수식: {context.nearby_formula}")
        if context.nearby_table:
            lines.append(f"근처 표: {context.nearby_table}")
        if context.figure_ocr:
            lines.append("그림 내부 텍스트: " + " / ".join(context.figure_ocr))

        trace = {
            "use_caption": use_caption,
            "use_previous": bool(previous),
            "use_next": bool(next_),
            "use_context_window": len(previous) > 1 or len(next_) > 1,
        }
        if not lines:
            body = "주변 교과서 문맥은 확인되지 않았습니다. 이미지에서 직접 확인되는 내용만 설명하세요."
            return body, trace
        return "교과서 문맥:\n" + "\n".join(lines), trace

    def _type_signal_section(self, type_signals: TypeSignals | None) -> tuple[str, bool]:
        if type_signals is None or not type_signals.has_any():
            return "", False
        lines = [
            f"{label}: {value}"
            for field_name, label in _TYPE_SIGNAL_LABELS
            if (value := getattr(type_signals, field_name, None))
        ]
        if type_signals.legend:
            lines.append("범례 후보: " + " / ".join(type_signals.legend))
        if type_signals.components:
            lines.append("구성 요소 후보: " + " / ".join(type_signals.components))
        if type_signals.objects:
            lines.append("대상 후보: " + " / ".join(type_signals.objects))
        if not lines:
            return "", False
        return (
            "이미지에서 기계적으로 추출된 후보(오탐 가능성이 있으므로 실제로 이미지에서 보일 때만 "
            "사용하세요):\n" + "\n".join(lines),
            True,
        )

    def _role_instruction(self, context: FigureContext) -> tuple[str, bool]:
        if not context.role_hint:
            return _DEFAULT_ROLE_HINT_INSTRUCTION, False
        instruction = _ROLE_HINT_INSTRUCTIONS.get(context.role_hint, _DEFAULT_ROLE_HINT_INSTRUCTION)
        return instruction, True
