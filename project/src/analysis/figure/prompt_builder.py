from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .context_builder import FigureContext
from .type_signals import TypeSignals

_GOAL_INSTRUCTION = (
    "당신은 한국 경제수학 교과서의 그림을 설명하는 교육 콘텐츠 작성자입니다. "
    "이 그림에 무엇이 그려져 있는지를 나열하지 말고, 이 그림이 교과서에서 '무엇을 설명하기 위해' "
    "사용되었는지를 중심으로 설명하세요. "
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
    "- 이미지가 실제로는 작은 색상 도형, 구분선, 불릿 아이콘처럼 정보가 없는 단순 장식이라면, "
    "주변 문맥을 끌어와 억지로 의미나 주제를 부여하지 말고 단순 장식용 아이콘이라는 사실 그대로 설명한다."
)

_OUTPUT_INSTRUCTION = (
    "목록이나 마크다운, 제목 없이 자연스러운 한국어 2~4문장의 문단 하나로만 답하세요. "
    "문장은 교과서 본문처럼 '~다/~이다/~한다'로 끝나는 평서형으로 쓰고, '~습니다', '~해요'와 같은 "
    "존댓말 종결어미는 사용하지 마세요."
)

_TYPE_INSTRUCTIONS: dict[str, str] = {
    "graph": (
        "그래프 유형 지침: 이 그래프가 무엇과 무엇을 비교하는지, 값이 증가하는지 감소하는지, "
        "최댓값과 최솟값은 어디인지, 전체적인 추세가 어떤지를 중심으로 설명하세요."
    ),
    "line_chart": (
        "그래프 유형 지침: 이 그래프가 무엇과 무엇을 비교하는지, 값이 증가하는지 감소하는지, "
        "최댓값과 최솟값은 어디인지, 전체적인 추세가 어떤지를 중심으로 설명하세요."
    ),
    "bar_chart": (
        "그래프 유형 지침: 막대들이 나타내는 항목을 서로 비교하고, 어떤 항목이 가장 크거나 작은지, "
        "전체적인 증가·감소 추세가 있는지를 중심으로 설명하세요."
    ),
    "pie_chart": (
        "그래프 유형 지침: 전체에서 각 부분이 차지하는 비중을 비교하고, 가장 큰 부분과 작은 부분을 "
        "중심으로 설명하세요."
    ),
    "scatter_plot": (
        "그래프 유형 지침: 점들이 흩어진 전반적인 경향과 두 값 사이의 관계(비례/반비례 등으로 보이는 "
        "패턴)를 중심으로 설명하세요."
    ),
    "table": (
        "표 유형 지침: 표의 행과 열이 나타내는 항목, 항목들 사이의 관계와 값의 변화를 중심으로 "
        "설명하세요."
    ),
    "mathematical_diagram": (
        "수학 도식 유형 지침: 도형의 구조, 각 구성 요소 사이의 관계, 표시된 기호나 문자가 의미하는 "
        "바를 중심으로 설명하세요."
    ),
    "diagram": (
        "도식 유형 지침: 구성 요소들의 구조, 서로의 관계, 흐름을 중심으로 설명하세요."
    ),
    "illustration": (
        "삽화 유형 지침: 이 삽화가 어떤 개념이나 상황을 설명하기 위해 그려졌는지를 중심으로 "
        "설명하세요."
    ),
    "photo": (
        "사진 유형 지침: 사진이 보여주는 상황이 무엇이고, 그것이 교과서 내용과 어떻게 연결되는지를 "
        "중심으로 설명하세요."
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
    "example": "이 그림은 '예제' 영역에 속합니다. 풀이 과정을 이해하는 데 필요한 정보를 중심으로 설명하세요.",
    "solution": "이 그림은 '풀이/확인' 영역에 속합니다. 풀이 과정을 이해하는 데 필요한 정보를 중심으로 설명하세요.",
    "problem": (
        "이 그림은 '문제' 영역에 속합니다. 정답을 추론하거나 알려주지 말고, 문제를 이해하는 데 "
        "필요한 정보만 제공하세요."
    ),
}
_DEFAULT_ROLE_HINT_INSTRUCTION = "이 그림은 개념 설명 영역에 속합니다. 개념 자체를 중심으로 설명하세요."

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
    ) -> PromptResult:
        title_section, use_title = self._title_section(context)
        context_lines_section, trace_partial = self._context_lines_section(context)
        type_signal_section, use_type_signals = self._type_signal_section(type_signals)
        role_instruction, use_role_hint = self._role_instruction(context)

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
            _OUTPUT_INSTRUCTION,
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
