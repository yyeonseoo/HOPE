import unittest

from src.analysis.figure.context_builder import FigureContext
from src.analysis.figure.prompt_builder import FigurePromptBuilder
from src.analysis.figure.type_signals import TypeSignals


class FigurePromptBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = FigurePromptBuilder()

    def _prompt(self, figure_type, context, type_signals=None):
        return self.builder.build(figure_type, context, type_signals).prompt

    def test_prompt_asks_for_purpose_not_a_visual_inventory(self):
        context = FigureContext(caption="반비례 관계를 나타낸 그래프")
        prompt = self._prompt("graph", context)

        self.assertIn("무엇을 설명하기 위해", prompt)

    def test_prompt_includes_available_context_fields(self):
        context = FigureContext(
            page_title="3단원 좌표평면과 그래프",
            previous_paragraph="토끼와 거북이가 경주를 한다.",
            caption="반비례 관계를 나타낸 그래프",
        )
        prompt = self._prompt("graph", context)

        self.assertIn("3단원 좌표평면과 그래프", prompt)
        self.assertIn("토끼와 거북이가 경주를 한다.", prompt)
        self.assertIn("반비례 관계를 나타낸 그래프", prompt)

    def test_prompt_omits_missing_context_fields(self):
        context = FigureContext(caption="반비례 관계를 나타낸 그래프")
        prompt = self._prompt("graph", context)

        self.assertNotIn("바로 앞 문단:", prompt)
        self.assertNotIn("근처 표:", prompt)

    def test_prompt_states_caption_has_priority(self):
        context = FigureContext(caption="반비례 관계를 나타낸 그래프")
        prompt = self._prompt("graph", context)

        self.assertIn("캡션", prompt)
        self.assertIn("가장 신뢰도가 높", prompt)

    def test_prompt_includes_figure_ocr_text(self):
        context = FigureContext(figure_ocr=("x축", "y축", "범례"))
        prompt = self._prompt("graph", context)

        self.assertIn("x축", prompt)
        self.assertIn("y축", prompt)
        self.assertIn("범례", prompt)

    def test_prompt_includes_hallucination_guard_rules(self):
        context = FigureContext()
        prompt = self._prompt("photo", context)

        self.assertIn("추측하지 않는다", prompt)
        self.assertIn("숫자를 임의로 생성하지 않는다", prompt)
        self.assertIn("모르면 모른다고 한다", prompt)

    def test_each_figure_type_gets_a_distinct_branch(self):
        context = FigureContext(caption="c")
        prompts = {
            figure_type: self._prompt(figure_type, context)
            for figure_type in ("graph", "table", "mathematical_diagram", "illustration", "photo", "icon")
        }
        self.assertEqual(len(set(prompts.values())), len(prompts))

    def test_graph_prompt_asks_about_trend_and_extremes(self):
        prompt = self._prompt("graph", FigureContext())

        self.assertIn("증가", prompt)
        self.assertIn("최댓값", prompt)
        self.assertIn("추세", prompt)

    def test_unknown_figure_type_falls_back_to_illustration_branch(self):
        context = FigureContext()
        fallback_prompt = self._prompt("unknown", context)
        illustration_prompt = self._prompt("illustration", context)

        self.assertEqual(fallback_prompt, illustration_prompt)

    def test_no_context_available_still_produces_a_usable_prompt(self):
        prompt = self._prompt("photo", FigureContext())

        self.assertIn("이미지에서 직접 확인되는 내용만", prompt)

    def test_context_window_renders_multiple_paragraphs(self):
        context = FigureContext(
            previous_paragraphs=("첫 번째 앞 문단", "두 번째 앞 문단"),
            next_paragraphs=("첫 번째 뒤 문단",),
        )
        prompt = self._prompt("graph", context)

        self.assertIn("첫 번째 앞 문단", prompt)
        self.assertIn("두 번째 앞 문단", prompt)
        self.assertIn("첫 번째 뒤 문단", prompt)

    def test_type_signals_are_included_when_present(self):
        signals = TypeSignals(x_axis="시간", y_axis="거리", trend="increasing")
        prompt = self._prompt("graph", FigureContext(), signals)

        self.assertIn("시간", prompt)
        self.assertIn("거리", prompt)
        self.assertIn("increasing", prompt)

    def test_role_hint_problem_forbids_revealing_the_answer(self):
        context = FigureContext(role_hint="problem")
        prompt = self._prompt("graph", context)

        self.assertIn("정답을 추론하거나 알려주지", prompt)

    def test_role_hint_example_focuses_on_solution_process(self):
        context = FigureContext(role_hint="example")
        prompt = self._prompt("graph", context)

        self.assertIn("풀이 과정", prompt)

    def test_trace_reflects_what_was_actually_used(self):
        context = FigureContext(caption="c", previous_paragraph="p", role_hint="problem")
        result = self.builder.build("graph", context)

        self.assertTrue(result.trace.use_caption)
        self.assertTrue(result.trace.use_previous)
        self.assertTrue(result.trace.use_role_hint)
        self.assertFalse(result.trace.use_next)
        self.assertFalse(result.trace.use_title)

    def test_trace_context_window_flag_only_true_with_more_than_one_paragraph(self):
        single = self.builder.build("graph", FigureContext(previous_paragraphs=("only one",)))
        windowed = self.builder.build(
            "graph", FigureContext(previous_paragraphs=("first", "second"))
        )

        self.assertFalse(single.trace.use_context_window)
        self.assertTrue(windowed.trace.use_context_window)

    def test_trace_use_ocr_and_type_signals(self):
        result = self.builder.build(
            "graph", FigureContext(figure_ocr=("x축",)), TypeSignals(trend="increasing")
        )

        self.assertTrue(result.trace.use_ocr)
        self.assertTrue(result.trace.use_type_signals)


if __name__ == "__main__":
    unittest.main()
