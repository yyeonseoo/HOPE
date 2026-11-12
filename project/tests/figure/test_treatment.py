import unittest

from src.analysis.figure.treatment import (
    combine_with_decision_tree,
    parse_decision_answers,
    suggest_visual_treatment,
    treatment_from_answers,
)


def answers(**flags):
    keys = ["q1_text_repeats", "q2_words_better", "q3_needs_sight", "q4_hard_to_touch", "q5_needed_for_task"]
    return {key: {"answer": "예" if flags.get(key) else "아니오", "reason": "이유"} for key in keys}


class SuggestVisualTreatmentTests(unittest.TestCase):
    def test_graph_defaults_to_tactile_graphic(self):
        result = suggest_visual_treatment("graph", ["좌표평면 위의 점을 살펴보자."])
        self.assertEqual(result["suggested_visual_treatment"], "tactile_graphic")
        self.assertEqual(result["treatment_rationale"], ["TG2022-1.7"])

    def test_problem_figure_cites_answer_leak_rule(self):
        result = suggest_visual_treatment("line_chart", ["위 그래프를 보고 걸린 시간을 구하시오."])
        self.assertEqual(result["suggested_visual_treatment"], "tactile_graphic")
        self.assertIn("TG2022-7.3", result["treatment_rationale"])

    def test_tiny_figure_is_omitted_whatever_its_type(self):
        result = suggest_visual_treatment("illustration", area_ratio=0.0033)
        self.assertEqual(result["suggested_visual_treatment"], "omit")
        self.assertEqual(suggest_visual_treatment("graph", area_ratio=0.016)["suggested_visual_treatment"], "tactile_graphic")

    def test_icon_is_omitted(self):
        self.assertEqual(suggest_visual_treatment("icon")["suggested_visual_treatment"], "omit")

    def test_photo_already_named_by_the_body_is_omitted(self):
        result = suggest_visual_treatment("photo", ["오른쪽 그림은 일정한 속력으로 회전하는 대관람차이다. 다음 그림은"])
        self.assertEqual(result["suggested_visual_treatment"], "omit")
        self.assertIn("TG2022-1.8", result["treatment_rationale"])

    def test_photo_with_many_numbers_is_treated_as_a_chart(self):
        texts = ["연도별 수입", "24,757", "15,497", "15,333", "2012년 2013년"]
        result = suggest_visual_treatment("photo", figure_texts=texts)
        self.assertEqual(result["suggested_visual_treatment"], "tactile_graphic")

    def test_photo_and_unknown_fall_back_to_transcriber_note(self):
        self.assertEqual(suggest_visual_treatment("photo")["suggested_visual_treatment"], "transcriber_note")
        self.assertEqual(suggest_visual_treatment(None)["suggested_visual_treatment"], "transcriber_note")


class DecisionTreeTests(unittest.TestCase):
    def test_walks_the_guideline_questions_in_order(self):
        self.assertEqual(treatment_from_answers(answers(q1_text_repeats=True, q5_needed_for_task=True))["treatment"], "omit")
        self.assertEqual(treatment_from_answers(answers())["treatment"], "omit")  # not needed for the task
        self.assertEqual(treatment_from_answers(answers(q2_words_better=True, q5_needed_for_task=True))["treatment"], "transcriber_note")
        self.assertEqual(treatment_from_answers(answers(q5_needed_for_task=True))["treatment"], "tactile_graphic")

    def test_incomplete_answers_give_no_decision(self):
        self.assertIsNone(treatment_from_answers({"q1_text_repeats": {"answer": "예"}}))

    def test_parses_json_inside_model_text(self):
        parsed = parse_decision_answers('답변: {"q1_text_repeats": {"answer": "아니오", "reason": "x"}} 끝')
        self.assertEqual(parsed["q1_text_repeats"]["answer"], "아니오")
        self.assertIsNone(parse_decision_answers("JSON 없음"))

    def test_tree_decides_unless_a_certain_rule_applies(self):
        rule = suggest_visual_treatment("photo")
        combined = combine_with_decision_tree(rule, answers(q5_needed_for_task=True))
        self.assertEqual(combined["suggested_visual_treatment"], "tactile_graphic")
        self.assertEqual(combined["rule_suggestion"], "transcriber_note")
        self.assertEqual(len(combined["decision_tree"]), 5)

        strong = suggest_visual_treatment("graph", ["거리를 구하시오."])
        kept = combine_with_decision_tree(strong, answers(q2_words_better=True, q5_needed_for_task=True))
        self.assertEqual(kept["suggested_visual_treatment"], "tactile_graphic")
        self.assertEqual(kept["tree_suggestion"], "transcriber_note")


if __name__ == "__main__":
    unittest.main()
