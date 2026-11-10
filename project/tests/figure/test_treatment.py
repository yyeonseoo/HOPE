import unittest

from src.analysis.figure.treatment import suggest_visual_treatment


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


if __name__ == "__main__":
    unittest.main()
