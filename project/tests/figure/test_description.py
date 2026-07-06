import unittest

from src.analysis.figure.description import build_context_free_description


class FigureDescriptionTests(unittest.TestCase):
    def test_description_uses_only_extracted_axes_and_points(self):
        description = build_context_free_description(
            {
                "status": "success",
                "result": {
                    "kind": "figure",
                    "figure_type": "line_chart",
                    "title": None,
                    "x_axis": {"label": "시간", "unit": "초"},
                    "y_axis": {"label": "거리", "unit": "m"},
                    "series": [{"name": "A", "points": [{"x": 0, "y": 0}, {"x": 1, "y": 2}]}],
                },
            }
        )

        self.assertEqual(description["status"], "success")
        self.assertIn("X축은 시간, 단위는 초", description["short_text"])
        self.assertIn("A: 0에서 0, 1에서 2.", description["long_text"])
        self.assertFalse(description["context_used"])

    def test_partial_data_requires_original_review(self):
        description = build_context_free_description(
            {
                "status": "partial",
                "result": {
                    "kind": "figure",
                    "figure_type": "other",
                    "title": None,
                    "x_axis": None,
                    "y_axis": None,
                    "series": [],
                },
            }
        )

        self.assertEqual(description["status"], "partial")
        self.assertEqual(description["review_status"], "unreviewed")
        self.assertIsNotNone(description["transcription_notes"])


if __name__ == "__main__":
    unittest.main()
