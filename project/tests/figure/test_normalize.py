import unittest

from src.analysis.figure.classifier import normalize_figure_type
from src.analysis.figure.normalize import build_figure_analysis


class FigureNormalizationTests(unittest.TestCase):
    def test_known_alias_is_normalized(self):
        self.assertEqual(normalize_figure_type("line-graph"), "line_chart")

    def test_unknown_model_label_is_not_guessed(self):
        self.assertEqual(normalize_figure_type("economic_magic_plot"), "unknown")

    def test_chart_without_extracted_data_is_partial(self):
        output = build_figure_analysis(
            {
                "model": {"name": "chart-model", "version": "1"},
                "confidence": 0.9,
                "figure_type": "bar_chart",
            }
        )

        self.assertEqual(output["analysis"]["status"], "partial")
        self.assertEqual(output["analysis"]["result"]["series"], [])
        self.assertTrue(output["warnings"])

    def test_other_figure_without_any_semantics_is_partial(self):
        output = build_figure_analysis(
            {
                "model": {"name": "figure-model", "version": "1"},
                "figure_type": "other",
                "title": None,
                "x_axis": None,
                "y_axis": None,
                "series": [],
            }
        )

        self.assertEqual(output["analysis"]["status"], "partial")
        self.assertTrue(output["warnings"])

    def test_series_keeps_only_explicit_points(self):
        output = build_figure_analysis(
            {
                "model": {"name": "chart-model", "version": "1"},
                "confidence": 2.0,
                "figure_type": "line_chart",
                "x_axis": {"label": "연도", "unit": "년"},
                "series": [
                    {
                        "name": "매출",
                        "points": [
                            {"x": "2024", "y": 100},
                            {"x": "missing-y"},
                        ],
                    }
                ],
            }
        )

        self.assertEqual(output["analysis"]["status"], "success")
        self.assertEqual(output["analysis"]["confidence"], 1.0)
        self.assertEqual(output["analysis"]["result"]["series"][0]["points"], [{"x": "2024", "y": 100}])


if __name__ == "__main__":
    unittest.main()
