import unittest

from src.analysis.figure.router import classify_figure_route


class FigureRouterTests(unittest.TestCase):
    def test_simple_axis_chart_routes_to_graph(self):
        evidence = {
            "words": [
                {"text": "시간", "x": 0.9, "y": 0.9},
                {"text": "거리", "x": 0.1, "y": 0.1},
            ],
            "paths": [
                {"points": [{"x": 0.1, "y": 0.1}, {"x": 0.8, "y": 0.8}]},
            ],
        }

        result = classify_figure_route(evidence)

        self.assertEqual(result["route_type"], "graph")
        self.assertEqual(result["usable_path_count"], 1)
        self.assertEqual(result["usable_path_indices"], [0])

    def test_colored_illustration_without_axes_routes_to_image(self):
        evidence = {
            "words": [{"text": "토끼", "x": 0.5, "y": 0.5}],
            "paths": [
                {"points": [{"x": 0.1, "y": 0.1}, {"x": 0.2, "y": 0.3}]},
            ],
        }

        result = classify_figure_route(evidence)

        self.assertEqual(result["route_type"], "image")
        self.assertIn("x_axis_label_missing_or_invalid", result["reasons"])

    def test_complex_multi_path_figure_routes_to_image(self):
        evidence = {
            "words": [
                {"text": "x", "x": 0.9, "y": 0.9},
                {"text": "y", "x": 0.1, "y": 0.1},
            ],
            "paths": [
                {"points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}
                for _ in range(10)
            ],
        }

        result = classify_figure_route(evidence)

        self.assertEqual(result["route_type"], "image")
        self.assertIn("data_path_count_out_of_range", result["reasons"])

    def test_corrupted_axis_label_routes_to_image(self):
        evidence = {
            "words": [
                {"text": ";2!;배", "x": 0.9, "y": 0.9},
                {"text": "y", "x": 0.1, "y": 0.1},
            ],
            "paths": [{"points": [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]}],
        }

        result = classify_figure_route(evidence)

        self.assertEqual(result["route_type"], "image")
        self.assertIn("x_axis_label_missing_or_invalid", result["reasons"])


if __name__ == "__main__":
    unittest.main()
