import unittest

from src.analysis.figure.pp_chart2table import PPChart2TableEngine, parse_chart_table


class FakeResult:
    json = {
        "res": {
            "result": "연도 | 매출 | 비용\n2023 | 1,200 | 800\n2024 | 1500.5 | 900"
        }
    }


class FakeModel:
    def predict(self, input, batch_size):
        self.input = input
        self.batch_size = batch_size
        return [FakeResult()]


class PPChart2TableTests(unittest.TestCase):
    def test_pipe_table_is_converted_to_multiple_series(self):
        parsed = parse_chart_table("연도 | 매출 | 비용\n2023 | 1,200 | 800\n2024 | 1500.5 | 900")

        self.assertEqual(parsed["x_axis"], {"label": "연도", "unit": None})
        self.assertEqual(parsed["series"][0]["name"], "매출")
        self.assertEqual(parsed["series"][0]["points"][0], {"x": 2023, "y": 1200})
        self.assertEqual(parsed["series"][0]["points"][1], {"x": 2024, "y": 1500.5})
        self.assertEqual(parsed["series"][1]["name"], "비용")

    def test_engine_adapts_paddle_result_without_loading_real_model(self):
        model = FakeModel()
        result = PPChart2TableEngine(model=model).analyze("chart.png")

        self.assertEqual(model.input, {"image": "chart.png"})
        self.assertEqual(result["model"]["name"], "PP-Chart2Table")
        self.assertEqual(result["figure_type"], "other")
        self.assertEqual(len(result["series"]), 2)

    def test_short_unstructured_output_does_not_invent_data(self):
        parsed = parse_chart_table("그래프")

        self.assertEqual(parsed["series"], [])
        self.assertTrue(parsed["warnings"])


if __name__ == "__main__":
    unittest.main()
