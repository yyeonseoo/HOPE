import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image

from src.analysis.figure import analyze_figure_blocks


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = json.loads((PROJECT_ROOT / "schemas" / "block_analysis.schema.json").read_text(encoding="utf-8"))


class FakeChartEngine:
    model_name = "fake-chart-model"
    model_version = "test-1"

    def analyze(self, image_path):
        self.last_image_path = image_path
        return {
            "confidence": 0.87,
            "figure_type": "line_chart",
            "title": "연도별 매출",
            "x_axis": {"label": "연도", "unit": "년"},
            "y_axis": {"label": "매출", "unit": "억원"},
            "series": [{"name": "매출", "points": [{"x": "2024", "y": 100}]}],
        }


class FailingEngine:
    model_name = "failing-model"
    model_version = "test-1"

    def analyze(self, image_path):
        raise RuntimeError("test failure")


class FakeCaptionEngine(FakeChartEngine):
    def analyze(self, image_path):
        output = super().analyze(image_path)
        output.update(
            {
                "description_text": "연도별 매출을 나타낸 선그래프다.",
                "description_model": {"name": "fake-captioner", "version": "test-2"},
                "description_confidence": 0.76,
                "generation_time_seconds": 0.42,
            }
        )
        return output


class FigureAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.validator = Draft202012Validator(SCHEMA)
        self.page = {
            "page_id": 5,
            "blocks": [
                {"block_id": "p5_b1", "type": "paragraph", "bbox": [0, 0, 90, 10]},
                {
                    "block_id": "p5_b2",
                    "type": "figure",
                    "bbox": [10, 10, 90, 70],
                    "score": 0.91,
                    "detector": "doclayout_yolo",
                },
                {"block_id": "p5_b3", "type": "caption", "bbox": [10, 72, 90, 78]},
            ],
        }

    def _page_image(self, directory):
        path = Path(directory) / "page.png"
        Image.new("RGB", (100, 80), "white").save(path)
        return path

    def test_only_figure_blocks_are_analyzed_and_schema_is_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = FakeChartEngine()
            results = analyze_figure_blocks(self.page, self._page_image(tmp), engine=engine)

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["analysis"]["status"], "success")
        self.assertEqual(result["analysis"]["result"]["figure_type"], "line_chart")
        self.assertEqual(result["context"]["caption_block_id"], "p5_b3")
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_missing_engine_returns_honest_partial_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp))[0]

        self.assertEqual(result["analysis"]["status"], "partial")
        self.assertEqual(result["analysis"]["result"]["figure_type"], "unknown")
        self.assertIn("not configured", " ".join(result["warnings"]))
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_generated_description_records_confidence_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=FakeCaptionEngine())[0]

        self.assertEqual(result["description"]["short_text"], "연도별 매출을 나타낸 선그래프다.")
        self.assertEqual(result["description"]["confidence"], 0.76)
        self.assertEqual(result["description"]["generation_time_seconds"], 0.42)
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_engine_failure_is_isolated_to_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=FailingEngine())[0]

        self.assertEqual(result["analysis"]["status"], "failed")
        self.assertIsNone(result["analysis"]["result"])
        self.assertIn("test failure", " ".join(result["warnings"]))
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_missing_page_image_returns_failed_schema_record(self):
        result = analyze_figure_blocks(self.page, None, engine=FakeChartEngine())[0]

        self.assertEqual(result["analysis"]["status"], "failed")
        self.assertIsNone(result["crop_path"])
        self.assertEqual(list(self.validator.iter_errors(result)), [])


if __name__ == "__main__":
    unittest.main()
