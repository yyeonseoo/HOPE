import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from PIL import Image

from src.analysis.figure import analyze_figure_blocks
from src.analysis.figure.captioners import CaptionOutput
from src.analysis.figure.openclip_classifier import RoutePrediction


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


class EvidenceAwareEngine(FakeChartEngine):
    def analyze(self, image_path, evidence=None):
        self.evidence = evidence
        return super().analyze(image_path)


class ContextAwareEngine(FakeChartEngine):
    def analyze(self, image_path, evidence=None, context=None):
        self.context = context
        output = super().analyze(image_path)
        output.update({
            "description_text": "주변 설명을 바탕으로 함수의 변화를 나타낸 그래프다.",
            "description_model": {"name": "context-captioner", "version": "test"},
            "description_confidence": 0.8,
            "generation_time_seconds": 0.1,
            "context_used": bool(context),
        })
        return output


class FigureAnalyzerTests(unittest.TestCase):
    def setUp(self):
        self.validator = Draft202012Validator(SCHEMA)
        self.page = {
            "page_id": 5,
            "blocks": [
                {
                    "block_id": "p5_b1", "type": "paragraph", "bbox": [0, 0, 90, 10],
                    "text": "함수 y=a/x의 그래프를 살펴보자.",
                },
                {
                    "block_id": "p5_b2",
                    "type": "figure",
                    "bbox": [10, 10, 90, 70],
                    "score": 0.91,
                    "detector": "doclayout_yolo",
                },
                {
                    "block_id": "p5_b3", "type": "caption", "bbox": [10, 72, 90, 78],
                    "text": "반비례 관계를 나타낸 그래프",
                },
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

    def test_figure_ocr_evidence_is_forwarded_to_supporting_engine(self):
        ocr_lines = [
            {"bbox": [20, 20, 40, 30], "text": "y=ax", "score": 0.99},
            {"bbox": [20, 35, 40, 45], "text": "(1, a)", "score": 0.95},
            {"bbox": [20, 50, 40, 60], "text": "unreliable", "score": 0.2},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            engine = EvidenceAwareEngine()
            analyze_figure_blocks(
                self.page,
                self._page_image(tmp),
                engine=engine,
                ocr_lines=ocr_lines,
            )

        self.assertEqual([item["text"] for item in engine.evidence], ["y=ax", "(1, a)"])
        self.assertEqual(engine.evidence[0]["id"], "t1")
        self.assertEqual(engine.evidence[0]["bbox"], [10, 10, 30, 20])
        self.assertEqual(
            engine.evidence[0]["relative_bbox"],
            [0.125, 1 / 6, 0.375, 1 / 3],
        )

    def test_engine_failure_is_isolated_to_the_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=FailingEngine())[0]

        self.assertEqual(result["analysis"]["status"], "failed")
        self.assertIsNone(result["analysis"]["result"])
        self.assertIn("test failure", " ".join(result["warnings"]))
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_nearby_textbook_context_is_forwarded_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ContextAwareEngine()
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=engine)[0]

        self.assertEqual([item["block_id"] for item in engine.context], ["p5_b3", "p5_b1"])
        self.assertTrue(result["description"]["context_used"])
        self.assertIn("p5_b1", result["context"]["nearby_block_ids"])
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_missing_page_image_returns_failed_schema_record(self):
        result = analyze_figure_blocks(self.page, None, engine=FakeChartEngine())[0]

        self.assertEqual(result["analysis"]["status"], "failed")
        self.assertIsNone(result["crop_path"])
        self.assertEqual(list(self.validator.iter_errors(result)), [])


class FakeRouteClassifier:
    model_name = "fake-openclip"
    model_version = "test-1"

    def __init__(self, route="graph"):
        self.route = route

    def classify(self, image_path):
        return RoutePrediction(route=self.route, confidence=0.82, scores={self.route: 0.82}, elapsed_seconds=0.01)


class FakePromptCaptioner:
    model_name = "fake-context-captioner"
    model_version = "test-2"

    def __init__(self, text="이 그래프는 토끼와 거북이의 경주를 나타낸다."):
        self.text = text
        self.calls = []

    def caption_with_prompt(self, image_path, prompt, evidence=None):
        self.calls.append((image_path, prompt, evidence))
        return CaptionOutput(
            text=self.text,
            confidence=0.88,
            generation_time_seconds=0.4,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class FakeGroundingScorer:
    def __init__(self, similarity=0.9):
        self.similarity = similarity
        self.calls = []

    def score(self, caption, context):
        self.calls.append((caption, context))
        return self.similarity if context else None


class ContextAwareEngineDouble:
    """Bundles classifier + captioner + grounding_scorer the way
    HuggingFaceFigureCaptionEngine does, so analyzer.py's duck-typed
    detection of the new pipeline picks it up without needing real models."""

    def __init__(self, captioner=None, classifier=None, grounding_scorer=None):
        self.classifier = classifier or FakeRouteClassifier()
        self.captioner = captioner or FakePromptCaptioner()
        self.grounding_scorer = grounding_scorer or FakeGroundingScorer()


class ContextAwarePipelineTests(unittest.TestCase):
    def setUp(self):
        self.validator = Draft202012Validator(SCHEMA)
        self.page = {
            "page_id": 5,
            "blocks": [
                {
                    "block_id": "p5_b1", "type": "paragraph", "bbox": [0, 0, 90, 10],
                    "text": "토끼와 거북이가 경주를 한다.",
                },
                {
                    "block_id": "p5_b2",
                    "type": "figure",
                    "bbox": [10, 10, 90, 70],
                    "score": 0.91,
                    "detector": "doclayout_yolo",
                },
                {
                    "block_id": "p5_b3", "type": "caption", "bbox": [10, 72, 90, 78],
                    "text": "반비례 관계를 나타낸 그래프",
                },
            ],
        }

    def _page_image(self, directory):
        path = Path(directory) / "page.png"
        Image.new("RGB", (100, 80), "white").save(path)
        return path

    def test_engine_with_prompt_capable_captioner_uses_the_new_pipeline(self):
        engine = ContextAwareEngineDouble()
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=engine)[0]

        self.assertEqual(result["figure_type"], "graph")
        self.assertEqual(result["education_context"]["caption"], "반비례 관계를 나타낸 그래프")
        self.assertEqual(result["education_context"]["previous_paragraph"], "토끼와 거북이가 경주를 한다.")
        self.assertIn("grounding", result)
        self.assertEqual(result["confidence"], 0.88)
        self.assertEqual(result["description"]["short_text"], engine.captioner.text)
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_prompt_sent_to_captioner_carries_the_built_context(self):
        engine = ContextAwareEngineDouble()
        with tempfile.TemporaryDirectory() as tmp:
            analyze_figure_blocks(self.page, self._page_image(tmp), engine=engine)

        prompt = engine.captioner.calls[0][1]
        self.assertIn("토끼와 거북이가 경주를 한다.", prompt)
        self.assertIn("반비례 관계를 나타낸 그래프", prompt)

    def test_legacy_engine_without_captioner_attribute_is_unaffected(self):
        # Same fake as the pre-existing FakeChartEngine tests above: no
        # `.classifier`/`.captioner`, so this must keep using the old
        # `engine.analyze(...)` path untouched.
        engine = FakeChartEngine()
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=engine)[0]

        self.assertNotIn("figure_type", result)
        self.assertNotIn("education_context", result)
        self.assertNotIn("grounding", result)
        self.assertEqual(list(self.validator.iter_errors(result)), [])

    def test_generation_failure_falls_back_to_legacy_engine_path(self):
        class ExplodingCaptioner(FakePromptCaptioner):
            def caption_with_prompt(self, image_path, prompt, evidence=None):
                raise RuntimeError("model exploded")

        engine = ContextAwareEngineDouble(captioner=ExplodingCaptioner())
        with tempfile.TemporaryDirectory() as tmp:
            result = analyze_figure_blocks(self.page, self._page_image(tmp), engine=engine)[0]

        # ContextAwareEngineDouble itself has no .analyze(), so run_figure_engine's
        # generic AttributeError path reports failure instead of crashing the page.
        self.assertEqual(result["analysis"]["status"], "failed")
        self.assertEqual(list(self.validator.iter_errors(result)), [])


if __name__ == "__main__":
    unittest.main()
