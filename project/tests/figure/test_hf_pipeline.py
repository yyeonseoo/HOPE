import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.analysis.figure.captioners import CaptionOutput, ChatGPTCaptioner, Florence2ImageCaptioner
from src.analysis.figure.hf_pipeline import HuggingFaceFigureCaptionEngine, create_openai_figure_engine
from src.analysis.figure.openclip_classifier import RoutePrediction


class FakeClassifier:
    model_name = "fake-openclip"
    model_version = "test-1"

    def __init__(self, route):
        self.route = route

    def classify(self, image_path):
        return RoutePrediction(
            route=self.route,
            confidence=0.82,
            scores={"graph": 0.82, "image": 0.18},
            elapsed_seconds=0.01,
        )


class FakeCaptioner:
    model_name = "fake-captioner"
    model_version = "test-2"

    def __init__(self, text):
        self.text = text
        self.calls = 0

    def caption(self, image_path, figure_type):
        self.calls += 1
        self.figure_type = figure_type
        return CaptionOutput(
            text=self.text,
            confidence=0.71,
            generation_time_seconds=0.25,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class EvidenceCaptioner(FakeCaptioner):
    def caption(self, image_path, figure_type, evidence=None):
        self.evidence = evidence
        return super().caption(image_path, figure_type)


class ContextCaptioner(FakeCaptioner):
    def caption(self, image_path, figure_type, evidence=None, context=None):
        self.context = context
        base = super().caption(image_path, figure_type)
        return CaptionOutput(
            text=base.text,
            confidence=base.confidence,
            generation_time_seconds=base.generation_time_seconds,
            model_name=base.model_name,
            model_version=base.model_version,
            context_block_ids=tuple(item["block_id"] for item in context or []),
        )


class InlineContextCaptioner(ContextCaptioner):
    handles_context_inline = True


class FakeGroundingScorer:
    def __init__(self, similarity):
        self.similarity = similarity
        self.calls = []

    def score(self, caption, context):
        self.calls.append((caption, context))
        return self.similarity


class HuggingFacePipelineTests(unittest.TestCase):
    def _image(self, directory):
        path = Path(directory) / "figure.png"
        Image.new("RGB", (40, 30), "white").save(path)
        return path

    def test_graph_route_is_passed_to_captioner(self):
        captioner = FakeCaptioner("선그래프가 증가한다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("graph"), captioner)
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp))

        self.assertEqual(captioner.calls, 1)
        self.assertEqual(captioner.figure_type, "graph")
        self.assertEqual(output["figure_type"], "graph")
        self.assertEqual(output["confidence"], 0.82)
        self.assertEqual(output["description_confidence"], 0.71)
        self.assertEqual(output["generation_time_seconds"], 0.25)
        self.assertTrue(output["description_only"])

    def test_illustration_route_is_passed_to_captioner(self):
        image = FakeCaptioner("두 사람이 걷는 삽화다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("illustration"), image)
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp))

        self.assertEqual(image.calls, 1)
        self.assertEqual(image.figure_type, "illustration")
        self.assertEqual(output["figure_type"], "illustration")
        self.assertEqual(output["description_model"]["name"], "fake-captioner")

    def test_grounding_evidence_is_passed_to_supporting_captioner(self):
        captioner = EvidenceCaptioner("직선 옆에 y=ax가 표시되어 있다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("graph"), captioner)
        with tempfile.TemporaryDirectory() as tmp:
            engine.analyze(self._image(tmp), evidence=["y=ax", "(1, a)"])

        self.assertEqual(captioner.evidence, ["y=ax", "(1, a)"])

    def test_context_is_passed_and_reported(self):
        captioner = ContextCaptioner("문맥을 이용한 설명이다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("graph"), captioner)
        context = [{"block_id": "p2_b4", "type": "paragraph", "score": 1.0, "text": "y=a/x의 그래프"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp), context=context)

        self.assertIsNone(captioner.context)
        self.assertTrue(output["context_used"])
        self.assertEqual(output["context_block_ids"], ["p2_b4"])

    def test_inline_context_captioner_receives_context_in_single_call(self):
        captioner = InlineContextCaptioner("문맥을 이용한 설명이다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("graph"), captioner)
        context = [{"block_id": "p2_b4", "type": "paragraph", "score": 1.0, "text": "y=a/x의 그래프"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp), context=context)

        self.assertEqual(captioner.context, context)
        self.assertTrue(output["context_used"])
        self.assertEqual(captioner.calls, 1)

    def test_photo_ignores_unrelated_paragraph_context(self):
        captioner = ContextCaptioner("해안 도로가 보이는 사진이다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("photo"), captioner)
        context = [{"block_id": "p2_b4", "type": "paragraph", "score": 1.0, "text": "전기 요금의 정비례 관계"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp), context=context)

        self.assertIsNone(captioner.context)
        self.assertFalse(output["context_used"])

    def test_photo_keeps_direct_caption_context(self):
        captioner = ContextCaptioner("해안 도로가 보이는 사진이다.")
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("photo"), captioner)
        context = [{"block_id": "p2_b5", "type": "caption", "score": 1.0, "text": "해안 도로"}]
        with tempfile.TemporaryDirectory() as tmp:
            engine.analyze(self._image(tmp), context=context)

        self.assertIsNone(captioner.context)

    def test_low_topic_similarity_adds_a_warning(self):
        captioner = ContextCaptioner("도로 위의 자동차 사진이다.")
        grounding_scorer = FakeGroundingScorer(0.05)
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("graph"), captioner, grounding_scorer)
        context = [{"block_id": "p2_b4", "type": "paragraph", "score": 1.0, "text": "x와 y가 정비례하는 관계"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp), context=context)

        self.assertTrue(any("topical similarity" in warning for warning in output["warnings"]))
        self.assertEqual(grounding_scorer.calls[0][0], "도로 위의 자동차 사진이다.")

    def test_normal_topic_similarity_adds_no_warning(self):
        captioner = ContextCaptioner("정비례 관계를 나타낸 표이다.")
        grounding_scorer = FakeGroundingScorer(0.8)
        engine = HuggingFaceFigureCaptionEngine(FakeClassifier("graph"), captioner, grounding_scorer)
        context = [{"block_id": "p2_b4", "type": "paragraph", "score": 1.0, "text": "x와 y가 정비례하는 관계"}]
        with tempfile.TemporaryDirectory() as tmp:
            output = engine.analyze(self._image(tmp), context=context)

        self.assertFalse(any("topical similarity" in warning for warning in output["warnings"]))

    def test_openai_factory_is_lazy_and_uses_chatgpt(self):
        engine = create_openai_figure_engine(device="cpu", api_key="test-key")

        self.assertIsInstance(engine.captioner, ChatGPTCaptioner)
        self.assertEqual(engine.captioner.model_name, "gpt-5")
        self.assertEqual(engine.captioner.api_key, "test-key")
        self.assertIsNone(engine.captioner._client)
        self.assertEqual(engine.classifier.device_request, "cpu")
        self.assertIsNone(engine.classifier._model)
        self.assertEqual(engine.grounding_scorer.device_request, "cpu")
        self.assertIsNone(engine.grounding_scorer._model)

    def test_florence_loader_uses_native_class_without_remote_code(self):
        calls = {}

        class FakeLoadedModel:
            def to(self, device):
                calls["device"] = device
                return self

            def eval(self):
                calls["eval"] = True

        class FakeModelClass:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls["model"] = (model_name, kwargs)
                return FakeLoadedModel()

        class FakeProcessorClass:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls["processor"] = (model_name, kwargs)
                return object()

        fake_transformers = SimpleNamespace(
            AutoProcessor=FakeProcessorClass,
            Florence2ForConditionalGeneration=FakeModelClass,
        )
        fake_torch = SimpleNamespace(float16="float16", float32="float32")
        captioner = Florence2ImageCaptioner(device="cpu", revision="test-revision")

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            captioner._load(fake_torch)

        self.assertEqual(calls["model"][0], "florence-community/Florence-2-base")
        self.assertEqual(calls["model"][1]["dtype"], "float32")
        self.assertNotIn("trust_remote_code", calls["model"][1])
        self.assertNotIn("trust_remote_code", calls["processor"][1])
        self.assertEqual(calls["processor"][1]["revision"], "test-revision")
        self.assertTrue(calls["eval"])


if __name__ == "__main__":
    unittest.main()
