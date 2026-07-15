import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.analysis.figure.captioners import CaptionOutput, Florence2ImageCaptioner, Qwen3VLCaptioner
from src.analysis.figure.hf_pipeline import HuggingFaceFigureCaptionEngine, create_huggingface_figure_engine
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

    def test_factory_is_lazy_and_uses_qwen(self):
        engine = create_huggingface_figure_engine(device="cpu")

        self.assertEqual(engine.captioner.model_name, "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(engine.classifier.device_request, "cpu")
        self.assertIsNone(engine.classifier._model)

    def test_qwen_loader_uses_native_transformers_class(self):
        calls = {}

        class FakeParameter:
            dtype = "bfloat16"

        class FakeLoadedModel:
            def to(self, device):
                calls["device"] = device
                return self

            def eval(self):
                calls["eval"] = True

            def parameters(self):
                return iter([FakeParameter()])

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
            Qwen3VLForConditionalGeneration=FakeModelClass,
        )
        captioner = Qwen3VLCaptioner(device="cpu", revision="test-revision")

        with patch.dict("sys.modules", {"transformers": fake_transformers}):
            captioner._load(SimpleNamespace())

        self.assertEqual(calls["model"][0], "Qwen/Qwen3-VL-2B-Instruct")
        self.assertEqual(calls["model"][1]["dtype"], "auto")
        self.assertEqual(calls["processor"][1]["revision"], "test-revision")
        self.assertTrue(calls["eval"])

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
