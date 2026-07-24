import unittest

from src.analysis.figure.captioners import CaptionOutput
from src.analysis.figure.generator import FigureDescriptionGenerator


class FakePromptCaptioner:
    model_name = "fake-context-captioner"
    model_version = "test-1"

    def __init__(self, text="이 그래프는 토끼와 거북이의 경주를 나타낸다."):
        self.text = text
        self.calls = []

    def caption_with_prompt(self, image_path, prompt, evidence=None):
        self.calls.append((image_path, prompt, evidence))
        return CaptionOutput(
            text=self.text,
            confidence=0.9,
            generation_time_seconds=0.5,
            model_name=self.model_name,
            model_version=self.model_version,
        )


class LegacyCaptioner:
    model_name = "legacy-captioner"
    model_version = None

    def caption(self, image_path, figure_type):
        raise AssertionError("legacy caption() should not be called by the generator")


class FigureDescriptionGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.generator = FigureDescriptionGenerator()

    def test_supports_returns_true_for_prompt_capable_captioner(self):
        self.assertTrue(self.generator.supports(FakePromptCaptioner()))

    def test_supports_returns_false_for_legacy_captioner(self):
        self.assertFalse(self.generator.supports(LegacyCaptioner()))

    def test_generate_forwards_prompt_and_evidence(self):
        captioner = FakePromptCaptioner()
        evidence = [{"id": "t1", "text": "y=ax"}]

        result = self.generator.generate(captioner, "crop.png", "prompt text", evidence=evidence)

        self.assertEqual(captioner.calls, [("crop.png", "prompt text", evidence)])
        self.assertEqual(result.text, "이 그래프는 토끼와 거북이의 경주를 나타낸다.")
        self.assertEqual(result.confidence, 0.9)
        self.assertEqual(result.generation_time_seconds, 0.5)
        self.assertEqual(result.model_name, "fake-context-captioner")
        self.assertEqual(result.warnings, [])

    def test_generate_returns_honest_empty_result_for_unsupported_captioner(self):
        result = self.generator.generate(LegacyCaptioner(), "crop.png", "prompt text")

        self.assertEqual(result.text, "")
        self.assertIsNone(result.confidence)
        self.assertTrue(result.warnings)


if __name__ == "__main__":
    unittest.main()
