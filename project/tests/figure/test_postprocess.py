import unittest

from src.analysis.figure.context_builder import FigureContext
from src.analysis.figure.generator import GeneratedDescription
from src.analysis.figure.grounding import GroundingScores
from src.analysis.figure.postprocess import build_context_aware_figure_record, split_additive_fields
from src.analysis.figure.prompt_builder import PromptTrace
from src.analysis.figure.summary import derive_summary
from src.analysis.figure.type_signals import TypeSignals


class PostprocessTests(unittest.TestCase):
    def setUp(self):
        self.figure_context = FigureContext(
            caption="반비례 관계를 나타낸 그래프",
            previous_paragraph="토끼와 거북이가 경주를 한다.",
        )
        self.description = GeneratedDescription(
            text="이 그래프는 토끼와 거북이의 경주에서 시간에 따른 이동 거리를 나타낸다.",
            confidence=0.87,
            generation_time_seconds=0.6,
            model_name="gpt-4o-context-aware",
            model_version=None,
            warnings=[],
        )
        self.grounding_scores = GroundingScores(caption_score=0.95, context_score=0.9, overall_score=0.93)

    def test_record_contains_schema_fields_and_additive_fields(self):
        record = build_context_aware_figure_record(
            figure_type="graph",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.82,
            description=self.description,
            grounding_scores=self.grounding_scores,
            figure_context=self.figure_context,
        )

        self.assertEqual(record["analysis"]["status"], "success")
        self.assertEqual(record["analysis"]["result"]["figure_type"], "graph")
        self.assertEqual(record["description"]["short_text"], self.description.text)
        self.assertEqual(record["figure_type"], "graph")
        self.assertEqual(record["education_context"]["caption"], "반비례 관계를 나타낸 그래프")
        self.assertEqual(record["grounding"], {"caption_score": 0.95, "context_score": 0.9, "overall_score": 0.93})
        self.assertEqual(record["confidence"], 0.87)
        self.assertEqual(record["summary"], derive_summary(self.description.text))
        self.assertTrue(record["context_used"])
        self.assertEqual(record["context_source"]["caption_block_id"], None)
        self.assertEqual(record["warning_codes"], [])
        self.assertNotIn("prompt_trace", record)
        self.assertNotIn("type_signals", record)

    def test_warning_codes_flag_missing_caption_and_low_grounding(self):
        context = FigureContext(previous_paragraph="어떤 문단")
        record = build_context_aware_figure_record(
            figure_type="graph",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.3,
            description=self.description,
            grounding_scores=GroundingScores(caption_score=None, context_score=0.1, overall_score=0.1),
            figure_context=context,
        )

        self.assertIn("no_caption", record["warning_codes"])
        self.assertIn("grounding_mismatch", record["warning_codes"])
        self.assertIn("figure_type_uncertain", record["warning_codes"])

    def test_prompt_trace_and_type_signals_are_attached_when_provided(self):
        trace = PromptTrace(use_caption=True, use_previous=True)
        signals = TypeSignals(x_axis="시간", trend="increasing")
        record = build_context_aware_figure_record(
            figure_type="graph",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.82,
            description=self.description,
            grounding_scores=self.grounding_scores,
            figure_context=self.figure_context,
            prompt_trace=trace,
            type_signals=signals,
        )

        self.assertEqual(record["prompt_trace"]["use_caption"], True)
        self.assertEqual(record["type_signals"]["x_axis"], "시간")

    def test_empty_type_signals_are_not_attached(self):
        record = build_context_aware_figure_record(
            figure_type="graph",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.82,
            description=self.description,
            grounding_scores=self.grounding_scores,
            figure_context=self.figure_context,
            type_signals=TypeSignals(),
        )

        self.assertNotIn("type_signals", record)

    def test_top_level_confidence_falls_back_to_grounding_when_description_confidence_missing(self):
        description = GeneratedDescription(
            text="설명",
            confidence=None,
            generation_time_seconds=0.1,
            model_name="m",
            model_version=None,
            warnings=[],
        )
        record = build_context_aware_figure_record(
            figure_type="illustration",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.5,
            description=description,
            grounding_scores=self.grounding_scores,
            figure_context=self.figure_context,
        )

        self.assertEqual(record["confidence"], 0.93)

    def test_empty_description_text_yields_failed_status_not_a_fabricated_success(self):
        description = GeneratedDescription(
            text="",
            confidence=None,
            generation_time_seconds=0.1,
            model_name="m",
            model_version=None,
            warnings=["ChatGPT returned an empty caption."],
        )
        record = build_context_aware_figure_record(
            figure_type="graph",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.5,
            description=description,
            grounding_scores=GroundingScores(None, None, None),
            figure_context=self.figure_context,
        )

        self.assertEqual(record["description"]["status"], "failed")
        self.assertIsNone(record["confidence"])

    def test_split_additive_fields_removes_all_new_keys_and_nothing_else(self):
        record = build_context_aware_figure_record(
            figure_type="graph",
            classifier_model={"name": "openclip", "version": None},
            classifier_confidence=0.82,
            description=self.description,
            grounding_scores=self.grounding_scores,
            figure_context=self.figure_context,
            prompt_trace=PromptTrace(use_caption=True),
            type_signals=TypeSignals(trend="increasing"),
        )

        additive = split_additive_fields(record)

        self.assertEqual(
            set(additive),
            {
                "figure_type",
                "education_context",
                "grounding",
                "confidence",
                "summary",
                "context_source",
                "warning_codes",
                "context_used",
                "prompt_trace",
                "type_signals",
            },
        )
        self.assertEqual(set(record), {"analysis", "warnings", "description"})


if __name__ == "__main__":
    unittest.main()
