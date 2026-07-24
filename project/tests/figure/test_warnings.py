import unittest

from src.analysis.figure.context_builder import FigureContext
from src.analysis.figure.generator import GeneratedDescription
from src.analysis.figure.grounding import GroundingScores
from src.analysis.figure.warnings import ALL_WARNING_CODES, WarningCode, derive_warning_codes


def _description(text="설명", warnings=None):
    return GeneratedDescription(
        text=text,
        confidence=0.9,
        generation_time_seconds=0.1,
        model_name="m",
        model_version=None,
        warnings=warnings or [],
    )


class DeriveWarningCodesTests(unittest.TestCase):
    def test_no_caption_is_flagged(self):
        context = FigureContext(previous_paragraph="문단")
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(),
            grounding_scores=GroundingScores(None, None, None),
        )
        self.assertIn(WarningCode.NO_CAPTION, codes)

    def test_caption_present_is_not_flagged(self):
        context = FigureContext(caption="캡션")
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(),
            grounding_scores=GroundingScores(None, None, None),
        )
        self.assertNotIn(WarningCode.NO_CAPTION, codes)

    def test_context_missing_when_nothing_found_nearby(self):
        context = FigureContext()
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(),
            grounding_scores=GroundingScores(None, None, None),
        )
        self.assertIn(WarningCode.CONTEXT_MISSING, codes)

    def test_low_ocr_confidence_flagged_from_evidence_scores(self):
        context = FigureContext(caption="캡션")
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(),
            grounding_scores=GroundingScores(None, None, None),
            evidence=[{"text": "y=ax", "score": 0.6}],
        )
        self.assertIn(WarningCode.LOW_OCR_CONFIDENCE, codes)

    def test_grounding_mismatch_flagged_below_threshold(self):
        context = FigureContext(caption="캡션")
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(),
            grounding_scores=GroundingScores(caption_score=0.1, context_score=None, overall_score=0.1),
        )
        self.assertIn(WarningCode.GROUNDING_MISMATCH, codes)

    def test_generation_failed_when_description_text_is_empty(self):
        context = FigureContext(caption="캡션")
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(text=""),
            grounding_scores=GroundingScores(None, None, None),
        )
        self.assertIn(WarningCode.GENERATION_FAILED, codes)

    def test_type_uncertain_flagged_from_low_classifier_confidence(self):
        context = FigureContext(caption="캡션")
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(),
            grounding_scores=GroundingScores(None, None, None),
            classifier_confidence=0.2,
        )
        self.assertIn(WarningCode.TYPE_UNCERTAIN, codes)

    def test_low_image_quality_is_never_fabricated(self):
        # No image-quality detector exists yet -- this code must never fire.
        context = FigureContext()
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(text=""),
            grounding_scores=GroundingScores(caption_score=0.0, context_score=0.0, overall_score=0.0),
            classifier_confidence=0.1,
            evidence=[{"text": "x", "score": 0.1}],
        )
        self.assertNotIn(WarningCode.LOW_IMAGE_QUALITY, codes)

    def test_all_returned_codes_are_in_the_taxonomy(self):
        context = FigureContext()
        codes = derive_warning_codes(
            figure_context=context,
            description=_description(text=""),
            grounding_scores=GroundingScores(caption_score=0.0, context_score=0.0, overall_score=0.0),
            classifier_confidence=0.1,
            evidence=[{"text": "x", "score": 0.1}],
        )
        self.assertTrue(set(codes) <= ALL_WARNING_CODES)


if __name__ == "__main__":
    unittest.main()
