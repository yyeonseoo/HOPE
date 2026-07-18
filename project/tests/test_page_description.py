import unittest
from dataclasses import dataclass, field

from src.page_description import build_page_description


@dataclass
class FakeGenerationResult:
    text: str
    confidence: float | None = 0.9
    generation_time_seconds: float = 0.5
    model_name: str = "fake-page-model"
    model_version: str | None = "test-1"
    warnings: list = field(default_factory=list)


class FakeGenerator:
    def __init__(self, text_or_fn):
        self._text_or_fn = text_or_fn
        self.calls = []

    def generate_page_description(self, draft_text: str) -> FakeGenerationResult:
        self.calls.append(draft_text)
        if callable(self._text_or_fn):
            return self._text_or_fn(draft_text)
        return FakeGenerationResult(text=self._text_or_fn)


class RaisingGenerator:
    def generate_page_description(self, draft_text: str) -> FakeGenerationResult:
        raise RuntimeError("model unavailable")


def _mixed_page():
    page_result = {
        "page_id": 1,
        "blocks": [
            {"block_id": "p1_b1", "type": "title", "text": "1단원 좌표평면", "reading_order": 0},
            {"block_id": "p1_b2", "type": "paragraph", "text": "이 단원에서는 좌표평면을 배운다.", "reading_order": 1},
            {"block_id": "p1_b3", "type": "formula", "text": "y=2x", "reading_order": 2},
            {"block_id": "p1_b4", "type": "table", "text": "raw table ocr glob", "reading_order": 3},
            {"block_id": "p1_b5", "type": "figure", "text": None, "reading_order": 4},
            {"block_id": "p1_b6", "type": "caption", "text": "그림 1", "reading_order": 5},
            {"block_id": "p1_b7", "type": "footer", "text": "12", "reading_order": 6},
        ],
    }
    semantic_analyses = [
        {
            "block_id": "p1_b3",
            "type": "formula",
            "description": {"long_text": "이 식은 y가 x의 2배임을 나타낸다.", "short_text": "y는 x의 2배."},
        },
        {
            "block_id": "p1_b4",
            "type": "table",
            "description": {"long_text": "표는 학년별 학생 수를 보여준다.", "short_text": "학년별 학생 수 표."},
        },
        {
            "block_id": "p1_b5",
            "type": "figure",
            "description": {"long_text": "그래프는 원점에서 시작해 증가한다.", "short_text": "증가하는 그래프."},
        },
    ]
    return page_result, semantic_analyses


class BuildPageDescriptionDraftTests(unittest.TestCase):
    def test_reading_order_assembly_with_mixed_block_types(self):
        page_result, semantic_analyses = _mixed_page()

        result = build_page_description(page_result, semantic_analyses)

        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["block_ids"], ["p1_b1", "p1_b2", "p1_b3", "p1_b4", "p1_b5", "p1_b6", "p1_b7"]
        )
        expected_order = [
            "1단원 좌표평면",
            "이 단원에서는 좌표평면을 배운다.",
            "이 식은 y가 x의 2배임을 나타낸다.",
            "표는 학년별 학생 수를 보여준다.",
            "그래프는 원점에서 시작해 증가한다.",
            "그림 1",
            "12",
        ]
        for fragment in expected_order:
            self.assertIn(fragment, result["draft_text"])
        # order preserved
        positions = [result["draft_text"].index(fragment) for fragment in expected_order]
        self.assertEqual(positions, sorted(positions))

    def test_formula_falls_back_to_raw_text_when_description_missing(self):
        page_result, semantic_analyses = _mixed_page()
        semantic_analyses[0]["description"] = {"long_text": None, "short_text": None}

        result = build_page_description(page_result, semantic_analyses)

        self.assertEqual(result["status"], "success")
        self.assertIn("y=2x", result["draft_text"])
        self.assertEqual(result["warnings"], [])

    def test_figure_without_description_is_omitted_with_warning(self):
        page_result, semantic_analyses = _mixed_page()
        semantic_analyses[2]["description"] = {"long_text": None, "short_text": None}

        result = build_page_description(page_result, semantic_analyses)

        self.assertEqual(result["status"], "partial")
        self.assertNotIn("p1_b5", result["block_ids"])
        self.assertTrue(any("figure" in warning for warning in result["warnings"]))

    def test_empty_page_is_failed(self):
        result = build_page_description({"page_id": 2, "blocks": []}, [])

        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["text"])
        self.assertIsNone(result["draft_text"])


class BuildPageDescriptionGeneratorTests(unittest.TestCase):
    def test_no_generator_returns_draft_only(self):
        page_result, semantic_analyses = _mixed_page()

        result = build_page_description(page_result, semantic_analyses, generator=None)

        self.assertEqual(result["text"], result["draft_text"])
        self.assertFalse(result["was_generated"])
        self.assertIsNone(result["model"])
        self.assertIsNone(result["confidence"])
        self.assertIsNone(result["generation_time_seconds"])

    def test_generator_output_is_used_when_grounded(self):
        page_result, semantic_analyses = _mixed_page()
        generator = FakeGenerator(
            "1단원 좌표평면에서는 좌표평면을 배운다. y는 x의 2배이며, 학년별 학생 수를 보여주는 표와 "
            "원점에서 시작해 증가하는 그래프가 함께 제시된다."
        )

        result = build_page_description(page_result, semantic_analyses, generator=generator)

        self.assertEqual(len(generator.calls), 1)
        self.assertTrue(result["was_generated"])
        self.assertEqual(result["model"], {"name": "fake-page-model", "version": "test-1"})
        self.assertEqual(result["confidence"], 0.9)
        self.assertIn("y는 x의 2배", result["text"])

    def test_generator_exception_falls_back_to_draft(self):
        page_result, semantic_analyses = _mixed_page()

        result = build_page_description(page_result, semantic_analyses, generator=RaisingGenerator())

        self.assertEqual(result["text"], result["draft_text"])
        self.assertFalse(result["was_generated"])
        self.assertTrue(any("model unavailable" in warning for warning in result["warnings"]))

    def test_draft_exceeding_max_chars_skips_generation(self):
        page_result, semantic_analyses = _mixed_page()
        generator = FakeGenerator("아무 텍스트")

        result = build_page_description(page_result, semantic_analyses, generator=generator, max_draft_chars=5)

        self.assertEqual(generator.calls, [])
        self.assertFalse(result["was_generated"])
        self.assertTrue(any("max_draft_chars" in warning for warning in result["warnings"]))


class GroundingVerificationTests(unittest.TestCase):
    def test_invented_number_is_stripped_and_flags_review(self):
        page_result = {
            "page_id": 3,
            "blocks": [
                {"block_id": "p3_b1", "type": "paragraph", "text": "이 페이지에는 3개의 그림과 2개의 표가 있다.", "reading_order": 0},
            ],
        }
        generator = FakeGenerator(
            "이 페이지에는 5개의 그림과 2개의 표가 있다. 그림들은 서로 관련된 내용을 보여준다."
        )

        result = build_page_description(page_result, [], generator=generator)

        self.assertNotIn("5개의 그림", result["text"])
        self.assertIn("그림들은 서로 관련된 내용을 보여준다.", result["text"])
        self.assertEqual(result["review_status"], "needs_review")
        self.assertTrue(any("unsupported claim" in warning for warning in result["warnings"]))

    def test_invented_equation_is_stripped(self):
        page_result = {
            "page_id": 4,
            "blocks": [
                {"block_id": "p4_b1", "type": "formula", "text": "y=2x", "reading_order": 0},
            ],
        }
        generator = FakeGenerator("이 식은 y=2x를 나타낸다. 또한 y=3x라는 관계도 성립한다.")

        result = build_page_description(page_result, [], generator=generator)

        self.assertIn("y=2x", result["text"])
        self.assertNotIn("y=3x", result["text"])

    def test_trailing_editorial_section_is_stripped(self):
        page_result = {
            "page_id": 7,
            "blocks": [
                {"block_id": "p7_b1", "type": "paragraph", "text": "이 페이지에는 3개의 그림이 있다.", "reading_order": 0},
            ],
        }
        generator = FakeGenerator("이 페이지에는 그림 3개가 있다.\n\n결론:\n이는 매우 중요한 내용이다.")

        result = build_page_description(page_result, [], generator=generator)

        self.assertEqual(result["text"], "이 페이지에는 그림 3개가 있다.")
        self.assertNotIn("결론", result["text"])
        self.assertTrue(any("참고" in warning or "결론" in warning for warning in result["warnings"]))

    def test_excessively_long_generation_falls_back_to_draft(self):
        page_result = {
            "page_id": 8,
            "blocks": [
                {"block_id": "p8_b1", "type": "paragraph", "text": "이 페이지에는 3개의 그림이 있다.", "reading_order": 0},
            ],
        }
        generator = FakeGenerator("이 페이지에는 3개의 그림이 있다. " * 6)

        result = build_page_description(page_result, [], generator=generator)

        self.assertEqual(result["text"], result["draft_text"])
        self.assertFalse(result["was_generated"])
        self.assertEqual(result["review_status"], "needs_review")
        self.assertTrue(any("padding or rambling" in warning for warning in result["warnings"]))

    def test_legitimate_paraphrase_keeping_same_numbers_is_not_stripped(self):
        page_result = {
            "page_id": 5,
            "blocks": [
                {"block_id": "p5_b1", "type": "paragraph", "text": "이 페이지에는 3개의 그림과 2개의 표가 있다.", "reading_order": 0},
            ],
        }
        generator = FakeGenerator("이 페이지는 그림 3개와 표 2개로 구성되어 있다.")

        result = build_page_description(page_result, [], generator=generator)

        self.assertEqual(result["text"], "이 페이지는 그림 3개와 표 2개로 구성되어 있다.")
        self.assertEqual(result["review_status"], "unreviewed")

    def test_fully_unsupported_generation_falls_back_to_draft(self):
        page_result = {
            "page_id": 6,
            "blocks": [
                {"block_id": "p6_b1", "type": "paragraph", "text": "이 페이지에는 3개의 그림이 있다.", "reading_order": 0},
            ],
        }
        generator = FakeGenerator("이 페이지에는 99개의 표와 42개의 수식이 있다.")

        result = build_page_description(page_result, [], generator=generator)

        self.assertEqual(result["text"], result["draft_text"])
        self.assertEqual(result["review_status"], "needs_review")


if __name__ == "__main__":
    unittest.main()
