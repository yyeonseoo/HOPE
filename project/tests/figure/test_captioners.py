import unittest

from src.analysis.figure.captioners import (
    _collapse_decimal_point_spacing,
    _find_incomplete_numbered_list,
    _find_invalid_month_mentions,
    _find_suspicious_caption_content,
    _postprocess_qwen_caption,
    _substitute_stray_hanja,
)


class SubstituteStrayHanjaTests(unittest.TestCase):
    def test_stray_hanja_is_converted_to_its_korean_reading(self):
        self.assertEqual(_substitute_stray_hanja("과程이 시작되며"), "과정이 시작되며")

    def test_parenthetical_hanja_gloss_is_converted(self):
        text = "이 그래프는 시간(時間)과 거리(距離)의 관계를 보여준다."
        expected = "이 그래프는 시간(시간)과 거리(거리)의 관계를 보여준다."
        self.assertEqual(_substitute_stray_hanja(text), expected)

    def test_pure_hangul_text_is_unchanged(self):
        text = "이 그래프는 시간과 거리의 관계를 보여준다."
        self.assertEqual(_substitute_stray_hanja(text), text)

    def test_postprocess_applies_hanja_substitution(self):
        result = _postprocess_qwen_caption("이후, 거리 값이 다시 감소하는 과程이 시작되며 끝난다.")
        self.assertIn("과정", result)
        self.assertNotIn("程", result)


class CollapseDecimalPointSpacingTests(unittest.TestCase):
    def test_removes_space_after_decimal_point(self):
        self.assertEqual(_collapse_decimal_point_spacing("0. 6"), "0.6")

    def test_handles_negative_numbers_and_units(self):
        text = "y축은 -0. 6℃에서 1. 0℃까지의 범위를 가집니다."
        expected = "y축은 -0.6℃에서 1.0℃까지의 범위를 가집니다."
        self.assertEqual(_collapse_decimal_point_spacing(text), expected)

    def test_normal_sentence_boundary_after_a_number_is_untouched(self):
        text = "정답은 5. 다음 문제로 넘어갑니다."
        self.assertEqual(_collapse_decimal_point_spacing(text), text)

    def test_postprocess_applies_decimal_spacing_fix(self):
        result = _postprocess_qwen_caption("y축은 -0. 6℃에서 1. 0℃까지입니다.")
        self.assertIn("-0.6", result)
        self.assertIn("1.0", result)


class PostprocessQwenCaptionTests(unittest.TestCase):
    def test_strips_leading_bullet_markers(self):
        text = "- 첫 문장입니다.\n- 두 번째 문장입니다."
        self.assertEqual(_postprocess_qwen_caption(text), "첫 문장입니다. 두 번째 문장입니다.")

    def test_strips_leading_header_marker(self):
        self.assertEqual(_postprocess_qwen_caption("# 제목입니다."), "제목입니다.")

    def test_strips_code_fences_and_bold_markers(self):
        text = "**중요**: 이 그림은 ```그래프```입니다."
        result = _postprocess_qwen_caption(text)
        self.assertNotIn("**", result)
        self.assertNotIn("```", result)

    def test_drops_incomplete_trailing_sentence(self):
        text = "이 그래프는 시간에 따라 증가한다. 그 다음에 감소하다가 다시"
        result = _postprocess_qwen_caption(text)
        self.assertEqual(result, "이 그래프는 시간에 따라 증가한다.")

    def test_keeps_single_incomplete_sentence_when_nothing_else_kept(self):
        text = "그 다음에 감소하다가 다시"
        result = _postprocess_qwen_caption(text)
        self.assertEqual(result, text)

    def test_collapses_immediately_repeated_sentence(self):
        text = "그래프는 증가한다. 그래프는 증가한다. 그래프는 증가한다."
        result = _postprocess_qwen_caption(text)
        self.assertEqual(result, "그래프는 증가한다.")

    def test_collapses_adjacent_repeated_word_within_a_sentence(self):
        text = "이 도형은 원통형 원통형 원통형 모양이다."
        result = _postprocess_qwen_caption(text)
        self.assertEqual(result, "이 도형은 원통형 모양이다.")


class InvalidMonthMentionTests(unittest.TestCase):
    def test_valid_months_are_not_flagged(self):
        text = "이 그래프는 1900년 3월부터 12월까지의 변화를 나타낸다."
        self.assertEqual(_find_invalid_month_mentions(text), [])

    def test_month_over_twelve_is_flagged(self):
        text = "1817년 97월에 가장 낮은 값을 가지며, 2035년 27월에 최고점을 기록합니다."
        warnings = _find_invalid_month_mentions(text)
        self.assertEqual(len(warnings), 2)
        self.assertIn("97월", warnings[0])

    def test_month_zero_is_flagged(self):
        self.assertTrue(_find_invalid_month_mentions("0월에 시작한다."))

    def test_duration_phrased_as_gaeworl_is_not_a_month_mention(self):
        text = "이 변화는 3개월 동안 지속되었다."
        self.assertEqual(_find_invalid_month_mentions(text), [])

    def test_duplicate_invalid_months_are_deduplicated(self):
        text = "97월과 97월 사이의 변화."
        self.assertEqual(len(_find_invalid_month_mentions(text)), 1)


class IncompleteNumberedListTests(unittest.TestCase):
    def test_no_declared_range_is_not_flagged(self):
        text = "이 그래프는 시간에 따라 거리가 증가한다."
        self.assertEqual(_find_incomplete_numbered_list(text), [])

    def test_fully_described_range_is_not_flagged(self):
        text = (
            "네 개의 그래프가 포함되어 있으며, 각각 (1)부터 (4)까지 번호가 붙어 있습니다. "
            "(1) 그래프는 증가한다. (2) 그래프는 일정하다. (3) 그래프는 감소한다. (4) 그래프는 진동한다."
        )
        self.assertEqual(_find_incomplete_numbered_list(text), [])

    def test_partially_described_range_is_flagged(self):
        text = (
            "네 개의 그래프가 포함되어 있으며, 각각 (1)부터 (4)까지 번호가 붙어 있습니다. "
            "(1) 그래프는 원점에서 시작하여 상승하는 직선입니다. "
            "(2) 그래프도 원점에서 출발하지만, 이후에 평행한 직선으로 변합니다."
        )
        warnings = _find_incomplete_numbered_list(text)
        self.assertEqual(len(warnings), 1)
        self.assertIn("[3, 4]", warnings[0])

    def test_unreasonably_large_declared_count_is_ignored(self):
        text = "(1)부터 (500)까지 이어지는 목록입니다."
        self.assertEqual(_find_incomplete_numbered_list(text), [])


class SuspiciousCaptionContentTests(unittest.TestCase):
    def test_combines_both_detectors(self):
        text = (
            "네 개의 그래프가 포함되어 있으며, 각각 (1)부터 (4)까지 번호가 붙어 있습니다. "
            "(1) 그래프는 1817년 97월에 시작한다."
        )
        warnings = _find_suspicious_caption_content(text)
        self.assertEqual(len(warnings), 2)

    def test_clean_caption_has_no_warnings(self):
        text = "이 그래프는 시간을 x축, 거리를 y축으로 하며 원점에서 시작해 증가한다."
        self.assertEqual(_find_suspicious_caption_content(text), [])


if __name__ == "__main__":
    unittest.main()
