import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from src.analysis.figure.captioners import (
    ChatGPTCaptioner,
    _collapse_decimal_point_spacing,
    _find_incomplete_numbered_list,
    _find_invalid_month_mentions,
    _find_suspicious_caption_content,
    _postprocess_caption_text,
    _remove_unsupported_exact_claims,
    _substitute_stray_hanja,
    _trusted_axis_labels,
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
        result = _postprocess_caption_text("이후, 거리 값이 다시 감소하는 과程이 시작되며 끝난다.")
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
        result = _postprocess_caption_text("y축은 -0. 6℃에서 1. 0℃까지입니다.")
        self.assertIn("-0.6", result)
        self.assertIn("1.0", result)


class PostprocessCaptionTextTests(unittest.TestCase):
    def test_strips_leading_bullet_markers(self):
        text = "- 첫 문장입니다.\n- 두 번째 문장입니다."
        self.assertEqual(_postprocess_caption_text(text), "첫 문장입니다. 두 번째 문장입니다.")

    def test_strips_leading_header_marker(self):
        self.assertEqual(_postprocess_caption_text("# 제목입니다."), "제목입니다.")

    def test_strips_code_fences_and_bold_markers(self):
        text = "**중요**: 이 그림은 ```그래프```입니다."
        result = _postprocess_caption_text(text)
        self.assertNotIn("**", result)
        self.assertNotIn("```", result)

    def test_removes_only_latex_math_delimiters(self):
        text = r"\(x\)축과 \(y=ax\)가 표시되어 있다."
        self.assertEqual(_postprocess_caption_text(text), "x축과 y=ax가 표시되어 있다.")

    def test_drops_incomplete_trailing_sentence(self):
        text = "이 그래프는 시간에 따라 증가한다. 그 다음에 감소하다가 다시"
        result = _postprocess_caption_text(text)
        self.assertEqual(result, "이 그래프는 시간에 따라 증가한다.")

    def test_keeps_single_incomplete_sentence_when_nothing_else_kept(self):
        text = "그 다음에 감소하다가 다시"
        result = _postprocess_caption_text(text)
        self.assertEqual(result, text)

    def test_collapses_immediately_repeated_sentence(self):
        text = "그래프는 증가한다. 그래프는 증가한다. 그래프는 증가한다."
        result = _postprocess_caption_text(text)
        self.assertEqual(result, "그래프는 증가한다.")

    def test_collapses_adjacent_repeated_word_within_a_sentence(self):
        text = "이 도형은 원통형 원통형 원통형 모양이다."
        result = _postprocess_caption_text(text)
        self.assertEqual(result, "이 도형은 원통형 모양이다.")

    def test_removes_orphan_korean_panel_marker(self):
        text = "세 도형과 세 그래프가 배치되어 있다. ㄱ."
        self.assertEqual(_postprocess_caption_text(text), "세 도형과 세 그래프가 배치되어 있다.")

    def test_normalizes_coordinate_axis_mixed_word(self):
        text = "두 곡선이 coordinate축에 가까워진다."
        self.assertEqual(_postprocess_caption_text(text), "두 곡선이 좌표축에 가까워진다.")


class TrustedAxisLabelTests(unittest.TestCase):
    def test_trusted_axis_labels_use_relative_figure_positions(self):
        evidence = [
            {"id": "y", "text": "거리", "relative_bbox": [0.03, 0.10, 0.18, 0.20]},
            {"id": "x", "text": "시간", "relative_bbox": [0.78, 0.78, 0.96, 0.90]},
            {"id": "menu", "text": "파일", "relative_bbox": [0.02, 0.01, 0.12, 0.06]},
        ]
        self.assertEqual(_trusted_axis_labels(evidence), ("시간", "거리"))

    def test_equations_and_coordinates_are_not_treated_as_axis_labels(self):
        evidence = [
            {"id": "eq", "text": "y=2x", "relative_bbox": [0.78, 0.80, 0.96, 0.90]},
            {"id": "coord", "text": "(1, 2)", "relative_bbox": [0.03, 0.10, 0.18, 0.20]},
        ]
        self.assertEqual(_trusted_axis_labels(evidence), (None, None))


class GroundedExactClaimTests(unittest.TestCase):
    def test_removes_hallucinated_equation_sentence(self):
        text = "우상향하는 직선이 표시되어 있다. 이 직선은 y=2x+2를 나타낸다."
        result, warnings = _remove_unsupported_exact_claims(text, ["x", "y", "O"])
        self.assertEqual(result, "우상향하는 직선이 표시되어 있다.")
        self.assertTrue(warnings)

    def test_keeps_equation_present_in_evidence(self):
        text = "직선 옆에 y=ax가 표시되어 있다."
        result, warnings = _remove_unsupported_exact_claims(text, ["y=ax", "(1, a)"])
        self.assertEqual(result, text)
        self.assertEqual(warnings, [])

    def test_removes_unsupported_coordinate_but_keeps_qualitative_sentence(self):
        text = "직선은 원점을 지난다. y축과의 교점은 (0, 1)이다."
        result, warnings = _remove_unsupported_exact_claims(text, ["x", "y", "1"])
        self.assertEqual(result, "직선은 원점을 지난다.")
        self.assertTrue(warnings)

    def test_does_not_accept_number_as_substring_of_another_value(self):
        text = "점의 값은 2이다."
        result, warnings = _remove_unsupported_exact_claims(text, ["12"])
        self.assertEqual(result, "")
        self.assertTrue(warnings)

    def test_removes_unsupported_function_notation(self):
        text = "곡선은 f(x)=2x+1을 나타낸다."
        result, warnings = _remove_unsupported_exact_claims(text, ["x", "y"])
        self.assertEqual(result, "")
        self.assertTrue(warnings)

    def test_keeps_parenthesized_panel_numbers_without_ocr_evidence(self):
        text = "도형 (1)은 위가 넓다. 도형 (2)는 가운데가 좁다. 도형 (3)은 세 부분으로 나뉜다."
        result, warnings = _remove_unsupported_exact_claims(text, [])
        self.assertEqual(result, text)
        self.assertEqual(warnings, [])

    def test_keeps_numbered_panel_list_markers(self):
        text = "1. 첫 번째 그래프는 곡선이다. 2. 두 번째 그래프는 꺾은선이다."
        result, warnings = _remove_unsupported_exact_claims(text, [])
        self.assertEqual(result, text)
        self.assertEqual(warnings, [])

    def test_still_removes_unsupported_data_number_sequence(self):
        text = "그래프의 값은 1, 2, 3이다."
        result, warnings = _remove_unsupported_exact_claims(text, [])
        self.assertEqual(result, "")
        self.assertTrue(warnings)

    def test_keeps_zero_when_it_only_identifies_the_origin(self):
        text = "첫 번째 그래프는 원점 0에서 시작해 일정하게 증가한다."
        result, warnings = _remove_unsupported_exact_claims(text, [])
        self.assertEqual(result, text)
        self.assertEqual(warnings, [])


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


class ChatGPTCaptionerTests(unittest.TestCase):
    def _image(self, directory):
        path = Path(directory) / "figure.png"
        Image.new("RGB", (20, 20), "white").save(path)
        return path

    def _fake_openai_module(self, calls, reply_text):
        class FakeCompletions:
            def create(self, **kwargs):
                calls["create_kwargs"] = kwargs
                message = SimpleNamespace(content=reply_text)
                choice = SimpleNamespace(message=message)
                return SimpleNamespace(choices=[choice])

        class FakeClient:
            def __init__(self, api_key=None, **kwargs):
                calls["api_key"] = api_key
                calls["client_kwargs"] = kwargs
                self.chat = SimpleNamespace(completions=FakeCompletions())

        return SimpleNamespace(OpenAI=FakeClient)

    def test_sends_prompt_and_image_and_returns_text(self):
        calls = {}
        captioner = ChatGPTCaptioner(api_key="test-key")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "우상향하는 그래프이다.")}
        ):
            output = captioner.caption_with_prompt(self._image(tmp), "이 그림을 설명하세요.")

        self.assertEqual(calls["api_key"], "test-key")
        self.assertEqual(calls["create_kwargs"]["model"], "gpt-5")
        self.assertEqual(calls["create_kwargs"]["max_completion_tokens"], 2000)
        self.assertNotIn("max_tokens", calls["create_kwargs"])
        self.assertNotIn("temperature", calls["create_kwargs"])
        messages = calls["create_kwargs"]["messages"]
        self.assertEqual(messages[0]["content"][0], {"type": "text", "text": "이 그림을 설명하세요."})
        self.assertEqual(messages[0]["content"][1]["type"], "image_url")
        self.assertTrue(messages[0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.assertEqual(output.text, "우상향하는 그래프이다.")
        self.assertEqual(output.model_name, "gpt-5-context-aware")

    def test_client_is_cached_across_calls(self):
        calls = {}
        captioner = ChatGPTCaptioner()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ):
            image = self._image(tmp)
            captioner.caption_with_prompt(image, "설명하세요.")
            first_client = captioner._client
            captioner.caption_with_prompt(image, "설명하세요.")

        self.assertIs(captioner._client, first_client)

    def test_removes_exact_claims_unsupported_by_evidence(self):
        calls = {}
        captioner = ChatGPTCaptioner()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "그래프는 y=3x+5를 나타낸다.")}
        ):
            output = captioner.caption_with_prompt(self._image(tmp), "설명하세요.", evidence=[])

        self.assertNotIn("y=3x+5", output.text)
        self.assertTrue(output.warnings)

    def test_client_gets_default_timeout_and_retries(self):
        calls = {}
        captioner = ChatGPTCaptioner()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ), patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HOPE_FIGURE_GPT_TIMEOUT_SECONDS", None)
            os.environ.pop("HOPE_FIGURE_GPT_MAX_RETRIES", None)
            captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(calls["client_kwargs"]["timeout"], 60.0)
        self.assertEqual(calls["client_kwargs"]["max_retries"], 3)

    def test_constructor_overrides_timeout_and_max_retries(self):
        calls = {}
        captioner = ChatGPTCaptioner(timeout=5.0, max_retries=1)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ):
            captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(calls["client_kwargs"]["timeout"], 5.0)
        self.assertEqual(calls["client_kwargs"]["max_retries"], 1)

    def test_env_vars_override_default_timeout_and_max_retries(self):
        calls = {}
        with patch.dict(
            os.environ,
            {"HOPE_FIGURE_GPT_TIMEOUT_SECONDS": "12.5", "HOPE_FIGURE_GPT_MAX_RETRIES": "7"},
        ):
            captioner = ChatGPTCaptioner()
            with tempfile.TemporaryDirectory() as tmp, patch.dict(
                "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
            ):
                captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(calls["client_kwargs"]["timeout"], 12.5)
        self.assertEqual(calls["client_kwargs"]["max_retries"], 7)

    def test_env_var_overrides_default_model_name(self):
        calls = {}
        with patch.dict(os.environ, {"HOPE_FIGURE_GPT_MODEL": "gpt-4o-mini"}):
            captioner = ChatGPTCaptioner()
            with tempfile.TemporaryDirectory() as tmp, patch.dict(
                "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
            ):
                output = captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(captioner.model_name, "gpt-4o-mini")
        self.assertEqual(calls["create_kwargs"]["model"], "gpt-4o-mini")
        self.assertEqual(output.model_name, "gpt-4o-mini-context-aware")

    def test_explicit_model_argument_takes_precedence_over_env_var(self):
        with patch.dict(os.environ, {"HOPE_FIGURE_GPT_MODEL": "gpt-4o-mini"}):
            captioner = ChatGPTCaptioner(model="gpt-4-turbo")

        self.assertEqual(captioner.model_name, "gpt-4-turbo")

    def test_reasoning_model_omits_unsupported_temperature(self):
        calls = {}
        captioner = ChatGPTCaptioner(model="gpt-5")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ):
            captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertNotIn("temperature", calls["create_kwargs"])

    def test_reasoning_model_gets_a_larger_default_token_budget(self):
        calls = {}
        captioner = ChatGPTCaptioner(model="gpt-5")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ):
            captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(calls["create_kwargs"]["max_completion_tokens"], 2000)

    def test_explicit_max_tokens_overrides_reasoning_model_default(self):
        calls = {}
        captioner = ChatGPTCaptioner(model="gpt-5", max_tokens=500)
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ):
            captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(calls["create_kwargs"]["max_completion_tokens"], 500)

    def test_non_reasoning_model_still_sends_temperature(self):
        calls = {}
        captioner = ChatGPTCaptioner(model="gpt-4o-mini")
        with tempfile.TemporaryDirectory() as tmp, patch.dict(
            "sys.modules", {"openai": self._fake_openai_module(calls, "설명이다.")}
        ):
            captioner.caption_with_prompt(self._image(tmp), "설명하세요.")

        self.assertEqual(calls["create_kwargs"]["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
