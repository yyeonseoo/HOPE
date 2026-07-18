import unittest

from src.analysis.figure.captioners import (
    QWEN_COMPLEX_GRAPH_PROMPT,
    QWEN_GRAPH_DESCRIPTION_PROMPT,
    QWEN_TYPE_PROMPTS,
    _collapse_decimal_point_spacing,
    _find_incomplete_numbered_list,
    _find_invalid_month_mentions,
    _find_suspicious_caption_content,
    _fallback_after_exact_claim_filter,
    _filter_unsupported_context_graph_claims,
    _graph_needs_structured_review,
    _graph_context_prompt,
    _caption_context_candidates,
    _minimal_figure_fallback,
    _deduplicated_matches,
    _EVIDENCE_EQUATION_PATTERN,
    _postprocess_qwen_caption,
    _parse_structured_graph_response,
    _parse_context_fusion_response,
    _preserves_visual_anchor,
    _reconcile_structured_graph,
    _remove_unsupported_exact_claims,
    _restore_truncated_context_equation,
    _restore_trusted_context_terms,
    _trusted_axis_labels,
    _anchor_trusted_axis_labels,
    _visible_panel_count,
    _substitute_stray_hanja,
)


class ComplexGraphPromptTests(unittest.TestCase):
    def test_requires_each_numbered_diagram_to_be_checked_separately(self):
        self.assertIn("도형도 왼쪽부터 하나씩 따로 관찰", QWEN_COMPLEX_GRAPH_PROMPT)
        self.assertIn("상단·중앙·하단", QWEN_COMPLEX_GRAPH_PROMPT)
        self.assertIn("실제 개수", QWEN_COMPLEX_GRAPH_PROMPT)
        self.assertIn("정확한 명칭이 확실하지 않으면 이름을 추정하지", QWEN_COMPLEX_GRAPH_PROMPT)

    def test_axis_meaning_requires_ocr_evidence(self):
        graph_prompt = QWEN_TYPE_PROMPTS["graph"]
        self.assertIn("OCR 근거에 명확한 이름", graph_prompt)
        self.assertIn("좌측·우측 축의 의미를 만들지", graph_prompt)


class ExactClaimFallbackTests(unittest.TestCase):
    def test_empty_filtered_graph_uses_qualitative_visual_caption(self):
        from src.analysis.figure.graph_visual import GraphVisualCue

        cue = GraphVisualCue(
            "plotted", "increasing", 0.96, mark_type="line", series_count=1,
            path_shape="straight_segments",
        )
        result, used = _fallback_after_exact_claim_filter("", cue, [])
        self.assertTrue(used)
        self.assertIn("우상향", result)
        self.assertNotIn("y=", result)

    def test_nonempty_caption_is_not_replaced(self):
        from src.analysis.figure.graph_visual import GraphVisualCue

        cue = GraphVisualCue("plotted", "increasing", 0.96, mark_type="line", series_count=1)
        result, used = _fallback_after_exact_claim_filter("점들이 오른쪽으로 갈수록 높아진다.", cue, [])
        self.assertFalse(used)
        self.assertEqual(result, "점들이 오른쪽으로 갈수록 높아진다.")


class InlineContextTests(unittest.TestCase):
    def test_graph_prompt_requires_each_series_and_standard_axis_names(self):
        self.assertIn("서로 다른 선·곡선·점 계열", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("각각을", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("하나의 경향으로 합치지", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("수평축(x축)과 수직축(y축)", QWEN_GRAPH_DESCRIPTION_PROMPT)

    def test_graph_context_is_only_auxiliary_visual_structure(self):
        prompt = _graph_context_prompt([{"block_id": "p1_b2", "text": "A는 거북이이다."}])
        self.assertIn("보조 자료", prompt)
        self.assertIn("문맥보다 이미지를 우선", prompt)
        self.assertIn("A는 거북이이다", prompt)

    def test_graph_context_exposes_normalized_equation_as_exact_fact(self):
        prompt = _graph_context_prompt([{"block_id": "p13_b10", "text": "y=6/x의 그래프"}])
        self.assertIn("정확히 확인된 수식은 y=6/x", prompt)
        self.assertIn("분모를 생략하지 말고 그대로", prompt)

    def test_disconnected_branches_are_not_merged_into_one_trend(self):
        self.assertIn("연결되지 않은 곡선 가지", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("하나의 연속 구간처럼 합치지", QWEN_GRAPH_DESCRIPTION_PROMPT)

    def test_graph_prompt_separates_context_identity_from_visual_description(self):
        self.assertIn("도입에 간단히 반영", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("라벨의 의미는 해당 계열을 설명할 때 정확히 사용", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("점 사이에 실제 선이 보이지 않으면", QWEN_GRAPH_DESCRIPTION_PROMPT)
        self.assertIn("점들이 이동한다고 표현하지", QWEN_GRAPH_DESCRIPTION_PROMPT)

    def test_trusted_axis_labels_use_relative_figure_positions(self):
        evidence = [
            {"id": "y", "text": "거리", "relative_bbox": [0.03, 0.10, 0.18, 0.20]},
            {"id": "x", "text": "시간", "relative_bbox": [0.78, 0.78, 0.96, 0.90]},
            {"id": "menu", "text": "파일", "relative_bbox": [0.02, 0.01, 0.12, 0.06]},
        ]
        self.assertEqual(_trusted_axis_labels(evidence), ("시간", "거리"))

    def test_trusted_axis_anchor_removes_conflicting_axis_sentence(self):
        evidence = [
            {"id": "y", "text": "거리", "relative_bbox": [0.03, 0.10, 0.18, 0.20]},
            {"id": "x", "text": "시간", "relative_bbox": [0.78, 0.78, 0.96, 0.90]},
        ]
        text = "시간을 수평축으로, 전압을 수직축으로 표현한다. 직선은 오른쪽 위로 향한다."
        result = _anchor_trusted_axis_labels(text, evidence)
        self.assertIn("시간에 따른 거리", result)
        self.assertNotIn("전압", result)
        self.assertIn("오른쪽 위", result)

    def test_context_entity_near_miss_is_restored_only_from_explicit_mapping(self):
        context = [{"block_id": "p1_b9", "text": "A는 거북이이고 B는 토끼를 나타낸다."}]
        self.assertEqual(
            _restore_trusted_context_terms("토큰과 거북이의 경주이다.", context),
            "토끼와 거북이의 경주이다.",
        )

    def test_numbered_evidence_declares_all_four_panels(self):
        evidence = [{"text": value} for value in ("(1)", "(2)", "(3)", "(4)")]
        self.assertEqual(_visible_panel_count(evidence), 4)

    def test_non_graph_routes_do_not_receive_graph_context(self):
        context = [{"block_id": "p1_b8", "text": "A와 B의 그래프"}]
        self.assertEqual(_caption_context_candidates(context, "illustration"), [])
        self.assertEqual(_caption_context_candidates(context, "photo"), [])
        self.assertEqual(_caption_context_candidates(context, "graph")[0]["block_id"], "p1_b8")

    def test_only_complex_graphs_require_qwen_review(self):
        from src.analysis.figure.graph_visual import GraphVisualCue

        simple = GraphVisualCue(
            "plotted", "increasing", 0.96, variation="monotonic",
            coordinate_plane=True, mark_type="line", series_count=1,
        )
        multiple = GraphVisualCue(
            "plotted", "increasing", 0.96, variation="monotonic",
            coordinate_plane=True, mark_type="line", series_count=2,
        )
        self.assertFalse(_graph_needs_structured_review(simple, []))
        self.assertTrue(_graph_needs_structured_review(multiple, []))

    def test_context_path_has_nonempty_last_resort_caption(self):
        self.assertEqual(
            _minimal_figure_fallback("illustration"),
            "주변 교과서 문맥에서 다루는 내용을 보여 주는 삽화이다.",
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

    def test_removes_only_latex_math_delimiters(self):
        text = r"\(x\)축과 \(y=ax\)가 표시되어 있다."
        self.assertEqual(_postprocess_qwen_caption(text), "x축과 y=ax가 표시되어 있다.")

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

    def test_removes_orphan_korean_panel_marker(self):
        text = "세 도형과 세 그래프가 배치되어 있다. ㄱ."
        self.assertEqual(_postprocess_qwen_caption(text), "세 도형과 세 그래프가 배치되어 있다.")

    def test_normalizes_coordinate_axis_mixed_word(self):
        text = "두 곡선이 coordinate축에 가까워진다."
        self.assertEqual(_postprocess_qwen_caption(text), "두 곡선이 좌표축에 가까워진다.")


class ContextEquationRestorationTests(unittest.TestCase):
    def test_restores_bare_slash_from_single_exact_context_equation(self):
        context = [{"block_id": "p13_b10", "text": "y=6/x의 그래프"}]
        text = "이미지에는 y=6/의 그래프가 두 갈래의 곡선으로 표시되어 있다."
        self.assertEqual(
            _restore_truncated_context_equation(text, context),
            "이미지에는 y=6/x의 그래프가 두 갈래의 곡선으로 표시되어 있다.",
        )

    def test_restores_bare_equal_sign_from_single_exact_context_equation(self):
        context = [{"block_id": "p13_b10", "text": "y=6/x의 그래프"}]
        text = "점들은 y=의 형태를 보여 준다."
        self.assertEqual(
            _restore_truncated_context_equation(text, context),
            "점들은 y=6/x의 형태를 보여 준다.",
        )

    def test_does_not_guess_when_context_contains_conflicting_equations(self):
        context = [
            {"block_id": "p1", "text": "y=6/x의 그래프"},
            {"block_id": "p2", "text": "y=3/x의 그래프"},
        ]
        text = "이미지에는 y=6/의 그래프가 있다."
        self.assertEqual(_restore_truncated_context_equation(text, context), text)

    def test_does_not_change_complete_equation(self):
        context = [{"block_id": "p13_b10", "text": "y=6/x의 그래프"}]
        text = "이미지에는 y=6/x의 그래프가 있다."
        self.assertEqual(_restore_truncated_context_equation(text, context), text)

    def test_removes_wrong_coordinate_and_derived_sign_claim_but_keeps_visual_prose(self):
        context = [{
            "block_id": "p13_b5",
            "text": "y=6/x, (-6, -1), (-3, -2), (1, 6), (2, 3)",
        }]
        text = (
            "y=6/x의 그래프이다. 점들이 두 구역에 나뉘어 표시되어 있다. "
            "좌표 (-6, 6)에서 시작한다. x가 음의 값이면 y는 양수이다."
        )
        result, warnings = _filter_unsupported_context_graph_claims(text, [], context)
        self.assertEqual(result, "y=6/x의 그래프이다. 점들이 두 구역에 나뉘어 표시되어 있다.")
        self.assertTrue(warnings)

    def test_keeps_coordinate_explicitly_present_in_context(self):
        context = [{"block_id": "p13_b5", "text": "점 (-6, -1)을 표시한다."}]
        text = "점 (-6, -1)이 표시되어 있다."
        result, warnings = _filter_unsupported_context_graph_claims(text, [], context)
        self.assertEqual(result, text)
        self.assertEqual(warnings, [])


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


class StructuredGraphResponseTests(unittest.TestCase):
    def test_parses_json_without_accepting_free_form_claims(self):
        text = (
            '```json\n{"arrangement":"single","plots":[{"shape":"points","dirs":["up"],'
            '"net":"up","x":"t2","y":"t1","name":null}]}\n```'
        )
        evidence = [{"id": "t1", "text": "온도"}, {"id": "t2", "text": "시간"}]
        result = _parse_structured_graph_response(text, evidence)
        self.assertEqual(result["plots"][0]["shape"], "points")
        self.assertEqual(result["plots"][0]["x"], "t2")

    def test_accepts_visible_axis_labels_and_composite_context(self):
        text = (
            '{"arrangement":"panels","panel_count":3,'
            '"context":{"kind":"solid_diagrams","count":3,"position":"above","paired":true,'
            '"items":[{"description":"위가 넓고 아래가 좁은 입체도형"}]},"quadrants":[],'
            '"plots":[{"shape":"straight_segments","dirs":["up"],"net":"up",'
            '"x":null,"y":null,"name":null,"x_text":"시간(초)","y_text":"높이"}]} '
        )
        result = _parse_structured_graph_response(text)
        self.assertEqual(result["panel_count"], 3)
        self.assertEqual(result["context"]["kind"], "solid_diagrams")
        self.assertEqual(result["context"]["items"], ["위가 넓고 아래가 좁은 입체도형"])
        self.assertEqual(result["plots"][0]["x_text"], "시간(초)")

    def test_accepts_only_valid_quadrant_numbers(self):
        text = (
            '{"arrangement":"single","quadrants":[3,1,7,"2"],"plots":['
            '{"shape":"smooth_curve","dirs":["down"],"net":"down"}]}'
        )
        result = _parse_structured_graph_response(text)
        self.assertEqual(result["quadrants"], [1, 3])


class ContextFusionResponseTests(unittest.TestCase):
    def test_accepts_only_known_context_ids(self):
        candidates = [{"block_id": "p2_b3", "text": "y=a/x"}]
        raw = '{"relevant_context_ids":["p2_b3","missing"],"caption":"두 갈래의 곡선이다."}'
        result = _parse_context_fusion_response(raw, candidates)
        self.assertEqual(result["relevant_context_ids"], ["p2_b3"])

    def test_rejects_non_json_fusion(self):
        self.assertIsNone(_parse_context_fusion_response("설명만 출력", []))

    def test_visual_anchor_rejects_context_takeover(self):
        base = "해안 도로와 다리가 보이는 사진이다."
        contaminated = "전기 사용량과 요금의 관계를 나타낸 사진이다."
        self.assertFalse(_preserves_visual_anchor(base, contaminated))

    def test_visual_anchor_allows_contextual_graph_rewrite(self):
        base = "좌표평면에 두 갈래의 곡선이 있다."
        fused = "좌표평면에 두 갈래의 곡선이 있으며, 문맥상 y=a/x의 그래프이다."
        self.assertTrue(_preserves_visual_anchor(base, fused))

    def test_connected_line_cue_overrides_point_shape(self):
        from src.analysis.figure.graph_visual import GraphVisualCue

        cue = GraphVisualCue("plotted", "increasing", 0.96, mark_type="line", series_count=1)
        structure = {
            "arrangement": "single",
            "plots": [{"shape": "points", "dirs": ["up", "flat", "down"], "net": "flat"}],
        }
        result = _reconcile_structured_graph(cue, structure)
        self.assertEqual(result["plots"][0]["shape"], "straight_segments")

    def test_detected_polyline_overrides_smooth_curve_shape(self):
        from src.analysis.figure.graph_visual import GraphVisualCue

        cue = GraphVisualCue(
            "plotted", "increasing", 0.96, mark_type="line", series_count=1,
            path_shape="straight_segments",
        )
        structure = {
            "arrangement": "single",
            "plots": [{"shape": "smooth_curve", "dirs": ["up", "up"], "net": "up"}],
        }
        result = _reconcile_structured_graph(cue, structure)
        self.assertEqual(result["plots"][0]["shape"], "straight_segments")

    def test_rejects_values_outside_fixed_schema(self):
        text = (
            '{"arrangement":"single","plots":[{"shape":"point_at_1_2","dirs":["up"],'
            '"net":"up","x":null,"y":null,"name":null}]}'
        )
        self.assertIsNone(_parse_structured_graph_response(text))

    def test_extracts_slash_and_latex_fraction_equations(self):
        text = r"y=a/x, f(x)=\frac{1}{x}"
        result = _deduplicated_matches(_EVIDENCE_EQUATION_PATTERN, text)
        self.assertEqual(result, ["y=a/x", r"f(x)=\frac{1}{x}"])

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
