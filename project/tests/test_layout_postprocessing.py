import unittest
from unittest.mock import patch

from src.layout_detection import (
    _drop_broad_figures_covering_content,
    _expand_paragraphs_with_nearby_ocr_lines,
    _looks_like_table_text,
    _merge_side_badges_into_paragraphs,
    _merge_numbered_items_with_parallel_explanations,
    _normalize_paragraph_text_order,
    _group_lines_by_content,
    _is_recoverable_low_conf_figure,
    _postprocess_blocks,
    _promote_role_title_text,
    _split_mixed_role_blocks,
    _split_answer_lines_from_paragraphs,
    _supplement_nested_formula_lines,
    detect_layout,
    refine_blocks_after_ocr,
)


class LayoutPostprocessingTests(unittest.TestCase):
    def test_low_confidence_figure_recovery_uses_relative_size(self):
        self.assertTrue(
            _is_recoverable_low_conf_figure("figure", 0.163, [1217, 379, 1713, 644], 2000, 2600)
        )
        self.assertFalse(
            _is_recoverable_low_conf_figure("figure", 0.163, [508, 319, 1806, 1369], 2000, 2600)
        )
        self.assertFalse(
            _is_recoverable_low_conf_figure("figure", 0.06, [1217, 379, 1713, 644], 2000, 2600)
        )

    def test_broad_false_figure_does_not_hide_smaller_visual(self):
        broad = {"type": "figure", "bbox": [225, 160, 902, 692], "score": 0.339}
        cylinder = {"type": "figure", "bbox": [609, 189, 857, 322], "score": 0.154}
        content = [
            {"type": "paragraph", "bbox": [292, 191, 596, 283], "score": 0.8},
            {"type": "paragraph", "bbox": [303, 375, 582, 460], "score": 0.8},
            {"type": "paragraph", "bbox": [302, 512, 583, 657], "score": 0.8},
        ]

        result = _drop_broad_figures_covering_content([broad, cylinder, *content], 995, 1326)

        self.assertNotIn(broad, result)
        self.assertIn(cylinder, result)

    def test_parallel_explanation_merges_but_answer_stays_separate(self):
        left_item = {
            "type": "paragraph",
            "bbox": [50, 390, 340, 430],
            "text": "⑶ 속력이 0이 되면 정지한다.",
            "score": 0.9,
        }
        right_first = {
            "type": "section_title",
            "bbox": [470, 382, 840, 425],
            "text": "240초 후 속력이 0이 되었으므로 정",
            "score": 0.8,
        }
        right_tail_and_answer = {
            "type": "paragraph",
            "bbox": [468, 427, 820, 505],
            "text": "지할 때까지 걸린 시간은 240초이다.\n답 ⑴ 20m/s ⑵ 150초 ⑶ 240초",
            "score": 0.8,
        }
        ocr_lines = [
            {"bbox": [470, 430, 815, 460], "text": "지할 때까지 걸린 시간은 240초이다.", "score": 0.9},
            {"bbox": [470, 480, 815, 500], "text": "답 ⑴ 20m/s ⑵ 150초 ⑶ 240초", "score": 0.9},
        ]

        split = _split_answer_lines_from_paragraphs(
            [left_item, right_first, right_tail_and_answer], ocr_lines
        )
        result = _merge_numbered_items_with_parallel_explanations(split)
        paragraphs = [block for block in result if block["type"] == "paragraph"]

        self.assertEqual(len(paragraphs), 2)
        self.assertIn("⑶ 속력이 0이 되면 정지한다.", paragraphs[0]["text"])
        self.assertIn("240초 후 속력이 0이 되었으므로 정", paragraphs[0]["text"])
        self.assertIn("지할 때까지 걸린 시간은 240초이다.", paragraphs[0]["text"])
        self.assertIn("정지할 때까지", paragraphs[0]["text"])
        self.assertEqual(paragraphs[1]["context"]["semantic_role"], "answer")
        self.assertTrue(paragraphs[1]["text"].startswith("답 ⑴"))

    def test_answer_text_splits_without_separate_ocr_line(self):
        block = {
            "type": "paragraph",
            "bbox": [470, 427, 820, 505],
            "text": "정지할 때까지 걸린 시간은 240초이다.\n답 (1) 20m/s (2) 150초 (3) 240초",
            "score": 0.8,
        }

        result = _split_answer_lines_from_paragraphs([block], [])

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "정지할 때까지 걸린 시간은 240초이다.")
        self.assertTrue(result[1]["text"].startswith("답 (1)"))
        self.assertEqual(result[1]["context"]["semantic_role"], "answer")

    def test_badge_directly_above_merges_into_paragraph(self):
        badge = {"type": "title", "bbox": [614, 1378, 747, 1419], "text": "", "score": 0.7}
        paragraph = {
            "type": "paragraph",
            "bbox": [563, 1429, 1603, 1538],
            "text": "⑴ 그래프에서 속력이 가장 빠른 경우를 찾는다.",
            "score": 0.9,
        }

        result = _merge_side_badges_into_paragraphs([badge, paragraph])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["bbox"], [563, 1378, 1603, 1538])
        self.assertEqual(result[0]["context"]["label_source"], "side_badge")

    def test_short_explanation_fragment_is_not_a_section_title(self):
        block = {
            "type": "section_title",
            "bbox": [470, 382, 840, 425],
            "text": "240초 후 속력이 0이 되었으므로 정",
            "score": 0.8,
        }

        result = _postprocess_blocks([block], None, None)

        self.assertEqual(result[0]["type"], "paragraph")

    def test_small_side_badge_merges_into_adjacent_paragraph(self):
        badge = {"type": "title", "bbox": [100, 105, 140, 145], "text": "생각열기", "score": 0.7}
        paragraph = {
            "type": "paragraph",
            "bbox": [148, 100, 650, 170],
            "text": "오른쪽 그래프를 보고 다음 물음에 답하여라.",
            "score": 0.9,
        }

        result = _merge_side_badges_into_paragraphs([badge, paragraph])

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "paragraph")
        self.assertEqual(result[0]["bbox"], [100, 100, 650, 170])
        self.assertEqual(result[0]["context"]["label_text"], "생각열기")
        self.assertEqual(result[0]["context"]["role_hint"], "problem")

    def test_page_footer_does_not_merge_into_distant_paragraph(self):
        paragraph = {"type": "paragraph", "bbox": [100, 600, 650, 680], "text": "본문", "score": 0.9}
        footer = {"type": "footer", "bbox": [100, 900, 260, 925], "text": "120 좌표평면", "score": 0.8}

        result = _merge_side_badges_into_paragraphs([paragraph, footer])

        self.assertEqual([block["type"] for block in result], ["paragraph", "footer"])

    def test_model_only_detection_skips_all_supplements(self):
        raw_blocks = [{"type": "paragraph", "bbox": [10, 20, 100, 80], "score": 0.91}]
        with patch("src.layout_detection._detect_with_yolo", return_value=raw_blocks), patch(
            "src.layout_detection._postprocess_blocks"
        ) as postprocess:
            result = detect_layout("unused.png", yolo_model_path="model.pt", use_supplements=False)

        self.assertEqual(result, raw_blocks)
        postprocess.assert_not_called()

    def test_numeric_prose_is_not_a_table(self):
        text = (
            "예제 2 담뱃세 인상으로 담배 한 갑당 국민 건강 증진 부담금이 354원에서 841원으로\n"
            "오르면서 2015년 한 해에만 8,473억 원의 부담금이 더 증가했다. 정부가 순수하게 국가 금연\n"
            "지원 서비스 사업에 쓴 예산은 부담금 증가분의 1,475억 원으로 나타났다. 부담금 전체 증가분을\n"
            "기준으로 정부가 지원 사업에 쓴 예산은 몇 퍼센트인지 구하여라."
        )
        block = {"type": "table", "bbox": [100, 100, 720, 250], "text": text, "score": 0.5}

        result = _postprocess_blocks([block], None, None)

        self.assertEqual(result[0]["type"], "paragraph")

    def test_explanatory_calculation_is_a_paragraph(self):
        text = (
            "풀이 전체 부담금 증가분이 8,473억 원이므로 1,475 / 8,473 = 0.174이고 17.4%이다.\n"
            "담배 부담금 전체는 24,757억 원이므로 1,475 / 24,757 = 0.059, 약 5.9%이다."
        )
        block = {"type": "table", "bbox": [100, 100, 720, 190], "text": text, "score": 0.5}

        result = _postprocess_blocks([block], None, None)

        self.assertEqual(result[0]["type"], "paragraph")

    def test_cell_oriented_numeric_text_remains_a_table(self):
        text = "국가\n2008년\n2009년\n2010년\n대한민국\n2,829\n0.708\n6.497\n그리스\n-0.214\n-3.136\n-4.943"

        self.assertTrue(_looks_like_table_text(text))

    def test_two_formulas_on_one_row_are_recovered_as_one_block(self):
        paragraph = {
            "type": "paragraph",
            "bbox": [100, 100, 700, 220],
            "text": "다음 함수의 미분계수를 구하여라.",
            "score": 0.9,
        }
        ocr_lines = [
            {"bbox": [120, 155, 310, 178], "text": "(1) f(x) = 4x - 3", "score": 0.9},
            {"bbox": [390, 155, 590, 178], "text": "(2) f(x) = 3x + 1", "score": 0.9},
        ]

        result = _supplement_nested_formula_lines([paragraph], ocr_lines)
        formulas = [block for block in result if block["type"] == "formula"]

        self.assertEqual(len(formulas), 1)
        self.assertEqual(formulas[0]["bbox"], [120, 155, 590, 178])
        self.assertIn("(1) f(x) = 4x - 3", formulas[0]["text"])
        self.assertIn("(2) f(x) = 3x + 1", formulas[0]["text"])

    def test_short_continuation_attaches_to_the_nearest_paragraph(self):
        upper = {
            "type": "paragraph",
            "bbox": [100, 90, 745, 145],
            "text": "함수의 그래프에서 미분계수를 알아보자.",
            "score": 0.9,
        }
        lower = {
            "type": "paragraph",
            "bbox": [119, 151, 462, 172],
            "text": "함수 y=f(x)에서 x의 값이 a에서 a+dx까지",
            "score": 0.55,
            "detector": "ocr_paragraph_recovery",
        }
        formula = {"type": "formula", "bbox": [250, 188, 520, 220], "text": "f'(a)=lim", "score": 0.8}
        ocr_lines = [
            {"bbox": lower["bbox"], "text": lower["text"], "score": 0.9},
            {"bbox": [120, 174, 300, 190], "text": "변할 때의 평균변화율은", "score": 0.9},
            {"bbox": [255, 190, 515, 217], "text": "f'(a)=lim", "score": 0.9},
        ]

        result = _expand_paragraphs_with_nearby_ocr_lines([upper, lower, formula], ocr_lines)
        paragraphs = [block for block in result if block["type"] == "paragraph"]

        self.assertNotIn("평균변화율", paragraphs[0]["text"])
        self.assertEqual(paragraphs[1]["bbox"], [119, 151, 462, 190])
        self.assertIn("변할 때의 평균변화율은", paragraphs[1]["text"])

    def test_short_role_region_can_split_prose_and_formula(self):
        block = {
            "type": "paragraph",
            "bbox": [100, 100, 700, 220],
            "text": "문제 다음 함수의 미분계수를 구하여라.",
            "score": 0.9,
            "role": "problem",
        }
        ocr_lines = [
            {"bbox": [110, 110, 680, 135], "text": "문제 다음 함수의 미분계수를 구하여라.", "score": 0.9},
            {"bbox": [120, 155, 590, 178], "text": "(1) f(x) = 4x - 3    (2) f(x) = 3x + 1", "score": 0.9},
        ]

        result = _split_mixed_role_blocks([block], ocr_lines)

        self.assertEqual([item["type"] for item in result], ["paragraph", "formula"])

    def test_numeric_sentence_groups_merge_back_into_one_paragraph(self):
        lines = [
            {
                "bbox": [170, 100, 710, 120],
                "text": "회사는 29,000원의 기본요금에 km당 70원을 더한 값으로 하루 동안",
                "score": 0.9,
            },
            {"bbox": [100, 122, 165, 138], "text": "문제 9", "score": 0.9},
            {"bbox": [100, 140, 500, 160], "text": "관광버스를 빌려준다. 다음 물음에 답하여라.", "score": 0.9},
        ]

        groups = _group_lines_by_content(lines)

        self.assertEqual(len(groups), 1)
        reordered = _promote_role_title_text("기본요금은 29,000원이다.\n문제9\n다음 물음에 답하여라.")
        self.assertTrue(reordered.startswith("문제9\n"))
        block = {"type": "paragraph", "bbox": [100, 100, 710, 160], "text": reordered, "score": 0.5}
        self.assertTrue(_normalize_paragraph_text_order([block])[0]["text"].startswith("문제9\n"))

    def test_unit3_profile_merges_split_paragraph_fragments(self):
        blocks = [
            {
                "type": "paragraph",
                "bbox": [100, 100, 520, 135],
                "text": "(3) 자동차가 움직이기 시작해서 정지할 때까지 걸린 시간을 구하시오.",
                "score": 0.91,
            },
            {
                "type": "section_title",
                "bbox": [530, 108, 760, 132],
                "text": "240초 후 속력이 0이 되었으므로",
                "score": 0.80,
            },
            {
                "type": "paragraph",
                "bbox": [530, 138, 760, 165],
                "text": "정지할 때까지 걸린 시간은 240초이다.",
                "score": 0.88,
            },
        ]

        result = refine_blocks_after_ocr(blocks, [], correction_profile="unit3")
        paragraphs = [block for block in result if block["type"] == "paragraph"]

        self.assertEqual(len(paragraphs), 1)
        self.assertIn("(3) 자동차가 움직이기 시작", paragraphs[0]["text"])
        self.assertIn("정지할 때까지 걸린 시간은 240초이다.", paragraphs[0]["text"])

    def test_unit3_profile_does_not_merge_across_figure(self):
        blocks = [
            {"type": "paragraph", "bbox": [100, 100, 420, 130], "text": "그래프를 보고 답하시오.", "score": 0.9},
            {"type": "figure", "bbox": [120, 150, 420, 360], "text": "", "score": 0.9},
            {"type": "paragraph", "bbox": [100, 380, 420, 410], "text": "아래 물음에 답하시오.", "score": 0.9},
        ]

        result = refine_blocks_after_ocr(blocks, [], correction_profile="unit3")
        paragraphs = [block for block in result if block["type"] == "paragraph"]

        self.assertEqual(len(paragraphs), 2)


if __name__ == "__main__":
    unittest.main()
