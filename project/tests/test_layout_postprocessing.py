import unittest

from src.layout_detection import (
    _expand_paragraphs_with_nearby_ocr_lines,
    _looks_like_table_text,
    _normalize_paragraph_text_order,
    _group_lines_by_content,
    _postprocess_blocks,
    _promote_role_title_text,
    _split_mixed_role_blocks,
    _supplement_nested_formula_lines,
)


class LayoutPostprocessingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
