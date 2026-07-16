import unittest

from src.analysis.figure.context import (
    _normalize_stacked_fraction,
    augment_caption_with_context,
    build_figure_context,
)


class FigureContextTests(unittest.TestCase):
    def test_normalizes_pdf_stacked_fraction_generically(self):
        self.assertEqual(_normalize_stacked_fraction("y= 6\nx의 그래프"), "y=6/x의 그래프")
        self.assertEqual(_normalize_stacked_fraction("y= a\nx 의 관계"), "y=a/x 의 관계")

    def test_prefers_adjacent_caption_and_paragraph(self):
        blocks = [
            {"block_id": "p3_b1", "type": "paragraph", "bbox": [0, 0, 200, 40], "text": "y=a/x를 알아보자."},
            {"block_id": "p3_b2", "type": "figure", "bbox": [20, 50, 180, 180]},
            {"block_id": "p3_b3", "type": "caption", "bbox": [20, 185, 180, 205], "text": "반비례 그래프"},
            {"block_id": "p3_b4", "type": "paragraph", "bbox": [0, 500, 200, 540], "text": "다른 문제의 설명"},
        ]
        context = build_figure_context(blocks, 1)
        self.assertEqual([item["block_id"] for item in context[:2]], ["p3_b3", "p3_b1"])

    def test_uses_formula_semantic_result_when_block_text_is_empty(self):
        blocks = [
            {"block_id": "p4_b1", "type": "formula", "bbox": [0, 0, 100, 30]},
            {"block_id": "p4_b2", "type": "figure", "bbox": [0, 35, 180, 180]},
        ]
        semantic = [{
            "block_id": "p4_b1",
            "analysis": {"result": {"kind": "formula", "plain_text": "y=a/x", "latex": "y=\\frac{a}{x}"}},
        }]
        context = build_figure_context(blocks, 1, semantic)
        self.assertEqual(context[0]["text"], "y=a/x")

    def test_pairs_side_by_side_figures_with_their_own_column_text(self):
        blocks = [
            {"block_id": "p2_b2", "type": "figure", "bbox": [293, 143, 535, 308]},
            {"block_id": "p2_b3", "type": "figure", "bbox": [615, 143, 854, 309]},
            {"block_id": "p2_b4", "type": "paragraph", "bbox": [315, 329, 532, 408], "text": "건전지 개수에 따른 전압 변화"},
            {"block_id": "p2_b5", "type": "paragraph", "bbox": [636, 329, 859, 408], "text": "시간에 따른 이동 거리 변화"},
        ]

        left = build_figure_context(blocks, 0)
        right = build_figure_context(blocks, 1)

        self.assertIn("p2_b4", [item["block_id"] for item in left])
        self.assertNotIn("p2_b5", [item["block_id"] for item in left])
        self.assertEqual(right[0]["block_id"], "p2_b5")
        self.assertNotIn("p2_b4", [item["block_id"] for item in right])

    def test_keeps_full_width_context_when_no_same_row_column_pair_exists(self):
        blocks = [
            {"block_id": "p1_b1", "type": "figure", "bbox": [300, 100, 850, 400]},
            {"block_id": "p1_b2", "type": "paragraph", "bbox": [90, 430, 900, 520], "text": "A는 거북이이고 B는 토끼이다."},
        ]
        context = build_figure_context(blocks, 0)
        self.assertEqual(context[0]["block_id"], "p1_b2")

    def test_unrelated_context_cannot_replace_photo_caption(self):
        base = "해안과 산을 따라 놓인 도로와 다리가 보이는 사진이다."
        context = [{
            "block_id": "p2_b4", "type": "paragraph", "score": 1.0,
            "text": "전기 사용량이 증가하면 전기 요금도 일정한 비율로 증가한다.",
        }]
        result, used = augment_caption_with_context(base, "photo", context)
        self.assertEqual(result, base)
        self.assertEqual(used, ())

    def test_relevant_paragraph_only_appends_to_photo_caption(self):
        base = "해안과 산을 따라 놓인 도로와 다리가 보이는 사진이다."
        context = [{
            "block_id": "p2_b5", "type": "paragraph", "score": 1.0,
            "text": "해안 도로와 다리는 산과 바다 사이를 연결한다.",
        }]
        result, used = augment_caption_with_context(base, "photo", context)
        self.assertTrue(result.startswith(base))
        self.assertIn("주변 설명에서는", result)
        self.assertEqual(used, ("p2_b5",))

    def test_graph_formula_is_appended_with_context_attribution(self):
        base = "좌표평면에 서로 마주 보는 두 갈래의 곡선이 있다."
        context = [{
            "block_id": "p14_b3", "type": "paragraph", "score": 1.0,
            "text": "함수 y=a/x의 그래프를 살펴보자.",
        }]
        result, used = augment_caption_with_context(base, "graph", context)
        self.assertEqual(result, base + " 주변 설명에서는 y=a/x를 다룬다.")
        self.assertEqual(used, ("p14_b3",))


if __name__ == "__main__":
    unittest.main()
