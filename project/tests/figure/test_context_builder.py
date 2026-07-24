import unittest

from src.analysis.figure.context_builder import FigureContextBuilder


class FigureContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = FigureContextBuilder()
        self.blocks = [
            {"block_id": "p5_b0", "type": "title", "text": "3단원 좌표평면과 그래프"},
            {"block_id": "p5_b1", "type": "section_title", "text": "1. 정비례 관계"},
            {"block_id": "p5_b2", "type": "paragraph", "text": "토끼와 거북이가 경주를 한다."},
            {
                "block_id": "p5_b3",
                "type": "formula",
                "text": "y=ax",
            },
            {"block_id": "p5_b4", "type": "figure", "bbox": [10, 10, 90, 70]},
            {"block_id": "p5_b5", "type": "caption", "text": "반비례 관계를 나타낸 그래프"},
            {"block_id": "p5_b6", "type": "paragraph", "text": "A는 거북이, B는 토끼를 의미한다."},
            {"block_id": "p5_b7", "type": "table", "text": "x | y\n1 | 2"},
        ]
        self.figure_block = self.blocks[4]

    def test_collects_all_nearby_context_fields(self):
        context = self.builder.build(self.blocks, self.figure_block, page_id=5)

        self.assertEqual(context.page_title, "3단원 좌표평면과 그래프")
        self.assertEqual(context.nearest_section_title, "1. 정비례 관계")
        self.assertEqual(context.previous_paragraph, "토끼와 거북이가 경주를 한다.")
        self.assertEqual(context.next_paragraph, "A는 거북이, B는 토끼를 의미한다.")
        self.assertEqual(context.caption, "반비례 관계를 나타낸 그래프")
        self.assertEqual(context.nearby_formula, "y=ax")
        self.assertEqual(context.nearby_table, "x | y\n1 | 2")
        self.assertEqual(context.page_number, 5)
        self.assertEqual(context.figure_block_id, "p5_b4")


    def test_caption_connects_across_a_single_intervening_paragraph(self):
        blocks = [
            {"block_id": "b0", "type": "caption", "text": "한 칸 건너 캡션"},
            {"block_id": "b1", "type": "paragraph", "text": "사이에 낀 문단"},
            {"block_id": "b2", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[2], page_id=1)

        self.assertEqual(context.caption, "한 칸 건너 캡션")
        self.assertEqual(context.context_source.caption_block_id, "b0")

    def test_caption_does_not_cross_a_non_paragraph_boundary(self):
        blocks = [
            {"block_id": "b0", "type": "caption", "text": "표 너머 캡션"},
            {"block_id": "b1", "type": "table", "text": "x | y"},
            {"block_id": "b2", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[2], page_id=1)

        self.assertIsNone(context.caption)

    def test_caption_beyond_max_distance_is_not_connected(self):
        blocks = [
            {"block_id": "b0", "type": "caption", "text": "너무 먼 캡션"},
            {"block_id": "b1", "type": "paragraph", "text": "문단 1"},
            {"block_id": "b2", "type": "paragraph", "text": "문단 2"},
            {"block_id": "b3", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[3], page_id=1)

        self.assertIsNone(context.caption)

    def test_missing_context_fields_are_none_not_fabricated(self):
        blocks = [{"block_id": "b0", "type": "figure", "bbox": [0, 0, 10, 10]}]
        context = self.builder.build(blocks, blocks[0], page_id=1)

        self.assertIsNone(context.page_title)
        self.assertIsNone(context.caption)
        self.assertIsNone(context.previous_paragraph)
        self.assertIsNone(context.next_paragraph)
        self.assertFalse(context.has_any_text())

    def test_figure_ocr_only_includes_text_inside_the_figure_bbox(self):
        blocks = [{"block_id": "b0", "type": "figure", "bbox": [10, 10, 50, 50]}]
        ocr_lines = [
            {"bbox": [15, 15, 25, 25], "text": "y=ax"},
            {"bbox": [100, 100, 120, 120], "text": "바깥쪽 텍스트"},
        ]
        context = self.builder.build(blocks, blocks[0], page_id=1, ocr_lines=ocr_lines)

        self.assertEqual(context.figure_ocr, ("y=ax",))

    def test_respects_reading_order_field_over_list_position(self):
        # Listed out of reading order on purpose; reading_order values say the
        # figure actually comes after the paragraph.
        blocks = [
            {"block_id": "b0", "type": "figure", "bbox": [0, 0, 10, 10], "reading_order": 2},
            {"block_id": "b1", "type": "paragraph", "text": "앞 문단", "reading_order": 1},
        ]
        context = self.builder.build(blocks, blocks[0], page_id=1)

        self.assertEqual(context.previous_paragraph, "앞 문단")

    def test_to_dict_matches_education_context_shape(self):
        context = self.builder.build(self.blocks, self.figure_block, page_id=5)

        self.assertEqual(
            set(context.to_dict()),
            {
                "page_title",
                "chapter_title",
                "section_title",
                "subsection_title",
                "nearest_section_title",
                "previous_paragraph",
                "next_paragraph",
                "previous_paragraphs",
                "next_paragraphs",
                "caption",
                "nearby_formula",
                "nearby_table",
                "figure_ocr",
                "page_number",
                "role_hint",
            },
        )

    def test_window_size_collects_multiple_nearest_paragraphs_in_reading_order(self):
        blocks = [
            {"block_id": "b0", "type": "paragraph", "text": "첫 번째 앞 문단"},
            {"block_id": "b1", "type": "paragraph", "text": "두 번째 앞 문단"},
            {"block_id": "b2", "type": "figure", "bbox": [0, 0, 10, 10]},
            {"block_id": "b3", "type": "paragraph", "text": "첫 번째 뒤 문단"},
            {"block_id": "b4", "type": "paragraph", "text": "두 번째 뒤 문단"},
        ]
        context = self.builder.build(blocks, blocks[2], page_id=1, window_size=2)

        self.assertEqual(context.previous_paragraphs, ("첫 번째 앞 문단", "두 번째 앞 문단"))
        self.assertEqual(context.next_paragraphs, ("첫 번째 뒤 문단", "두 번째 뒤 문단"))
        # Backward-compatible scalar fields still resolve to the nearest one.
        self.assertEqual(context.previous_paragraph, "두 번째 앞 문단")
        self.assertEqual(context.next_paragraph, "첫 번째 뒤 문단")

    def test_default_window_size_matches_old_single_paragraph_behavior(self):
        blocks = [
            {"block_id": "b0", "type": "paragraph", "text": "첫 번째 앞 문단"},
            {"block_id": "b1", "type": "paragraph", "text": "두 번째 앞 문단"},
            {"block_id": "b2", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[2], page_id=1)

        self.assertEqual(context.previous_paragraphs, ("두 번째 앞 문단",))

    def test_title_hierarchy_splits_section_and_subsection_when_both_present(self):
        blocks = [
            {"block_id": "b0", "type": "title", "text": "3단원 좌표평면과 그래프"},
            {"block_id": "b1", "type": "section_title", "text": "1. 정비례 관계"},
            {"block_id": "b2", "type": "section_title", "text": "1-1. 그래프 그리기"},
            {"block_id": "b3", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[3], page_id=1)

        self.assertEqual(context.chapter_title, "3단원 좌표평면과 그래프")
        self.assertEqual(context.section_title, "1. 정비례 관계")
        self.assertEqual(context.subsection_title, "1-1. 그래프 그리기")
        self.assertEqual(context.nearest_section_title, "1-1. 그래프 그리기")

    def test_title_hierarchy_leaves_section_none_when_only_one_heading_found(self):
        blocks = [
            {"block_id": "b0", "type": "title", "text": "3단원 좌표평면과 그래프"},
            {"block_id": "b1", "type": "section_title", "text": "1. 정비례 관계"},
            {"block_id": "b2", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[2], page_id=1)

        self.assertEqual(context.subsection_title, "1. 정비례 관계")
        self.assertIsNone(context.section_title)

    def test_role_hint_read_from_figure_blocks_own_context(self):
        blocks = [{"block_id": "b0", "type": "figure", "bbox": [0, 0, 10, 10], "context": {"role_hint": "problem"}}]
        context = self.builder.build(blocks, blocks[0], page_id=1)

        self.assertEqual(context.role_hint, "problem")
        self.assertEqual(context.context_source.role_hint_block_id, "b0")

    def test_role_hint_read_from_nearby_block_when_figure_has_none(self):
        blocks = [
            {"block_id": "b0", "type": "paragraph", "text": "예제 문단", "context": {"role_hint": "example"}},
            {"block_id": "b1", "type": "figure", "bbox": [0, 0, 10, 10]},
        ]
        context = self.builder.build(blocks, blocks[1], page_id=1)

        self.assertEqual(context.role_hint, "example")
        self.assertEqual(context.context_source.role_hint_block_id, "b0")

    def test_context_source_matches_the_resolved_fields(self):
        context = self.builder.build(self.blocks, self.figure_block, page_id=5)
        source = context.context_source

        self.assertEqual(source.caption_block_id, "p5_b5")
        self.assertEqual(source.previous_block_ids, ("p5_b2",))
        self.assertEqual(source.next_block_ids, ("p5_b6",))
        self.assertEqual(source.nearby_formula_block_id, "p5_b3")
        self.assertEqual(source.nearby_table_block_id, "p5_b7")
        self.assertEqual(
            set(source.to_dict()),
            {
                "title_block_id",
                "section_block_id",
                "subsection_block_id",
                "caption_block_id",
                "nearby_formula_block_id",
                "nearby_table_block_id",
                "previous_block_ids",
                "next_block_ids",
                "role_hint_block_id",
            },
        )


if __name__ == "__main__":
    unittest.main()
