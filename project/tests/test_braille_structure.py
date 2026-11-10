import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from braille_structure import apply_braille_structure  # noqa: E402


def block(block_id, block_type, bbox, text="", order=0):
    return {"block_id": block_id, "type": block_type, "bbox": bbox, "text": text, "reading_order": order}


class ApplyBrailleStructureTests(unittest.TestCase):
    def test_side_by_side_captions_are_split_out_of_a_paragraph(self):
        page = {"page_id": 10, "blocks": [
            block("p10_b2", "figure", [317, 214, 610, 438], order=1),
            block("p10_b3", "figure", [712, 218, 875, 434], order=2),
            block("p10_b4", "paragraph", [291, 437, 900, 537], "[그림 1]      [그림 2]\n[그림 1]을 확대하면", order=3),
        ]}
        lines = [{"bbox": [294, 437, 822, 459], "text": "[그림 1]          [그림 2]"}]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertEqual(
            [(b["block_id"], b["type"], b["text"]) for b in blocks[:4]],
            [("p10_b2", "figure", ""), ("p10_cap1", "caption", "[그림 1]"),
             ("p10_b3", "figure", ""), ("p10_cap2", "caption", "[그림 2]")],
        )
        self.assertEqual(blocks[0]["caption_block_id"], "p10_cap1")
        self.assertEqual(blocks[2]["caption_block_id"], "p10_cap2")
        # Only the caption line is removed; the body sentence that mentions
        # "[그림 1]" stays.
        self.assertEqual(blocks[4]["text"], "[그림 1]을 확대하면")

    def test_long_or_question_lines_under_a_figure_are_not_captions(self):
        page = {"page_id": 3, "blocks": [
            block("p3_b5", "figure", [100, 100, 300, 300], order=1),
            block("p3_b6", "paragraph", [100, 305, 300, 330], "⑴ 속력을 구하시오.", order=2),
        ]}
        lines = [{"bbox": [100, 305, 300, 330], "text": "⑴ 속력을 구하시오."}]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertNotIn("caption", [b["type"] for b in blocks])

    def test_axis_label_of_next_figure_and_wrapped_sidebar_line_are_not_captions(self):
        page = {"page_id": 6, "blocks": [
            block("photo", "figure", [610, 192, 857, 391], order=1),
            block("graph", "figure", [612, 405, 881, 550], order=2),
            block("badge", "figure", [211, 173, 273, 244], order=3),
            block("side", "paragraph", [111, 259, 260, 375], "대관람차는 바퀴 모양\n의 둘레에 작은 공간을", order=4),
        ]}
        lines = [
            {"bbox": [616, 407, 641, 422], "text": "높이"},
            {"bbox": [111, 259, 260, 279], "text": "대관람차는 바퀴 모양"},
        ]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertNotIn("caption", [b["type"] for b in blocks])

    def test_run_of_two_column_blocks_reads_left_column_then_right(self):
        page = {"page_id": 3, "blocks": [
            block("b9", "paragraph", [281, 670, 803, 769], "섞인 글", 1),
            block("b10", "paragraph", [280, 776, 859, 832], "섞인 글", 2),
            block("after", "paragraph", [280, 900, 859, 930], "다음 글", 3),
        ]}
        lines = [
            {"bbox": [587, 713, 798, 735], "text": "가장 빠른 속력은 20 m/s이다."},
            {"bbox": [284, 717, 564, 735], "text": "⑴그래프에서 가장 빠른 경우를"},
            {"bbox": [308, 747, 360, 765], "text": "찾는다."},
            {"bbox": [587, 775, 856, 797], "text": "일정한 속력으로 움직인 시간은"},
            {"bbox": [284, 778, 564, 797], "text": "⑵그래프에서 일정한 구간을"},
            {"bbox": [308, 809, 410, 827], "text": "찾는다."},
            {"bbox": [280, 905, 850, 925], "text": "다음 글"},
        ]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertEqual([b["block_id"] for b in blocks], ["b9", "b10", "b9_r", "b10_r", "after"])
        self.assertEqual(blocks[0]["text"], "⑴그래프에서 가장 빠른 경우를 찾는다.")
        self.assertEqual(blocks[2]["text"], "가장 빠른 속력은 20 m/s이다.")

    def test_single_column_paragraph_with_indented_line_is_not_split(self):
        page = {"page_id": 3, "blocks": [block("p", "paragraph", [280, 700, 600, 770], "글", 1)]}
        lines = [
            {"bbox": [284, 717, 564, 735], "text": "⑴그래프에서 속력이 가장 빠른 경우를"},
            {"bbox": [308, 747, 360, 765], "text": "찾는다."},
        ]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertEqual([b["block_id"] for b in blocks], ["p"])

    def test_wrapped_lines_rejoin_without_breaking_words_or_formulas(self):
        page = {"page_id": 3, "blocks": [
            block("a", "paragraph", [0, 0, 10, 10], "알맞은 상황을 이야기로 만\n드시오.", 1),
            block("b", "paragraph", [0, 20, 10, 30], "거리의 변화를\n나타낸 것이다.", 2),
            block("c", "paragraph", [0, 40, 10, 50], "y=ax의 그래프는 원점 (0, 0)\n을 지난다.", 3),
        ]}

        blocks = apply_braille_structure(page, [], dpi=120)["blocks"]

        self.assertEqual(blocks[0]["text"], "알맞은 상황을 이야기로 만드시오.")
        self.assertEqual(blocks[1]["text"], "거리의 변화를 나타낸 것이다.")
        self.assertEqual(blocks[2]["text"], "y=ax의 그래프는 원점 (0, 0) 을 지난다.")

    def test_invisible_pdf_characters_are_removed(self):
        page = {"page_id": 3, "blocks": [block("q", "paragraph", [0, 0, 10, 10], "⑴ \u200c\x07속력을 구하시오.", 1)]}
        blocks = apply_braille_structure(page, [], dpi=120)["blocks"]
        self.assertEqual(blocks[0]["text"], "⑴ 속력을 구하시오.")

    def test_source_line_goes_after_the_figure_caption(self):
        page = {"page_id": 4, "blocks": [
            block("fig", "figure", [100, 100, 300, 300], order=1),
            block("body", "paragraph", [100, 302, 300, 360], "[그림 3]\n출처: 통계청, 2020\n본문이 이어진다.", order=2),
        ]}
        lines = [
            {"bbox": [100, 305, 200, 322], "text": "[그림 3]"},
            {"bbox": [100, 325, 260, 342], "text": "출처: 통계청, 2020"},
        ]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertEqual([(b["block_id"], b.get("role")) for b in blocks],
                         [("fig", None), ("p4_cap1", None), ("fig_src", "figure_source"), ("body", None)])
        self.assertEqual(blocks[3]["text"], "본문이 이어진다.")

    def test_margin_badge_becomes_a_type_word_title_and_drops_the_badge_figure(self):
        page = {"page_id": 6, "blocks": [
            block("badge", "figure", [211, 173, 273, 244], order=1),
            block("stem", "paragraph", [290, 191, 583, 317], "오른쪽 그림은 대관람차이다.", 2),
            block("q1", "paragraph", [291, 325, 581, 385], "⑴ 높이를 구하시오.", 3),
            block("next", "paragraph", [290, 834, 607, 950], "오른쪽 그림은 회전목마이다.", 4),
        ]}
        texts = {191: "함께 풀기4", 834: "6 문제 해결"}

        def read_region(box):
            return next((text for top, text in texts.items() if box[1] <= top <= box[3]), "")

        blocks = apply_braille_structure(page, [], dpi=120, read_region=read_region)["blocks"]

        self.assertEqual(
            [(b["block_id"], b.get("text")) for b in blocks],
            [("p6_badge1", "(함께 풀기)"), ("stem", "오른쪽 그림은 대관람차이다."), ("q1", "⑴ 높이를 구하시오."),
             ("p6_badge2", "(문제 해결)"), ("next", "오른쪽 그림은 회전목마이다.")],
        )
        self.assertEqual(blocks[3]["context"], {"role_hint": "problem"})

    def test_bracket_caption_inside_the_figure_box_is_pulled_out(self):
        page = {"page_id": 98, "blocks": [block("graph", "figure", [476, 496, 647, 694], order=1)]}
        lines = [
            {"bbox": [480, 600, 520, 615], "text": "4"},
            {"bbox": [520, 670, 600, 690], "text": "〈한계효용곡선〉"},
        ]

        blocks = apply_braille_structure(page, lines, dpi=120)["blocks"]

        self.assertEqual([(b["type"], b["text"]) for b in blocks], [("figure", ""), ("caption", "〈한계효용곡선〉")])
        self.assertEqual(blocks[0]["caption_block_id"], blocks[1]["block_id"])

    def test_margin_note_is_read_after_the_problem_questions(self):
        page = {"page_id": 6, "blocks": [
            block("stem", "paragraph", [290, 191, 583, 317], "오른쪽 그림은 대관람차이다. 다음 물음에 답하시오.", 1),
            block("side", "paragraph", [111, 259, 260, 375], "대관람차는 회전식 놀이 기구이다.", 2),
            block("q1", "paragraph", [291, 325, 581, 385], "⑴ 높이를 구하시오.", 3),
            block("q2", "paragraph", [292, 392, 579, 451], "⑵ 시간을 구하시오.", 4),
            block("solve", "paragraph", [301, 566, 879, 758], "⑴그래프에서 가장 높은 곳을 찾는다.", 5),
        ]}

        blocks = apply_braille_structure(page, [], dpi=120)["blocks"]

        self.assertEqual([b["block_id"] for b in blocks], ["stem", "q1", "q2", "side", "solve"])
        self.assertEqual(blocks[3]["role"], "sidebar")

    def test_figure_inside_a_question_run_moves_before_the_first_question(self):
        page = {"page_id": 6, "blocks": [
            block("stem", "paragraph", [0, 0, 10, 10], "다음 물음에 답하시오.", 1),
            block("q1", "paragraph", [0, 20, 10, 30], "⑴ 높이를 구하시오.", 2),
            block("q2", "paragraph", [0, 40, 10, 50], "⑵ 시간을 구하시오.", 3),
            block("fig", "figure", [50, 40, 90, 80], "", 4),
            block("q3", "paragraph", [0, 60, 10, 70], "⑶ 횟수를 구하시오.", 5),
        ]}

        blocks = apply_braille_structure(page, [], dpi=120)["blocks"]

        self.assertEqual([b["block_id"] for b in blocks], ["stem", "fig", "q1", "q2", "q3"])
        changes = page["structure_changes"]
        self.assertEqual([(c["kind"], c["block_id"]) for c in changes], [("moved", "fig")])
        self.assertEqual([b["reading_order"] for b in blocks], [1, 2, 3, 4, 5])
        self.assertEqual([b["original_reading_order"] for b in blocks], [1, 4, 2, 3, 5])

    def test_lowest_page_line_paragraph_becomes_footer(self):
        page = {"page_id": 10, "blocks": [
            block("body", "paragraph", [0, 0, 100, 50], "본문이다.", 1),
            block("line", "paragraph", [752, 1231, 901, 1253], "1. 좌표평면과 그래프  127", 2),
        ]}

        blocks = apply_braille_structure(page, [], dpi=120)["blocks"]

        self.assertEqual(blocks[1]["type"], "footer")
        self.assertEqual(blocks[0]["type"], "paragraph")


if __name__ == "__main__":
    unittest.main()
