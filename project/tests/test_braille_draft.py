import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from braille_draft import CELL_GAP, EMPTY_CELL, NOTE_MARK, edit_text_to_braille_draft  # noqa: E402


def fake(text):
    return f"<{text}>"


class BrailleDraftTests(unittest.TestCase):
    def test_markers_structure_and_formulas(self):
        text = "\n".join([
            "[원본 쪽 120]",
            "[제목] 함수",
            "본문이다.",
            ",'그림 생략,'",
            "[촉도 1]",
            "[수식] y=2x",
            "[수식] \\frac{1}{2}",
            "[표]",
            "x | 1 | ==",
            "[표 끝]",
        ])

        lines = edit_text_to_braille_draft(text, fake).splitlines()

        self.assertTrue(lines[0].startswith("# 자동 점역 초안"))
        self.assertEqual(lines[1:], [
            "[원본 쪽 120]",
            "[제목] <함수>",
            "<본문이다.>",
            f"{NOTE_MARK}<그림 생략>{NOTE_MARK}",
            "[촉도 1]",
            "<y=2x>",
            "[수식 점역 필요] \\frac{1}{2}",
            "[표]",
            CELL_GAP.join(["<x>", "<1>", EMPTY_CELL]),
            "[표 끝]",
        ])

    def test_unsupported_characters_do_not_sink_the_draft(self):
        draft = edit_text_to_braille_draft("❶ a>0일 때 ☃ 표시")
        body = draft.splitlines()[1]
        self.assertIn("[원문: ☃]", body)
        self.assertNotIn("❶", body)

    def test_real_braillify_translates_korean(self):
        draft = edit_text_to_braille_draft("가장 빠른 속력은 20 m/s이다.")
        body = draft.splitlines()[1]
        self.assertTrue(body)
        self.assertTrue(all(0x2800 <= ord(ch) <= 0x28FF for ch in body))


if __name__ == "__main__":
    unittest.main()
