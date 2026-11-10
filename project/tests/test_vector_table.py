import sys
import unittest
from pathlib import Path

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vector_table import cells_from_layout  # noqa: E402


def word(x0, y0, text):
    return fitz.Rect(x0, y0, x0 + 6, y0 + 10), text


class CellsFromLayoutTests(unittest.TestCase):
    def test_worksheet_table_with_header_boxes_and_blank_boxes(self):
        boxes = [
            fitz.Rect(358, 185, 375, 200.5), fitz.Rect(358, 200.5, 375, 216),  # row headers
            fitz.Rect(429, 203, 449, 214), fitz.Rect(454, 203, 474, 214),  # blank answer boxes
        ]
        words = [
            word(362, 187, "x"), word(382, 187, "1"), word(408, 187, "2"), word(433, 187, "3"), word(459, 187, "4"),
            word(362, 203, "y"), word(382, 203, "4"), word(408, 203, "8"),
            word(396, 170, "2배"),  # arrow note outside the table rows
        ]

        cells = cells_from_layout(boxes, words)

        rows = [[c["text"] for c in cells if c["row"] == r] for r in range(2)]
        self.assertEqual(rows, [["x", "1", "2", "3", "4"], ["y", "4", "8", "", ""]])
        self.assertTrue(all(c["is_header"] == (c["column"] == 0) for c in cells))

    def test_graph_without_stacked_header_boxes_is_not_a_table(self):
        boxes = [fitz.Rect(10, 10, 20, 20), fitz.Rect(100, 50, 110, 60)]
        self.assertIsNone(cells_from_layout(boxes, [word(12, 12, "O")]))


if __name__ == "__main__":
    unittest.main()
