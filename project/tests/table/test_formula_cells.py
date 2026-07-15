import unittest
from unittest.mock import patch

import numpy as np

from src.analysis.table.formula_cells import looks_like_formula_cell, merge_formula_cell_ocr


class LooksLikeFormulaCellTests(unittest.TestCase):
    def test_none_or_empty_text_is_not_formula(self):
        self.assertFalse(looks_like_formula_cell(None))
        self.assertFalse(looks_like_formula_cell(""))

    def test_plain_short_number_or_word_is_not_formula(self):
        self.assertFalse(looks_like_formula_cell("90"))
        self.assertFalse(looks_like_formula_cell("학년"))

    def test_math_symbols_are_formula_signals(self):
        self.assertTrue(looks_like_formula_cell("√2"))
        self.assertTrue(looks_like_formula_cell("a^2"))

    def test_long_bare_digit_run_is_formula_candidate(self):
        self.assertTrue(looks_like_formula_cell("128"))

    def test_short_bare_digit_run_is_not_flagged(self):
        self.assertFalse(looks_like_formula_cell("12"))


class MergeFormulaCellOcrTests(unittest.TestCase):
    def setUp(self):
        self.table_crop = np.zeros((100, 100, 3), dtype=np.uint8)

    def _cell(self, text, bbox):
        return {
            "row": 0,
            "column": 0,
            "row_span": 1,
            "column_span": 1,
            "is_header": False,
            "text": text,
            "bbox": bbox,
        }

    def test_replaces_text_for_flagged_cell_when_recognition_succeeds(self):
        cells = [self._cell("128", [0, 0, 20, 20]), self._cell("학년", [20, 0, 40, 20])]
        with patch(
            "src.analysis.table.formula_cells.recognize_formula_from_crop",
            return_value={"latex": "\\frac{1}{2}", "plain_text": None, "warnings": []},
        ) as mock_recognize:
            merged = merge_formula_cell_ocr(cells, self.table_crop)

        mock_recognize.assert_called_once()
        self.assertEqual(merged[0]["text"], "\\frac{1}{2}")
        self.assertEqual(merged[1]["text"], "학년")

    def test_keeps_original_text_when_recognition_finds_nothing(self):
        cells = [self._cell("128", [0, 0, 20, 20])]
        with patch(
            "src.analysis.table.formula_cells.recognize_formula_from_crop",
            return_value={"latex": None, "plain_text": None, "warnings": []},
        ):
            merged = merge_formula_cell_ocr(cells, self.table_crop)

        self.assertEqual(merged[0]["text"], "128")

    def test_missing_bbox_is_left_untouched_without_calling_recognizer(self):
        cells = [self._cell("128", None)]
        with patch("src.analysis.table.formula_cells.recognize_formula_from_crop") as mock_recognize:
            merged = merge_formula_cell_ocr(cells, self.table_crop)

        mock_recognize.assert_not_called()
        self.assertEqual(merged[0]["text"], "128")

    def test_none_table_crop_returns_cells_unchanged(self):
        cells = [self._cell("128", [0, 0, 20, 20])]
        merged = merge_formula_cell_ocr(cells, None)
        self.assertEqual(merged, cells)

    def test_non_formula_cells_never_call_recognizer(self):
        cells = [self._cell("학년", [0, 0, 20, 20]), self._cell("90", [20, 0, 40, 20])]
        with patch("src.analysis.table.formula_cells.recognize_formula_from_crop") as mock_recognize:
            merged = merge_formula_cell_ocr(cells, self.table_crop)

        mock_recognize.assert_not_called()
        self.assertEqual([cell["text"] for cell in merged], ["학년", "90"])


if __name__ == "__main__":
    unittest.main()
