import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.analysis.table.analyzer import analyze_table_blocks


class ReclassifyAsTableTests(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self._tmp_dir.name) / "page.png"
        cv2.imwrite(str(self.image_path), np.zeros((200, 200, 3), dtype=np.uint8))

    def tearDown(self):
        self._tmp_dir.cleanup()

    def _page_with_block(self, block_type):
        return {
            "page_id": 1,
            "blocks": [
                {
                    "block_id": "p1_b1",
                    "type": block_type,
                    "bbox": [0, 0, 100, 100],
                    "detector": "doclayout_yolo",
                    "score": 0.9,
                },
            ],
        }

    def test_figure_block_reclassified_when_real_table_found(self):
        page = self._page_with_block("figure")
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={
                "html": "<table><tr><td>A1</td><td>A2</td></tr><tr><td>B1</td><td>B2</td></tr></table>",
                "confidence": None,
            },
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["type"], "table")
        self.assertEqual(results[0]["block_id"], "p1_b1")
        self.assertTrue(any("재분류" in warning for warning in results[0]["warnings"]))

    def test_formula_block_not_reclassified_when_grid_too_small(self):
        page = self._page_with_block("formula")
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": "<table><tr><td>y=ax</td></tr></table>", "confidence": None},
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results, [])

    def test_figure_block_not_reclassified_when_engine_finds_nothing(self):
        page = self._page_with_block("figure")
        with patch("src.analysis.table.analyzer.run_table_engine", return_value=None):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results, [])

    def test_other_block_types_are_ignored_entirely(self):
        page = self._page_with_block("paragraph")
        with patch("src.analysis.table.analyzer.run_table_engine") as mock_run:
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        mock_run.assert_not_called()
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
