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

    def test_formula_block_reclassified_when_real_table_found(self):
        page = self._page_with_block("formula")
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

    def test_reclassified_record_always_forces_needs_review(self):
        # A reclassification is always a candidate, not a confirmed fact --
        # even a fully-filled, clean-content grid should still come back
        # flagged for a human to confirm.
        page = self._page_with_block("formula")
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={
                "html": "<table><tr><td>A1</td><td>A2</td></tr><tr><td>B1</td><td>B2</td></tr></table>",
                "confidence": None,
            },
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results[0]["description"]["review_status"], "needs_review")

    def test_figure_blocks_are_never_reclassified(self):
        # A full 17-page real-PDF batch review (2026-07-07) showed `figure`
        # reclassification kept promoting coordinate-plane graphs to
        # "table" -- a distance-time graph's axis labels, a y=a/x graph's
        # equation/coordinate annotation, then a CO2-concentration line
        # chart's legend text each slipped past a different round of
        # content-signal checks. Rather than keep chasing new graph
        # vocabulary, `figure` was dropped from RECLASSIFIABLE_TYPES
        # entirely -- it's never even given to the table engine, regardless
        # of how table-like its content would look.
        page = self._page_with_block("figure")
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={
                "html": "<table><tr><td>A1</td><td>A2</td></tr><tr><td>B1</td><td>B2</td></tr></table>",
                "confidence": None,
            },
        ) as mock_run:
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        mock_run.assert_not_called()
        self.assertEqual(results, [])

    def test_formula_block_not_reclassified_when_grid_too_small(self):
        page = self._page_with_block("formula")
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": "<table><tr><td>y=ax</td></tr></table>", "confidence": None},
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results, [])

    def test_formula_block_not_reclassified_when_grid_is_mostly_blank(self):
        # Mimics a chart's axis gridlines getting misread as a table: a
        # decent-sized grid, but only a couple of cells actually have text
        # (axis tick labels), the rest blank. Should NOT be reclassified
        # even though row/column counts alone would pass.
        page = self._page_with_block("formula")
        sparse_html = (
            "<table>"
            "<tr><td></td><td></td><td>0</td></tr>"
            "<tr><td></td><td></td><td></td></tr>"
            "<tr><td>10</td><td></td><td></td></tr>"
            "</table>"
        )
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": sparse_html, "confidence": None},
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results, [])

    def test_formula_block_not_reclassified_when_cells_are_graph_annotations(self):
        # Regression case: a y=a/x coordinate-plane graph's axis labels,
        # equation, and labeled point ("y= a x o", "(1, a)", etc.) densely
        # filled a 3x2 grid (passing the filled-ratio check) but is not an
        # actual data table -- real production output that was wrongly
        # reclassified before the formula-signal check was added.
        page = self._page_with_block("formula")
        graph_html = (
            "<table>"
            "<tr><td>y= a x o</td><td>1-a x</td></tr>"
            "<tr><td rowspan=\"2\">-1 a</td><td>(1, a)</td></tr>"
            "<tr><td></td></tr>"
            "</table>"
        )
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": graph_html, "confidence": None},
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results, [])

    def test_formula_block_not_reclassified_when_cells_are_graph_axis_labels(self):
        # Regression case: real output from a distance-time coordinate-plane
        # graph. Its axis labels/origin marker ("거리", "시간", "O", "↑")
        # densely filled a 2x2 grid (passing filled-ratio) and contained no
        # "=" or coordinate-pair syntax (passing contains_formula_signal),
        # but is still not an actual data table.
        page = self._page_with_block("formula")
        graph_html = (
            "<table>"
            "<tr><td>2 거리↑ 0</td><td></td></tr>"
            "<tr><td>0</td><td>시간</td></tr>"
            "</table>"
        )
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": graph_html, "confidence": None},
        ):
            results = analyze_table_blocks(page, str(self.image_path), engine=object())

        self.assertEqual(results, [])

    def test_formula_block_not_reclassified_when_engine_finds_nothing(self):
        page = self._page_with_block("formula")
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
