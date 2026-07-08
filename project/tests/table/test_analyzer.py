import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.analysis.table.analyzer import analyze_table_block


class AnalyzeTableBlockTests(unittest.TestCase):
    def setUp(self):
        self._tmp_dir = tempfile.TemporaryDirectory()
        self.image_path = Path(self._tmp_dir.name) / "page.png"
        cv2.imwrite(str(self.image_path), np.zeros((200, 200, 3), dtype=np.uint8))

    def tearDown(self):
        self._tmp_dir.cleanup()

    def test_success_path_uses_injected_engine(self):
        fake_engine = object()
        with patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": "<table><tr><td>A1</td></tr></table>", "confidence": None},
        ) as mock_run:
            output = analyze_table_block(self.image_path, [0, 0, 100, 100], engine=fake_engine)

        mock_run.assert_called_once()
        self.assertIs(mock_run.call_args[0][0], fake_engine)
        self.assertEqual(output["analysis"]["status"], "success")
        self.assertEqual(output["analysis"]["result"]["cells"][0]["text"], "A1")

    def test_lazy_loads_engine_when_none_injected(self):
        with patch(
            "src.analysis.table.analyzer._load_table_engine", return_value="loaded-engine"
        ) as mock_load, patch(
            "src.analysis.table.analyzer.run_table_engine",
            return_value={"html": "<table><tr><td>A1</td></tr></table>", "confidence": None},
        ) as mock_run:
            analyze_table_block(self.image_path, [0, 0, 100, 100])

        mock_load.assert_called_once()
        self.assertEqual(mock_run.call_args[0][0], "loaded-engine")

    def test_invalid_bbox_fails_without_calling_engine(self):
        with patch("src.analysis.table.analyzer.run_table_engine") as mock_run:
            output = analyze_table_block(self.image_path, [50, 50, 50, 50], engine=object())

        mock_run.assert_not_called()
        self.assertEqual(output["analysis"]["status"], "failed")
        self.assertIsNone(output["analysis"]["result"])
        self.assertEqual(len(output["warnings"]), 1)

    def test_engine_exception_is_absorbed_as_failed_status(self):
        with patch(
            "src.analysis.table.analyzer.run_table_engine", side_effect=RuntimeError("boom")
        ):
            output = analyze_table_block(self.image_path, [0, 0, 100, 100], engine=object())

        self.assertEqual(output["analysis"]["status"], "failed")
        self.assertIsNone(output["analysis"]["result"])
        self.assertIn("boom", output["warnings"][0])

    def test_engine_finds_no_table(self):
        with patch("src.analysis.table.analyzer.run_table_engine", return_value=None):
            output = analyze_table_block(self.image_path, [0, 0, 100, 100], engine=object())

        self.assertEqual(output["analysis"]["status"], "failed")
        self.assertIsNone(output["analysis"]["result"])


if __name__ == "__main__":
    unittest.main()
