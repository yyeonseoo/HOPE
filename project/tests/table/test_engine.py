import unittest
from unittest.mock import MagicMock

import numpy as np

from src.analysis.table.engine import run_table_engine


class RunTableEngineTests(unittest.TestCase):
    def _make_engine(self, predict_return):
        engine = MagicMock()
        engine.predict.return_value = iter(predict_return)
        return engine

    def test_returns_none_when_predict_yields_nothing(self):
        engine = self._make_engine([])
        self.assertIsNone(run_table_engine(engine, np.zeros((10, 10, 3), dtype=np.uint8)))

    def test_returns_none_when_no_table_regions_found(self):
        engine = self._make_engine([{"table_res_list": []}])
        self.assertIsNone(run_table_engine(engine, np.zeros((10, 10, 3), dtype=np.uint8)))

    def test_returns_html_for_single_table(self):
        engine = self._make_engine(
            [
                {
                    "table_res_list": [
                        {
                            "pred_html": "<table><tr><td>A1</td></tr></table>",
                            "cell_box_list": [[0, 0, 10, 10]],
                        }
                    ]
                }
            ]
        )
        result = run_table_engine(engine, np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result["html"], "<table><tr><td>A1</td></tr></table>")
        self.assertIsNone(result["confidence"])

    def test_picks_largest_table_when_multiple_found(self):
        small = {
            "pred_html": "<table><tr><td>small</td></tr></table>",
            "cell_box_list": [[0, 0, 5, 5]],
        }
        large = {
            "pred_html": "<table><tr><td>large</td></tr></table>",
            "cell_box_list": [[0, 0, 100, 100]],
        }
        engine = self._make_engine([{"table_res_list": [small, large]}])
        result = run_table_engine(engine, np.zeros((10, 10, 3), dtype=np.uint8))
        self.assertEqual(result["html"], large["pred_html"])

    def test_returns_none_when_best_table_has_no_html(self):
        engine = self._make_engine(
            [{"table_res_list": [{"pred_html": "", "cell_box_list": [[0, 0, 5, 5]]}]}]
        )
        self.assertIsNone(run_table_engine(engine, np.zeros((10, 10, 3), dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
