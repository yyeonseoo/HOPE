import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from evaluate_treatments import score  # noqa: E402


class ScoreTests(unittest.TestCase):
    def test_scores_each_method_against_acceptable_sets(self):
        suggestions = {
            "a": {"rule": "tactile_graphic", "tree": "transcriber_note", "hybrid": "tactile_graphic"},
            "b": {"rule": "transcriber_note", "tree": "omit", "hybrid": "omit"},
            "c": {"rule": "omit", "tree": None, "hybrid": "omit"},
        }
        acceptable = {"a": ["tactile_graphic"], "b": ["omit", "transcriber_note"], "d": ["omit"]}

        report = score(suggestions, acceptable)

        self.assertEqual((report["rule"]["correct"], report["rule"]["n"]), (2, 2))
        self.assertEqual((report["tree"]["correct"], report["tree"]["n"]), (1, 2))
        self.assertEqual(report["tree"]["wrong"], ["a"])
        self.assertEqual(report["missing_from_results"], ["d"])


if __name__ == "__main__":
    unittest.main()
