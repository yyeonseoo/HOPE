import unittest

from src.analysis.formula.formula_analyzer import (
    analyze_formula_blocks,
    normalize_formula_text,
)


class TestFormulaAnalyzer(unittest.TestCase):
    def test_analyze_formula_blocks_returns_only_formula_results(self):
        page = {
            "page_id": 9,
            "blocks": [
                {
                    "block_id": "p9_b1",
                    "type": "paragraph",
                    "bbox": [10, 10, 200, 50],
                    "text": "정비례 관계를 알아보자.",
                    "score": 0.95,
                    "detector": "doclayout_yolo",
                },
                {
                    "block_id": "p9_b2",
                    "type": "formula",
                    "bbox": [20, 80, 180, 120],
                    "text": "y = ax",
                    "score": 0.91,
                    "detector": "doclayout_yolo",
                },
            ],
        }

        results = analyze_formula_blocks(page)

        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result["schema_version"], "1.0.0")
        self.assertEqual(result["page_id"], 9)
        self.assertEqual(result["block_id"], "p9_b2")
        self.assertEqual(result["type"], "formula")
        self.assertEqual(result["bbox"], [20, 80, 180, 120])

        self.assertEqual(result["analysis"]["status"], "success")
        self.assertEqual(result["analysis"]["result"]["kind"], "formula")
        self.assertEqual(result["analysis"]["result"]["latex"], "y=ax")
        self.assertEqual(result["analysis"]["result"]["plain_text"], "y=ax")

        self.assertEqual(result["context"]["previous_block_id"], "p9_b1")
        self.assertIsNone(result["context"]["next_block_id"])

    def test_formula_without_text_returns_partial(self):
        page = {
            "page_id": 10,
            "blocks": [
                {
                    "block_id": "p10_b1",
                    "type": "formula",
                    "bbox": [50, 100, 250, 140],
                    "text": "",
                    "score": 0.8,
                    "detector": "doclayout_yolo",
                }
            ],
        }

        results = analyze_formula_blocks(page)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["analysis"]["status"], "partial")
        self.assertIsNone(results[0]["analysis"]["result"]["latex"])
        self.assertGreater(len(results[0]["warnings"]), 0)

    def test_normalize_formula_text(self):
        self.assertEqual(normalize_formula_text(" y = 2 × x "), r"y=2\timesx")
        self.assertEqual(normalize_formula_text("y ÷ x"), r"y\divx")
        self.assertEqual(normalize_formula_text(""), None)
        self.assertEqual(normalize_formula_text(None), None)


if __name__ == "__main__":
    unittest.main()