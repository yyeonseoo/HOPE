import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.analysis.formula.formula_analyzer import (
    analyze_formula_blocks,
    crop_formula_block,
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
        self.assertEqual(result["analysis"]["result"]["plain_text"], "y = ax")

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

    def test_parse_formula_with_condition_text(self):
        page = {
            "page_id": 9,
            "blocks": [
                {
                    "block_id": "p9_b2",
                    "type": "paragraph",
                    "bbox": [286, 168, 875, 195],
                    "text": "일반적으로 x와 y가 정비례할 때, x와 y 사이에는 다음과 같은 식이 성립한다.",
                    "score": 0.856,
                    "detector": "doclayout_yolo",
                },
                {
                    "block_id": "p9_b3",
                    "type": "formula",
                    "bbox": [469, 213, 683, 240],
                    "text": "y=ax (단, a는 0이 아니다.)",
                    "score": 0.584,
                    "detector": "doclayout_yolo",
                },
            ],
        }

        results = analyze_formula_blocks(page)

        self.assertEqual(len(results), 1)

        result = results[0]
        self.assertEqual(result["page_id"], 9)
        self.assertEqual(result["block_id"], "p9_b3")
        self.assertEqual(result["type"], "formula")
        self.assertEqual(result["analysis"]["status"], "success")
        self.assertEqual(result["analysis"]["result"]["latex"], "y=ax")
        self.assertEqual(
            result["analysis"]["result"]["plain_text"],
            "y=ax (단, a는 0이 아니다.)",
        )

    def test_crop_formula_block_creates_image_file(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("Pillow is not installed.")

        with TemporaryDirectory() as temp_dir:
            temp_dir_path = Path(temp_dir)
            page_image_path = temp_dir_path / "page9.png"

            image = Image.new("RGB", (100, 100), "white")
            image.save(page_image_path)

            block = {
                "block_id": "p9_b3",
                "type": "formula",
                "bbox": [10, 20, 60, 50],
                "text": "y=ax",
            }

            crop_path = crop_formula_block(
                page_image_path=str(page_image_path),
                block=block,
                page_id=9,
            )

            self.assertIsNotNone(crop_path)

            crop_file = Path(crop_path)
            self.assertTrue(crop_file.exists())

            with Image.open(crop_file) as cropped:
                self.assertEqual(cropped.size, (50, 30))

            crop_file.unlink(missing_ok=True)

    def test_normalize_formula_text(self):
        self.assertEqual(normalize_formula_text(" y = 2 × x "), r"y=2\timesx")
        self.assertEqual(normalize_formula_text("y ÷ x"), r"y\divx")
        self.assertEqual(normalize_formula_text(""), None)
        self.assertEqual(normalize_formula_text(None), None)


if __name__ == "__main__":
    unittest.main()