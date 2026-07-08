import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from jsonschema import Draft202012Validator

from src.analysis.table.analyzer import analyze_table_blocks, get_neighbor_block_id
from src.analysis.table.crop import crop_and_save_table_block

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "block_analysis.schema.json"


def _page(blocks):
    return {"page_id": 9, "blocks": blocks}


class GetNeighborBlockIdTests(unittest.TestCase):
    def test_returns_block_id_at_index(self):
        blocks = [{"block_id": "p9_b1"}, {"block_id": "p9_b2"}]
        self.assertEqual(get_neighbor_block_id(blocks, 1), "p9_b2")

    def test_returns_none_out_of_range(self):
        blocks = [{"block_id": "p9_b1"}]
        self.assertIsNone(get_neighbor_block_id(blocks, -1))
        self.assertIsNone(get_neighbor_block_id(blocks, 1))


class AnalyzeTableBlocksTests(unittest.TestCase):
    def test_returns_only_table_results_with_full_schema_shape(self):
        page = _page(
            [
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
                    "type": "table",
                    "bbox": [20, 80, 180, 120],
                    "text": "1 2\n3 4",
                    "score": 0.88,
                    "detector": "doclayout_yolo",
                },
            ]
        )

        with patch(
            "src.analysis.table.analyzer.analyze_table_block",
            return_value={
                "analysis": {
                    "status": "success",
                    "model": {"name": "TableRecognitionPipelineV2", "version": "paddleocr-3.7"},
                    "confidence": None,
                    "result": {
                        "kind": "table",
                        "row_count": 2,
                        "column_count": 2,
                        "cells": [
                            {"row": 0, "column": 0, "row_span": 1, "column_span": 1, "text": "1", "is_header": False},
                            {"row": 0, "column": 1, "row_span": 1, "column_span": 1, "text": "2", "is_header": False},
                            {"row": 1, "column": 0, "row_span": 1, "column_span": 1, "text": "3", "is_header": False},
                            {"row": 1, "column": 1, "row_span": 1, "column_span": 1, "text": "4", "is_header": False},
                        ],
                    },
                },
                "warnings": [],
            },
        ):
            results = analyze_table_blocks(page, page_image_path="dummy.png")

        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result["schema_version"], "1.0.0")
        self.assertEqual(result["page_id"], 9)
        self.assertEqual(result["block_id"], "p9_b2")
        self.assertEqual(result["type"], "table")
        self.assertEqual(result["bbox"], [20, 80, 180, 120])
        self.assertEqual(result["detection"]["model"]["name"], "doclayout_yolo")
        self.assertEqual(result["detection"]["confidence"], 0.88)
        self.assertEqual(result["analysis"]["status"], "success")
        self.assertEqual(result["context"]["previous_block_id"], "p9_b1")
        self.assertIsNone(result["context"]["next_block_id"])
        self.assertEqual(result["context"]["nearby_block_ids"], ["p9_b1"])
        self.assertEqual(result["warnings"], [])

    def test_no_page_image_path_fails_without_calling_engine(self):
        page = _page(
            [{"block_id": "p9_b1", "type": "table", "bbox": [0, 0, 10, 10], "score": 0.9, "detector": "doclayout_yolo"}]
        )

        with patch("src.analysis.table.analyzer.analyze_table_block") as mock_analyze:
            results = analyze_table_blocks(page, page_image_path=None)

        mock_analyze.assert_not_called()
        self.assertEqual(results[0]["analysis"]["status"], "failed")
        self.assertIsNone(results[0]["crop_path"])
        self.assertEqual(len(results[0]["warnings"]), 1)

    def test_warns_when_layout_tagged_table_does_not_look_like_a_real_table(self):
        # Regression case: the layout model itself tagged a fill-in-the-blank
        # paragraph ("2배, 3배... 1/2배, 1/3배" -- repeated short fraction-y
        # fragments) as `table` with high confidence. We can't retag it back
        # to paragraph from here (schema requires a table-type record for a
        # table-type block), but we can flag it for review the same way a
        # bad figure/formula reclassification would be rejected.
        page = _page(
            [
                {
                    "block_id": "p16_b6",
                    "type": "table",
                    "bbox": [180, 364, 685, 489],
                    "score": 0.94,
                    "detector": "doclayout_yolo",
                },
            ]
        )
        with patch(
            "src.analysis.table.analyzer.analyze_table_block",
            return_value={
                "analysis": {
                    "status": "success",
                    "model": {"name": "TableRecognitionPipelineV2", "version": "paddleocr-3.7"},
                    "confidence": None,
                    "result": {
                        "kind": "table",
                        "row_count": 1,
                        "column_count": 1,
                        "cells": [
                            {
                                "row": 0,
                                "column": 0,
                                "row_span": 1,
                                "column_span": 1,
                                "text": "두 변수 x, y에서 x가 2배, 3배, 4배로 변함에 따라 y가 1/2배, 1/3배로 변한다",
                                "is_header": False,
                            }
                        ],
                    },
                },
                "warnings": [],
            },
        ):
            results = analyze_table_blocks(page, page_image_path="dummy.png")

        self.assertEqual(len(results), 1)
        self.assertTrue(any("검수" in warning for warning in results[0]["warnings"]))

    def test_missing_detector_and_score_default_to_null(self):
        page = _page([{"block_id": "p9_b1", "type": "table", "bbox": [0, 0, 10, 10]}])

        with patch(
            "src.analysis.table.analyzer.analyze_table_block",
            return_value={
                "analysis": {"status": "failed", "model": {"name": "x", "version": None}, "confidence": None, "result": None},
                "warnings": ["표 영역을 인식하지 못했습니다."],
            },
        ):
            results = analyze_table_blocks(page, page_image_path="dummy.png")

        self.assertEqual(results[0]["detection"]["model"]["name"], "model-a")
        self.assertIsNone(results[0]["detection"]["confidence"])


class CropAndSaveTableBlockTests(unittest.TestCase):
    def test_creates_padded_image_file(self):
        with TemporaryDirectory() as temp_dir:
            page_image_path = Path(temp_dir) / "page9.png"
            cv2.imwrite(str(page_image_path), np.full((100, 100, 3), 255, dtype=np.uint8))

            block = {"block_id": "p9_b3", "type": "table", "bbox": [10, 20, 60, 50]}
            crop_path = crop_and_save_table_block(str(page_image_path), block, page_id=9)

            self.assertIsNotNone(crop_path)
            crop_file = Path(crop_path)
            self.assertTrue(crop_file.exists())

            saved = cv2.imread(str(crop_file))
            # bbox [10,20,60,50] padded by 30 on each side, clamped to 100x100 source:
            # x: [10-30, 60+30) -> [0, 90) -> 90 wide, y: [20-30, 50+30) -> [0, 80) -> 80 tall
            self.assertEqual(saved.shape[:2], (80, 90))
            crop_file.unlink(missing_ok=True)

    def test_returns_none_without_page_image_path(self):
        block = {"block_id": "p9_b1", "bbox": [0, 0, 10, 10]}
        self.assertIsNone(crop_and_save_table_block(None, block, page_id=9))


class AnalyzeTableBlocksSchemaConformanceTests(unittest.TestCase):
    """Unlike test_schema_conformance.py (which wraps build_table_analysis's
    partial output in a hand-built dummy record), this validates
    analyze_table_blocks' own complete record end-to-end, since it now
    assembles page_id/detection/context/crop_path itself."""

    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_full_record_validates_without_page_image(self):
        page = _page(
            [
                {"block_id": "p9_b1", "type": "paragraph", "bbox": [0, 0, 10, 10], "score": 0.9, "detector": "doclayout_yolo"},
                {"block_id": "p9_b2", "type": "table", "bbox": [20, 20, 60, 60], "score": 0.8, "detector": "doclayout_yolo"},
            ]
        )
        results = analyze_table_blocks(page, page_image_path=None)
        self.assertEqual(list(self.validator.iter_errors(results[0])), [])

    def test_full_record_validates_with_success_result(self):
        page = _page(
            [{"block_id": "p9_b2", "type": "table", "bbox": [20, 20, 60, 60], "score": 0.8, "detector": "doclayout_yolo"}]
        )
        with patch(
            "src.analysis.table.analyzer.analyze_table_block",
            return_value={
                "analysis": {
                    "status": "success",
                    "model": {"name": "TableRecognitionPipelineV2", "version": "paddleocr-3.7"},
                    "confidence": None,
                    "result": {
                        "kind": "table",
                        "row_count": 1,
                        "column_count": 1,
                        "cells": [{"row": 0, "column": 0, "row_span": 1, "column_span": 1, "text": "1", "is_header": False}],
                    },
                },
                "warnings": [],
            },
        ):
            results = analyze_table_blocks(page, page_image_path="dummy.png")
        self.assertEqual(list(self.validator.iter_errors(results[0])), [])


if __name__ == "__main__":
    unittest.main()
