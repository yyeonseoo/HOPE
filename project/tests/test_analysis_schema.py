import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "block_analysis.schema.json"


class AnalysisSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_formula_analysis_contract(self):
        payload = {
            "schema_version": "1.0.0",
            "page_id": 15,
            "block_id": "p15_b4",
            "type": "formula",
            "bbox": [105, 329, 745, 426],
            "crop_path": "crops/p15_b4.png",
            "detection": {"model": {"name": "DocLayout-YOLO", "version": None}, "confidence": 0.94},
            "analysis": {
                "status": "success",
                "model": {"name": "PP-FormulaNet", "version": "server"},
                "confidence": 0.88,
                "result": {
                    "kind": "formula",
                    "latex": "\\frac{f(b)-f(a)}{b-a}",
                    "mathml": None,
                    "plain_text": None,
                },
            },
            "context": {
                "previous_block_id": "p15_b3",
                "next_block_id": "p15_b5",
                "caption_block_id": None,
                "nearby_block_ids": ["p15_b3", "p15_b5"],
            },
            "warnings": [],
        }

        self.assertEqual(list(self.validator.iter_errors(payload)), [])

    def test_unknown_fields_are_rejected(self):
        payload = {
            "schema_version": "1.0.0",
            "page_id": 1,
            "block_id": "p1_b1",
            "type": "formula",
            "bbox": [0, 0, 10, 10],
            "detection": {"model": {"name": "detector", "version": None}, "confidence": 0.5},
            "analysis": {
                "status": "failed",
                "model": {"name": "recognizer", "version": None},
                "confidence": None,
                "result": None,
            },
            "context": {
                "previous_block_id": None,
                "next_block_id": None,
                "caption_block_id": None,
                "nearby_block_ids": [],
            },
            "warnings": ["Recognition failed"],
            "unexpected": True,
        }

        self.assertTrue(list(self.validator.iter_errors(payload)))

    def test_formula_block_rejects_table_result(self):
        payload = {
            "schema_version": "1.0.0",
            "page_id": 1,
            "block_id": "p1_b1",
            "type": "formula",
            "bbox": [0, 0, 10, 10],
            "detection": {"model": {"name": "detector", "version": None}, "confidence": 0.5},
            "analysis": {
                "status": "success",
                "model": {"name": "recognizer", "version": None},
                "confidence": 0.8,
                "result": {"kind": "table", "row_count": 0, "column_count": 0, "cells": []},
            },
            "context": {
                "previous_block_id": None,
                "next_block_id": None,
                "caption_block_id": None,
                "nearby_block_ids": [],
            },
            "warnings": [],
        }

        self.assertTrue(list(self.validator.iter_errors(payload)))

    def test_optional_accessibility_description(self):
        payload = {
            "schema_version": "1.0.0",
            "page_id": 2,
            "block_id": "p2_b3",
            "type": "figure",
            "bbox": [20, 30, 400, 300],
            "detection": {"model": {"name": "detector", "version": "1"}, "confidence": 0.9},
            "analysis": {
                "status": "partial",
                "model": {"name": "chart-model", "version": "1"},
                "confidence": 0.7,
                "result": {
                    "kind": "figure",
                    "figure_type": "line_chart",
                    "title": None,
                    "x_axis": {"label": "연도", "unit": "년"},
                    "y_axis": None,
                    "series": [],
                },
            },
            "context": {
                "previous_block_id": "p2_b2",
                "next_block_id": "p2_b4",
                "caption_block_id": None,
                "nearby_block_ids": ["p2_b2", "p2_b4"],
            },
            "description": {
                "status": "not_started",
                "model": None,
                "short_text": None,
                "long_text": None,
                "transcription_notes": None,
                "context_used": False,
                "review_status": "unreviewed",
            },
            "warnings": ["Y축을 인식하지 못함"],
        }

        self.assertEqual(list(self.validator.iter_errors(payload)), [])


if __name__ == "__main__":
    unittest.main()
