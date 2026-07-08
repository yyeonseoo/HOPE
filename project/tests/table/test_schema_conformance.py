import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from src.analysis.table.normalize import build_table_analysis

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "block_analysis.schema.json"


def _wrap_as_block_record(analysis_output: dict) -> dict:
    """Embed a build_table_analysis() output into a full block_analysis
    record with dummy page/detection/context fields, matching the shape the
    integration layer is expected to assemble around it. This lets us verify
    our own output slots into the shared contract without ever modifying
    schemas/block_analysis.schema.json."""
    return {
        "schema_version": "1.0.0",
        "page_id": 8,
        "block_id": "p8_b3",
        "type": "table",
        "bbox": [87, 689, 905, 1324],
        "crop_path": None,
        "detection": {"model": {"name": "DocLayout-YOLO", "version": None}, "confidence": 0.81},
        "analysis": analysis_output["analysis"],
        "context": {
            "previous_block_id": None,
            "next_block_id": None,
            "caption_block_id": None,
            "nearby_block_ids": [],
        },
        "warnings": analysis_output["warnings"],
    }


class TableSchemaConformanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema)

    def test_success_output_validates_against_shared_schema(self):
        raw = {
            "html": "<table><tr><th>학년</th><th>점수</th></tr><tr><td>1</td><td>90</td></tr></table>",
            "confidence": 0.9,
        }
        output = build_table_analysis(raw, model_name="ppstructure_v3", model_version="server")
        record = _wrap_as_block_record(output)
        self.assertEqual(list(self.validator.iter_errors(record)), [])

    def test_partial_output_validates_against_shared_schema(self):
        raw = {"html": "<table><tr><td></td><td>A2</td></tr></table>", "confidence": 0.4}
        output = build_table_analysis(raw, model_name="ppstructure_v3")
        record = _wrap_as_block_record(output)
        self.assertEqual(list(self.validator.iter_errors(record)), [])

    def test_failed_output_validates_against_shared_schema(self):
        output = build_table_analysis(None, model_name="ppstructure_v3")
        record = _wrap_as_block_record(output)
        self.assertEqual(list(self.validator.iter_errors(record)), [])


if __name__ == "__main__":
    unittest.main()
