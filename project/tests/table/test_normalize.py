import unittest

from src.analysis.table.normalize import build_table_analysis


class BuildTableAnalysisTests(unittest.TestCase):
    def test_success_path_full_table(self):
        raw = {
            "html": "<table><tr><td>A1</td><td>A2</td></tr><tr><td>B1</td><td>B2</td></tr></table>",
            "confidence": 0.92,
        }
        output = build_table_analysis(raw, model_name="ppstructure_v3", model_version="1.0")

        analysis = output["analysis"]
        self.assertEqual(analysis["status"], "success")
        self.assertEqual(analysis["model"], {"name": "ppstructure_v3", "version": "1.0"})
        self.assertEqual(analysis["confidence"], 0.92)
        self.assertEqual(analysis["result"]["kind"], "table")
        self.assertEqual(analysis["result"]["row_count"], 2)
        self.assertEqual(analysis["result"]["column_count"], 2)
        self.assertEqual(len(analysis["result"]["cells"]), 4)
        self.assertEqual(output["warnings"], [])

    def test_partial_path_missing_cell_text(self):
        raw = {
            "html": "<table><tr><td></td><td>A2</td></tr></table>",
            "confidence": 0.5,
        }
        output = build_table_analysis(raw, model_name="ppstructure_v3")

        analysis = output["analysis"]
        self.assertEqual(analysis["status"], "partial")
        self.assertIsNotNone(analysis["result"])
        self.assertEqual(len(output["warnings"]), 1)
        self.assertIn("1개", output["warnings"][0])

    def test_failed_path_no_table_found(self):
        output = build_table_analysis(None, model_name="ppstructure_v3", model_version=None)

        analysis = output["analysis"]
        self.assertEqual(analysis["status"], "failed")
        self.assertEqual(analysis["model"], {"name": "ppstructure_v3", "version": None})
        self.assertIsNone(analysis["confidence"])
        self.assertIsNone(analysis["result"])
        self.assertEqual(len(output["warnings"]), 1)

    def test_failed_path_empty_html(self):
        output = build_table_analysis({"html": "", "confidence": None}, model_name="ppstructure_v3")
        self.assertEqual(output["analysis"]["status"], "failed")
        self.assertIsNone(output["analysis"]["result"])

    def test_failed_path_unparseable_html(self):
        output = build_table_analysis(
            {"html": "<div>not a table</div>", "confidence": 0.3}, model_name="ppstructure_v3"
        )
        self.assertEqual(output["analysis"]["status"], "failed")
        self.assertIsNone(output["analysis"]["result"])

    def test_confidence_defaults_to_none_when_engine_omits_it(self):
        raw = {"html": "<table><tr><td>A1</td></tr></table>"}
        output = build_table_analysis(raw, model_name="ppstructure_v3")
        self.assertIsNone(output["analysis"]["confidence"])


if __name__ == "__main__":
    unittest.main()
