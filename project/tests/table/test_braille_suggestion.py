import unittest

from src.analysis.table.analyzer import _assemble_table_record


def output(cells):
    return {
        "analysis": {
            "status": "partial",
            "model": {"name": "test", "version": None},
            "confidence": None,
            "result": {"kind": "table", "row_count": 2, "column_count": 2, "cells": cells},
        },
        "warnings": [],
    }


def cell(row, column, text):
    return {"row": row, "column": column, "row_span": 1, "column_span": 1, "text": text, "is_header": False}


class TableBrailleSuggestionTests(unittest.TestCase):
    def _record(self, cells):
        block = {"block_id": "p9_b12", "type": "table", "bbox": [0, 0, 10, 10]}
        return _assemble_table_record(9, block, [block], 0, block["bbox"], None, output(cells))

    def test_mostly_unreadable_table_is_suggested_for_omission(self):
        record = self._record([cell(0, 0, "파일"), cell(0, 1, ""), cell(1, 0, ""), cell(1, 1, "")])
        self.assertEqual(record["braille_review"]["suggested_visual_treatment"], "omit")

    def test_readable_table_gets_no_suggestion(self):
        record = self._record([cell(0, 0, "x"), cell(0, 1, "1"), cell(1, 0, "y"), cell(1, 1, "")])
        self.assertNotIn("braille_review", record)


if __name__ == "__main__":
    unittest.main()
