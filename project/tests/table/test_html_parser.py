import unittest

from src.analysis.table.html_parser import grid_dimensions, parse_html_table


class HtmlTableParserTests(unittest.TestCase):
    def test_plain_2x2_grid(self):
        html = """
        <table>
          <tr><td>A1</td><td>A2</td></tr>
          <tr><td>B1</td><td>B2</td></tr>
        </table>
        """
        cells = parse_html_table(html)
        self.assertEqual(len(cells), 4)
        self.assertEqual(grid_dimensions(cells), (2, 2))

        by_pos = {(cell["row"], cell["column"]): cell for cell in cells}
        self.assertEqual(by_pos[(0, 0)]["text"], "A1")
        self.assertEqual(by_pos[(0, 1)]["text"], "A2")
        self.assertEqual(by_pos[(1, 0)]["text"], "B1")
        self.assertEqual(by_pos[(1, 1)]["text"], "B2")
        for cell in cells:
            self.assertEqual(cell["row_span"], 1)
            self.assertEqual(cell["column_span"], 1)
            self.assertFalse(cell["is_header"])

    def test_colspan_header_merge(self):
        html = """
        <table>
          <tr><th colspan="2">제목</th></tr>
          <tr><td>A1</td><td>A2</td></tr>
        </table>
        """
        cells = parse_html_table(html)
        header_cells = [cell for cell in cells if cell["is_header"]]
        self.assertEqual(len(header_cells), 1)
        header = header_cells[0]
        self.assertEqual(header["row"], 0)
        self.assertEqual(header["column"], 0)
        self.assertEqual(header["column_span"], 2)
        self.assertEqual(header["text"], "제목")
        self.assertEqual(grid_dimensions(cells), (2, 2))

    def test_rowspan_merge(self):
        html = """
        <table>
          <tr><td rowspan="2">A</td><td>B1</td></tr>
          <tr><td>B2</td></tr>
        </table>
        """
        cells = parse_html_table(html)
        by_text = {cell["text"]: cell for cell in cells}
        self.assertEqual(by_text["A"]["row"], 0)
        self.assertEqual(by_text["A"]["column"], 0)
        self.assertEqual(by_text["A"]["row_span"], 2)
        # B2 must be pushed to column 1 since column 0 on row 1 is occupied
        # by the rowspan from "A".
        self.assertEqual(by_text["B2"]["row"], 1)
        self.assertEqual(by_text["B2"]["column"], 1)
        self.assertEqual(grid_dimensions(cells), (2, 2))

    def test_thead_th_marks_header_rows(self):
        html = """
        <table>
          <thead><tr><th>학년</th><th>점수</th></tr></thead>
          <tbody><tr><td>1</td><td>90</td></tr></tbody>
        </table>
        """
        cells = parse_html_table(html)
        header_row = [cell for cell in cells if cell["row"] == 0]
        body_row = [cell for cell in cells if cell["row"] == 1]
        self.assertTrue(all(cell["is_header"] for cell in header_row))
        self.assertTrue(all(not cell["is_header"] for cell in body_row))

    def test_empty_html_returns_no_cells(self):
        self.assertEqual(parse_html_table(""), [])
        self.assertEqual(grid_dimensions([]), (0, 0))

    def test_empty_cell_text_becomes_none(self):
        html = "<table><tr><td></td><td>value</td></tr></table>"
        cells = parse_html_table(html)
        by_pos = {(cell["row"], cell["column"]): cell for cell in cells}
        self.assertIsNone(by_pos[(0, 0)]["text"])
        self.assertEqual(by_pos[(0, 1)]["text"], "value")


if __name__ == "__main__":
    unittest.main()
