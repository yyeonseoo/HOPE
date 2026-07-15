from __future__ import annotations

from html.parser import HTMLParser
from typing import Dict, List, Optional, Sequence

import numpy as np


class _TableHTMLParser(HTMLParser):
    """Parses a single <table> HTML string into a flat list of physical
    <td>/<th> cells in document order, each tagged with its rowspan/colspan
    and whether it appeared inside <thead> or as a <th>.

    This only tracks physical cells (one entry per tag), not the final
    row/column grid position — grid placement (accounting for spans from
    previous rows) happens in `parse_html_table`.
    """

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.physical_rows: List[List[Dict]] = []
        self._current_row: Optional[List[Dict]] = None
        self._current_cell: Optional[Dict] = None
        self._in_thead = False
        self._thead_depth = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        attrs_dict = dict(attrs)
        if tag == "thead":
            self._in_thead = True
            self._thead_depth += 1
        elif tag == "tr":
            self._current_row = []
        elif tag in ("td", "th"):
            self._current_cell = {
                "is_header": tag == "th" or self._in_thead,
                "row_span": _safe_int(attrs_dict.get("rowspan"), default=1),
                "col_span": _safe_int(attrs_dict.get("colspan"), default=1),
                "text": "",
            }

    def handle_endtag(self, tag: str) -> None:
        if tag == "thead":
            self._thead_depth = max(0, self._thead_depth - 1)
            self._in_thead = self._thead_depth > 0
        elif tag == "tr":
            if self._current_row is not None:
                self.physical_rows.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th"):
            if self._current_cell is not None and self._current_row is not None:
                self._current_cell["text"] = self._current_cell["text"].strip()
                self._current_row.append(self._current_cell)
            self._current_cell = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell["text"] += data


def _safe_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def parse_html_table(html: str, cell_box_list: Optional[Sequence] = None) -> List[List[Dict]]:
    """Parse a <table> HTML string (rowspan/colspan, <thead>/<th> aware) into
    a row-major grid of cell dicts: {row, column, row_span, column_span,
    is_header, text, bbox}.

    Merged cells occupy a single grid cell at their top-left (row, column);
    the cells they span over are not repeated as separate entries. Grid
    placement accounts for spans carried over from previous rows.

    `cell_box_list`, if given, is the engine's raw per-physical-cell pixel
    boxes (crop-local, one entry per <td>/<th> in document order -- see
    engine.py::run_table_engine). Each grid cell's `bbox` is filled in by
    zipping physical traversal order against this list; on any length
    mismatch (or when omitted) `bbox` is left None rather than guessed, so
    callers that don't need per-cell crops (the normal schema-conformant
    path) are unaffected.

    Returns an empty list if no <tr> rows were found.
    """
    parser = _TableHTMLParser()
    parser.feed(html)

    boxes = list(cell_box_list) if cell_box_list else None

    grid_cells: List[List[Dict]] = []
    # occupied[(row, column)] = True once a cell (or a span from an earlier
    # row) claims that grid position, so later cells in the same row skip
    # past it when assigning columns.
    occupied: Dict[tuple, bool] = {}
    physical_index = 0

    for row_index, physical_row in enumerate(parser.physical_rows):
        column_index = 0
        for cell in physical_row:
            while occupied.get((row_index, column_index)):
                column_index += 1

            row_span = cell["row_span"]
            col_span = cell["col_span"]

            bbox = None
            if boxes is not None and physical_index < len(boxes):
                bbox = _bbox_from_box(boxes[physical_index])

            grid_cells.append(
                {
                    "row": row_index,
                    "column": column_index,
                    "row_span": row_span,
                    "column_span": col_span,
                    "is_header": cell["is_header"],
                    "text": cell["text"] or None,
                    "bbox": bbox,
                }
            )

            physical_index += 1

            for span_row in range(row_index, row_index + row_span):
                for span_col in range(column_index, column_index + col_span):
                    occupied[(span_row, span_col)] = True

            column_index += col_span

    return grid_cells


def _bbox_from_box(box) -> Optional[List[float]]:
    """Normalize one raw cell box (flat [x1,y1,x2,y2] or 4 corner points,
    per engine.py::_as_points) into a plain [x1, y1, x2, y2] list, or None
    if it can't be parsed as coordinates."""
    try:
        flat = np.asarray(box, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None

    if flat.size == 4:
        xs, ys = flat[[0, 2]], flat[[1, 3]]
    elif flat.size >= 8:
        points = flat.reshape(-1, 2)
        xs, ys = points[:, 0], points[:, 1]
    else:
        return None

    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def grid_dimensions(cells: List[Dict]) -> tuple:
    """Return (row_count, column_count) covering every cell's full span."""
    if not cells:
        return (0, 0)
    max_row = max(cell["row"] + cell["row_span"] for cell in cells)
    max_col = max(cell["column"] + cell["column_span"] for cell in cells)
    return (max_row, max_col)
