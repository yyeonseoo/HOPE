"""Recover worksheet-style tables from the PDF itself.

Some textbook tables (e.g. "x | 1 2 3 4 5 / y | 4 8 □ □ □") have no ruling
lines: the row headers are filled boxes, the blanks are small boxes, and the
values are PDF text. Image table recognition misreads them (arrow notes such
as "2배" become cells), and the layout model often calls them figures, so
braille gets a prose note instead of a table (KIM2010 제2장: tables must be
transcribed as tables, empty cells as ==).

ponytail: only this layout -- a stacked column of row-header boxes on the
left, values as text, blanks as small boxes. Ruled tables are left to the
image table recognizer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

_MIN_COLUMNS = 3


def read_vector_table(pdf_path: str | Path, page_number: int, bbox: Sequence[float], dpi: int) -> List[Dict[str, Any]] | None:
    import fitz

    scale = 72 / dpi
    area = fitz.Rect(*[v * scale for v in bbox])
    with fitz.open(pdf_path) as document:
        page = document[page_number - 1]
        boxes = [
            item[1] for drawing in page.get_drawings() if area.intersects(drawing["rect"])
            for item in drawing["items"] if item[0] == "re" and area.contains(item[1])
        ]
        words = [
            (fitz.Rect(span["bbox"]), _symbol_text(span["font"], span["text"].strip()))
            for block in page.get_text("dict", clip=area)["blocks"]
            for line in block.get("lines", [])
            for span in line["spans"]
            if span["text"].strip()
        ]
    return cells_from_layout(boxes, words)


# Hancom equation symbol fonts store symbols under Latin codes: "…" is "y"
# in EHyak. ponytail: only the glyph seen so far; add others as they appear.
_SYMBOL_FONT_GLYPHS = {("EHyak", "y"): "…"}


def _symbol_text(font: str, text: str) -> str:
    for (prefix, code), glyph in _SYMBOL_FONT_GLYPHS.items():
        if font.startswith(prefix) and text == code:
            return glyph
    return text


def cells_from_layout(boxes: Sequence[Any], words: Sequence[tuple[Any, str]]) -> List[Dict[str, Any]] | None:
    """boxes: rects with x0/y0/x1/y1; words: (rect, text). Pure, for testing."""
    if len(boxes) < 2:
        return None
    left = min(b.x0 for b in boxes)
    headers = sorted((b for b in boxes if abs(b.x0 - left) < 1 and b.height >= 8), key=lambda b: b.y0)
    if len(headers) < 2 or any(abs(a.y1 - b.y0) > 1.5 for a, b in zip(headers, headers[1:])):
        return None
    bands = [(h.y0, h.y1) for h in headers]

    def band_of(y: float) -> int | None:
        return next((i for i, (top, bottom) in enumerate(bands) if top <= y <= bottom), None)

    rows: List[List[tuple[float, str]]] = [[] for _ in bands]
    for rect, text in words:
        row = band_of((rect.y0 + rect.y1) / 2)
        if row is not None:
            rows[row].append(((rect.x0 + rect.x1) / 2, text))
    for box in boxes:
        row = band_of((box.y0 + box.y1) / 2)
        if row is not None and box not in headers:
            rows[row].append(((box.x0 + box.x1) / 2, ""))  # blank answer box

    columns = sorted(x for x, _ in max(rows, key=len))
    if len(columns) < _MIN_COLUMNS:
        return None
    cells = []
    for r, items in enumerate(rows):
        texts = [""] * len(columns)
        for x, text in items:
            column = min(range(len(columns)), key=lambda c: abs(columns[c] - x))
            texts[column] = (texts[column] + " " + text).strip() if text else texts[column]
        cells.extend(
            {"row": r, "column": c, "row_span": 1, "column_span": 1, "text": text, "is_header": c == 0}
            for c, text in enumerate(texts)
        )
    return cells
