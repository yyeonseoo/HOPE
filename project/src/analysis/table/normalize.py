from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .formula_cells import merge_formula_cell_ocr
from .html_parser import grid_dimensions, parse_html_table


def build_table_analysis(
    raw_result: Optional[Dict],
    model_name: str,
    model_version: Optional[str] = None,
    table_crop: Optional[np.ndarray] = None,
) -> Dict:
    """Normalize one engine's raw table-recognition output into the
    `analysis` object defined by schemas/block_analysis.schema.json (plus a
    sibling `warnings` list, which lives at the top level of that schema).

    `raw_result` is expected to look like `{"html": "<table>...</table>",
    "confidence": float | None}` when the engine found a table region, or be
    None/empty when it found nothing. This function never raises — any
    unparseable or missing input degrades to a `status="failed"` result with
    `result=None`, per the "unknown values are null, never guessed" contract.

    `model_name`/`model_version` identify the table-structure-recognition
    model itself (not the upstream layout detector) and are always recorded,
    even on failure, so downstream consumers know which model produced/failed
    to produce this analysis.

    `table_crop`, if given, enables the cell-level formula-OCR merge step
    (formula_cells.merge_formula_cell_ocr): cells whose general-OCR text
    looks like a mangled formula get re-recognized from their own
    sub-image via feature/formula-analysis's crop-based recognizer, and the
    merged text (not the raw OCR text) is what ends up in the returned
    `cells`. Omit it (the default) to keep the original, formula-OCR-free
    behavior -- e.g. when `raw_result` didn't carry a `cell_box_list` to
    crop from in the first place.
    """
    model = {"name": model_name, "version": model_version}
    warnings: List[str] = []

    html = (raw_result or {}).get("html") if raw_result else None
    if not html:
        return {
            "analysis": {
                "status": "failed",
                "model": model,
                "confidence": None,
                "result": None,
            },
            "warnings": ["표 영역을 인식하지 못했습니다."],
        }

    cell_box_list = (raw_result or {}).get("cell_box_list")
    cells = parse_html_table(html, cell_box_list=cell_box_list)
    if not cells:
        return {
            "analysis": {
                "status": "failed",
                "model": model,
                "confidence": None,
                "result": None,
            },
            "warnings": ["표 HTML을 셀 구조로 파싱하지 못했습니다."],
        }

    if table_crop is not None:
        cells = merge_formula_cell_ocr(cells, table_crop)

    row_count, column_count = grid_dimensions(cells)
    missing_text_cells = [cell for cell in cells if not cell["text"]]
    if missing_text_cells:
        warnings.append(f"{len(missing_text_cells)}개 셀의 텍스트를 인식하지 못했습니다.")

    status = "partial" if missing_text_cells else "success"
    confidence = (raw_result or {}).get("confidence")

    return {
        "analysis": {
            "status": status,
            "model": model,
            "confidence": confidence,
            "result": {
                "kind": "table",
                "row_count": row_count,
                "column_count": column_count,
                "cells": [
                    {
                        "row": cell["row"],
                        "column": cell["column"],
                        "row_span": cell["row_span"],
                        "column_span": cell["column_span"],
                        "text": cell["text"],
                        "is_header": cell["is_header"],
                    }
                    for cell in cells
                ],
            },
        },
        "warnings": warnings,
    }
