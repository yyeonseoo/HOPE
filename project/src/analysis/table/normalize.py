from __future__ import annotations

from typing import Dict, List, Optional

from .html_parser import grid_dimensions, parse_html_table


def build_table_analysis(
    raw_result: Optional[Dict],
    model_name: str,
    model_version: Optional[str] = None,
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

    cells = parse_html_table(html)
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
