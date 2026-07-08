from __future__ import annotations

from typing import Any, Dict, List, Optional

DESCRIPTION_MODEL_NAME = "table-description"
DESCRIPTION_MODEL_VERSION = None


def _row_texts(cells: List[Dict[str, Any]], row_count: int) -> List[List[Optional[str]]]:
    """Group cell text by row index, ordered by column, for `row_count` rows."""
    rows: List[List[Optional[str]]] = [[] for _ in range(row_count)]
    for cell in sorted(cells, key=lambda cell: (cell["row"], cell["column"])):
        row_index = cell["row"]
        if 0 <= row_index < len(rows):
            rows[row_index].append(cell.get("text"))
    return rows


def generate_table_description(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Build the accessibility `description` block for a table analysis result.

    Mirrors feature/formula-analysis's `generate_formula_description`, but
    reads the `analysis` object produced by `build_table_analysis`
    (status/model/confidence/result) instead of a bare latex string, since a
    table's row/column structure -- not a single value -- is what needs
    describing here. Output matches the shared `#/$defs/description` schema
    in schemas/block_analysis.schema.json.
    """
    status = analysis.get("status")
    result = analysis.get("result")

    if status == "failed" or result is None:
        return {
            "status": "not_started",
            "model": None,
            "short_text": None,
            "long_text": None,
            "transcription_notes": None,
            "context_used": False,
            "review_status": "unreviewed",
        }

    row_count = result.get("row_count", 0)
    column_count = result.get("column_count", 0)
    cells: List[Dict[str, Any]] = result.get("cells", [])

    rows = _row_texts(cells, row_count)

    # The table-recognition engine's HTML rarely marks <th>/<thead>, so
    # is_header is almost always False on real output. Fall back to the
    # first row's actual recognized text so the description still reflects
    # real content instead of only dimensions -- this reads the first row,
    # it does not assert those cells are semantically headers.
    explicit_header_texts = [
        cell["text"] for cell in cells if cell.get("is_header") and cell.get("text")
    ]
    first_row_texts = [text for text in rows[0] if text] if rows else []
    header_texts = explicit_header_texts or first_row_texts
    used_header_heuristic = not explicit_header_texts and bool(first_row_texts)

    short_text = f"{row_count}행 {column_count}열로 이루어진 표입니다."
    if header_texts:
        short_text += f" 첫 행에는 {', '.join(header_texts)}가 있습니다."

    long_text_parts = [f"이 표는 {row_count}행 {column_count}열로 구성되어 있습니다."]
    if header_texts:
        long_text_parts.append(f"첫 번째 행에는 {', '.join(header_texts)}가 있습니다.")

    sample_row = next((row for row in rows[1:] if any(row)), None)
    if sample_row:
        sample_texts = [text for text in sample_row if text]
        if sample_texts:
            long_text_parts.append(f"예를 들어 다음 행에는 {', '.join(sample_texts)}가 있습니다.")

    long_text = " ".join(long_text_parts)

    has_merged_cells = any(
        cell.get("row_span", 1) > 1 or cell.get("column_span", 1) > 1 for cell in cells
    )

    if has_merged_cells:
        transcription_notes = (
            "일부 셀이 여러 행 또는 열에 걸쳐 병합되어 있으므로, "
            "각 셀을 읽을 때 병합 범위를 함께 안내합니다."
        )
    else:
        transcription_notes = "행과 열 순서대로 각 셀의 내용을 점역합니다."

    missing_text_cells = [cell for cell in cells if not cell.get("text")]
    review_status = (
        "needs_review" if missing_text_cells or used_header_heuristic else "unreviewed"
    )

    return {
        "status": status,
        "model": {"name": DESCRIPTION_MODEL_NAME, "version": DESCRIPTION_MODEL_VERSION},
        "short_text": short_text,
        "long_text": long_text,
        "transcription_notes": transcription_notes,
        "context_used": False,
        "review_status": review_status,
    }
