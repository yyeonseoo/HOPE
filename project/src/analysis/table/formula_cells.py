from __future__ import annotations

import re
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.analysis.formula.formula_recognizer import recognize_formula_from_crop

# Cheap text-side signal that a cell's *general*-OCR text is likely a
# mangled formula (fraction, root, integral, sum/product, matrix-ish
# notation) rather than plain prose/numbers. This is intentionally a text
# heuristic, not an image classifier: PaddleOCR's general text model still
# reliably reads these specific unicode glyphs even when it garbles the
# surrounding structure (stacked numerator/denominator, nested radicals).
_FORMULA_SIGNAL_PATTERN = re.compile(r"[√∫∑∏±≤≥≠^]|frac|\\int|\\sum")

# A cell whose text is nothing but digits with no separator/operator at all
# (e.g. "12" where a fraction like "1/2" had its "/" dropped by OCR, or a
# stacked numerator/denominator lost its line break) is also a candidate --
# but only from 3+ digits, since 1-2 digit numbers are extremely common
# plain table data and would otherwise false-positive constantly.
_BARE_DIGIT_RUN_PATTERN = re.compile(r"^\d{3,}$")


def looks_like_formula_cell(text: Optional[str]) -> bool:
    """Heuristic: does this cell's recognized text look like a formula that
    general OCR likely mangled, and so is worth re-running through
    formula-analysis's crop-based recognizer?

    This never asserts a cell *is* a formula -- callers must still treat the
    re-recognition result as a candidate replacement, not a certainty.
    """
    if not text:
        return False
    compact = text.replace(" ", "")
    if _FORMULA_SIGNAL_PATTERN.search(compact):
        return True
    return bool(_BARE_DIGIT_RUN_PATTERN.fullmatch(compact))


def crop_cell_image(table_crop: np.ndarray, bbox: Optional[List[float]]) -> Optional[np.ndarray]:
    """Crop a single cell's region out of the table's own crop image (not
    the full page image -- `bbox` is already in the table crop's local
    pixel space, per html_parser.parse_html_table's `cell_box_list`
    handling). Returns None if bbox is missing or the crop would be empty.
    """
    if bbox is None or table_crop is None:
        return None

    height, width = table_crop.shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    x1 = max(0, min(x1, width))
    x2 = max(0, min(x2, width))
    y1 = max(0, min(y1, height))
    y2 = max(0, min(y2, height))
    if x2 <= x1 or y2 <= y1:
        return None

    cell_image = table_crop[y1:y2, x1:x2]
    return cell_image if cell_image.size > 0 else None


def merge_formula_cell_ocr(
    cells: List[Dict[str, Any]],
    table_crop: Optional[np.ndarray],
) -> List[Dict[str, Any]]:
    """For cells whose text looks like a mangled formula, re-run
    feature/formula-analysis's `recognize_formula_from_crop` on just that
    cell's sub-image and replace the cell's `text` with the recognized
    latex/plain_text when recognition succeeds.

    Cells without a usable `bbox` (e.g. `cell_box_list` wasn't available or
    didn't line up with the HTML) or that don't look formula-like pass
    through unchanged. Never raises: any per-cell recognition failure just
    keeps the original OCR text, consistent with the rest of this module's
    "degrade, don't crash" contract.
    """
    if table_crop is None:
        return cells

    merged: List[Dict[str, Any]] = []
    for cell in cells:
        if not looks_like_formula_cell(cell.get("text")):
            merged.append(cell)
            continue

        cell_image = crop_cell_image(table_crop, cell.get("bbox"))
        if cell_image is None:
            merged.append(cell)
            continue

        crop_path = _save_temp_cell_crop(cell_image)
        try:
            recognition = recognize_formula_from_crop(
                crop_path=crop_path,
                fallback_text=cell.get("text"),
            )
        except Exception:  # noqa: BLE001 - one bad cell can't fail the whole table
            merged.append(cell)
            continue
        finally:
            Path(crop_path).unlink(missing_ok=True)

        recognized_text = recognition.get("latex") or recognition.get("plain_text")
        if recognized_text:
            merged.append({**cell, "text": recognized_text})
        else:
            merged.append(cell)

    return merged


def _save_temp_cell_crop(cell_image: np.ndarray) -> str:
    """Save a cell's cropped ndarray to a throwaway PNG so it can be handed
    to `recognize_formula_from_crop`, which (like the rest of
    feature/formula-analysis) reads crops from disk rather than accepting
    an in-memory array."""
    temp_dir = Path(tempfile.gettempdir()) / "hope_table_formula_cells"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / f"{uuid.uuid4().hex}.png"
    cv2.imwrite(str(temp_path), cell_image)
    return str(temp_path)
