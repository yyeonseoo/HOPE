from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


def crop_and_save_table_block(
    page_image_path: Optional[str | Path],
    block: Dict[str, Any],
    page_id: Optional[int],
    padding: int = 30,
) -> Optional[str]:
    """Crop `block`'s bbox (with padding) out of the page image and save it
    to `outputs/crops/table/p{page_id}_{block_id}.png`, returning the path as
    a string. Mirrors feature/formula-analysis's `crop_formula_block` (same
    output convention, same 30px default padding) so both analyzers save
    crops the same way for the integration layer. Returns None (never
    raises) if the page image, bbox, or block_id is missing/invalid.
    """
    if not page_image_path:
        return None

    bbox = block.get("bbox")
    block_id = block.get("block_id")
    if not bbox or len(bbox) != 4 or not block_id:
        return None

    image = cv2.imread(str(page_image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1 = max(0, min(x1 - padding, width))
    y1 = max(0, min(y1 - padding, height))
    x2 = max(0, min(x2 + padding, width))
    y2 = max(0, min(y2 + padding, height))
    if x2 <= x1 or y2 <= y1:
        return None

    output_dir = Path("outputs") / "crops" / "table"
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_page_id = page_id if page_id is not None else "unknown"
    output_path = output_dir / f"p{safe_page_id}_{block_id}.png"

    cv2.imwrite(str(output_path), image[y1:y2, x1:x2])
    return str(output_path).replace("\\", "/")


def crop_table_image(image_path: str | Path, bbox: List[int]) -> Optional[np.ndarray]:
    """Crop `bbox` out of the page image at `image_path`, returning a BGR
    ndarray. Coordinates are clamped to the image bounds. Returns None if the
    source image can't be read or the crop is empty, so callers can degrade
    to a `status="failed"` analysis instead of raising.

    Self-contained on purpose: duplicates the small clamp-crop logic already
    used by ocr.py::crop_and_ocr rather than importing from ocr.py, so this
    module has no dependency on files owned by other analyzers.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), width))
    y1 = max(0, min(int(y1), height))
    x2 = max(0, min(int(x2), width))
    y2 = max(0, min(int(y2), height))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    return crop if crop.size > 0 else None
