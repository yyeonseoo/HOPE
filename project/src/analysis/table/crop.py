from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


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
