from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from PIL import Image


def crop_figure_image(image_path: str | Path, bbox: Sequence[float]) -> Optional[Image.Image]:
    """Return a clamped RGB crop, or ``None`` for an invalid crop."""
    if len(bbox) != 4:
        return None

    try:
        with Image.open(image_path) as source:
            width, height = source.size
            x1, y1, x2, y2 = (int(round(float(value))) for value in bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            return source.crop((x1, y1, x2, y2)).convert("RGB")
    except (FileNotFoundError, OSError, TypeError, ValueError):
        return None


def crop_and_save_figure_block(
    page_image_path: str | Path | None,
    block: Mapping[str, Any],
    page_id: int | None,
    output_dir: str | Path | None = None,
) -> Optional[str]:
    """Save a deterministic crop for one figure block and return its path."""
    bbox = block.get("bbox")
    if not page_image_path or not isinstance(bbox, (list, tuple)):
        return None

    crop = crop_figure_image(page_image_path, bbox)
    if crop is None:
        return None

    destination_dir = Path(output_dir) if output_dir else Path(page_image_path).parent / "crops" / "figure"
    destination_dir.mkdir(parents=True, exist_ok=True)
    page_label = str(page_id if page_id is not None else "unknown")
    block_label = str(block.get("block_id") or "figure").replace("/", "_").replace("\\", "_")
    destination = destination_dir / f"p{page_label}_{block_label}.png"
    crop.save(destination, format="PNG")
    return str(destination)
