from __future__ import annotations

import io
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
    *,
    pdf_path: str | Path | None = None,
    source_dpi: int | None = None,
    caption_dpi: int = 180,
) -> Optional[str]:
    """Save a deterministic crop for one figure block and return its path."""
    bbox = block.get("bbox")
    if not page_image_path or not isinstance(bbox, (list, tuple)):
        return None

    crop = _render_high_resolution_figure(
        pdf_path, page_id, bbox, source_dpi, caption_dpi
    )
    if crop is None:
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


def _render_high_resolution_figure(
    pdf_path: str | Path | None,
    page_number: int | None,
    bbox: Sequence[float],
    source_dpi: int | None,
    caption_dpi: int,
) -> Optional[Image.Image]:
    """Render only the detected Figure region at captioning resolution.

    Layout coordinates remain tied to the original page DPI.  Re-rendering a
    small clip avoids increasing DocLayout/OCR cost for the whole page while
    preserving labels and point markers for the vision-language model.
    """
    if (
        not pdf_path
        or page_number is None
        or not isinstance(source_dpi, int)
        or source_dpi <= 0
        or caption_dpi <= source_dpi
        or len(bbox) != 4
    ):
        return None
    try:
        import fitz

        scale_to_points = 72.0 / source_dpi
        clip = fitz.Rect(*(float(value) * scale_to_points for value in bbox))
        with fitz.open(pdf_path) as document:
            page = document[int(page_number) - 1]
            clip &= page.rect
            if clip.is_empty or clip.is_infinite:
                return None
            matrix = fitz.Matrix(caption_dpi / 72.0, caption_dpi / 72.0)
            pixmap = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
            return Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
    except (FileNotFoundError, OSError, TypeError, ValueError, IndexError, RuntimeError):
        return None
