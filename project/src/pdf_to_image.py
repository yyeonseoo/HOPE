from __future__ import annotations

from pathlib import Path
from typing import List

import fitz


def pdf_to_images(pdf_path: str | Path, output_dir: str | Path, dpi: int = 200) -> List[Path]:
    """Render every PDF page to a PNG image."""
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    image_paths: List[Path] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        image_path = output_dir / f"page_{page_index + 1:04d}.png"
        pix.save(str(image_path))
        image_paths.append(image_path)

    doc.close()
    return image_paths


def get_page_count(pdf_path: str | Path) -> int:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def render_pdf_page(
    pdf_path: str | Path,
    page_number: int,
    output_path: str | Path,
    dpi: int = 200,
) -> Path:
    """Render one 1-based PDF page to a PNG image."""
    pdf_path = Path(pdf_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"page_number must be between 1 and {len(doc)}")
        zoom = dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        pix.save(str(output_path))
        return output_path
    finally:
        doc.close()
