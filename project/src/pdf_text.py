from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import fitz


def extract_pdf_text_lines(pdf_path: str | Path, page_number: int, dpi: int = 200) -> List[Dict]:
    """Extract text lines from a PDF text layer and scale bboxes to rendered pixels."""
    pdf_path = Path(pdf_path)
    scale = dpi / 72.0
    doc = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > len(doc):
            raise ValueError(f"page_number must be between 1 and {len(doc)}")
        page = doc.load_page(page_number - 1)
        data = page.get_text("dict")
    finally:
        doc.close()

    lines: List[Dict] = []
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if not text:
                continue
            x1, y1, x2, y2 = line["bbox"]
            lines.append(
                {
                    "bbox": [int(x1 * scale), int(y1 * scale), int(x2 * scale), int(y2 * scale)],
                    "text": text,
                    "score": 1.0,
                    "source": "pdf_text",
                }
            )

    lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
    return lines
