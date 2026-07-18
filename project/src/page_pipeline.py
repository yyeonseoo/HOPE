from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

from export_json import build_page_result
from layout_detection import detect_layout, refine_blocks_after_ocr
from main import visualize_blocks
from ocr import _load_paddleocr, ocr_blocks, ocr_image
from pdf_text import extract_pdf_text_lines
from pdf_to_image import render_pdf_page
from reading_order import sort_reading_order


def process_single_page(
    pdf_path: Path,
    page_number: int,
    work_dir: Path,
    dpi: int = 200,
    yolo_model_path: Optional[str] = None,
    lang: str = "korean",
    ocr_engine=None,
    prefer_pdf_text: bool = True,
    model_only: bool = False,
    correction_profile: Optional[str] = None,
) -> Dict:
    work_dir.mkdir(parents=True, exist_ok=True)
    page_image_path = work_dir / f"page_{page_number:04d}.png"
    visualization_path = work_dir / f"page_{page_number:04d}_layout.png"

    render_pdf_page(pdf_path, page_number, page_image_path, dpi=dpi)
    ocr_source = "pdf_text"
    ocr_lines = extract_pdf_text_lines(pdf_path, page_number, dpi=dpi) if prefer_pdf_text else []
    if len(ocr_lines) < 3:
        ocr_source = "paddleocr"
        engine = ocr_engine or _load_paddleocr(lang=lang)
        ocr_lines = ocr_image(page_image_path, ocr_engine=engine, lang=lang)
    blocks = detect_layout(
        page_image_path,
        ocr_lines=ocr_lines,
        yolo_model_path=yolo_model_path,
        use_supplements=not model_only,
    )
    blocks = ocr_blocks(page_image_path, blocks, ocr_lines=ocr_lines, ocr_engine=ocr_engine, lang=lang)
    if not model_only:
        blocks = refine_blocks_after_ocr(blocks, ocr_lines=ocr_lines, correction_profile=correction_profile)
    blocks = sort_reading_order(blocks)
    page_result = build_page_result(page_number, blocks)
    visualize_blocks(page_image_path, blocks, visualization_path)

    return {
        "page": page_result,
        "page_image_path": page_image_path,
        "visualization_path": visualization_path,
        # Internal grounding evidence for downstream figure captioning.
        # This is intentionally kept outside the exported page schema.
        "ocr_lines": ocr_lines,
        "ocr_source": ocr_source,
        "layout_mode": "model_only" if model_only else (correction_profile or "enhanced"),
    }
