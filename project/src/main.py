from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import cv2

from export_json import build_page_result, export_pages
from layout_detection import detect_layout, refine_blocks_after_ocr
from ocr import _load_paddleocr, ocr_blocks, ocr_image
from pdf_to_image import pdf_to_images
from reading_order import sort_reading_order


def visualize_blocks(image_path: Path, blocks: List[Dict], output_path: Path) -> Path:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    palette = {
        "title": (30, 80, 220),
        "section_title": (30, 150, 220),
        "paragraph": (80, 80, 80),
        "formula": (180, 70, 180),
        "table": (40, 140, 40),
        "figure": (20, 150, 150),
        "caption": (110, 110, 20),
        "footer": (120, 120, 120),
        "page_number": (0, 0, 0),
    }

    for block in blocks:
        x1, y1, x2, y2 = block["bbox"]
        color = palette.get(block["type"], (0, 0, 255))
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
        label = f'{block.get("reading_order", "?")} {block["type"]}'
        cv2.putText(image, label, (x1, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return output_path


def process_pdf(
    pdf_path: Path,
    output_dir: Path,
    dpi: int = 200,
    yolo_model_path: str | None = None,
    lang: str = "korean",
) -> Path:
    pages_dir = output_dir / "pages"
    vis_dir = output_dir / "visualizations"
    json_path = output_dir / "layout.json"

    image_paths = pdf_to_images(pdf_path, pages_dir, dpi=dpi)
    ocr_engine = _load_paddleocr(lang=lang)

    page_results = []
    for page_index, image_path in enumerate(image_paths, start=1):
        ocr_lines = ocr_image(image_path, ocr_engine=ocr_engine, lang=lang)
        blocks = detect_layout(image_path, ocr_lines=ocr_lines, yolo_model_path=yolo_model_path)
        blocks = ocr_blocks(image_path, blocks, ocr_lines=ocr_lines, ocr_engine=ocr_engine, lang=lang)
        blocks = refine_blocks_after_ocr(blocks, ocr_lines=ocr_lines)
        blocks = sort_reading_order(blocks)
        page_results.append(build_page_result(page_index, blocks))
        visualize_blocks(image_path, blocks, vis_dir / f"page_{page_index:04d}_layout.png")

    return export_pages(page_results, json_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Economics math textbook PDF layout parser")
    parser.add_argument("--pdf", required=True, help="Input PDF path")
    parser.add_argument("--output", default="outputs", help="Output directory")
    parser.add_argument("--dpi", type=int, default=200, help="PDF render DPI")
    parser.add_argument("--yolo-model", default=None, help="Optional DocLayout-YOLO/Ultralytics model path")
    parser.add_argument("--lang", default="korean", help="PaddleOCR language")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = process_pdf(
        pdf_path=Path(args.pdf),
        output_dir=Path(args.output),
        dpi=args.dpi,
        yolo_model_path=args.yolo_model,
        lang=args.lang,
    )
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
