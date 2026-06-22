from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


def _load_paddleocr(lang: str = "korean"):
    project_cache_dir = Path(__file__).resolve().parents[1] / ".cache"
    cache_dir = project_cache_dir / "paddlex"
    matplotlib_cache_dir = project_cache_dir / "matplotlib"
    paddle_home = project_cache_dir / "paddle"
    project_home = project_cache_dir / "home"
    for path in (cache_dir, matplotlib_cache_dir, paddle_home, project_home):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(project_home)
    os.environ["USERPROFILE"] = str(project_home)
    os.environ["XDG_CACHE_HOME"] = str(project_cache_dir)
    os.environ["PADDLE_HOME"] = str(paddle_home)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise RuntimeError("PaddleOCR is not installed. Run: pip install -r requirements.txt") from exc

    return PaddleOCR(
        lang=lang,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        device="cpu",
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _poly_to_bbox(poly) -> List[int]:
    points = np.array(poly, dtype=np.float32)
    x1, y1 = points.min(axis=0)
    x2, y2 = points.max(axis=0)
    return [int(x1), int(y1), int(x2), int(y2)]


def _normalize_ocr_result(raw_result) -> List[Dict]:
    lines: List[Dict] = []
    if not raw_result:
        return lines

    page_result = raw_result[0] if isinstance(raw_result, list) and raw_result else raw_result
    if not page_result:
        return lines

    if hasattr(page_result, "get") and "rec_texts" in page_result:
        texts = page_result.get("rec_texts") or []
        scores = page_result.get("rec_scores") or []
        polys = page_result.get("rec_polys") or page_result.get("dt_polys") or []
        for index, text in enumerate(texts):
            text = str(text).strip()
            if not text or index >= len(polys):
                continue
            score = float(scores[index]) if index < len(scores) else 0.0
            lines.append({"bbox": _poly_to_bbox(polys[index]), "text": text, "score": score})
        return lines

    for item in page_result:
        if len(item) < 2:
            continue
        poly, rec = item[0], item[1]
        text = rec[0] if isinstance(rec, (list, tuple)) else str(rec)
        score = float(rec[1]) if isinstance(rec, (list, tuple)) and len(rec) > 1 else 0.0
        if text.strip():
            lines.append({"bbox": _poly_to_bbox(poly), "text": text.strip(), "score": score})
    return lines


def ocr_image(image_path: str | Path, ocr_engine=None, lang: str = "korean") -> List[Dict]:
    engine = ocr_engine or _load_paddleocr(lang)
    raw_result = engine.ocr(str(image_path))
    return _normalize_ocr_result(raw_result)


def ocr_blocks(
    image_path: str | Path,
    blocks: List[Dict],
    ocr_lines: Optional[List[Dict]] = None,
    ocr_engine=None,
    lang: str = "korean",
) -> List[Dict]:
    """Attach OCR text to text-like layout blocks."""
    if ocr_lines is None:
        ocr_lines = ocr_image(image_path, ocr_engine=ocr_engine, lang=lang)

    text_types = {
        "title",
        "section_title",
        "paragraph",
        "formula",
        "caption",
        "footer",
        "page_number",
        "table",
    }

    for block in blocks:
        if block["type"] not in text_types:
            block.setdefault("text", "")
            continue
        block["text"] = lines_text_inside_bbox(ocr_lines, block["bbox"])
    return blocks


def lines_text_inside_bbox(ocr_lines: List[Dict], bbox: List[int]) -> str:
    x1, y1, x2, y2 = bbox
    matched = []
    for line in ocr_lines:
        lx1, ly1, lx2, ly2 = line["bbox"]
        cx = (lx1 + lx2) / 2
        cy = (ly1 + ly2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            matched.append(line)

    matched.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    return "\n".join(line["text"] for line in matched).strip()


def crop_and_ocr(image_path: str | Path, bbox: List[int], ocr_engine=None, lang: str = "korean") -> str:
    """OCR a single crop. Useful when line-level OCR misses boxed content."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    x1, y1, x2, y2 = bbox
    crop = image[max(y1, 0) : max(y2, 0), max(x1, 0) : max(x2, 0)]
    if crop.size == 0:
        return ""

    tmp_path = Path(image_path).with_suffix(".crop.tmp.png")
    cv2.imwrite(str(tmp_path), crop)
    try:
        lines = ocr_image(tmp_path, ocr_engine=ocr_engine, lang=lang)
        return "\n".join(line["text"] for line in lines)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
