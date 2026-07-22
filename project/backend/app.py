from __future__ import annotations

import asyncio
import base64
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for _path in (ROOT_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ocr import _load_paddleocr
from pdf_text import extract_pdf_text_lines
from page_pipeline import process_single_page
from analysis.formula.formula_analyzer import analyze_formula_blocks
from analysis.figure import analyze_figure_blocks, create_huggingface_figure_engine
from analysis.table import analyze_table_blocks
from page_description import build_page_description
from page_confidence import build_page_confidence

app = FastAPI(title="Textbook Layout Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_OCR_ENGINES = {}
_ANALYSIS_LOCK = asyncio.Lock()
_FIGURE_ENGINES = {}
_FIGURE_ANALYSIS_LOCK = asyncio.Lock()
LAYOUT_MODES = {"doclayout_yolo", "doclayout_yolo_raw", "doclayout_yolo_unit3"}


def _get_figure_engine(request_enabled: bool):
    enabled_by_environment = os.getenv("HOPE_FIGURE_CAPTIONING", "0").strip().lower() in {"1", "true", "yes", "on"}
    enabled = request_enabled or enabled_by_environment
    if not enabled:
        return None
    device = os.getenv("HOPE_FIGURE_DEVICE", "auto")
    cache_key = (device, "qwen3-vl-2b")
    if cache_key not in _FIGURE_ENGINES:
        _FIGURE_ENGINES[cache_key] = create_huggingface_figure_engine(device=device)
    return _FIGURE_ENGINES[cache_key]


def _get_ocr_engine(lang: str):
    if lang not in _OCR_ENGINES:
        _OCR_ENGINES[lang] = _load_paddleocr(lang=lang)
    return _OCR_ENGINES[lang]


def _save_upload(uploaded_file: UploadFile, target_dir: Path) -> Path:
    if not uploaded_file.filename or not uploaded_file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files can be uploaded.")

    target_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = target_dir / Path(uploaded_file.filename).name
    pdf_path.write_bytes(uploaded_file.file.read())
    return pdf_path


def _count_pages(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return len(doc)
    finally:
        doc.close()


def _image_data_url(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _analyze_saved_pdf(
    pdf_path: Path,
    work_dir: Path,
    page_number: int,
    dpi: int,
    lang: str,
    layout_model: str,
    yolo_model_path: Optional[str],
):
    page_count = _count_pages(pdf_path)
    if page_number < 1 or page_number > page_count:
        raise HTTPException(status_code=400, detail=f"Page number must be between 1 and {page_count}.")

    pdf_text_lines = extract_pdf_text_lines(pdf_path, page_number, dpi=dpi)
    prefer_pdf_text = len(pdf_text_lines) >= 3
    ocr_engine = None if prefer_pdf_text else _get_ocr_engine(lang)
    selected_model_path = yolo_model_path.strip() if yolo_model_path else None
    if layout_model not in LAYOUT_MODES:
        raise HTTPException(status_code=400, detail=f"Unknown layout model: {layout_model}")
    selected_model_path = "hf:juliozhao/DocLayout-YOLO-DocStructBench"

    result = process_single_page(
        pdf_path=pdf_path,
        page_number=page_number,
        work_dir=work_dir,
        dpi=dpi,
        yolo_model_path=selected_model_path,
        lang=lang,
        ocr_engine=ocr_engine,
        prefer_pdf_text=prefer_pdf_text,
        model_only=layout_model == "doclayout_yolo_raw",
        correction_profile="unit3" if layout_model == "doclayout_yolo_unit3" else None,
    )
    return page_count, result


@app.get("/api/health")
def health():
    return {"ok": True}


@app.post("/api/page-count")
async def page_count(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory(prefix="textbook_page_count_") as tmp:
        pdf_path = _save_upload(file, Path(tmp))
        try:
            count = _count_pages(pdf_path)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not open PDF: {exc}") from exc
        return {"page_count": count}


@app.post("/api/analyze")
async def analyze_page(
    file: UploadFile = File(...),
    page_number: int = Form(...),
    dpi: int = Form(120),
    lang: str = Form("korean"),
    layout_model: str = Form("doclayout_yolo"),
    yolo_model_path: Optional[str] = Form(None),
    figure_captioning: bool = Form(False),
):
    with tempfile.TemporaryDirectory(prefix="textbook_layout_") as tmp:
        tmp_dir = Path(tmp)
        pdf_path = _save_upload(file, tmp_dir / "uploads")
        try:
            async with _ANALYSIS_LOCK:
                page_count, result = await asyncio.to_thread(
                    _analyze_saved_pdf,
                    pdf_path,
                    tmp_dir / "results",
                    page_number,
                    dpi,
                    lang,
                    layout_model,
                    yolo_model_path,
                )
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        semantic_analyses = analyze_formula_blocks(
            result["page"],
            page_image_path=result["page_image_path"],
        )
        semantic_analyses.extend(
            analyze_table_blocks(result["page"], str(result["page_image_path"]))
        )
        figure_engine = _get_figure_engine(figure_captioning)
        if figure_engine is not None:
            async with _FIGURE_ANALYSIS_LOCK:
                figure_analyses = await asyncio.to_thread(
                    analyze_figure_blocks,
                    result["page"],
                    result["page_image_path"],
                    figure_engine,
                    ocr_lines=result.get("ocr_lines"),
                    semantic_analyses=semantic_analyses,
                    pdf_path=pdf_path,
                    source_dpi=dpi,
                )
            semantic_analyses.extend(figure_analyses)

        # Deterministic reading-order text only -- no model. The optional
        # Qwen-based rewrite (`generate_page_description`) exists and is
        # tested, but isn't wired in here yet: on real pages it produced
        # garbled vocabulary (not just fabricated numbers/equations, which
        # build_page_description's grounding check catches) that slipped
        # through as a "clean" result, so it isn't reliable enough to expose
        # through this endpoint on the current 2B/CPU setup.
        page_description_result = build_page_description(result["page"], semantic_analyses)

        page_confidence_result = build_page_confidence(
            result["page"],
            semantic_analyses,
            page_description_result,
        )

        return {
            "page_count": page_count,
            "page": result["page"],
            "semantic_analyses": semantic_analyses,
            "page_description": page_description_result,
            "page_confidence": page_confidence_result,
            "page_image": _image_data_url(result["page_image_path"]),
            "visualization_image": _image_data_url(result["visualization_path"]),
            "ocr_source": result["ocr_source"],
            "layout_mode": result["layout_mode"],
            "figure_captioning_enabled": figure_engine is not None,
            "figure_caption_model": "Qwen/Qwen3-VL-2B-Instruct" if figure_engine is not None else None,
        }
