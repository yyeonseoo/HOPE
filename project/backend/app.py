from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path
from typing import Optional

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ocr import _load_paddleocr
from pdf_text import extract_pdf_text_lines
from page_pipeline import process_single_page


app = FastAPI(title="Textbook Layout Parser API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_OCR_ENGINES = {}


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
    yolo_model_path: Optional[str] = Form(None),
):
    with tempfile.TemporaryDirectory(prefix="textbook_layout_") as tmp:
        tmp_dir = Path(tmp)
        pdf_path = _save_upload(file, tmp_dir / "uploads")
        try:
            page_count = _count_pages(pdf_path)
            if page_number < 1 or page_number > page_count:
                raise HTTPException(status_code=400, detail=f"Page number must be between 1 and {page_count}.")

            pdf_text_lines = extract_pdf_text_lines(pdf_path, page_number, dpi=dpi)
            prefer_pdf_text = len(pdf_text_lines) >= 3
            ocr_engine = None if prefer_pdf_text else _get_ocr_engine(lang)
            result = process_single_page(
                pdf_path=pdf_path,
                page_number=page_number,
                work_dir=tmp_dir / "results",
                dpi=dpi,
                yolo_model_path=yolo_model_path.strip() if yolo_model_path else None,
                lang=lang,
                ocr_engine=ocr_engine,
                prefer_pdf_text=prefer_pdf_text,
            )
        except HTTPException:
            raise
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}") from exc

        return {
            "page_count": page_count,
            "page": result["page"],
            "page_image": _image_data_url(result["page_image_path"]),
            "visualization_image": _image_data_url(result["visualization_path"]),
            "ocr_source": result["ocr_source"],
        }
