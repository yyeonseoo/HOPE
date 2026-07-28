from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Optional

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env", override=False)

SRC_DIR = ROOT_DIR / "src"
for _path in (ROOT_DIR, SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ocr import _load_paddleocr
from pdf_text import extract_pdf_text_lines
from page_pipeline import process_single_page
from analysis.formula.formula_analyzer import analyze_formula_blocks
from analysis.figure import analyze_figure_blocks, create_openai_figure_engine
from analysis.table import analyze_table_blocks, reconcile_reclassified_table_blocks
from page_description import build_page_description

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
    cache_key = (device, "gpt-5")
    if cache_key not in _FIGURE_ENGINES:
        _FIGURE_ENGINES[cache_key] = create_openai_figure_engine(device=device)
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


def _save_page_image(uploaded_file: UploadFile, target_dir: Path) -> Path:
    content_type = (uploaded_file.content_type or "").lower()
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="page_image must be an image.")
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / "page.png"
    image_path.write_bytes(uploaded_file.file.read())
    if not image_path.stat().st_size:
        raise HTTPException(status_code=400, detail="page_image was empty.")
    return image_path


def _parse_json_form(value: str, field_name: str):
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"{field_name} must be valid JSON.") from exc


def _ocr_lines_from_page(page: dict) -> list[dict]:
    return [
        {
            "text": block.get("text"),
            "bbox": block.get("bbox"),
            "score": block.get("score"),
        }
        for block in page.get("blocks", [])
        if isinstance(block, dict) and block.get("text") and block.get("bbox")
    ]


def _page_for_single_block(page: dict, block_id: str, target_type: str) -> dict:
    blocks = page.get("blocks")
    if not isinstance(blocks, list):
        raise HTTPException(status_code=400, detail="page.blocks must be a list.")
    if not any(isinstance(block, dict) and block.get("block_id") == block_id for block in blocks):
        raise HTTPException(status_code=404, detail=f"Block {block_id!r} was not found.")

    if target_type == "figure":
        # Keep surrounding text blocks for context while ensuring that the
        # figure analyzer calls GPT for the selected block only.
        selected_blocks = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            copied = dict(block)
            if copied.get("block_id") == block_id:
                copied["type"] = target_type
            elif copied.get("type") == "figure":
                copied["type"] = "_context_figure"
            selected_blocks.append(copied)
    else:
        target = next(
            dict(block)
            for block in blocks
            if isinstance(block, dict) and block.get("block_id") == block_id
        )
        target["type"] = target_type
        selected_blocks = [target]
    return {**page, "blocks": selected_blocks}


def _analysis_description_text(record: dict) -> str | None:
    description = record.get("description")
    if not isinstance(description, dict):
        return None
    return description.get("long_text") or description.get("short_text")


def _summarize_api_usage(records: list[dict]) -> dict | None:
    usages = [record.get("api_usage") for record in records if isinstance(record.get("api_usage"), dict)]
    if not usages:
        return None
    summary = {
        "provider": "openai",
        "model": usages[0].get("model"),
        "api_calls": 0,
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "estimated_cost_usd": 0.0,
    }
    for usage in usages:
        for key in ("api_calls", "input_tokens", "cached_input_tokens", "output_tokens", "total_tokens"):
            summary[key] += int(usage.get(key, 0) or 0)
        summary["estimated_cost_usd"] += float(usage.get("estimated_cost_usd", 0) or 0)
    summary["estimated_cost_usd"] = round(summary["estimated_cost_usd"], 8)
    return summary


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
        semantic_analyses = reconcile_reclassified_table_blocks(
            result["page"],
            semantic_analyses,
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

        # Deterministic reading-order text only -- no model rewrite is wired
        # in here.
        page_description_result = build_page_description(result["page"], semantic_analyses)

        return {
            "page_count": page_count,
            "page": result["page"],
            "semantic_analyses": semantic_analyses,
            "page_description": page_description_result,
            "page_image": _image_data_url(result["page_image_path"]),
            "visualization_image": _image_data_url(result["visualization_path"]),
            "ocr_source": result["ocr_source"],
            "layout_mode": result["layout_mode"],
            "figure_captioning_enabled": figure_engine is not None,
            "figure_caption_model": figure_engine.captioner.model_name if figure_engine is not None else None,
            "api_usage": _summarize_api_usage(semantic_analyses),
        }


@app.post("/api/analyze-block")
async def analyze_block(
    page_image: UploadFile = File(...),
    page_json: str = Form(...),
    semantic_analyses_json: str = Form("[]"),
    block_id: str = Form(...),
    target_type: str = Form(...),
):
    """Reanalyze one manually retyped block without rerunning the page."""
    if target_type not in {"figure", "table", "formula"}:
        raise HTTPException(status_code=400, detail="target_type must be figure, table, or formula.")

    page = _parse_json_form(page_json, "page_json")
    semantic_analyses = _parse_json_form(semantic_analyses_json, "semantic_analyses_json")
    if not isinstance(page, dict) or not isinstance(semantic_analyses, list):
        raise HTTPException(status_code=400, detail="Invalid page or semantic analysis data.")

    with tempfile.TemporaryDirectory(prefix="textbook_block_analysis_") as tmp:
        image_path = _save_page_image(page_image, Path(tmp))
        analysis_page = _page_for_single_block(page, block_id, target_type)
        existing = [
            item for item in semantic_analyses
            if isinstance(item, dict) and item.get("block_id") != block_id
        ]
        try:
            if target_type == "formula":
                records = await asyncio.to_thread(
                    analyze_formula_blocks, analysis_page, str(image_path)
                )
            elif target_type == "table":
                records = await asyncio.to_thread(
                    analyze_table_blocks, analysis_page, str(image_path)
                )
            else:
                figure_engine = _get_figure_engine(True)
                async with _FIGURE_ANALYSIS_LOCK:
                    records = await asyncio.to_thread(
                        analyze_figure_blocks,
                        analysis_page,
                        image_path,
                        figure_engine,
                        None,
                        _ocr_lines_from_page(page),
                        existing,
                    )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Block analysis failed: {exc}") from exc

    if not records:
        raise HTTPException(status_code=500, detail="The selected block did not produce an analysis.")
    record = records[0]
    return {
        "analysis": record,
        "description_text": _analysis_description_text(record),
        "api_usage": record.get("api_usage"),
        "analysis_engine": "openai" if target_type == "figure" else "local",
    }
