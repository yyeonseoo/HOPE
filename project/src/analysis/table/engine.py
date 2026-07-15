from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

MODEL_NAME = "TableRecognitionPipelineV2"
MODEL_VERSION = "paddleocr-3.7"

_ENGINE_CACHE: Dict[str, object] = {}


def _load_table_engine(lang: str = "korean"):
    """Lazy-load and cache a PaddleOCR table-recognition pipeline.

    Mirrors src/ocr.py::_load_paddleocr's cache-dir env-var redirection so
    both engines share the same on-disk model cache under project/.cache
    instead of silently writing model weights outside the repo. This module
    duplicates that setup rather than importing ocr.py, to stay
    self-contained within src/analysis/table/ per OWNERSHIP.md.
    """
    if lang in _ENGINE_CACHE:
        return _ENGINE_CACHE[lang]

    project_cache_dir = Path(__file__).resolve().parents[3] / ".cache"
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
    os.environ["PADDLE_PDX_CACHE_HOME"] = str(cache_dir)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")

    try:
        from paddleocr import TableRecognitionPipelineV2
    except ImportError as exc:
        raise RuntimeError(
            "paddleocr is not installed. Run: pip install -r requirements.txt"
        ) from exc

    engine = TableRecognitionPipelineV2(
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
        device="cpu",
        # Without this, CPU oneDNN raises "ConvertPirAttribute2RuntimeAttribute
        # not support [pir::ArrayAttribute<pir::DoubleAttribute>]" for the
        # SLANeXt table-structure model (paddlepaddle 3.3.1 / paddlex 3.7.2).
        enable_mkldnn=False,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_layout_detection=False,
        use_ocr_model=True,
    )
    _ENGINE_CACHE[lang] = engine
    return engine


def run_table_engine(engine, crop: np.ndarray) -> Optional[Dict]:
    """Run the table-recognition pipeline on a single crop (BGR ndarray).

    Returns `{"html": str, "confidence": None}` for the largest table region
    found, or None if the pipeline found no table at all. The pipeline is
    called with `use_layout_detection=False` because the crop is already a
    single table region (isolated by our own upstream layout detector) — the
    whole crop is treated as one table box rather than re-detecting layout.

    `confidence` is always None: PaddleOCR's TableRecognitionPipelineV2 does
    not surface a table-structure confidence score in its final per-table
    result (only an internal, discarded `structure_score`), so reporting one
    here would mean inventing a number rather than reading a real value —
    against the "unknown values are null, never guessed" contract.
    """
    results = list(
        engine.predict(
            crop,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_layout_detection=False,
            use_ocr_model=True,
            use_table_orientation_classify=False,
        )
    )
    if not results:
        return None

    table_res_list = results[0].get("table_res_list") or []
    if not table_res_list:
        return None

    best = max(table_res_list, key=_table_cell_area)
    html = best.get("pred_html")
    if not html:
        return None

    return {
        "html": html,
        "confidence": None,
        # Crop-local pixel boxes, one per physical <td>/<th> cell, in the
        # same document order pix2struct-style table models emit their HTML
        # cells in. Used by formula_cells.py to crop individual cells for
        # formula-OCR re-recognition; None/mismatched-length callers simply
        # skip that step (see html_parser.parse_html_table).
        "cell_box_list": best.get("cell_box_list") or [],
    }


def _table_cell_area(table_res: Dict) -> float:
    cell_boxes = table_res.get("cell_box_list") or []
    if not cell_boxes:
        return 0.0
    points = [_as_points(box) for box in cell_boxes]
    points = [p for p in points if p.size]
    if not points:
        return 0.0
    all_points = np.concatenate(points, axis=0)
    xs, ys = all_points[:, 0], all_points[:, 1]
    return float((xs.max() - xs.min()) * (ys.max() - ys.min()))


def _as_points(box) -> np.ndarray:
    """Normalize one cell box into an (N, 2) ndarray of (x, y) points.

    PaddleX's cell_box_list entries are numpy arrays, either a flat
    [x1, y1, x2, y2] (size 4) or four (x, y) corner points (size 8 once
    flattened) — shape, not element dtype, is what distinguishes them. An
    earlier version checked `isinstance(value, (int, float))`, which is
    False for numpy scalar dtypes (float32/int64/...), so real model output
    always fell through to the wrong branch and crashed with "invalid index
    to scalar variable" once a box happened to be the flat 4-number form.
    """
    flat = np.asarray(box, dtype=float).reshape(-1)
    if flat.size == 0:
        return flat.reshape(0, 2)
    if flat.size == 4:
        x1, y1, x2, y2 = flat
        return np.array([[x1, y1], [x2, y2]])
    return flat.reshape(-1, 2)
