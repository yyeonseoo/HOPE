from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping


MODEL_NAME = "PP-Chart2Table"
MODEL_VERSION = "paddleocr-3.7"
_MODEL = None


def _configure_project_cache() -> None:
    project_dir = Path(__file__).resolve().parents[3]
    cache_dir = project_dir / ".cache"
    paths = {
        "HOME": cache_dir / "home",
        "USERPROFILE": cache_dir / "home",
        "XDG_CACHE_HOME": cache_dir,
        "PADDLE_HOME": cache_dir / "paddle",
        "PADDLE_PDX_CACHE_HOME": cache_dir / "paddlex",
        "MPLCONFIGDIR": cache_dir / "matplotlib",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        os.environ[name] = str(path)
    os.environ.setdefault("FLAGS_use_mkldnn", "0")
    os.environ.setdefault("FLAGS_enable_pir_api", "0")


def _load_model():
    global _MODEL
    if _MODEL is None:
        _configure_project_cache()
        try:
            from paddleocr import ChartParsing
        except ImportError as exc:
            raise RuntimeError("PaddleOCR ChartParsing is unavailable.") from exc
        model_dir = Path(__file__).resolve().parents[3] / ".cache" / "models" / MODEL_NAME
        if not model_dir.exists():
            raise RuntimeError(
                "PP-Chart2Table model is not installed. "
                "Download and extract it to project/.cache/models/PP-Chart2Table."
            )
        _MODEL = ChartParsing(
            model_name=MODEL_NAME,
            model_dir=str(model_dir),
            device="cpu",
            engine="paddle_dynamic",
        )
    return _MODEL


class PPChart2TableEngine:
    """PaddleOCR chart-to-table adapter for the shared figure engine API."""

    model_name = MODEL_NAME
    model_version = MODEL_VERSION

    def __init__(self, model=None):
        self._model = model

    def analyze(self, image_path: str | Path) -> Mapping[str, Any]:
        model = self._model or _load_model()
        results = model.predict(input={"image": str(image_path)}, batch_size=1)
        first = next(iter(results), None)
        if first is None:
            raise RuntimeError("PP-Chart2Table returned no result.")

        payload = _result_payload(first)
        table_text = _find_table_text(payload)
        if not table_text:
            raise RuntimeError("PP-Chart2Table result did not contain chart data.")

        parsed = parse_chart_table(table_text)
        parsed.update(
            {
                "model": {"name": self.model_name, "version": self.model_version},
                "confidence": None,
                "figure_type": "other",
                "title": None,
            }
        )
        parsed.setdefault("warnings", []).append(
            "PP-Chart2Table extracts chart data but does not provide a reliable visual chart type."
        )
        return parsed


def _result_payload(result: Any) -> Any:
    payload = getattr(result, "json", result)
    return payload() if callable(payload) else payload


def _find_table_text(payload: Any) -> str | None:
    if isinstance(payload, str):
        return payload.strip() or None
    if not isinstance(payload, Mapping):
        return None
    direct = payload.get("result")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()
    nested = payload.get("res")
    return _find_table_text(nested)


def parse_chart_table(text: str) -> dict[str, Any]:
    """Convert PP-Chart2Table's pipe-delimited output into axis and series data."""
    rows = []
    for line in text.splitlines():
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) >= 2 and any(cells):
            rows.append(cells)

    if len(rows) < 2:
        return {
            "x_axis": None,
            "y_axis": None,
            "series": [],
            "warnings": ["Chart table contained fewer than two usable rows."],
        }

    headers = rows[0]
    series = []
    for column in range(1, len(headers)):
        points = []
        for row in rows[1:]:
            if column >= len(row):
                continue
            x_value = _scalar(row[0])
            y_value = _scalar(row[column])
            if x_value is None or y_value is None:
                continue
            points.append({"x": x_value, "y": y_value})
        series.append({"name": headers[column] or None, "points": points})

    return {
        "x_axis": {"label": headers[0] or None, "unit": None},
        "y_axis": None,
        "series": series,
        "warnings": [],
    }


def _scalar(value: str) -> str | int | float | None:
    cleaned = value.strip()
    if not cleaned:
        return None
    numeric = cleaned.replace(",", "").replace("−", "-")
    if re.fullmatch(r"[-+]?\d+", numeric):
        return int(numeric)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", numeric):
        return float(numeric)
    return cleaned
