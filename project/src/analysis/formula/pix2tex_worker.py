"""Run pix2tex in its isolated virtual environment.

This module is intentionally dependency-light so the main application can keep
the modern timm version required by OpenCLIP while pix2tex uses its pinned timm
version in a separate interpreter.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


RESULT_PREFIX = "__PIX2TEX_RESULT__"


def main() -> int:
    if len(sys.argv) != 2:
        print(f"{RESULT_PREFIX}{json.dumps({'error': 'image path is required'})}")
        return 2

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"{RESULT_PREFIX}{json.dumps({'error': 'image file not found'})}")
        return 2

    try:
        from PIL import Image
        from pix2tex.cli import LatexOCR

        result = LatexOCR()(Image.open(image_path))
        payload = {"latex": str(result).strip() if result else None}
    except Exception as exc:
        payload = {"error": f"{type(exc).__name__}: {exc}"}

    print(f"{RESULT_PREFIX}{json.dumps(payload, ensure_ascii=False)}")
    return 0 if payload.get("latex") else 1


if __name__ == "__main__":
    raise SystemExit(main())
