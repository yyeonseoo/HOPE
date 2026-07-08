from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from .analyzer import analyze_figure_blocks
from .description import build_context_free_description
from .pp_chart2table import PPChart2TableEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PP-Chart2Table on one verified chart crop.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with Image.open(args.image) as image:
        width, height = image.size
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"status": "running", "input": str(args.image)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    page = {
        "page_id": 1,
        "blocks": [
            {
                "block_id": "trial_figure_1",
                "type": "figure",
                "bbox": [0, 0, width, height],
                "score": 1.0,
                "detector": "human_verified_crop",
            }
        ],
    }

    started = time.perf_counter()
    record = analyze_figure_blocks(
        page,
        page_image_path=args.image,
        engine=PPChart2TableEngine(),
        output_dir=args.output.parent / "crops",
    )[0]
    record["description"] = build_context_free_description(record["analysis"])
    payload = {"elapsed_seconds": round(time.perf_counter() - started, 2), "record": record}

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
