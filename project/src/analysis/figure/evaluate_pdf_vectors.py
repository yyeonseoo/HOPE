from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analysis.figure.pdf_vector import analyze_pdf_vector_figure
from page_pipeline import process_single_page


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate native PDF vector analysis on all detected figures.")
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()

    import fitz

    with fitz.open(args.pdf) as document:
        page_count = len(document)

    records = []
    with tempfile.TemporaryDirectory(prefix="figure_vector_eval_") as tmp:
        work_root = Path(tmp)
        for page_number in range(1, page_count + 1):
            print(f"[{page_number}/{page_count}] layout detection", flush=True)
            page_output = process_single_page(
                pdf_path=args.pdf,
                page_number=page_number,
                work_dir=work_root / f"page_{page_number:04d}",
                dpi=args.dpi,
                yolo_model_path="hf:juliozhao/DocLayout-YOLO-DocStructBench",
                prefer_pdf_text=True,
            )
            for block in page_output["page"]["blocks"]:
                if block.get("type") != "figure":
                    continue
                try:
                    output = analyze_pdf_vector_figure(
                        pdf_path=args.pdf,
                        page_number=page_number,
                        bbox=block["bbox"],
                        dpi=args.dpi,
                        block_id=block["block_id"],
                        detection_score=block.get("score"),
                        detector=block.get("detector", "layout detector"),
                    )
                    records.append(output)
                except Exception as exc:
                    records.append(
                        {
                            "record": {
                                "page_id": page_number,
                                "block_id": block.get("block_id"),
                                "type": "figure",
                                "bbox": block.get("bbox"),
                                "error": str(exc),
                            },
                            "evidence": None,
                        }
                    )

    statuses = {}
    for item in records:
        status = item["record"].get("analysis", {}).get("status", "error")
        statuses[status] = statuses.get(status, 0) + 1
    payload = {
        "pdf": args.pdf.name,
        "page_count": page_count,
        "figure_count": len(records),
        "status_counts": statuses,
        "results": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("pdf", "page_count", "figure_count", "status_counts")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
