"""Export every [확인 필요] flag (and a sample of unflagged values) to a CSV
a person can grade, to report the precision of the grounding filter.

Usage:
  python eval/export_flag_review.py --results <dir with p*_result.json> --out flags.csv

Fill the "정답" column with 맞음 (the value is correct in the figure) or 틀림.
Precision of the flag = share of flagged values that are 틀림.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

FLAG = "[확인 필요]"
VALUE_BEFORE_FLAG = re.compile(r"(\S{0,20}?)(\(?[-+]?\d[\d.,]*(?:,\s*[-+]?\d[\d.,]*)?\)?)" + re.escape(FLAG))


def rows(results_dir: Path):
    for path in sorted(results_dir.glob("p*_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for analysis in data.get("semantic_analyses", []):
            if analysis.get("type") != "figure":
                continue
            text = (analysis.get("description") or {}).get("long_text") or ""
            evidence = " / ".join((analysis.get("education_context") or {}).get("figure_ocr") or [])
            for match in VALUE_BEFORE_FLAG.finditer(text):
                start, end = max(0, match.start() - 30), min(len(text), match.end() + 30)
                yield {
                    "쪽": data.get("page_number") or data.get("page", {}).get("page_id"),
                    "블록": analysis.get("block_id"),
                    "표시된 값": match.group(2),
                    "문맥": text[start:end].replace("\n", " "),
                    "그림 안 OCR 근거": evidence,
                    "정답": "",
                }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    found = list(rows(args.results))
    with args.out.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["쪽", "블록", "표시된 값", "문맥", "그림 안 OCR 근거", "정답"])
        writer.writeheader()
        writer.writerows(found)
    print(f"{len(found)} flagged values -> {args.out}")


if __name__ == "__main__":
    main()
