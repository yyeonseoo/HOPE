"""Compare braille treatment suggestions (rule-only, decision-tree-only,
hybrid) against a reference.

Two kinds of reference:
  --reference eval/reference_unit3_dev.json   acceptable treatments per block
  --review-log <교과서>_검수기록.json          the reviewer's final choices,
                                               exported from the review UI

Usage:
  python eval/evaluate_treatments.py --results <dir with p*_result.json> --reference eval/reference_unit3_dev.json
  python eval/evaluate_treatments.py --review-log 교과서_검수기록.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

METHODS = ("rule", "tree", "hybrid")


def suggestions_from_results(results_dir: Path) -> dict[str, dict[str, str | None]]:
    out: dict[str, dict[str, str | None]] = {}
    for path in sorted(results_dir.glob("p*_result.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for analysis in data.get("semantic_analyses", []):
            review = analysis.get("braille_review") or {}
            if not review.get("suggested_visual_treatment"):
                continue
            rule = review.get("rule_suggestion") or review["suggested_visual_treatment"]
            out[analysis["block_id"]] = {
                "rule": rule,
                "tree": review.get("tree_suggestion"),
                "hybrid": review["suggested_visual_treatment"],
            }
    return out


def score(suggestions: dict[str, dict[str, str | None]], acceptable: dict[str, list[str]]) -> dict[str, dict]:
    report = {}
    for method in METHODS:
        judged = [(block, s[method]) for block, s in suggestions.items() if block in acceptable and s.get(method)]
        correct = [block for block, value in judged if value in acceptable[block]]
        report[method] = {
            "n": len(judged),
            "correct": len(correct),
            "accuracy": round(len(correct) / len(judged), 3) if judged else None,
            "wrong": sorted({block for block, _ in judged} - set(correct)),
        }
    report["missing_from_results"] = sorted(set(acceptable) - set(suggestions))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--review-log", type=Path)
    args = parser.parse_args()

    if args.review_log:
        exported = json.loads(args.review_log.read_text(encoding="utf-8"))
        decisions = [d for page in exported["pages"] for d in page.get("decisions", []) if d.get("reviewer")]
        acceptable = {d["block_id"]: [d["reviewer"]] for d in decisions}
        suggestions = {
            d["block_id"]: {"rule": d.get("rule") or d["suggested"], "tree": d.get("tree"), "hybrid": d["suggested"]}
            for d in decisions
        }
        source = f"reviewer decisions ({args.review_log.name})"
    else:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        acceptable = reference["acceptable"]
        suggestions = suggestions_from_results(args.results)
        source = f"{args.reference.name} (split: {reference.get('split', '?')})"

    print(json.dumps({"reference": source, **score(suggestions, acceptable)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
