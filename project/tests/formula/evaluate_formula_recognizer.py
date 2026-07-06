import json
from collections import defaultdict
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.analysis.formula.formula_recognizer import recognize_formula_from_crop


SAMPLES_PATH = Path(__file__).with_name("evaluation_samples.json")


def main():
    samples = json.loads(SAMPLES_PATH.read_text(encoding="utf-8"))

    total_count = 0
    correct_count = 0
    type_total_counts = defaultdict(int)
    type_correct_counts = defaultdict(int)
    failures = []

    for sample in samples:
        sample_id = sample["id"]
        input_text = sample["input_text"]
        expected_latex = sample["expected_latex"]
        case_type = sample["case_type"]

        result = recognize_formula_from_crop(
            crop_path=None,
            fallback_text=input_text,
        )

        predicted_latex = result["latex"]
        is_correct = predicted_latex == expected_latex

        total_count += 1
        type_total_counts[case_type] += 1

        if is_correct:
            correct_count += 1
            type_correct_counts[case_type] += 1
        else:
            failures.append(
                {
                    "id": sample_id,
                    "case_type": case_type,
                    "input_text": input_text,
                    "expected_latex": expected_latex,
                    "predicted_latex": predicted_latex,
                    "warnings": result["warnings"],
                }
            )

    accuracy = correct_count / total_count if total_count else 0

    print("=" * 80)
    print("Formula Recognizer Evaluation")
    print("=" * 80)
    print(f"Total: {total_count}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nCase Type Accuracy")
    print("-" * 80)

    for case_type in sorted(type_total_counts):
        type_total = type_total_counts[case_type]
        type_correct = type_correct_counts[case_type]
        type_accuracy = type_correct / type_total if type_total else 0

        print(f"{case_type}: {type_correct}/{type_total} ({type_accuracy:.2%})")

    print("\nFailures")
    print("-" * 80)

    if not failures:
        print("No failures.")
    else:
        for failure in failures:
            print(f"[{failure['id']}] {failure['case_type']}")
            print(f"input: {failure['input_text']}")
            print(f"expected: {failure['expected_latex']}")
            print(f"predicted: {failure['predicted_latex']}")
            print(f"warnings: {failure['warnings']}")
            print("-" * 80)


if __name__ == "__main__":
    main()