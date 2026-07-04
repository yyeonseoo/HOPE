from pathlib import Path
from typing import Any, Dict, List, Optional


def recognize_formula_from_crop(
    crop_path: Optional[str],
    fallback_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    수식 crop 이미지를 입력받아 LaTeX/MathML 후보를 반환한다.

    현재 단계에서는 실제 수식 인식 모델을 붙이기 전이므로,
    Model A가 제공한 fallback_text를 LaTeX 후보로 사용한다.

    이후 PP-FormulaNet, UniMERNet 같은 수식 인식 모델을 연결할 때
    이 함수 내부만 교체하면 된다.
    """

    warnings: List[str] = []

    if crop_path is None:
        warnings.append("Formula crop path was not provided.")
    else:
        crop_file = Path(crop_path)
        if not crop_file.exists():
            warnings.append(f"Formula crop file does not exist: {crop_path}")

    plain_text = normalize_plain_text(fallback_text)

    if plain_text is not None and not contains_formula_signal(plain_text):
        warnings.append("Detected formula block does not contain a formula-like expression.")
        latex = None
    else:
        latex = normalize_latex_candidate(plain_text)

    if latex is None:
        warnings.append("Formula LaTeX could not be recognized from crop or fallback text.")

    mathml = convert_latex_to_mathml(latex)

    return {
        "latex": latex,
        "mathml": mathml,
        "plain_text": plain_text if plain_text is not None else latex,
        "confidence": None,
        "model": {
            "name": "formula-recognizer-fallback",
            "version": None,
        },
        "warnings": warnings,
    }

def normalize_plain_text(text: Optional[str]) -> Optional[str]:
    """
    fallback text를 사람이 읽을 수 있는 형태로 정리한다.
    """

    if text is None:
        return None

    cleaned = str(text).strip()

    if not cleaned:
        return None

    cleaned = " ".join(cleaned.split())

    if not cleaned:
        return None

    return cleaned


def contains_formula_signal(text: str) -> bool:
    """
    텍스트가 실제 수식 관계식인지 대략 판단한다.

    예:
    - y=ax → True
    - y=-2x → True
    - a>0 → True
    - ⑴ -2, -1, 0, 1, 2 / ⑵ 수 전체 → False
    """

    compact = text.replace(" ", "")

    formula_signals = [
        "=",
        "<",
        ">",
        "≤",
        "≥",
        "≠",
        "^",
        "√",
        r"\frac",
    ]

    if any(signal in compact for signal in formula_signals):
        return True

    # a/x, 6/x 같은 반비례식 후보
    if "/" in compact and any(variable in compact for variable in ["x", "y", "a", "b"]):
        return True

    # y2, x1 같은 단순 변수+숫자만으로는 부족하므로 여기서는 제외
    return False

def normalize_latex_candidate(text: Optional[str]) -> Optional[str]:
    """
    OCR 또는 fallback text를 LaTeX 후보 문자열로 최소 정리한다.
    """

    if text is None:
        return None

    cleaned = str(text).strip()

    if not cleaned:
        return None

    # 설명 조건 분리: "y=ax (단, a는 0이 아니다.)" -> "y=ax"
    for separator in ["(단", "（단", "( 단", "단,"]:
        if separator in cleaned:
            cleaned = cleaned.split(separator)[0]
            break

    replacements = {
        "×": r"\times",
        "÷": r"\div",
        "Ö": r"\div",
        "−": "-",
        "－": "-",
        "＝": "=",
        "＋": "+",
        " ": "",
    }

    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)

    cleaned = cleaned.strip()

    if not cleaned:
        return None

    return cleaned


def convert_latex_to_mathml(latex: Optional[str]) -> Optional[str]:
    """
    LaTeX를 MathML로 변환한다.

    현재는 외부 변환 라이브러리를 연결하지 않았기 때문에 None을 반환한다.
    이후 latex2mathml 같은 라이브러리를 도입하면 이 함수 내부를 교체하면 된다.
    """

    if latex is None:
        return None

    return None