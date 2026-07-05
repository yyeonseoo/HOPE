import re
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

def is_axis_or_unit_label(text: str) -> bool:
    """
    그래프 축 라벨이나 단위 표기는 수식 분석 대상에서 제외한다.

    예:
    x(m/s), y(원), y(km), 시간(초), 거리(km)
    """

    axis_or_unit_patterns = [
        r"^[xy]\([a-zA-Z가-힣/]+\)$",
        r"^[가-힣]+\([a-zA-Z가-힣/]+\)$",
        r"^\([a-zA-Z가-힣/]+\)$",
    ]

    return any(re.fullmatch(pattern, text) for pattern in axis_or_unit_patterns)

def is_table_like_formula_noise(text: str) -> bool:
    """
    표, 그래프 축, 숫자 나열이 formula block으로 잡힌 경우 제외한다.

    예:
    x(m/s)102030405060y(초)
    x102030405060x(m/s)4800y(초)48002401601209680x
    """

    has_equation_signal = any(signal in text for signal in ["=", "<", ">", "≤", "≥", "≠"])

    if has_equation_signal:
        return False

    numbers = re.findall(r"[+-]?\d+(?:\.\d+)?", text)

    if len(numbers) >= 5:
        return True

    return False

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

    if is_axis_or_unit_label(compact):
        return False
    
    if is_table_like_formula_noise(text):
        return False
    
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
    
    # (1,a), (-2,-4), (x,y) 같은 좌표/순서쌍 표현
    if re.fullmatch(r"\([+-]?[0-9.]+,[a-zA-Z가-힣]\)", compact):
        return True

    if re.fullmatch(r"\([a-zA-Z가-힣],[a-zA-Z가-힣]\)", compact):
        return True

    if re.fullmatch(r"\([+-]?[0-9.]+,[+-]?[0-9.]+\)", compact):
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

    cleaned = cleaned.replace(" ", "")
    cleaned = remove_formula_prefix(cleaned)
    cleaned = remove_formula_suffix(cleaned)
    cleaned = normalize_fraction_artifacts(cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    return cleaned

def remove_formula_prefix(text: str) -> str:
    """
    교과서 문항 번호나 설명 접두어가 수식 앞에 붙은 경우 제거한다.
    여러 수식이 한 블록에 붙어 있으면 중간 문항 번호를 구분자로 바꾼다.

    예:
    ⑴y=4x -> y=4x
    ⑵y=-3x -> y=-3x
    식y=800x -> y=800x
    ⑴y=4x⑵y=-3x -> y=4x;y=-3x
    (1)y=4x(2)y=-3x -> y=4x;y=-3x
    """

    # 맨 앞 문항 번호 제거
    text = re.sub(r"^[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]", "", text)
    text = re.sub(r"^\((?:1|2|3|4|5|6|7|8|9|10)\)(?=[a-zA-Z가-힣])", "", text)

    # 중간 문항 번호는 수식 구분자로 변환
    text = re.sub(r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]", ";", text)
    text = re.sub(r"\((?:1|2|3|4|5|6|7|8|9|10)\)(?=[a-zA-Z가-힣])", ";", text)

    # 설명 접두어 제거
    text = re.sub(r"^식(?=[a-zA-Z가-힣]*=)", "", text)

    # 구분자 정리
    text = re.sub(r";+", ";", text)
    text = text.strip(";")

    return text

def remove_formula_suffix(text: str) -> str:
    """
    수식 뒤에 붙은 설명 문구나 OCR 잡음을 제거한다.

    예:
    y=ax의그래프 -> y=ax
    y=ax` -> y=ax
    """

    text = re.sub(r"`+$", "", text)
    text = re.sub(r"의그래프.*$", "", text)
    text = re.sub(r"그래프.*$", "", text)

    return text

def normalize_fraction_artifacts(text: str) -> str:
    """
    교과서 OCR에서 자주 깨지는 분수 표기를 LaTeX 형태로 보정한다.

    예:
    y=;2!;x -> y=\\frac{1}{2}x
    """

    # ;2!; -> \frac{1}{2}
    text = re.sub(r";(\d+)!;", r"\\frac{1}{\1}", text)

    # ;3@; -> \frac{2}{3}, ;4#; -> \frac{3}{4} 형태 확장
    numerator_map = {
        "!": "1",
        "@": "2",
        "#": "3",
        "$": "4",
        "%": "5",
    }

    def replace_special_fraction(match: re.Match) -> str:
        denominator = match.group(1)
        numerator_symbol = match.group(2)
        numerator = numerator_map.get(numerator_symbol)

        if numerator is None:
            return match.group(0)

        return rf"\frac{{{numerator}}}{{{denominator}}}"

    text = re.sub(r";(\d+)([!@#$%]);", replace_special_fraction, text)

    return text

def convert_latex_to_mathml(latex: Optional[str]) -> Optional[str]:
    """
    LaTeX를 MathML로 변환한다.

    현재는 외부 변환 라이브러리를 연결하지 않았기 때문에 None을 반환한다.
    이후 latex2mathml 같은 라이브러리를 도입하면 이 함수 내부를 교체하면 된다.
    """

    if latex is None:
        return None

    return None