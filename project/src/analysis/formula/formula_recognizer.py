import re
from pathlib import Path
from typing import Any, Dict, List, Optional

def count_formula_parts(latex: Optional[str]) -> int:
    if latex is None:
        return 0

    return len([part for part in latex.split(";") if part])

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

    fallback_source_text = fallback_text if fallback_text is not None else plain_text
    fallback_latex_candidate = normalize_latex_candidate(fallback_source_text)

    model_result = recognize_with_optional_pix2tex(crop_path)

    used_fallback_due_to_part_count = False

    if model_result is not None:
        extracted_model_result = extract_formula_from_model_latex(model_result)

        if extracted_model_result is not None and is_reliable_model_latex(extracted_model_result):
            latex = normalize_latex_candidate(extracted_model_result)

            if latex is not None:
                if count_formula_parts(fallback_latex_candidate) > count_formula_parts(latex):
                    warnings.append(
                        "Pix2tex recognized fewer formula parts than fallback text; fallback recognizer was used."
                    )
                    used_fallback_due_to_part_count = True
                else:
                    return {
                        "latex": latex,
                        "mathml": convert_latex_to_mathml(latex),
                        "plain_text": plain_text,
                        "confidence": None,
                        "model": {
                            "name": "pix2tex",
                            "version": None,
                        },
                        "warnings": warnings,
                    }
        if not used_fallback_due_to_part_count:
            warnings.append(
                "Pix2tex output was rejected as unreliable; fallback recognizer was used."
            )
    elif crop_path is not None:
        warnings.append(
            "Pix2tex was unavailable or failed; fallback recognizer was used."
        )

    if plain_text is not None and not contains_formula_signal(plain_text):
        warnings.append("Detected formula block does not contain a formula-like expression.")
        latex = None
    else:
        latex = fallback_latex_candidate
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

def recognize_with_optional_pix2tex(crop_path: Optional[str]) -> Optional[str]:
    """
    pix2tex가 설치되어 있고 crop 이미지가 있으면 이미지 기반 LaTeX 인식을 시도한다.
    설치되어 있지 않거나 실패하면 None을 반환하여 기존 fallback 로직을 사용하게 한다.
    """

    if crop_path is None:
        return None

    image_path = Path(crop_path)

    if not image_path.exists():
        return None

    try:
        from PIL import Image
        from pix2tex.cli import LatexOCR
    except Exception:
        return None

    try:
        model = LatexOCR()
        image = Image.open(image_path)
        result = model(image)

        if not result:
            return None

        return str(result).strip()
    except Exception:
        return None

def extract_formula_from_model_latex(model_latex: str) -> Optional[str]:
    """
    pix2tex 결과가 array, subscript, 장식 명령 등을 포함하더라도
    그 안에서 실제 교과서 수식으로 보이는 부분만 추출한다.

    예:
    \\begin{array} ... (1) y=\\frac{8}{x} ... \\end{array}
    -> y=\\frac{8}{x}

    \\bigcup_{y=-{\\frac{8}{x}}}
    -> y=-\\frac{8}{x}
    """

    if not model_latex:
        return None

    text = model_latex.replace(" ", "")

    # y=-{\frac{8}{x}} 또는 y={\frac{8}{x}} 형태
    braced_fraction_match = re.search(
        r"([a-zA-Z])=([+-]?)\{?\\frac\{([^{}]+)\}\{([^{}]+)\}\}?",
        text,
    )

    if braced_fraction_match:
        left = braced_fraction_match.group(1)
        sign = braced_fraction_match.group(2)
        numerator = braced_fraction_match.group(3)
        denominator = braced_fraction_match.group(4)

        return rf"{left}={sign}\frac{{{numerator}}}{{{denominator}}}"

    # y=-{\frac{8}{x}} 처럼 음수 부호 뒤에 중괄호 분수가 오는 형태
    negative_braced_fraction_match = re.search(
        r"([a-zA-Z])=-\{\\frac\{([^{}]+)\}\{([^{}]+)\}\}",
        text,
    )

    if negative_braced_fraction_match:
        left = negative_braced_fraction_match.group(1)
        numerator = negative_braced_fraction_match.group(2)
        denominator = negative_braced_fraction_match.group(3)

        return rf"{left}=-\frac{{{numerator}}}{{{denominator}}}"

    # y=\frac{8}{x} 형태
    fraction_match = re.search(
        r"([a-zA-Z])=([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}",
        text,
    )

    if fraction_match:
        left = fraction_match.group(1)
        sign = fraction_match.group(2)
        numerator = fraction_match.group(3)
        denominator = fraction_match.group(4)

        return rf"{left}={sign}\frac{{{numerator}}}{{{denominator}}}"

    # y=4x, y=-3x, y=ax 형태
    linear_match = re.search(r"([a-zA-Z])=([+-]?\d*|[a-zA-Z])([a-zA-Z])", text)

    if linear_match:
        left = linear_match.group(1)
        coefficient = linear_match.group(2)
        variable = linear_match.group(3)

        return f"{left}={coefficient}{variable}"

    return model_latex

def is_reliable_model_latex(latex: str) -> bool:
    """
    이미지 기반 수식 인식 모델 결과가 지나치게 복잡하거나 잡음이 많으면 사용하지 않는다.
    불안정한 모델 결과는 fallback text 기반 후처리로 넘긴다.
    """

    if not latex:
        return False

    compact = latex.replace(" ", "")

    # 너무 긴 결과는 작은 수식 crop에서 나온 결과로 보기 어렵다.
    if len(compact) > 80:
        return False

    suspicious_tokens = [
        r"\stackrel",
        r"\operatorname",
        r"\mathsf",
        r"\scriptscriptstyle",
        r"\textstyle",
        r"\underline",
        r"\emptyset",
        r"\Phi",
        r"\odot",
        r"\subseteq",
    ]

    if any(token in compact for token in suspicious_tokens):
        return False

    # 간단한 교과서 수식에서 LaTeX 명령이 과도하게 많이 나오면 잡음 가능성이 높다.
    latex_command_count = len(re.findall(r"\\[a-zA-Z]+", compact))

    if latex_command_count >= 4:
        return False

    return True

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

    numbered_formula = normalize_numbered_formula_parts(text)

    if numbered_formula is not None:
        return numbered_formula

    stacked_fraction = normalize_stacked_fraction(text)

    if stacked_fraction is not None:
        return stacked_fraction
    
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
    cleaned = normalize_inline_fraction(cleaned)
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

def normalize_inline_fraction(text: Optional[str]) -> Optional[str]:
    """
    y=a/x, y = a / x, y=a÷x 같은 한 줄 분수 표현을 LaTeX 분수로 변환한다.
    y=ax처럼 분수 기호가 없는 정비례식은 변환하지 않는다.
    """

    if text is None:
        return None

    normalized = text.replace("÷", "/").replace("\\div", "/")

    parts = normalized.split(";")
    converted_parts = [normalize_inline_fraction_part(part) for part in parts]

    return ";".join(converted_parts)

def normalize_stacked_fraction(text: Optional[str]) -> Optional[str]:
    """
    y= a
    x 처럼 줄바꿈으로 분자/분모가 분리된 세로 분수 OCR 결과를 LaTeX 분수로 변환한다.
    여러 개의 세로 분수가 함께 있는 경우 ;로 연결한다.
    y=ax처럼 한 줄로 붙어 있는 정비례식은 변환하지 않는다.
    """

    if text is None:
        return None

    expression = re.split(r"\(단|단,", text)[0]
    expression = expression.replace("`", "").strip()

    stacked_fraction_matches = re.findall(
        r"[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]?\s*"
        r"(?:\([0-9]+\))?\s*"
        r"([a-zA-Z])\s*=\s*([+-]?)\s*([0-9a-zA-Z]+)\s*\n\s*([0-9a-zA-Z]+)",
        expression,
    )

    if not stacked_fraction_matches:
        return None

    formulas = []

    for left, sign, numerator, denominator in stacked_fraction_matches:
        formulas.append(rf"{left}={sign}\frac{{{numerator}}}{{{denominator}}}")

    return ";".join(formulas)

def normalize_numbered_formula_parts(text: Optional[str]) -> Optional[str]:
    """
    ⑴, ⑵ 또는 (1), (2)처럼 번호가 붙은 여러 수식을 각각 분리해 정규화한다.
    일반 수식과 세로 분수 수식이 한 블록에 함께 있는 경우를 처리한다.
    """

    if text is None:
        return None

    expression = re.split(r"\(단|단,", text)[0]
    expression = expression.replace("`", "").strip()

    numbered_parts = re.split(
        r"(?:[⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽]|\([0-9]+\))",
        expression,
    )

    numbered_parts = [part.strip() for part in numbered_parts if part.strip()]

    if len(numbered_parts) < 2:
        return None

    formulas = []

    for part in numbered_parts:
        stacked_formula = normalize_stacked_fraction(part)

        if stacked_formula is not None:
            formulas.append(stacked_formula)
            continue

        normalized_part = normalize_latex_candidate(part)

        if normalized_part is not None:
            formulas.append(normalized_part)

    if len(formulas) < 2:
        return None

    return ";".join(formulas)

def normalize_inline_fraction_part(text: str) -> str:
    inline_fraction_match = re.fullmatch(
        r"([a-zA-Z])=([+-]?)([0-9a-zA-Z]+)\/([0-9a-zA-Z]+)",
        text,
    )

    if not inline_fraction_match:
        return text

    left = inline_fraction_match.group(1)
    sign = inline_fraction_match.group(2)
    numerator = inline_fraction_match.group(3)
    denominator = inline_fraction_match.group(4)

    return rf"{left}={sign}\frac{{{numerator}}}{{{denominator}}}"

def convert_latex_to_mathml(latex: Optional[str]) -> Optional[str]:
    """
    간단한 교과서 수식을 MathML 문자열로 변환한다.
    실제 수식 인식 모델 연결 전까지 사용하는 기본 변환기이다.

    지원 예:
    y=ax
    y=4x
    y=-3x
    y=8/x
    y=4x;y=-3x
    """

    if latex is None:
        return None

    formulas = [part for part in latex.split(";") if part]

    if not formulas:
        return None

    mathml_parts = []

    for formula in formulas:
        formula_mathml = convert_single_formula_to_mathml(formula)

        if formula_mathml is None:
            return None

        mathml_parts.append(formula_mathml)

    joined = "<mo>;</mo>".join(mathml_parts)

    return f"<math><mrow>{joined}</mrow></math>"

def convert_single_formula_to_mathml(formula: str) -> Optional[str]:
    """
    단일 수식을 간단한 MathML mrow로 변환한다.
    """

    formula = formula.strip()

    if not formula:
        return None

    # 좌표/순서쌍: (1,a), (-2,-4), (x,y)
    coordinate_match = re.fullmatch(
        r"\(([+-]?[0-9.]+|[a-zA-Z가-힣]),([+-]?[0-9.]+|[a-zA-Z가-힣])\)",
        formula,
    )

    if coordinate_match:
        left = convert_math_token_to_mathml(coordinate_match.group(1))
        right = convert_math_token_to_mathml(coordinate_match.group(2))

        return (
            "<mrow>"
            "<mo>(</mo>"
            f"{left}"
            "<mo>,</mo>"
            f"{right}"
            "<mo>)</mo>"
            "</mrow>"
        )

    # y=\frac{1}{2}x, y=\frac{2}{3}x 형태
    coefficient_fraction_match = re.fullmatch(
        r"([a-zA-Z])=\\frac\{([^{}]+)\}\{([^{}]+)\}([a-zA-Z])",
        formula,
    )

    if coefficient_fraction_match:
        left = coefficient_fraction_match.group(1)
        numerator = coefficient_fraction_match.group(2)
        denominator = coefficient_fraction_match.group(3)
        variable = coefficient_fraction_match.group(4)

        return (
            "<mrow>"
            f"<mi>{left}</mi>"
            "<mo>=</mo>"
            "<mfrac>"
            f"{convert_math_token_to_mathml(numerator)}"
            f"{convert_math_token_to_mathml(denominator)}"
            "</mfrac>"
            f"<mi>{variable}</mi>"
            "</mrow>"
        )

    # y=\frac{8}{x}, y=-\frac{8}{x}, y=\frac{a}{x} 형태
    inverse_fraction_match = re.fullmatch(
        r"([a-zA-Z])=([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}",
        formula,
    )

    if inverse_fraction_match:
        left = inverse_fraction_match.group(1)
        sign = inverse_fraction_match.group(2)
        numerator = inverse_fraction_match.group(3)
        denominator = inverse_fraction_match.group(4)

        sign_mathml = ""

        if sign == "-":
            sign_mathml = "<mo>-</mo>"
        elif sign == "+":
            sign_mathml = "<mo>+</mo>"

        return (
            "<mrow>"
            f"<mi>{left}</mi>"
            "<mo>=</mo>"
            f"{sign_mathml}"
            "<mfrac>"
            f"{convert_math_token_to_mathml(numerator)}"
            f"{convert_math_token_to_mathml(denominator)}"
            "</mfrac>"
            "</mrow>"
        )

    if inverse_fraction_match:
        left = inverse_fraction_match.group(1)
        numerator = inverse_fraction_match.group(2)
        denominator = inverse_fraction_match.group(3)

        return (
            "<mrow>"
            f"<mi>{left}</mi>"
            "<mo>=</mo>"
            "<mfrac>"
            f"{convert_math_token_to_mathml(numerator)}"
            f"{convert_math_token_to_mathml(denominator)}"
            "</mfrac>"
            "</mrow>"
        )
    
    # y=8/x 형태
    fraction_match = re.fullmatch(r"([a-zA-Z])=([+-]?\d+)/([a-zA-Z])", formula)

    if fraction_match:
        left = fraction_match.group(1)
        numerator = fraction_match.group(2)
        denominator = fraction_match.group(3)

        return (
            "<mrow>"
            f"<mi>{left}</mi>"
            "<mo>=</mo>"
            "<mfrac>"
            f"<mn>{numerator}</mn>"
            f"<mi>{denominator}</mi>"
            "</mfrac>"
            "</mrow>"
        )

    # y=ax, y=4x, y=-3x 형태
    linear_match = re.fullmatch(r"([a-zA-Z])=([+-]?\d*|[a-zA-Z])([a-zA-Z])", formula)

    if linear_match:
        left = linear_match.group(1)
        coefficient = linear_match.group(2)
        variable = linear_match.group(3)

        coefficient_mathml = ""

        if coefficient and coefficient not in ["+", "-"]:
            if re.fullmatch(r"[+-]?\d+", coefficient):
                coefficient_mathml = f"<mn>{coefficient}</mn>"
            else:
                coefficient_mathml = f"<mi>{coefficient}</mi>"
        elif coefficient == "-":
            coefficient_mathml = "<mo>-</mo>"

        return (
            "<mrow>"
            f"<mi>{left}</mi>"
            "<mo>=</mo>"
            f"{coefficient_mathml}"
            f"<mi>{variable}</mi>"
            "</mrow>"
        )

    return None

def convert_math_token_to_mathml(token: str) -> str:
    """
    숫자/문자 토큰을 MathML 태그로 변환한다.
    """

    if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token):
        return f"<mn>{token}</mn>"

    return f"<mi>{token}</mi>"

def formula_to_accessible_text(latex: str) -> str:
    """
    접근성 설명에 LaTeX 원문을 그대로 넣지 않고 자연어 표현으로 바꾼다.
    """

    latex_fraction_match = re.fullmatch(
        r"([a-zA-Z])=([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}",
        latex,
    )

    if latex_fraction_match:
        left = latex_fraction_match.group(1)
        sign = latex_fraction_match.group(2)
        numerator = latex_fraction_match.group(3)
        denominator = latex_fraction_match.group(4)

        if sign == "-":
            return f"{left}는 음수 {numerator}를 {denominator}로 나눈 값"

        return f"{left}는 {numerator}를 {denominator}로 나눈 값"

    return latex

def generate_formula_description(latex: Optional[str]) -> Dict[str, str]:
    """
    수식 LaTeX를 바탕으로 접근성 설명을 생성한다.
    스크린리더/점역 참고용으로 사용할 수 있는 기본 설명이다.
    """

    if latex is None:
        return {
            "status": "not_started",
            "short_text": None,
            "long_text": None,
            "transcription_notes": None,
            "review_status": "auto",
        }

    if ";" in latex:
        parts = [part for part in latex.split(";") if part]
        readable_parts = ", ".join(formula_to_accessible_text(part) for part in parts)

        return {
            "status": "generated",
            "short_text": f"여러 개의 수식입니다: {readable_parts}",
            "long_text": f"이 영역에는 {len(parts)}개의 수식이 포함되어 있습니다. {readable_parts}입니다.",
            "transcription_notes": "여러 수식은 세미콜론으로 구분하여 점역하며, 각 수식의 분자와 분모를 구분해 확인합니다.",
            "review_status": "auto",
        }

    if latex.startswith("(") and latex.endswith(")") and "," in latex:
        return {
            "status": "generated",
            "short_text": f"좌표 또는 순서쌍 {latex}입니다.",
            "long_text": f"{latex}는 괄호 안에 두 값을 쉼표로 구분하여 나타낸 좌표 또는 순서쌍입니다.",
            "transcription_notes": "괄호, 쉼표, 각 항을 순서대로 점역합니다.",
            "review_status": "auto",
        }
    
    arithmetic_division_match = re.fullmatch(
        r"([0-9]+)\s*(?:/|\\div|÷)\s*([0-9]+)=([0-9]+)(?:\(([^()]+)\))?",
        latex,
    )

    if arithmetic_division_match:
        dividend = arithmetic_division_match.group(1)
        divisor = arithmetic_division_match.group(2)
        quotient = arithmetic_division_match.group(3)
        unit = arithmetic_division_match.group(4)

        unit_note = f" 괄호 안의 '{unit}'은 결과 단위로 확인합니다." if unit else ""

        return {
            "status": "generated",
            "short_text": f"{dividend}을 {divisor}으로 나누면 {quotient}입니다.",
            "long_text": (
                f"이 수식은 {dividend}을 {divisor}으로 나눈 결과가 "
                f"{quotient}임을 나타냅니다."
            ),
            "transcription_notes": (
                "나눗셈 기호와 등호를 구분하여 점역합니다."
                f"{unit_note}"
            ),
            "review_status": "auto",
        }

    arithmetic_operation_match = re.fullmatch(
        r"([0-9]+)\s*(\+|-|\\times|×|\*)\s*([0-9]+)=([0-9]+)(?:\(([^()]+)\))?",
        latex,
    )

    if arithmetic_operation_match:
        left_number = arithmetic_operation_match.group(1)
        operator = arithmetic_operation_match.group(2)
        right_number = arithmetic_operation_match.group(3)
        result_number = arithmetic_operation_match.group(4)
        unit = arithmetic_operation_match.group(5)

        operator_text_map = {
            "+": "더하면",
            "-": "빼면",
            r"\times": "곱하면",
            "×": "곱하면",
            "*": "곱하면",
        }

        operator_text = operator_text_map.get(operator, "계산하면")
        unit_text = f" {unit}" if unit else ""

        return {
            "status": "generated",
            "short_text": f"{left_number}과 {right_number}을 계산하면 {result_number}{unit_text}입니다.",
            "long_text": (
                f"이 수식은 {left_number}에 {right_number}을 {operator_text} "
                f"{result_number}{unit_text}이 된다는 의미입니다."
            ),
            "transcription_notes": "연산 기호와 등호를 구분하여 점역하고, 괄호 안 단위가 있으면 결과 단위로 함께 확인합니다.",
            "review_status": "auto",
        }

    inequality_match = re.fullmatch(
        r"([a-zA-Z0-9]+)\s*([<>≤≥])\s*([a-zA-Z0-9]+)",
        latex,
    )

    if inequality_match:
        left_value = inequality_match.group(1)
        operator = inequality_match.group(2)
        right_value = inequality_match.group(3)

        operator_text_map = {
            ">": "보다 큽니다",
            "<": "보다 작습니다",
            "≤": "보다 작거나 같습니다",
            "≥": "보다 크거나 같습니다",
        }

        operator_text = operator_text_map.get(operator, "비교 관계입니다")

        return {
            "status": "generated",
            "short_text": f"{left_value}는 {right_value}{operator_text}.",
            "long_text": (
                f"이 부등식은 {left_value}와 {right_value}의 크기 관계를 나타냅니다. "
                f"{left_value}는 {right_value}{operator_text}."
            ),
            "transcription_notes": "부등호 방향이 의미를 결정하므로 점역 전 부등호 방향을 확인합니다.",
            "review_status": "auto",
        }

    latex_fraction_match = re.fullmatch(
        r"([a-zA-Z])=([+-]?)\\frac\{([^{}]+)\}\{([^{}]+)\}",
        latex,
    )

    if latex_fraction_match:
        left = latex_fraction_match.group(1)
        sign = latex_fraction_match.group(2)
        numerator = latex_fraction_match.group(3)
        denominator = latex_fraction_match.group(4)

        if sign == "-":
            relation_text = f"{left}와 {denominator}의 곱이 -{numerator}로 일정한 반비례 관계입니다."
            long_text = (
                f"수식은 {left}가 음수 {numerator}를 {denominator}로 나눈 값과 같다는 의미입니다. "
                f"즉, {denominator}의 값이 커질수록 {left}의 절댓값은 작아지는 반비례 관계를 나타냅니다."
            )
        else:
            relation_text = f"{left}와 {denominator}의 곱이 {numerator}로 일정한 반비례 관계입니다."
            long_text = (
                f"수식은 {left}가 {numerator}를 {denominator}로 나눈 값과 같다는 의미입니다. "
                f"즉, {denominator}의 값이 커질수록 {left}의 값은 작아지는 반비례 관계를 나타냅니다."
            )

        return {
            "status": "generated",
            "short_text": relation_text,
            "long_text": long_text,
            "transcription_notes": "분수 구조는 분자와 분모를 구분하여 점역하며, 반비례 관계임을 함께 설명합니다.",
            "review_status": "auto",
        }
    
    fraction_match = re.fullmatch(r"([a-zA-Z])=([+-]?\d+)/([a-zA-Z])", latex)

    if fraction_match:
        left = fraction_match.group(1)
        numerator = fraction_match.group(2)
        denominator = fraction_match.group(3)

        return {
            "status": "generated",
            "short_text": f"{left}는 {numerator}를 {denominator}로 나눈 값입니다.",
            "long_text": f"수식은 {left}가 {numerator} 나누기 {denominator}와 같다는 의미입니다.",
            "transcription_notes": "분수 구조는 분자와 분모를 구분하여 점역합니다.",
            "review_status": "auto",
        }

    linear_match = re.fullmatch(r"([a-zA-Z])=([+-]?\d*|[a-zA-Z])([a-zA-Z])", latex)

    if linear_match:
        left = linear_match.group(1)
        coefficient = linear_match.group(2)
        variable = linear_match.group(3)

        if coefficient in ["", "+"]:
            coefficient_value = "1"
            short_text = f"{latex}는 {left}와 {variable}가 정비례하는 관계입니다."
            long_text = (
                f"수식 {latex}는 {left}가 {variable}와 같다는 의미입니다. "
                f"{variable}의 값이 1 증가하면 {left}도 1만큼 증가합니다."
            )
        elif coefficient == "-":
            coefficient_value = "-1"
            short_text = f"{latex}는 음의 정비례 관계입니다."
            long_text = (
                f"수식 {latex}는 {left}가 {variable}에 -1을 곱한 값과 같다는 의미입니다. "
                f"{variable}의 값이 증가하면 {left}의 값은 반대 방향으로 변합니다."
            )
        else:
            coefficient_value = coefficient
            short_text = f"{latex}는 {left}와 {variable}가 정비례하는 관계입니다."
            long_text = (
                f"수식 {latex}는 {left}가 {variable}에 {coefficient_value}를 곱한 값과 같다는 의미입니다. "
                f"{variable}의 값이 1 증가하면 {left}는 {coefficient_value}만큼 변합니다."
            )

        return {
            "status": "generated",
            "short_text": short_text,
            "long_text": long_text,
            "transcription_notes": "등호를 기준으로 좌변과 우변을 구분하고, 계수와 변수의 관계를 함께 설명합니다.",
            "review_status": "auto",
        }
    simple_equation_match = re.fullmatch(
        r"(.+)=([^=]+)",
        latex,
    )

    if simple_equation_match and any(char.isdigit() for char in latex):
        left_expression = simple_equation_match.group(1)
        right_expression = simple_equation_match.group(2)

        return {
            "status": "generated",
            "short_text": f"{left_expression}의 값은 {right_expression}입니다.",
            "long_text": (
                f"이 수식은 등호를 기준으로 왼쪽 식 {left_expression}과 "
                f"오른쪽 값 {right_expression}이 같다는 의미입니다."
            ),
            "transcription_notes": "등호를 기준으로 좌변과 우변을 구분하여 점역합니다.",
            "review_status": "auto",
        }