import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .formula_recognizer import recognize_formula_from_crop

def analyze_formula_blocks(
    page: Dict[str, Any],
    page_image_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Model A의 page 결과에서 formula 블록만 골라
    Model B/C 통합용 semantic analysis 결과 형식으로 변환한다.
    """

    page_id = page.get("page_id")
    blocks = page.get("blocks", [])

    results = []

    for index, block in enumerate(blocks):
        if block.get("type") != "formula":
            continue

        result = analyze_single_formula_block(
            page_id=page_id,
            block=block,
            blocks=blocks,
            block_index=index,
            page_image_path=page_image_path,
        )
        results.append(result)

    return results


def analyze_single_formula_block(
    page_id: int,
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    page_image_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    formula 블록 하나를 공통 결과 형식으로 변환한다.

    현재 단계에서는 Model A가 제공한 text 값을 latex 후보로 사용한다.
    이후 실제 수식 OCR/LaTeX 복원 모델을 붙이면
    latex를 만드는 부분만 교체하면 된다.
    """

    raw_text = block.get("text")
    formula_info = parse_formula_text(raw_text)
    plain_text = formula_info["plain_text"]

    previous_block_id = get_neighbor_block_id(blocks, block_index - 1)
    next_block_id = get_neighbor_block_id(blocks, block_index + 1)

    crop_path = crop_formula_block(
        page_image_path=page_image_path,
        block=block,
        page_id=page_id,
    )

    recognition = recognize_formula_from_crop(
        crop_path=crop_path,
        fallback_text=plain_text,
    )

    latex = recognition.get("latex")
    mathml = recognition.get("mathml")
    plain_text = recognition.get("plain_text")
    analysis_confidence = recognition.get("confidence")
    analysis_model = recognition.get(
        "model",
        {
            "name": "formula-analysis",
            "version": None,
        },
    )

    status = "success" if latex else "partial"
    warnings = recognition.get("warnings", [])

    if not latex and "Formula text was not available from Model A output." not in warnings:
        warnings.append("Formula text was not available from Model A output.")

    return {
        "schema_version": "1.0.0",
        "page_id": page_id,
        "block_id": block.get("block_id"),
        "type": "formula",
        "bbox": block.get("bbox"),
        "crop_path": crop_path,
        "detection": {
            "model": {
                "name": block.get("detector", "model-a"),
                "version": None,
            },
            "confidence": block.get("score"),
        },
        "analysis": {
            "status": status,
            "model": analysis_model,
            "confidence": analysis_confidence,
            "result": {
                "kind": "formula",
                "latex": latex,
                "mathml": mathml,
                "plain_text": plain_text,
            },
        },
        "context": {
            "previous_block_id": previous_block_id,
            "next_block_id": next_block_id,
            "caption_block_id": None,
            "nearby_block_ids": [
                block_id
                for block_id in [previous_block_id, next_block_id]
                if block_id is not None
            ],
        },
        "warnings": warnings,
    }

def parse_formula_text(text: Optional[str]) -> Dict[str, Optional[str]]:
    """
    Model A가 준 formula text에서 실제 수식 부분과 설명 문장을 분리한다.

    예:
    "y=ax (단, a는 0이 아니다.)"
    -> latex: "y=ax"
    -> plain_text: "y=ax (단, a는 0이 아니다.)"
    """

    if text is None:
        return {
            "latex": None,
            "plain_text": None,
        }

    plain_text = normalize_plain_text(text)

    if not plain_text:
        return {
            "latex": None,
            "plain_text": None,
        }

    formula_part = plain_text

    for separator in ["(단", "（단", "( 단", "단,"]:
        if separator in formula_part:
            formula_part = formula_part.split(separator)[0]
            break

    latex = normalize_formula_text(formula_part)

    return {
        "latex": latex,
        "plain_text": plain_text,
    }


def normalize_plain_text(text: Optional[str]) -> Optional[str]:
    """
    OCR 결과에 섞인 특수 공백과 줄바꿈을 일반 텍스트로 정리한다.
    """

    if text is None:
        return None

    cleaned = str(text)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        return None

    return cleaned

def crop_formula_block(
    page_image_path: Optional[str],
    block: Dict[str, Any],
    page_id: Optional[int],
) -> Optional[str]:
    """
    Model A가 제공한 bbox를 이용해 페이지 이미지에서 formula 영역을 crop한다.

    page_image_path가 없으면 crop을 만들 수 없으므로 None을 반환한다.
    """

    if not page_image_path:
        return None

    bbox = block.get("bbox")
    block_id = block.get("block_id")

    if not bbox or len(bbox) != 4 or not block_id:
        return None

    image_path = Path(page_image_path)

    if not image_path.exists():
        return None

    try:
        from PIL import Image
    except ImportError:
        return None

    x1, y1, x2, y2 = [int(value) for value in bbox]

    if x2 <= x1 or y2 <= y1:
        return None

    output_dir = Path("outputs") / "crops" / "formula"
    output_dir.mkdir(parents=True, exist_ok=True)

    safe_page_id = page_id if page_id is not None else "unknown"
    output_path = output_dir / f"p{safe_page_id}_{block_id}.png"

    with Image.open(image_path) as image:
        width, height = image.size

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        cropped = image.crop((x1, y1, x2, y2))
        cropped.save(output_path)

    return str(output_path).replace("\\", "/")

def normalize_formula_text(text: Optional[str]) -> Optional[str]:
    """
    Model A OCR 결과로 들어온 수식 텍스트를 최소한으로 정리한다.
    """

    if text is None:
        return None

    cleaned = str(text).strip()

    if not cleaned:
        return None

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

    return cleaned


def get_neighbor_block_id(
    blocks: List[Dict[str, Any]],
    index: int,
) -> Optional[str]:
    """
    현재 formula 블록의 앞/뒤 블록 ID를 가져온다.
    범위를 벗어나면 None을 반환한다.
    """

    if index < 0 or index >= len(blocks):
        return None

    return blocks[index].get("block_id")