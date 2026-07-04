from typing import Any, Dict, List, Optional


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
    latex = normalize_formula_text(raw_text)

    status = "success" if latex else "partial"
    warnings = []

    if not latex:
        warnings.append("Formula text was not available from Model A output.")

    previous_block_id = get_neighbor_block_id(blocks, block_index - 1)
    next_block_id = get_neighbor_block_id(blocks, block_index + 1)

    return {
        "schema_version": "1.0.0",
        "page_id": page_id,
        "block_id": block.get("block_id"),
        "type": "formula",
        "bbox": block.get("bbox"),
        "crop_path": None,
        "detection": {
            "model": {
                "name": block.get("detector", "model-a"),
                "version": None,
            },
            "confidence": block.get("score"),
        },
        "analysis": {
            "status": status,
            "model": {
                "name": "formula-analysis",
                "version": None,
            },
            "confidence": None,
            "result": {
                "kind": "formula",
                "latex": latex,
                "mathml": None,
                "plain_text": latex,
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