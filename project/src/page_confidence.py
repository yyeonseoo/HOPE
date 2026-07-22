from __future__ import annotations

import math
import re
from statistics import mean
from typing import Any, Mapping, Sequence


ANALYZED_TYPES = {"formula", "table", "figure"}
CONTEXT_TYPES = {"title", "section_title", "paragraph", "caption"}
COVERAGE_TYPES = {"title", "section_title", "paragraph", "caption", "formula", "table", "figure"}

WEIGHTS = {
    "semantic_context_similarity": 0.55,
    "detection": 0.25,
    "coverage": 0.10,
    "warning": 0.10,
}

EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_UNIT_PATTERN = re.compile(
    r"(?:km|cm|mm|m|kg|g|초|분|시간|원|만원|kwh|kcal|mb|번|배|명|개|장|톤|l)",
    re.IGNORECASE,
)

_EMBEDDING_MODEL = None
_EMBEDDING_MODEL_LOAD_FAILED = False


def build_page_confidence(
    page_result: Mapping[str, Any],
    semantic_analyses: Sequence[Mapping[str, Any]],
    page_description: Mapping[str, Any],
) -> dict[str, Any]:
    """
    페이지 전체 설명의 신뢰도를 계산한다.

    기준:
    - figure/formula/table 설명과 주변 paragraph/caption의 문장 의미 유사도
    - 숫자/수식/단위 불일치 penalty
    - layout detector의 탐지 신뢰도
    - page_description에 포함된 블록 비율
    - warnings 안정성
    """

    blocks = list(page_result.get("blocks", []))
    block_by_id = {
        str(block.get("block_id")): block
        for block in blocks
        if block.get("block_id") is not None
    }

    semantic_items = [
        item
        for item in semantic_analyses
        if item.get("type") in ANALYZED_TYPES and item.get("block_id") is not None
    ]

    block_scores = [
        _score_semantic_block(item, block_by_id, blocks)
        for item in semantic_items
    ]

    semantic_context_similarity = _average(
        [score["semantic_context_similarity"] for score in block_scores],
        default=1.0,
    )

    detection_score = _average(
        [score["detection"] for score in block_scores if score["detection"] is not None],
        default=0.5,
    )

    coverage_score = _compute_coverage_score(blocks, page_description)
    warning_score = _compute_warning_score(semantic_analyses, page_description)

    components = {
        "semantic_context_similarity": round(semantic_context_similarity, 3),
        "detection": round(detection_score, 3),
        "coverage": round(coverage_score, 3),
        "warning": round(warning_score, 3),
    }

    raw_score = sum(components[key] * WEIGHTS[key] for key in WEIGHTS)
    score = int(round(raw_score * 100))

    level, label, review_status = _resolve_level(score)

    return {
        "score": score,
        "level": level,
        "label": label,
        "method": "sentence_embedding_with_claim_penalty",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "components": components,
        "weights": WEIGHTS,
        "review_status": review_status,
        "reasons": _build_reasons(components, score),
        "block_scores": block_scores,
    }


def _score_semantic_block(
    item: Mapping[str, Any],
    block_by_id: Mapping[str, Mapping[str, Any]],
    blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    block_id = str(item.get("block_id"))
    block = block_by_id.get(block_id, {})
    block_type = item.get("type") or block.get("type")

    description_text = _description_text(item)
    context_text = _context_text_for_block(item, block, blocks, block_by_id)

    embedding_similarity = _sentence_similarity(description_text, context_text)
    claim_penalty = _claim_mismatch_penalty(description_text, context_text)

    semantic_context_similarity = max(
        0.0,
        embedding_similarity - claim_penalty,
    )

    return {
        "block_id": block_id,
        "type": block_type,
        "semantic_context_similarity": round(semantic_context_similarity, 3),
        "embedding_similarity": round(embedding_similarity, 3),
        "claim_penalty": round(claim_penalty, 3),
        "detection": _detection_score(item, block),
        "has_description": bool(description_text),
        "has_context": bool(context_text),
    }


def _description_text(item: Mapping[str, Any]) -> str:
    description = item.get("description") or {}
    text = description.get("long_text") or description.get("short_text") or ""
    return _normalize_text(text)


def _context_text_for_block(
    item: Mapping[str, Any],
    block: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    block_by_id: Mapping[str, Mapping[str, Any]],
) -> str:
    context = item.get("context") or {}
    context_ids: list[str] = []

    for key in ["caption_block_id", "previous_block_id", "next_block_id"]:
        value = context.get(key)
        if value:
            context_ids.append(str(value))

    for value in context.get("nearby_block_ids") or []:
        if value:
            context_ids.append(str(value))

    context_texts = []

    for context_id in dict.fromkeys(context_ids):
        context_block = block_by_id.get(context_id)
        if context_block and context_block.get("type") in CONTEXT_TYPES:
            text = _normalize_text(context_block.get("text"))
            if text:
                context_texts.append(text)

    if context_texts:
        return " ".join(context_texts)

    return _neighbor_context_text(block, blocks)


def _neighbor_context_text(
    block: Mapping[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    window: int = 2,
) -> str:
    block_id = block.get("block_id")
    ordered_blocks = sorted(blocks, key=lambda value: value.get("reading_order", 0))

    target_index = None

    for index, candidate in enumerate(ordered_blocks):
        if candidate.get("block_id") == block_id:
            target_index = index
            break

    if target_index is None:
        return ""

    start = max(0, target_index - window)
    end = min(len(ordered_blocks), target_index + window + 1)

    texts = []

    for candidate in ordered_blocks[start:end]:
        if candidate.get("block_id") == block_id:
            continue

        if candidate.get("type") not in CONTEXT_TYPES:
            continue

        text = _normalize_text(candidate.get("text"))

        if text:
            texts.append(text)

    return " ".join(texts)


def _sentence_similarity(description_text: str, context_text: str) -> float:
    if not description_text:
        return 0.0

    if not context_text:
        return 0.5

    model = _get_embedding_model()

    if model is None:
        return _fallback_token_similarity(description_text, context_text)

    try:
        embeddings = model.encode([description_text, context_text])
        return _cosine_similarity(embeddings[0], embeddings[1])
    except Exception:
        return _fallback_token_similarity(description_text, context_text)


def _get_embedding_model():
    global _EMBEDDING_MODEL
    global _EMBEDDING_MODEL_LOAD_FAILED

    if _EMBEDDING_MODEL is not None:
        return _EMBEDDING_MODEL

    if _EMBEDDING_MODEL_LOAD_FAILED:
        return None

    try:
        from sentence_transformers import SentenceTransformer

        _EMBEDDING_MODEL = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return _EMBEDDING_MODEL
    except Exception:
        _EMBEDDING_MODEL_LOAD_FAILED = True
        return None


def _cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    dot_product = sum(float(a) * float(b) for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vector_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity = dot_product / (norm_a * norm_b)

    return max(0.0, min(1.0, similarity))


def _fallback_token_similarity(description_text: str, context_text: str) -> float:
    description_tokens = _tokens(description_text)
    context_tokens = _tokens(context_text)

    if not description_tokens:
        return 0.0

    if not context_tokens:
        return 0.5

    overlap = description_tokens & context_tokens

    return min(1.0, len(overlap) / max(len(description_tokens), 1) / 0.35)


def _tokens(text: str) -> set[str]:
    normalized = _normalize_text(text)
    raw_tokens = re.findall(r"[가-힣A-Za-z0-9]+", normalized)

    return {
        token.lower()
        for token in raw_tokens
        if len(token) >= 2
    }


def _claim_mismatch_penalty(description_text: str, context_text: str) -> float:
    """
    문장 임베딩은 전체 의미를 잘 보지만 숫자/수식/단위 오류에는 둔감할 수 있다.
    점역 자료에서는 숫자와 수식 오류가 중요하므로 별도 penalty를 적용한다.
    """

    if not description_text or not context_text:
        return 0.0

    penalty = 0.0

    description_numbers = _numbers(description_text)
    context_numbers = _numbers(context_text)

    if description_numbers and context_numbers:
        if not description_numbers & context_numbers:
            penalty += 0.15

    description_equations = _equations(description_text)
    context_equations = _equations(context_text)

    if description_equations and context_equations:
        if not description_equations & context_equations:
            penalty += 0.20

    description_units = _units(description_text)
    context_units = _units(context_text)

    if description_units and context_units:
        if not description_units & context_units:
            penalty += 0.10

    return min(0.30, penalty)


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"[-+]?\d+(?:\.\d+)?", text))


def _equations(text: str) -> set[str]:
    matches = re.findall(
        r"[A-Za-z]\s*=\s*[-+]?(?:\\frac\{[^{}]+\}\{[^{}]+\}|[A-Za-z0-9/+\-*]+)",
        text,
    )

    return {re.sub(r"\s+", "", match) for match in matches}


def _units(text: str) -> set[str]:
    return {match.group(0).lower() for match in _UNIT_PATTERN.finditer(text)}


def _detection_score(
    item: Mapping[str, Any],
    block: Mapping[str, Any],
) -> float | None:
    detection = item.get("detection") or {}
    confidence = detection.get("confidence")

    if confidence is None:
        confidence = block.get("score")

    if confidence is None:
        return None

    try:
        return max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        return None


def _compute_coverage_score(
    blocks: Sequence[Mapping[str, Any]],
    page_description: Mapping[str, Any],
) -> float:
    candidate_ids = {
        str(block.get("block_id"))
        for block in blocks
        if block.get("type") in COVERAGE_TYPES and block.get("block_id") is not None
    }

    if not candidate_ids:
        return 1.0 if page_description.get("text") else 0.0

    included_ids = {
        str(block_id)
        for block_id in page_description.get("block_ids", [])
    }

    return len(candidate_ids & included_ids) / len(candidate_ids)


def _compute_warning_score(
    semantic_analyses: Sequence[Mapping[str, Any]],
    page_description: Mapping[str, Any],
) -> float:
    warnings: list[str] = []

    for item in semantic_analyses:
        warnings.extend(str(warning) for warning in item.get("warnings", []))

    warnings.extend(str(warning) for warning in page_description.get("warnings", []))

    if not warnings:
        return 1.0

    penalty = 0.0

    for warning in warnings:
        lowered = warning.lower()

        if any(keyword in lowered for keyword in ["failed", "could not", "not available", "omitted"]):
            penalty += 0.20
        elif any(keyword in lowered for keyword in ["unreliable", "unsupported", "needs review"]):
            penalty += 0.15
        elif "fallback" in lowered:
            penalty += 0.08
        else:
            penalty += 0.05

    return max(0.0, 1.0 - penalty)


def _resolve_level(score: int) -> tuple[str, str, str]:
    if score >= 80:
        return "high", "높음", "ok"

    if score >= 60:
        return "medium", "보통", "needs_review"

    return "low", "낮음", "needs_review"


def _build_reasons(components: Mapping[str, float], score: int) -> list[str]:
    reasons = []

    if components["semantic_context_similarity"] >= 0.8:
        reasons.append("주변 문단과 figure/formula/table 설명의 문장 의미가 잘 일치합니다.")
    elif components["semantic_context_similarity"] >= 0.6:
        reasons.append("주변 문단과 분석 설명의 문장 의미가 일부 일치하지만 검수가 필요합니다.")
    else:
        reasons.append("주변 문단과 분석 설명의 의미 유사도가 낮아 원본 대조가 필요합니다.")

    if components["detection"] >= 0.75:
        reasons.append("탐지 신뢰도 평균이 양호합니다.")
    elif components["detection"] >= 0.5:
        reasons.append("탐지 신뢰도 평균이 보통 수준입니다.")
    else:
        reasons.append("탐지 신뢰도가 낮은 블록이 있어 위치 검수가 필요합니다.")

    if components["coverage"] < 0.8:
        reasons.append("전체 페이지 설명에 포함되지 않은 블록이 있어 누락 여부 확인이 필요합니다.")

    if components["warning"] < 0.8:
        reasons.append("fallback, 누락, 검증 관련 warning이 있어 점역 전 확인이 필요합니다.")

    if score >= 80:
        reasons.append("전체 페이지 설명은 비교적 안정적으로 사용할 수 있습니다.")
    elif score >= 60:
        reasons.append("전체 페이지 설명은 사용할 수 있으나 점역교정사의 검수가 권장됩니다.")
    else:
        reasons.append("전체 페이지 설명의 신뢰도가 낮아 원본 이미지와 구조화 결과를 함께 확인해야 합니다.")

    return reasons


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""

    return re.sub(r"\s+", " ", str(value)).strip()


def _average(values: Sequence[float], default: float) -> float:
    if not values:
        return default

    return mean(values)