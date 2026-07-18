from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


CONTEXT_BLOCK_TYPES = {"caption", "paragraph", "formula", "table", "title", "section_title"}
_MAX_CONTEXT_ITEMS = 6
_MAX_CONTEXT_CHARACTERS = 1800
_EQUATION_PATTERN = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9_]*(?:\([^()]{1,20}\))?\s*=\s*(?:\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}|[^\s,.;。!?]{1,40}))"
)
_GENERIC_TOKENS = {
    "그림", "그래프", "이미지", "설명", "관계", "내용", "Figure", "figure",
    "나타낸", "나타내는", "보여", "사용", "대한", "있다", "한다", "된다",
}


def build_figure_context(
    blocks: Sequence[Mapping[str, Any]],
    figure_index: int,
    semantic_analyses: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select nearby textbook context without mixing in the entire page."""
    if not 0 <= figure_index < len(blocks):
        return []
    figure = blocks[figure_index]
    figure_bbox = _bbox(figure.get("bbox"))
    semantic_by_block = {
        str(item.get("block_id")): item
        for item in semantic_analyses or []
        if isinstance(item, Mapping) and item.get("block_id")
    }
    candidates: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if index == figure_index or block.get("type") not in CONTEXT_BLOCK_TYPES:
            continue
        block_id = str(block.get("block_id") or "").strip()
        if not block_id:
            continue
        text = _context_text(block, semantic_by_block.get(block_id))
        if not text:
            continue
        score = _context_score(figure_bbox, _bbox(block.get("bbox")), figure_index, index, str(block.get("type")))
        if score < 0.30:
            continue
        candidates.append({
            "block_id": block_id,
            "type": str(block.get("type")),
            "text": text,
            "score": round(score, 3),
            "relative_position": "before" if index < figure_index else "after",
            "_bbox": _bbox(block.get("bbox")),
        })

    # Textbook pages commonly place two figures side by side and put each
    # explanation directly underneath its own column.  Reading order alone
    # makes the left explanation appear adjacent to the right figure, so drop
    # that cross-column competitor when an aligned explanation exists on the
    # same row.  This is deliberately limited to competing blocks of the same
    # type and row; genuine side captions remain available when there is no
    # aligned alternative.
    candidates = [
        item for item in candidates
        if not _is_cross_column_competitor(figure_bbox, item, candidates)
    ]

    type_priority = {"caption": 0, "formula": 1, "table": 2, "paragraph": 3, "section_title": 4, "title": 5}
    candidates.sort(key=lambda item: (
        -item["score"],
        type_priority.get(item["type"], 9),
        abs(_block_number(item["block_id"]) - figure_index),
    ))
    selected: list[dict[str, Any]] = []
    used_characters = 0
    for item in candidates:
        remaining = _MAX_CONTEXT_CHARACTERS - used_characters
        if remaining <= 0 or len(selected) >= _MAX_CONTEXT_ITEMS:
            break
        text = item["text"][:remaining].strip()
        if not text:
            continue
        selected.append({
            key: value
            for key, value in {**item, "id": f"c{len(selected) + 1}", "text": text}.items()
            if key != "_bbox"
        })
        used_characters += len(text)
    return selected


def augment_caption_with_context(
    caption: str,
    figure_type: str,
    context: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, tuple[str, ...]]:
    """Append context without ever rewriting the image-first caption."""
    base = caption.strip()
    if not base:
        return base, ()
    base_tokens = _meaningful_tokens(base)
    for item in context or []:
        text = str(item.get("text") or "").strip()
        block_id = str(item.get("block_id") or "").strip()
        if not text or not block_id or float(item.get("score") or 0.0) < 0.70:
            continue

        equations = list(dict.fromkeys(_clean_equation(item) for item in _EQUATION_PATTERN.findall(text)))
        if figure_type in {"graph", "mathematical_diagram"} and equations:
            equation = equations[0]
            if equation not in base:
                return f"{base} 주변 설명에서는 {equation}를 다룬다.", (block_id,)

        sentence = _most_relevant_sentence(text, base_tokens)
        if sentence is None:
            continue
        if item.get("type") == "caption":
            supplement = f"주변 캡션은 ‘{sentence}’라고 제시한다."
        else:
            supplement = f"주변 설명에서는 ‘{sentence}’라고 설명한다."
        return f"{base} {supplement}", (block_id,)
    return base, ()


def _most_relevant_sentence(text: str, base_tokens: set[str]) -> str | None:
    best: tuple[int, str] | None = None
    for raw in re.split(r"(?<=[.!?。！？])\s+|\n+", text):
        sentence = " ".join(raw.strip().split()).strip("‘’\" ")
        if not sentence or len(sentence) > 180:
            continue
        overlap = len(base_tokens & _meaningful_tokens(sentence))
        if overlap < 2:
            continue
        if best is None or overlap > best[0]:
            best = (overlap, sentence.rstrip(".!?。！？"))
    return best[1] if best else None


def _meaningful_tokens(text: str) -> set[str]:
    return {
        normalized
        for token in re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?", text)
        if (normalized := _normalize_token(token)) not in _GENERIC_TOKENS and len(normalized) >= 2
    }


def _normalize_token(token: str) -> str:
    if re.fullmatch(r"[가-힣]+", token):
        for suffix in ("에서는", "으로", "에서", "에게", "까지", "부터", "처럼", "보다", "에는", "은", "는", "이", "가", "을", "를", "에", "와", "과", "도"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 2:
                return token[:-len(suffix)]
    return token


def _clean_equation(value: str) -> str:
    return re.sub(r"[은는이가을를의]$", "", value.strip())


def _context_text(block: Mapping[str, Any], semantic: Mapping[str, Any] | None) -> str:
    parts: list[str] = []
    raw = block.get("text")
    if isinstance(raw, str) and raw.strip():
        parts.append(_normalize_stacked_fraction(raw.strip()))
    if semantic:
        result = semantic.get("analysis", {}).get("result") if isinstance(semantic.get("analysis"), Mapping) else None
        if isinstance(result, Mapping):
            if result.get("kind") == "formula":
                for key in ("plain_text", "latex"):
                    value = result.get(key)
                    if isinstance(value, str) and value.strip():
                        parts.append(value.strip())
                        break
            elif result.get("kind") == "table" and isinstance(result.get("cells"), list):
                cell_text = [
                    str(cell.get("text") or "").strip()
                    for cell in result["cells"]
                    if isinstance(cell, Mapping) and str(cell.get("text") or "").strip()
                ]
                if cell_text:
                    parts.append(" | ".join(cell_text[:30]))
        description = semantic.get("description")
        if isinstance(description, Mapping):
            text = description.get("short_text")
            if isinstance(text, str) and text.strip():
                parts.append(text.strip())
    return " ".join(dict.fromkeys(parts))[:900]


def _normalize_stacked_fraction(text: str) -> str:
    """Normalize PDF text where a simple fraction is split across lines."""
    atom = r"[+-]?(?:\d+(?:\.\d+)?|[A-Za-z][A-Za-z0-9_]*)"
    return re.sub(
        rf"([A-Za-z][A-Za-z0-9_]*)\s*=\s*({atom})\s*\n\s*({atom})",
        r"\1=\2/\3",
        text,
    )


def _context_score(
    figure_bbox: tuple[float, float, float, float] | None,
    candidate_bbox: tuple[float, float, float, float] | None,
    figure_index: int,
    candidate_index: int,
    block_type: str,
) -> float:
    index_distance = abs(candidate_index - figure_index)
    score = max(0.0, 0.72 - 0.11 * max(0, index_distance - 1))
    if block_type == "caption":
        score += 0.45
    elif block_type in {"formula", "table"}:
        score += 0.12
    elif block_type in {"title", "section_title"}:
        score += 0.08
    if index_distance == 1:
        score += 0.20
    if figure_bbox and candidate_bbox:
        fx1, fy1, fx2, fy2 = figure_bbox
        cx1, cy1, cx2, cy2 = candidate_bbox
        horizontal_overlap = max(0.0, min(fx2, cx2) - max(fx1, cx1)) / max(1.0, min(fx2 - fx1, cx2 - cx1))
        vertical_gap = max(0.0, max(fy1, cy1) - min(fy2, cy2))
        scale = max(1.0, fy2 - fy1)
        if horizontal_overlap >= 0.35:
            score += 0.18
        score -= min(0.35, 0.10 * vertical_gap / scale)
    return min(1.0, max(0.0, score))


def _is_cross_column_competitor(
    figure_bbox: tuple[float, float, float, float] | None,
    candidate: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
) -> bool:
    candidate_bbox = candidate.get("_bbox")
    if figure_bbox is None or candidate_bbox is None:
        return False
    candidate_overlap = _horizontal_overlap_ratio(figure_bbox, candidate_bbox)
    if candidate_overlap >= 0.15:
        return False
    candidate_center_y = (candidate_bbox[1] + candidate_bbox[3]) / 2
    candidate_height = max(1.0, candidate_bbox[3] - candidate_bbox[1])
    for other in candidates:
        if other is candidate or other.get("type") != candidate.get("type"):
            continue
        other_bbox = other.get("_bbox")
        if other_bbox is None or _horizontal_overlap_ratio(figure_bbox, other_bbox) < 0.35:
            continue
        other_center_y = (other_bbox[1] + other_bbox[3]) / 2
        other_height = max(1.0, other_bbox[3] - other_bbox[1])
        if abs(candidate_center_y - other_center_y) <= max(candidate_height, other_height) * 0.65:
            return True
    return False


def _horizontal_overlap_ratio(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    overlap = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    return overlap / max(1.0, min(first[2] - first[0], second[2] - second[0]))


def _bbox(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        return None
    return tuple(float(item) for item in value)


def _block_number(block_id: str) -> int:
    match = re.search(r"_b(\d+)$", block_id)
    return int(match.group(1)) if match else math.inf
