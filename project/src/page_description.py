"""Assemble a whole-page accessibility description from block-level results.

The deterministic reading-order draft (see `_build_draft`) is always available
and requires no model. An optional `PageDescriptionGenerator` (any object
with a `generate_page_description` method) can rewrite that draft into a more
natural narrative; its output is verified against the draft afterward so it
can't introduce facts, numbers, or equations that weren't already there.
No current captioner implements this -- see backend/app.py's `build_page_description`
call site.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

_TEXT_ONLY_TYPES = {"title", "section_title", "paragraph", "caption", "footer", "page_number"}
_ANALYZED_TYPES = {"formula", "table", "figure"}
_RAW_TEXT_FALLBACK_TYPES = {"formula", "table"}
_SECTION_LABEL_TYPES = {"footer", "page_number"}
_DECORATIVE_TITLE_TYPES = {"title", "section_title"}
# The chapter's running roman-numeral marker (already surfaced once via the
# footer/page_number header) sometimes gets mistagged as its own title block
# floating mid-page -- e.g. a lone "Ⅲ" with no other text.
_BARE_ROMAN_NUMERAL_PATTERN = re.compile(r"^[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ]$")

_SENTENCE_SPLIT_PATTERN = re.compile(r".+?(?:[.!?。！？]+|$)", re.DOTALL)
_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9가-힣])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9])")
_EQUATION_PATTERN = re.compile(r"[A-Za-z]\s*=\s*[-+]?[A-Za-z0-9\\/^().+\-* ]{1,24}")
_EDITORIAL_SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:-{3,}|(?:참고|결론|요약|정리|마무리)\s*[:：])"
)
# A verified rewrite should stay close to the draft's length; a much longer
# result is a sign the model padded or rambled beyond what was asked.
_MAX_LENGTH_RATIO = 1.8


@runtime_checkable
class PageDescriptionGenerator(Protocol):
    def generate_page_description(self, draft_text: str) -> "GenerationResultLike": ...


@runtime_checkable
class GenerationResultLike(Protocol):
    text: str
    confidence: float | None
    generation_time_seconds: float
    model_name: str
    model_version: str | None
    warnings: Sequence[str]


def _nullable_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_block_text(
    block: Mapping[str, Any], analyses_by_id: Mapping[str, Mapping[str, Any]]
) -> tuple[str | None, str | None]:
    """Return (text, skip_reason); skip_reason is only set when text is None."""
    block_type = block.get("type")
    block_id = block.get("block_id")

    if block_type not in _ANALYZED_TYPES:
        return _nullable_text(block.get("text")), None

    record = analyses_by_id.get(str(block_id))
    description = (record or {}).get("description") or {}
    text = _nullable_text(description.get("long_text")) or _nullable_text(description.get("short_text"))
    if text:
        return text, None

    if block_type in _RAW_TEXT_FALLBACK_TYPES:
        raw_text = _nullable_text(block.get("text"))
        if raw_text:
            return raw_text, None

    return None, f"{block_type} block {block_id!r} has no usable description or text; omitted from page description."


def _collapse_whitespace(text: str) -> str:
    """Join OCR line-wraps within a block into flowing text.

    A paragraph block's text is built by stacking separately-detected OCR
    lines (see ocr.py's lines_text_inside_bbox), so it often contains
    mid-word/mid-sentence newlines that have nothing to do with real
    sentence breaks. Collapsing them to spaces is pure formatting -- no
    words are added, removed, or reordered.
    """
    return re.sub(r"\s+", " ", text).strip()


def _extract_section_label(text: str) -> str | None:
    """Strip a bare page number from footer/page_number text, keeping only a
    real running header if one remains.

    The page number can sit on either side depending on the textbook's
    left/right-page layout -- "118 Ⅲ. 좌표평면과 그래프" on one page,
    "1. 좌표평면과 그래프 119" on the next (that leading "1." is a section
    number, not a page number, so it must not be stripped). Only a digit run
    with whitespace on its outer side is treated as the page number; a
    footer that's nothing but a number returns None so the caller can drop
    the block entirely instead of surfacing noise.
    """
    if re.fullmatch(r"\d+", text.strip()):
        return None
    stripped = re.sub(r"^\d+\s+", "", text)
    stripped = re.sub(r"\s+\d+$", "", stripped).strip()
    return stripped or None


def _build_draft(
    page_result: Mapping[str, Any], semantic_analyses: Sequence[Mapping[str, Any]]
) -> tuple[str, list[str], list[str]]:
    """Return (draft_text, block_ids, warnings), reading_order-sorted.

    Each contributing block is rendered as its own "[type] text" line, in
    reading order, so the source of every sentence is explicit rather than
    blended into anonymous prose. footer/page_number blocks are handled
    separately: a bare page number is dropped as noise, but a chapter label
    surviving after the number is stripped is surfaced once as a page-level
    header instead of an inline "[footer]" line.
    """
    analyses_by_id = {
        str(item.get("block_id")): item for item in semantic_analyses if item.get("block_id") is not None
    }
    blocks = sorted(page_result.get("blocks", []), key=lambda b: b.get("reading_order", 0))

    section_labels: list[str] = []
    seen_labels: set[str] = set()
    lines: list[str] = []
    block_ids: list[str] = []
    warnings: list[str] = []
    for block in blocks:
        block_type = block.get("type")
        text, skip_reason = _resolve_block_text(block, analyses_by_id)
        if not text:
            if skip_reason:
                warnings.append(skip_reason)
            continue
        text = _collapse_whitespace(text)
        if not text:
            continue

        block_id = block.get("block_id")

        if block_type in _SECTION_LABEL_TYPES:
            label = _extract_section_label(text)
            if label and label not in seen_labels:
                seen_labels.add(label)
                section_labels.append(label)
                if block_id is not None:
                    block_ids.append(str(block_id))
            continue

        if block_type in _DECORATIVE_TITLE_TYPES and _BARE_ROMAN_NUMERAL_PATTERN.match(text):
            continue

        lines.append(f"[{block_type}] {text}")
        if block_id is not None:
            block_ids.append(str(block_id))

    header = "\n".join(section_labels)
    body = "\n".join(lines)
    draft = "\n\n".join(part for part in (header, body) if part).strip()
    return draft, block_ids, warnings


def _split_sentences(text: str) -> list[str]:
    return [part.strip() for part in _SENTENCE_SPLIT_PATTERN.findall(text) if part.strip()]


def _extract_claims(text: str) -> set[str]:
    claims = {match.group(0) for match in _NUMBER_PATTERN.finditer(text)}
    claims |= {re.sub(r"\s+", "", match.group(0)) for match in _EQUATION_PATTERN.finditer(text)}
    return claims


def _verify_and_strip_unsupported_claims(generated_text: str, draft_text: str) -> tuple[str, list[str]]:
    """Drop generated sentences containing a number/equation absent from draft_text."""
    draft_claims = _extract_claims(draft_text)
    draft_compact = re.sub(r"\s+", "", draft_text)

    kept: list[str] = []
    warnings: list[str] = []
    for sentence in _split_sentences(generated_text):
        unsupported = {
            claim
            for claim in _extract_claims(sentence)
            if claim not in draft_claims and re.sub(r"\s+", "", claim) not in draft_compact
        }
        if unsupported:
            warnings.append(
                "Dropped a generated sentence containing unsupported claim(s) not present in the "
                f"source text: {sorted(unsupported)}."
            )
            continue
        kept.append(sentence)

    return " ".join(kept).strip(), warnings


def _strip_trailing_editorial_section(text: str) -> tuple[str, bool]:
    """Cut off a '참고:'/'결론:'/'---' section the model appended despite being
    told not to -- these tend to introduce ungrounded generalizations that
    aren't tied to any specific claim the number/equation check can catch."""
    match = _EDITORIAL_SECTION_PATTERN.search(text)
    if not match:
        return text, False
    return text[: match.start()].strip(), True


def build_page_description(
    page_result: Mapping[str, Any],
    semantic_analyses: Sequence[Mapping[str, Any]],
    *,
    generator: PageDescriptionGenerator | None = None,
    max_draft_chars: int = 4000,
) -> dict[str, Any]:
    draft_text, block_ids, draft_warnings = _build_draft(page_result, semantic_analyses)
    page_id = page_result.get("page_id")

    if not draft_text:
        return {
            "page_id": page_id,
            "status": "failed",
            "text": None,
            "draft_text": None,
            "was_generated": False,
            "model": None,
            "confidence": None,
            "generation_time_seconds": None,
            "block_ids": [],
            "review_status": "unreviewed",
            "warnings": draft_warnings or ["Page had no blocks with usable text."],
        }

    result: dict[str, Any] = {
        "page_id": page_id,
        "status": "partial" if draft_warnings else "success",
        "text": draft_text,
        "draft_text": draft_text,
        "was_generated": False,
        "model": None,
        "confidence": None,
        "generation_time_seconds": None,
        "block_ids": block_ids,
        "review_status": "unreviewed",
        "warnings": list(draft_warnings),
    }

    if generator is None:
        return result

    if len(draft_text) > max_draft_chars:
        result["warnings"].append(
            f"Draft text ({len(draft_text)} chars) exceeded max_draft_chars ({max_draft_chars}); "
            "generation was skipped."
        )
        return result

    try:
        generated = generator.generate_page_description(draft_text)
    except Exception as exc:  # noqa: BLE001 -- any generator failure must fall back, not crash the page
        result["warnings"].append(f"Page description generation failed and was skipped: {exc}")
        return result

    generated_text, stripped_editorial_section = _strip_trailing_editorial_section(generated.text)
    verified_text, verification_warnings = _verify_and_strip_unsupported_claims(generated_text, draft_text)
    if stripped_editorial_section:
        verification_warnings.append(
            "Removed a trailing '참고'/'결론'-style section the model added despite being told not to."
        )

    if verified_text and len(verified_text) > _MAX_LENGTH_RATIO * len(draft_text):
        verification_warnings.append(
            f"Generated text ({len(verified_text)} chars) was more than {_MAX_LENGTH_RATIO}x the draft "
            f"({len(draft_text)} chars), suggesting padding or rambling; used the deterministic draft instead."
        )
        verified_text = ""

    if not verified_text:
        result["warnings"].append(
            "Generated description failed grounding verification entirely; used the deterministic draft."
        )
        result["warnings"].extend(verification_warnings)
        result["warnings"].extend(generated.warnings)
        if verification_warnings:
            result["review_status"] = "needs_review"
        return result

    result.update(
        text=verified_text,
        was_generated=True,
        model={"name": generated.model_name, "version": generated.model_version},
        confidence=generated.confidence,
        generation_time_seconds=generated.generation_time_seconds,
    )
    result["warnings"].extend(generated.warnings)
    result["warnings"].extend(verification_warnings)
    if verification_warnings:
        result["review_status"] = "needs_review"
        if result["status"] == "success":
            result["status"] = "partial"

    return result
