"""Braille-oriented structure fixes applied after layout and reading order.

Three rules from the braille guideline audit (reports/HOPE 점역 지침 대조 평가.md, 5.1):
- figure captions sit inside neighbouring paragraphs; pull them out as
  caption blocks so the "caption, then graphic" order can apply;
- a figure must not split a run of numbered questions (BF2016 §6.2.3);
- the printed page line ("1. 좌표평면과 그래프 127") is a footer, not body text.

Block ids assigned by build_page_result are kept; new caption blocks get
"p{page}_cap{n}" so evaluations keyed on block ids stay comparable.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Mapping, Sequence

_QUESTION_MARKER = re.compile(r"^\s*(?:[⑴-⒇]|\(\d{1,2}\))")
# Zero-width and control characters from the PDF text layer (U+200C, U+0007 after "⑴")
# show up as stray marks in the review UI and as stray braille cells.
_INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\u200b-\u200f\u2060\ufeff]")
_FIGURE_LABEL = re.compile(r"^\[?\s*(?:그림|표|그래프)\s*\d+")
_PAGE_LINE = re.compile(r"^\d{1,4}\s+\S.{0,38}$|^.{1,38}\S\s+\d{1,4}$")
_CAPTION_MAX_CHARS = 30
_MIN_COLUMN_CHARS = 8
# ponytail: caption search band below the figure is a fixed 0.25 inch; tune
# per textbook if captions sit further away or above figures.
_CAPTION_GAP_INCH = 0.25


def apply_braille_structure(
    page: Dict[str, Any],
    ocr_lines: Sequence[Mapping[str, Any]] | None,
    dpi: int,
    read_region: Callable[[Sequence[float]], str] | None = None,
) -> Dict[str, Any]:
    blocks = sorted(page.get("blocks", []), key=lambda b: b.get("reading_order", 0))
    for block in blocks:
        if isinstance(block.get("text"), str):
            block["text"] = _INVISIBLE.sub("", block["text"])
    _mark_page_line_as_footer(blocks)
    blocks = _split_two_column_blocks(blocks, ocr_lines or [])
    blocks = _extract_figure_captions(blocks, ocr_lines or [], page.get("page_id"), dpi)
    if read_region is not None:
        blocks = _badge_titles(blocks, page.get("page_id"), dpi, read_region)
    # Print order before any braille move, so a moved figure can get the
    # "originally here" transcriber's note (NLD2022 3장 1절 1. 7).
    for order, block in enumerate(blocks, start=1):
        block["original_reading_order"] = order
    blocks = _keep_figures_out_of_question_runs(blocks)
    blocks = _place_sidebars_after_their_unit(blocks)
    for block in blocks:
        if block.get("type") in {"paragraph", "caption"} and "\n" in str(block.get("text") or ""):
            block["text"] = _rejoin_wrapped_lines(str(block["text"]))
    for order, block in enumerate(blocks, start=1):
        block["reading_order"] = order
    page["blocks"] = blocks
    return page


def _split_two_column_blocks(
    blocks: List[Dict[str, Any]],
    ocr_lines: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Undo paragraphs that merged two side-by-side columns line by line
    (e.g. 사고 과정 | 풀이 과정). Equal columns read left then right
    (NLD2022 2장 2절 5.), so a run of such blocks becomes all left parts,
    then all right parts."""
    result: List[Dict[str, Any]] = []
    pending_right: List[Dict[str, Any]] = []
    for block in blocks:
        parts = _column_parts(block, blocks, ocr_lines) if block.get("type") == "paragraph" else None
        if parts is None:
            result.extend(pending_right)
            pending_right = []
            result.append(block)
            continue
        left, right = parts
        result.append(left)
        pending_right.append(right)
    result.extend(pending_right)
    return result


def _column_parts(
    block: Dict[str, Any],
    blocks: Sequence[Mapping[str, Any]],
    ocr_lines: Sequence[Mapping[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]] | None:
    box = _bbox(block)
    if not box:
        return None
    area = (box[2] - box[0]) * (box[3] - box[1])
    lines = []
    for line in ocr_lines:
        lb = line.get("bbox")
        if not isinstance(lb, (list, tuple)) or len(lb) != 4 or not str(line.get("text") or "").strip():
            continue
        if _block_at([block], lb) is None:
            continue
        # A line inside a smaller block (formula, answer) belongs to that block.
        owner = [b for b in blocks if b is not block and _bbox(b) and _block_at([b], lb) is not None]
        if any((_bbox(b)[2] - _bbox(b)[0]) * (_bbox(b)[3] - _bbox(b)[1]) < area for b in owner):
            continue
        lines.append(line)
    if len(lines) < 2:
        return None
    lines.sort(key=lambda l: l["bbox"][0])
    for cut in range(1, len(lines)):
        left, right = lines[:cut], lines[cut:]
        if max(l["bbox"][2] for l in left) >= min(l["bbox"][0] for l in right):
            continue
        # Side by side, not stacked: the columns must share vertical space.
        if _overlap_y(left, right) <= 0:
            return None
        # A problem number ("2") or a word split by an inline icon ("을") is
        # not a column.
        if min(_char_count(left), _char_count(right)) < _MIN_COLUMN_CHARS:
            return None
        return _column_block(block, left, block["block_id"]), _column_block(block, right, f"{block['block_id']}_r")
    return None


def _char_count(lines: Sequence[Mapping[str, Any]]) -> int:
    return sum(len(re.sub(r"\s", "", str(l.get("text") or ""))) for l in lines)


def _overlap_y(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> float:
    top = max(min(l["bbox"][1] for l in left), min(l["bbox"][1] for l in right))
    bottom = min(max(l["bbox"][3] for l in left), max(l["bbox"][3] for l in right))
    return bottom - top


def _column_block(source: Mapping[str, Any], lines: Sequence[Mapping[str, Any]], block_id: str) -> Dict[str, Any]:
    lines = sorted(lines, key=lambda l: l["bbox"][1])
    return {
        **source,
        "block_id": block_id,
        "bbox": [
            min(l["bbox"][0] for l in lines), min(l["bbox"][1] for l in lines),
            max(l["bbox"][2] for l in lines), max(l["bbox"][3] for l in lines),
        ],
        "text": "\n".join(str(l["text"]).strip() for l in lines),
    }


def _extract_figure_captions(
    blocks: List[Dict[str, Any]],
    ocr_lines: Sequence[Mapping[str, Any]],
    page_id: Any,
    dpi: int,
) -> List[Dict[str, Any]]:
    figures = [b for b in blocks if b.get("type") == "figure" and _bbox(b)]
    gap = _CAPTION_GAP_INCH * dpi
    captions: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for line in ocr_lines:
        line_box = line.get("bbox")
        text = str(line.get("text") or "").strip()
        if not text or not isinstance(line_box, (list, tuple)) or len(line_box) != 4:
            continue
        # A source line ("출처: …") goes last in the figure group
        # (TG2022 §5.2: caption, note, graphic, source).
        if text.startswith("출처"):
            above = [
                f for f in figures
                if _bbox(f)[3] - gap / 5 <= line_box[1] <= _bbox(f)[3] + 2 * gap and _overlap_x(_bbox(f), line_box) > 0
            ]
            if above:
                figure = max(above, key=lambda f: _bbox(f)[3])
                sources[figure["block_id"]] = {"text": text, "line_box": list(line_box)}
            continue
        below = [
            f for f in figures
            if _bbox(f)[3] - gap / 5 <= line_box[1] <= _bbox(f)[3] + gap
            and _overlap_x(_bbox(f), line_box) > 0
            and f["block_id"] not in captions
        ]
        if not below:
            continue
        # An axis label of the next figure, a footer or a heading is not a caption.
        container = _block_at(blocks, line_box)
        if container is not None and container.get("type") in {"figure", "footer", "page_number", "title", "section_title"}:
            continue
        container_text = re.sub(r"\s+", " ", str((container or {}).get("text") or "")).strip()
        below.sort(key=lambda f: _bbox(f)[0])
        segments = re.split(r"\s{3,}", text)
        if len(segments) != len(below):
            if len(segments) != 1:
                continue
            below = [max(below, key=lambda f: _overlap_x(_bbox(f), line_box))]
        for figure, segment in zip(below, segments):
            if len(segment) > _CAPTION_MAX_CHARS or _QUESTION_MARKER.match(segment):
                continue
            # A caption is a complete unit, not the first line of a wrapped
            # paragraph or sidebar ("대관람차는 바퀴 모양 / 의 둘레에 ...").
            # ponytail: captions without a period (e.g. "그래프 ①") are missed.
            if not (_FIGURE_LABEL.match(segment) or segment.endswith(("다.", ".")) or container_text == segment):
                continue
            captions[figure["block_id"]] = {"text": segment, "line_box": list(line_box)}

    if not captions and not sources:
        return blocks
    result: List[Dict[str, Any]] = []
    for index, block in enumerate(blocks):
        figure_id = block.get("block_id")
        if block.get("type") != "figure" or (figure_id not in captions and figure_id not in sources):
            result.append(block)
            continue
        result.append(block)
        source = sources.get(figure_id)
        if source:
            _remove_from_source_block(blocks, source)
        if figure_id not in captions:
            if source:
                result.append(_source_block(page_id, figure_id, source))
            continue
        found = captions[figure_id]
        _remove_from_source_block(blocks, found)
        caption_id = f"p{page_id}_cap{len([b for b in result if b.get('type') == 'caption']) + 1}"
        # Explicit link: with two figures in a row, "nearest caption" would
        # hand the second figure the first figure's caption.
        block["caption_block_id"] = caption_id
        result.append({
            "block_id": caption_id,
            "type": "caption",
            "bbox": found["line_box"],
            "text": found["text"],
            "score": None,
            "detector": "braille_caption_rule",
        })
        if source:
            result.append(_source_block(page_id, figure_id, source))
    kept = [b for b in result if not b.get("_emptied")]
    for block in kept:
        block.pop("_emptied", None)
    return kept


def _source_block(page_id: Any, figure_id: str, source: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "block_id": f"{figure_id}_src",
        "type": "paragraph",
        "role": "figure_source",
        "bbox": source["line_box"],
        "text": source["text"],
        "score": None,
        "detector": "braille_caption_rule",
    }


def _remove_from_source_block(blocks: Sequence[Dict[str, Any]], found: Mapping[str, Any]) -> None:
    x1, y1, x2, y2 = found["line_box"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    for block in blocks:
        box = _bbox(block)
        if block.get("type") in {"figure", "caption"} or not box:
            continue
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3] and found["text"] in str(block.get("text") or ""):
            block["text"] = str(block["text"]).replace(found["text"], "", 1).strip()
            block["_emptied"] = not block["text"]
            return


def _keep_figures_out_of_question_runs(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def is_question(block: Mapping[str, Any]) -> bool:
        return block.get("type") != "figure" and bool(_QUESTION_MARKER.match(str(block.get("text") or "")))

    index = 1
    while index < len(blocks) - 1:
        block = blocks[index]
        group_end = index + 1
        while group_end < len(blocks) and blocks[group_end].get("type") == "caption":
            group_end += 1
        if block.get("type") == "figure" and is_question(blocks[index - 1]) and group_end < len(blocks) and is_question(blocks[group_end]):
            start = index - 1
            while start > 0 and is_question(blocks[start - 1]):
                start -= 1
            group = blocks[index:group_end]
            del blocks[index:group_end]
            blocks[start:start] = group
            index = group_end
        else:
            index += 1
    return blocks


# Activity badges are image text in the margin ("함께 풀기 4", "6 문제 해결").
# Braille keeps the type word in parentheses before the item (NLD2022 3장 3절).
# ponytail: dictionary of this textbook's badges; extend it per textbook.
_BADGES = {
    "함께풀기": ("함께 풀기", "example"),
    "문제해결": ("문제 해결", "problem"),
    "의사소통": ("의사소통", "problem"),
    "개념짚기": ("개념 짚기", None),
    "생각열기": ("생각 열기", None),
    "활동하기": ("활동하기", None),
    "확인하기": ("확인하기", "problem"),
}


def _badge_titles(
    blocks: List[Dict[str, Any]],
    page_id: Any,
    dpi: int,
    read_region: Callable[[Sequence[float]], str],
) -> List[Dict[str, Any]]:
    paragraphs = [b for b in blocks if b.get("type") == "paragraph" and _bbox(b)]
    if not paragraphs:
        return blocks
    lefts = sorted(_bbox(b)[0] for b in paragraphs)
    body_left = lefts[len(lefts) // 2]
    result: List[Dict[str, Any]] = []
    seen: set[str] = set()
    removed: set[int] = set()
    for block in blocks:
        box = _bbox(block)
        # Badge sits in the margin left of the body column, beside the first
        # line of an item (whose box may already start left, at "6").
        if block.get("type") == "paragraph" and box and box[0] <= body_left + 0.35 * dpi:
            strip = [body_left - 0.95 * dpi, box[1] - 0.15 * dpi, body_left - 2, box[1] + 0.65 * dpi]
            key = _badge_key(read_region(strip))
            if key and key not in seen:
                seen.add(key)
                label, role = _BADGES[key]
                # The badge image itself was often detected as a tiny figure.
                removed.update(
                    id(b) for b in blocks
                    if b.get("type") == "figure" and _bbox(b)
                    and _bbox(b)[2] - _bbox(b)[0] < dpi and _intersects(_bbox(b), strip)
                )
                result.append({
                    "block_id": f"p{page_id}_badge{len(seen)}",
                    "type": "section_title",
                    "bbox": strip,
                    "text": f"({label})",
                    "score": None,
                    "detector": "braille_badge_rule",
                    "context": {"role_hint": role} if role else {},
                })
        result.append(block)
    return [b for b in result if id(b) not in removed]


def _badge_key(text: str) -> str | None:
    compact = re.sub(r"[\s\d]", "", text or "")
    return next((key for key in _BADGES if key in compact), None)


def _intersects(a: Sequence[float], b: Sequence[float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


_ITEM_MARKER = re.compile(r"^\s*(?:[⑴-⒇❶-❿]|\(\d{1,2}\))")


def _place_sidebars_after_their_unit(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """A margin note (날개) beside a problem is read after that problem's
    stem, questions and figures, not wedged between them (BF2016 §12.3.1).
    ponytail: margin = narrow paragraph left of the body text column; a
    right-hand margin or a full sidebar box is not detected."""
    paragraphs = [b for b in blocks if b.get("type") == "paragraph" and _bbox(b)]
    if len(paragraphs) < 3:
        return blocks
    lefts = sorted(_bbox(b)[0] for b in paragraphs)
    body_left = lefts[len(lefts) // 2]
    span = max(_bbox(b)[2] for b in paragraphs) - min(lefts)
    sidebars = [
        b for b in paragraphs
        if _bbox(b)[2] < body_left - 5 and (_bbox(b)[2] - _bbox(b)[0]) < span * 0.3
    ]
    for sidebar in sidebars:
        sidebar["role"] = "sidebar"
        blocks.remove(sidebar)
        top, bottom = _bbox(sidebar)[1], _bbox(sidebar)[3]
        beside = [
            i for i, b in enumerate(blocks)
            if b.get("type") == "paragraph" and _bbox(b) and _bbox(b)[1] < bottom and _bbox(b)[3] > top
        ]
        index = (beside[0] + 1) if beside else len(blocks)
        # Skip the rest of the unit: numbered questions ("⑴ … 구하시오."),
        # figures and their captions. Worked solutions also start with "⑴"
        # but are not questions, so the margin note lands before them.
        while index < len(blocks) and (
            blocks[index].get("type") in {"figure", "caption"}
            or (
                _ITEM_MARKER.match(str(blocks[index].get("text") or ""))
                and "시오" in str(blocks[index].get("text") or "")
            )
        ):
            index += 1
        blocks.insert(index, sidebar)
    return blocks


def _mark_page_line_as_footer(blocks: List[Dict[str, Any]]) -> None:
    text_blocks = [b for b in blocks if b.get("type") in {"paragraph", "footer", "page_number"} and _bbox(b)]
    if not text_blocks:
        return
    lowest = max(text_blocks, key=lambda b: _bbox(b)[1])
    if lowest.get("type") == "paragraph" and _PAGE_LINE.match(str(lowest.get("text") or "").strip()):
        lowest["type"] = "footer"


_kiwi: Any = None
_HANGUL = re.compile(r"[가-힣]")


def _rejoin_wrapped_lines(text: str) -> str:
    """Join OCR line wraps. Korean textbooks wrap mid-word ("만\\n드시오"),
    so each wrap is judged by Kiwi's spacing model on just the two words
    around it; spacing elsewhere (and formulas like "y=ax") is untouched.
    Without kiwipiepy every wrap becomes a space, as before."""
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    if not lines:
        return ""
    joined = lines[0]
    for line in lines[1:]:
        left, right = joined.split(" ")[-1], line.split(" ")[0]
        glue = " "
        if _HANGUL.search(left[-1:]) and _HANGUL.search(right[:1]):
            kiwi = _load_kiwi()
            if kiwi is not None and kiwi.space(left + right, reset_whitespace=False) == left + right:
                glue = ""
        joined += glue + line
    return joined


def _load_kiwi() -> Any:
    global _kiwi
    if _kiwi is None:
        try:
            from kiwipiepy import Kiwi
        except ImportError:
            _kiwi = False
        else:
            _kiwi = Kiwi()
    return _kiwi or None


def _block_at(blocks: Sequence[Mapping[str, Any]], box: Sequence[float]) -> Mapping[str, Any] | None:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    # Figures first: a line inside a figure is figure text even if a
    # paragraph box also covers it.
    for block in sorted(blocks, key=lambda b: b.get("type") != "figure"):
        b = _bbox(block)
        if b and b[0] <= cx <= b[2] and b[1] <= cy <= b[3]:
            return block
    return None


def _bbox(block: Mapping[str, Any]) -> List[float] | None:
    box = block.get("bbox")
    return list(box) if isinstance(box, (list, tuple)) and len(box) == 4 else None


def _overlap_x(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
