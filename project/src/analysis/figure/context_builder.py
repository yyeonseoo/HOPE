from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_CAPTION_MAX_DISTANCE = 2
_ROLE_HINT_MAX_DISTANCE = 3
_DEFAULT_WINDOW_SIZE = 1


@dataclass(frozen=True)
class ContextSource:
    """Block ids behind every non-null `FigureContext` field, for
    explainability and debugging: given a generated description, this
    answers "which blocks was that built from?"."""

    title_block_id: str | None = None
    section_block_id: str | None = None
    subsection_block_id: str | None = None
    caption_block_id: str | None = None
    nearby_formula_block_id: str | None = None
    nearby_table_block_id: str | None = None
    previous_block_ids: tuple[str, ...] = ()
    next_block_ids: tuple[str, ...] = ()
    role_hint_block_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "title_block_id": self.title_block_id,
            "section_block_id": self.section_block_id,
            "subsection_block_id": self.subsection_block_id,
            "caption_block_id": self.caption_block_id,
            "nearby_formula_block_id": self.nearby_formula_block_id,
            "nearby_table_block_id": self.nearby_table_block_id,
            "previous_block_ids": list(self.previous_block_ids),
            "next_block_ids": list(self.next_block_ids),
            "role_hint_block_id": self.role_hint_block_id,
        }


@dataclass(frozen=True)
class FigureContext:
    """Textbook context surrounding one figure block, used to steer an
    educational (not merely visual) description.

    Every text field is raw block text, or ``None``/``()`` when nothing
    qualifying was found nearby -- callers must not invent a value for a
    missing field. ``context_source`` records which block each field came
    from, so downstream grounding/review can trace a claim back to its
    source.
    """

    figure_block_id: str | None = None
    page_id: int | None = None
    page_number: int | None = None

    # Title hierarchy. `page_title` is kept for backward compatibility with
    # the single-level lookup this shipped with first; `chapter_title` is
    # its hierarchy-aware equivalent (nearest 'title' block, not necessarily
    # the page's first). In the common case (one title per page) they match.
    page_title: str | None = None
    chapter_title: str | None = None
    section_title: str | None = None
    subsection_title: str | None = None
    # Kept for backward compatibility: the single nearest heading-class
    # block, same value `subsection_title` gets when one exists, else
    # whatever `section_title` resolves to.
    nearest_section_title: str | None = None

    previous_paragraph: str | None = None
    next_paragraph: str | None = None
    previous_paragraphs: tuple[str, ...] = ()
    next_paragraphs: tuple[str, ...] = ()

    caption: str | None = None
    nearby_formula: str | None = None
    nearby_table: str | None = None
    figure_ocr: tuple[str, ...] = field(default_factory=tuple)
    role_hint: str | None = None

    context_source: ContextSource = field(default_factory=ContextSource)

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_title": self.page_title,
            "chapter_title": self.chapter_title,
            "section_title": self.section_title,
            "subsection_title": self.subsection_title,
            "nearest_section_title": self.nearest_section_title,
            "previous_paragraph": self.previous_paragraph,
            "next_paragraph": self.next_paragraph,
            "previous_paragraphs": list(self.previous_paragraphs),
            "next_paragraphs": list(self.next_paragraphs),
            "caption": self.caption,
            "nearby_formula": self.nearby_formula,
            "nearby_table": self.nearby_table,
            "figure_ocr": list(self.figure_ocr),
            "page_number": self.page_number,
            "role_hint": self.role_hint,
        }

    def has_any_text(self) -> bool:
        return any(
            [
                self.page_title,
                self.nearest_section_title,
                self.previous_paragraph,
                self.next_paragraph,
                self.caption,
                self.nearby_formula,
                self.nearby_table,
                self.figure_ocr,
            ]
        )


class FigureContextBuilder:
    """Locate the figure among its page blocks and pull out nearby textbook
    context: title hierarchy, surrounding paragraphs (optionally windowed),
    caption, nearby formula/table, the figure's own OCR text, and the
    pedagogical role (role_hint) of the block/section it sits in."""

    def build(
        self,
        page_blocks: Sequence[Mapping[str, Any]],
        figure_block: Mapping[str, Any],
        *,
        page_id: int | None = None,
        ocr_lines: Sequence[Mapping[str, Any]] | None = None,
        window_size: int = _DEFAULT_WINDOW_SIZE,
    ) -> FigureContext:
        window_size = max(1, window_size)
        ordered = _reading_order(page_blocks)
        figure_position = _locate(ordered, figure_block)
        figure_ocr = tuple(_figure_ocr_texts(ocr_lines, figure_block.get("bbox")))

        if figure_position is None:
            page_title_block = _first_of_type(ordered, "title")
            role_hint, role_hint_block_id = _role_hint(figure_block, ordered, None)
            return FigureContext(
                figure_block_id=_block_id(figure_block),
                page_id=page_id,
                page_number=page_id,
                page_title=_text_of(page_title_block),
                figure_ocr=figure_ocr,
                role_hint=role_hint,
                context_source=ContextSource(
                    title_block_id=_id_of(page_title_block),
                    role_hint_block_id=role_hint_block_id,
                ),
            )

        page_title_block = _first_of_type(ordered, "title")
        chapter_block = _nearest_either(ordered, figure_position, "title") or page_title_block
        heading_stack = _heading_stack(ordered, figure_position)
        if not heading_stack:
            fallback = _nearest_after(ordered, figure_position, "section_title")
            heading_stack = [fallback] if fallback else []
        subsection_block = heading_stack[0] if heading_stack else None
        section_block = heading_stack[1] if len(heading_stack) >= 2 else None
        nearest_section_block = subsection_block

        previous_paragraph_blocks = _collect_before(ordered, figure_position, "paragraph", window_size)
        next_paragraph_blocks = _collect_after(ordered, figure_position, "paragraph", window_size)
        caption_block = _nearest_caption(ordered, figure_position, _CAPTION_MAX_DISTANCE)
        nearby_formula_block = _nearest_either(ordered, figure_position, "formula")
        nearby_table_block = _nearest_either(ordered, figure_position, "table")
        role_hint, role_hint_block_id = _role_hint(figure_block, ordered, figure_position)

        return FigureContext(
            figure_block_id=_block_id(figure_block),
            page_id=page_id,
            page_number=page_id,
            page_title=_text_of(page_title_block),
            chapter_title=_text_of(chapter_block),
            section_title=_text_of(section_block),
            subsection_title=_text_of(subsection_block),
            nearest_section_title=_text_of(nearest_section_block),
            previous_paragraph=_text_of(previous_paragraph_blocks[-1]) if previous_paragraph_blocks else None,
            next_paragraph=_text_of(next_paragraph_blocks[0]) if next_paragraph_blocks else None,
            previous_paragraphs=tuple(_text_of(block) for block in previous_paragraph_blocks),
            next_paragraphs=tuple(_text_of(block) for block in next_paragraph_blocks),
            caption=_text_of(caption_block),
            nearby_formula=_text_of(nearby_formula_block),
            nearby_table=_text_of(nearby_table_block),
            figure_ocr=figure_ocr,
            role_hint=role_hint,
            context_source=ContextSource(
                title_block_id=_id_of(chapter_block),
                section_block_id=_id_of(section_block),
                subsection_block_id=_id_of(subsection_block),
                caption_block_id=_id_of(caption_block),
                nearby_formula_block_id=_id_of(nearby_formula_block),
                nearby_table_block_id=_id_of(nearby_table_block),
                previous_block_ids=tuple(_id_of(block) for block in previous_paragraph_blocks if _id_of(block)),
                next_block_ids=tuple(_id_of(block) for block in next_paragraph_blocks if _id_of(block)),
                role_hint_block_id=role_hint_block_id,
            ),
        )


def _reading_order(page_blocks: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return blocks in reading order.

    `page["blocks"]` is already reading-order sorted by the time it reaches
    the analysis layer (see reading_order.py), so this is a defensive
    re-sort: a no-op when list order already matches `reading_order`, but
    correct if a caller ever hands over an unsorted list. Blocks without a
    `reading_order` value keep their original relative position.
    """
    indexed = list(enumerate(page_blocks))
    if all(block.get("reading_order") is not None for _, block in indexed):
        indexed.sort(key=lambda item: item[1]["reading_order"])
    return [block for _, block in indexed]


def _locate(ordered: Sequence[Mapping[str, Any]], figure_block: Mapping[str, Any]) -> int | None:
    figure_id = figure_block.get("block_id")
    if figure_id is not None:
        for index, block in enumerate(ordered):
            if block.get("block_id") == figure_id:
                return index
    for index, block in enumerate(ordered):
        if block is figure_block:
            return index
    return None


def _block_id(block: Mapping[str, Any]) -> str | None:
    value = block.get("block_id")
    return str(value) if value is not None else None


def _id_of(block: Mapping[str, Any] | None) -> str | None:
    return _block_id(block) if block is not None else None


def _block_text(block: Mapping[str, Any]) -> str | None:
    text = block.get("text")
    if not isinstance(text, str):
        return None
    cleaned = text.strip()
    return cleaned or None


def _text_of(block: Mapping[str, Any] | None) -> str | None:
    return _block_text(block) if block is not None else None


def _first_of_type(ordered: Sequence[Mapping[str, Any]], block_type: str) -> Mapping[str, Any] | None:
    for block in ordered:
        if block.get("type") == block_type and _block_text(block):
            return block
    return None


def _nearest_before(
    ordered: Sequence[Mapping[str, Any]], position: int, block_type: str
) -> Mapping[str, Any] | None:
    for index in range(position - 1, -1, -1):
        if ordered[index].get("type") == block_type and _block_text(ordered[index]):
            return ordered[index]
    return None


def _nearest_after(
    ordered: Sequence[Mapping[str, Any]], position: int, block_type: str
) -> Mapping[str, Any] | None:
    for index in range(position + 1, len(ordered)):
        if ordered[index].get("type") == block_type and _block_text(ordered[index]):
            return ordered[index]
    return None


def _nearest_either(
    ordered: Sequence[Mapping[str, Any]], position: int, block_type: str
) -> Mapping[str, Any] | None:
    before = _nearest_before(ordered, position, block_type)
    after = _nearest_after(ordered, position, block_type)
    if before is None:
        return after
    if after is None:
        return before
    before_distance = position - ordered.index(before)
    after_distance = ordered.index(after) - position
    return before if before_distance <= after_distance else after


def _collect_before(
    ordered: Sequence[Mapping[str, Any]], position: int, block_type: str, count: int
) -> list[Mapping[str, Any]]:
    """Nearest `count` matching blocks before `position`, returned oldest-first
    (natural reading order, ending with the one closest to the figure)."""
    found: list[Mapping[str, Any]] = []
    for index in range(position - 1, -1, -1):
        if ordered[index].get("type") == block_type and _block_text(ordered[index]):
            found.append(ordered[index])
            if len(found) >= count:
                break
    return list(reversed(found))


def _collect_after(
    ordered: Sequence[Mapping[str, Any]], position: int, block_type: str, count: int
) -> list[Mapping[str, Any]]:
    """Nearest `count` matching blocks after `position`, nearest-first (the
    order reading naturally continues away from the figure)."""
    found: list[Mapping[str, Any]] = []
    for index in range(position + 1, len(ordered)):
        if ordered[index].get("type") == block_type and _block_text(ordered[index]):
            found.append(ordered[index])
            if len(found) >= count:
                break
    return found


def _heading_stack(ordered: Sequence[Mapping[str, Any]], position: int) -> list[Mapping[str, Any]]:
    """Every 'section_title'-type block between the figure and the nearest
    preceding chapter-level 'title' block, nearest-first.

    The layout model has only one heading-class type ("section_title") below
    the page/chapter title, so a real section vs. subsection distinction
    only exists when two or more such headings stack up above the figure
    without an intervening chapter title -- the nearest is treated as the
    subsection, the next one out as the section. With zero or one heading
    found, that finer distinction genuinely isn't determinable and is left
    `None` rather than guessed (see `subsection_title`/`section_title`
    assignment in `build`).
    """
    stack: list[Mapping[str, Any]] = []
    for index in range(position - 1, -1, -1):
        block_type = ordered[index].get("type")
        if block_type == "title":
            break
        if block_type == "section_title" and _block_text(ordered[index]):
            stack.append(ordered[index])
    return stack


def _nearest_caption(
    ordered: Sequence[Mapping[str, Any]], position: int, max_distance: int
) -> Mapping[str, Any] | None:
    """Nearest caption within `max_distance`, allowing intervening blocks
    only if they are all paragraphs (a figure with one descriptive paragraph
    squeezed between it and its caption still connects to that caption; a
    formula/table/another figure in between signals a real boundary)."""
    for distance in range(1, max_distance + 1):
        for index in (position - distance, position + distance):
            if not 0 <= index < len(ordered):
                continue
            block = ordered[index]
            if block.get("type") != "caption" or not _block_text(block):
                continue
            between = ordered[min(index, position) + 1 : max(index, position)]
            if all(item.get("type") == "paragraph" for item in between):
                return block
    return None


def _role_hint(
    figure_block: Mapping[str, Any],
    ordered: Sequence[Mapping[str, Any]],
    position: int | None,
) -> tuple[str | None, str | None]:
    """Pedagogical role (e.g. example/problem/solution) governing the figure,
    read from block metadata rather than inferred from text -- see
    layout_detection.py's role_hint tagging. Checked on the figure block
    itself first, then the nearest surrounding blocks that carry one."""
    own_role = _role_hint_of(figure_block)
    if own_role:
        return own_role, _block_id(figure_block)
    if position is None:
        return None, None
    for distance in range(1, _ROLE_HINT_MAX_DISTANCE + 1):
        for index in (position - distance, position + distance):
            if 0 <= index < len(ordered):
                role = _role_hint_of(ordered[index])
                if role:
                    return role, _id_of(ordered[index])
    return None, None


def _role_hint_of(block: Mapping[str, Any]) -> str | None:
    context = block.get("context")
    if not isinstance(context, Mapping):
        return None
    role = context.get("role_hint")
    return str(role) if role else None


def _figure_ocr_texts(
    ocr_lines: Sequence[Mapping[str, Any]] | None, bbox: Any
) -> list[str]:
    """Text (axis names, legend, labels, ...) whose center lies inside the figure."""
    if not ocr_lines or not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return []
    x1, y1, x2, y2 = bbox
    texts: list[str] = []
    for line in ocr_lines:
        line_bbox = line.get("bbox")
        text = str(line.get("text") or "").strip()
        if not text or not isinstance(line_bbox, (list, tuple)) or len(line_bbox) != 4:
            continue
        lx1, ly1, lx2, ly2 = line_bbox
        center_x, center_y = (lx1 + lx2) / 2, (ly1 + ly2) / 2
        if x1 <= center_x <= x2 and y1 <= center_y <= y2 and text not in texts:
            texts.append(text)
    return texts
