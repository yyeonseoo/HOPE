from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_MODEL_CACHE: Dict[str, object] = {}

TARGET_CLASSES = [
    "title",
    "section_title",
    "paragraph",
    "formula",
    "table",
    "figure",
    "caption",
    "footer",
    "page_number",
]

ROLE_TYPES = {
    "example_box": "example",
    "problem_box": "problem",
    "solution_box": "solution",
}


def detect_layout(
    image_path: str | Path,
    ocr_lines: Optional[List[Dict]] = None,
    yolo_model_path: Optional[str | Path] = None,
    use_supplements: bool = True,
) -> List[Dict]:
    if yolo_model_path:
        try:
            raw_model_blocks = _detect_with_yolo(
                image_path,
                yolo_model_path,
                recover_low_conf_figures=use_supplements,
            )
            if not use_supplements:
                return raw_model_blocks

            model_blocks = _postprocess_blocks(raw_model_blocks, None, None)
            heuristic_blocks = _detect_with_heuristics(image_path, ocr_lines or [])
            supplemented = _supplement_model_blocks(model_blocks, heuristic_blocks)
            supplemented = _supplement_uncovered_ocr_text(image_path, ocr_lines or [], supplemented)
            return _merge_and_filter(supplemented)
        except Exception as exc:
            if str(yolo_model_path).startswith("hf:"):
                raise RuntimeError(f"Layout model inference failed: {exc}") from exc

    return _detect_with_heuristics(image_path, ocr_lines or [])


def _supplement_model_blocks(model_blocks: List[Dict], heuristic_blocks: List[Dict]) -> List[Dict]:
    supplemented = list(model_blocks)
    useful_supplement_types = {"formula", "caption", "footer", "page_number", "section_title"}
    for block in heuristic_blocks:
        if block["type"] not in useful_supplement_types:
            continue
        if _is_overlapping_any(block["bbox"], supplemented, threshold=0.45):
            continue
        block = dict(block)
        block["detector"] = block.get("detector", "heuristic_supplement")
        block["score"] = min(float(block.get("score", 0.35)), 0.50)
        supplemented.append(block)
    return supplemented


def _is_overlapping_any(bbox: List[int], blocks: List[Dict], threshold: float) -> bool:
    return any(_intersection_over_area(bbox, block["bbox"]) >= threshold for block in blocks)


def _supplement_uncovered_ocr_text(image_path: str | Path, ocr_lines: List[Dict], blocks: List[Dict]) -> List[Dict]:
    if not ocr_lines:
        return blocks

    image = cv2.imread(str(image_path))
    if image is None:
        return blocks
    height, width = image.shape[:2]

    result = list(blocks)
    result = _supplement_role_title_regions(ocr_lines, result, width, height, image)

    supplemental = _classify_text_regions(ocr_lines, width, height, result)
    for block in supplemental:
        if block["type"] not in {"paragraph", "formula", "caption", "section_title"}:
            continue
        if _is_noise_text(block.get("text", "")):
            continue
        if _is_overlapping_any(block["bbox"], result, threshold=0.35):
            continue
        block = dict(block)
        block["detector"] = "ocr_text_supplement"
        block["score"] = min(float(block.get("score", 0.35)), 0.50)
        result.append(block)
    return _merge_ocr_supplement_paragraphs(result)


def _supplement_role_title_regions(
    ocr_lines: List[Dict], blocks: List[Dict], page_width: int, page_height: int, image: Optional[np.ndarray] = None
) -> List[Dict]:
    result = list(blocks)
    role_titles = []
    for block in blocks:
        if block["type"] not in {"title", "section_title"}:
            continue
        if _area(block["bbox"]) >= page_width * page_height * 0.01:
            continue
        role_hint = _role_hint_for_title_block(block, ocr_lines)
        if role_hint in {"example", "solution", "problem"}:
            title_block = dict(block)
            title_block["_role_hint"] = role_hint
            role_titles.append(title_block)

    for title in role_titles:
        title_bbox = title["bbox"]
        role_hint = title["_role_hint"]
        next_y = min(
            [
                block["bbox"][1]
                for block in blocks
                if block is not title
                and block["bbox"][1] > title_bbox[3] + 20
                and block["type"] in {"paragraph", "figure", "table", "footer"}
            ]
            or [page_height],
        )
        y1 = max(0, title_bbox[1] - 32)
        y2 = min(page_height, next_y - 6, title_bbox[3] + 175)

        region_lines = []
        for line in ocr_lines:
            lx1, ly1, lx2, ly2 = line["bbox"]
            if ly2 < y1 or ly1 > y2:
                continue
            if lx2 < max(0, title_bbox[0] - 30) or lx1 > min(page_width, title_bbox[2] + page_width * 0.78):
                continue
            is_title_line = _intersection_over_area(line["bbox"], title_bbox) >= 0.15
            if _is_noise_text(line.get("text", "")) and not is_title_line and not _is_math_fragment(line.get("text", "")):
                continue
            region_lines.append(line)

        if len(region_lines) < 2 and not (
            role_hint == "solution" and region_lines and len(region_lines[0].get("text", "").replace(" ", "")) >= 20
        ):
            continue

        bbox = [
            min(line["bbox"][0] for line in region_lines),
            min(line["bbox"][1] for line in region_lines),
            max(line["bbox"][2] for line in region_lines),
            max(line["bbox"][3] for line in region_lines),
        ]
        bbox = _expand_role_bbox_with_visual_lines(image, bbox, title_bbox, y1, y2)
        if _is_overlapping_any(bbox, [block for block in result if block["type"] == "paragraph"], threshold=0.70):
            continue

        text = "\n".join(line.get("text", "") for line in _sort_ocr_lines_for_text(region_lines))
        block_type = "formula" if _looks_like_role_formula_region(text) else "paragraph"
        result.append(
            {
                "type": block_type,
                "bbox": bbox,
                "text": text,
                "score": 0.72,
                "detector": "role_region_supplement",
                "context": {"role_hint": role_hint},
            }
        )
    return result


def _expand_role_bbox_with_visual_lines(
    image: Optional[np.ndarray], bbox: List[int], title_bbox: List[int], y1: int, y2: int
) -> List[int]:
    if image is None:
        return bbox

    height, width = image.shape[:2]
    scan_y1 = max(0, y1 - 16)
    scan_y2 = min(height, y2 + 16)
    if scan_y2 <= scan_y1:
        return bbox

    gray = cv2.cvtColor(image[scan_y1:scan_y2], cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (35, 1))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    row_counts = np.count_nonzero(horizontal, axis=1)
    candidate_rows = np.where(row_counts > max(40, width * 0.20))[0]
    if candidate_rows.size == 0:
        return bbox

    x_values: List[int] = []
    y_values: List[int] = []
    for row in candidate_rows:
        absolute_y = scan_y1 + int(row)
        if absolute_y < title_bbox[1] - 45 or absolute_y > y2 + 8:
            continue
        xs = np.where(horizontal[row] > 0)[0]
        if xs.size == 0:
            continue
        if xs.max() < title_bbox[0] or xs.min() > bbox[2] + 80:
            continue
        x_values.extend([int(xs.min()), int(xs.max())])
        y_values.append(absolute_y)

    if not x_values:
        return bbox

    line_x1 = min(x_values)
    line_x2 = max(x_values)
    expanded_width = line_x2 - line_x1
    if expanded_width < (bbox[2] - bbox[0]) * 0.8:
        return bbox

    return [
        min(bbox[0], line_x1),
        min(bbox[1], min(y_values)),
        max(bbox[2], line_x2),
        max(bbox[3], max(y_values)),
    ]


def _sort_ocr_lines_for_text(lines: List[Dict]) -> List[Dict]:
    rows: List[List[Dict]] = []
    for line in sorted(lines, key=lambda item: ((item["bbox"][1] + item["bbox"][3]) / 2, item["bbox"][0])):
        cy = (line["bbox"][1] + line["bbox"][3]) / 2
        if rows:
            row_cy = sum((item["bbox"][1] + item["bbox"][3]) / 2 for item in rows[-1]) / len(rows[-1])
            if abs(cy - row_cy) <= 18:
                rows[-1].append(line)
                continue
        rows.append([line])

    ordered: List[Dict] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda item: item["bbox"][0]))
    return ordered


def _is_math_fragment(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if not compact:
        return False
    return any(char.isdigit() for char in compact) and any(char in compact for char in "0123456789=+-–×÷/%().")


def _looks_like_role_formula_region(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "".join(lines)
    content = compact.replace("풀이", "").replace("해설", "").replace("정답", "")
    if not content:
        return False
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in content)
    math_chars = sum(char in "=+-−–×÷/%()[]{}^∆Δ√∑" for char in content)
    digit_count = sum(char.isdigit() for char in content)
    formula_lines = sum(_looks_like_formula(line) or _is_math_fragment(line) for line in lines)
    sentence_like = korean_count >= 8 or any(marker in content for marker in ["이고", "이며", "한다", "된다", "구하여라"])
    return not sentence_like and (formula_lines >= 3 or (math_chars >= 5 and digit_count >= 2))


def _role_hint_for_title_block(block: Dict, ocr_lines: List[Dict]) -> str:
    context_role = (block.get("context") or {}).get("role_hint")
    if context_role in {"example", "solution", "problem"}:
        return context_role

    texts = []
    x1, y1, x2, y2 = block["bbox"]
    expanded = [x1 - 18, y1 - 18, x2 + 55, y2 + 18]
    for line in ocr_lines:
        if _intersection_over_area(line["bbox"], expanded) > 0:
            texts.append(line.get("text", ""))
    inferred = _infer_role("\n".join(texts) or block.get("text", ""))
    return inferred


def _merge_ocr_supplement_paragraphs(blocks: List[Dict]) -> List[Dict]:
    merged: List[Dict] = []
    buffer: List[Dict] = []

    def flush_buffer():
        if not buffer:
            return
        if len(buffer) == 1:
            merged.append(buffer[0])
        else:
            x1 = min(item["bbox"][0] for item in buffer)
            y1 = min(item["bbox"][1] for item in buffer)
            x2 = max(item["bbox"][2] for item in buffer)
            y2 = max(item["bbox"][3] for item in buffer)
            text = "\n".join(item.get("text", "") for item in buffer).strip()
            merged.append(
                {
                    "type": "paragraph",
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "score": 0.50,
                    "detector": "ocr_text_supplement",
                }
            )
        buffer.clear()

    for block in sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        is_supplement_paragraph = block.get("detector") == "ocr_text_supplement" and block["type"] == "paragraph"
        if not is_supplement_paragraph:
            flush_buffer()
            merged.append(block)
            continue
        if not buffer:
            buffer.append(block)
            continue
        prev = buffer[-1]
        gap = block["bbox"][1] - prev["bbox"][3]
        horizontal_overlap = min(block["bbox"][2], prev["bbox"][2]) - max(block["bbox"][0], prev["bbox"][0])
        min_width = max(1, min(block["bbox"][2] - block["bbox"][0], prev["bbox"][2] - prev["bbox"][0]))
        same_text_region = horizontal_overlap / min_width >= 0.35 or abs(block["bbox"][0] - prev["bbox"][0]) < 110
        if gap < 42 and same_text_region:
            buffer.append(block)
        else:
            flush_buffer()
            buffer.append(block)
    flush_buffer()
    return merged


def _is_noise_text(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if len(compact) < 4:
        return True
    informative = sum(char.isalnum() or "\uac00" <= char <= "\ud7a3" for char in compact)
    return informative / max(len(compact), 1) < 0.45


def _expand_paragraphs_with_nearby_ocr_lines(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    if not ocr_lines:
        return blocks

    result: List[Dict] = []
    used_line_ids = set()
    for block in sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        if block["type"] != "paragraph":
            result.append(block)
            continue
        if _starts_with_numbered_item(block.get("text", "")):
            result.append(block)
            continue

        bx1, by1, bx2, by2 = block["bbox"]
        nearby = []
        for index, line in enumerate(ocr_lines):
            if index in used_line_ids:
                continue
            lx1, ly1, lx2, ly2 = line["bbox"]
            gap_above = by1 - ly2
            gap_below = ly1 - by2
            is_above = ly2 <= by1 + 6 and -6 <= gap_above <= 38
            is_below = ly1 >= by2 - 6 and -6 <= gap_below <= 38
            if not (is_above or is_below):
                continue
            horizontal_overlap = min(bx2, lx2) - max(bx1, lx1)
            line_width = max(1, lx2 - lx1)
            same_text_flow = horizontal_overlap / line_width >= 0.25 or abs(lx1 - bx1) <= 90
            if not same_text_flow:
                continue
            if _has_closer_paragraph_for_line(line["bbox"], block, blocks):
                continue
            text = line.get("text", "")
            if _is_noise_text(text) or _looks_like_axis_or_legend_text(text):
                continue
            recoverable_paragraph = _looks_like_recoverable_paragraph_line(text)
            paragraph_continuation = _looks_like_paragraph_continuation(text)
            if is_below and not (recoverable_paragraph or paragraph_continuation):
                continue
            if any(
                other is not block
                and other["type"] == "paragraph"
                and _inside(line["bbox"], other["bbox"])
                for other in blocks
            ):
                continue
            if any(other["type"] == "paragraph" and _inside(line["bbox"], other["bbox"]) for other in result):
                continue
            overlaps_structural_block = _is_overlapping_any(
                line["bbox"],
                [other for other in blocks if other is not block and other["type"] in {"figure", "formula", "table"}],
                threshold=0.20,
            )
            if overlaps_structural_block and not (recoverable_paragraph or paragraph_continuation):
                continue
            nearby.append((index, line))

        if not nearby:
            result.append(block)
            continue

        lines = [line for _, line in sorted(nearby, key=lambda item: (item[1]["bbox"][1], item[1]["bbox"][0]))]
        for index, _ in nearby:
            used_line_ids.add(index)
        merged_lines = lines + [{"bbox": block["bbox"], "text": block.get("text", "")}]
        merged_lines.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))
        block = dict(block)
        block["bbox"] = [
            min(item["bbox"][0] for item in merged_lines),
            min(item["bbox"][1] for item in merged_lines),
            max(item["bbox"][2] for item in merged_lines),
            max(item["bbox"][3] for item in merged_lines),
        ]
        block["text"] = "\n".join(item.get("text", "").strip() for item in merged_lines if item.get("text", "").strip())
        result.append(block)
    return result


def _has_closer_paragraph_for_line(line_bbox: List[int], current: Dict, blocks: List[Dict]) -> bool:
    line_width = max(1, line_bbox[2] - line_bbox[0])

    def vertical_gap(bbox: List[int]) -> int:
        if line_bbox[3] < bbox[1]:
            return bbox[1] - line_bbox[3]
        if line_bbox[1] > bbox[3]:
            return line_bbox[1] - bbox[3]
        return 0

    current_gap = vertical_gap(current["bbox"])
    for other in blocks:
        if other is current or other["type"] != "paragraph":
            continue
        ox1, _, ox2, _ = other["bbox"]
        horizontal_overlap = min(line_bbox[2], ox2) - max(line_bbox[0], ox1)
        same_text_flow = horizontal_overlap / line_width >= 0.25 or abs(line_bbox[0] - ox1) <= 90
        if same_text_flow and vertical_gap(other["bbox"]) + 2 < current_gap:
            return True
    return False


def _looks_like_axis_or_legend_text(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if len(compact) <= 2 and any(char.isdigit() for char in compact):
        return True
    return compact.startswith(("y=", "x=", "+y=", "-y=")) and len(compact) <= 18


def _starts_with_numbered_item(text: str) -> bool:
    stripped = text.strip()
    return bool(re.match(r"^(?:[\(（]\s*\d{1,2}\s*[\)）]|[①-⑳⑴-⒇])", stripped))


def _supplement_missing_paragraph_lines(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    if not ocr_lines:
        return blocks

    result = list(blocks)
    for line in ocr_lines:
        text = line.get("text", "").strip()
        if _is_noise_text(text) or _looks_like_axis_or_legend_text(text):
            continue
        if _covered_by_existing_relaxed(line["bbox"], result):
            continue
        if not _looks_like_recoverable_paragraph_line(text):
            continue
        result.append(
            {
                "type": "paragraph",
                "bbox": line["bbox"],
                "text": text,
                "score": min(float(line.get("score", 0.45)), 0.55),
                "detector": "ocr_paragraph_recovery",
            }
        )
    return _merge_and_filter(result)


def _supplement_and_expand_paragraphs(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    recovered = _supplement_missing_paragraph_lines(blocks, ocr_lines)
    return _expand_paragraphs_with_nearby_ocr_lines(recovered, ocr_lines)


def _covered_by_existing_relaxed(bbox: List[int], blocks: List[Dict]) -> bool:
    for block in blocks:
        overlap = _intersection_over_area(bbox, block["bbox"])
        if overlap >= 0.55 or _inside(bbox, block["bbox"]):
            return True
    return False


def _looks_like_recoverable_paragraph_line(text: str) -> bool:
    compact = text.replace(" ", "")
    if len(compact) < 18:
        return False
    if _looks_like_choice_or_answer_list(text) or _looks_like_formula_block(text):
        return False
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    sentence_marker = any(marker in compact for marker in ["에서", "대한", "때", "하면", "이다", "한다", "있다", "은", "는", "을", "를"])
    return korean_count >= 8 and sentence_marker


def _looks_like_paragraph_continuation(text: str) -> bool:
    compact = text.replace(" ", "")
    if len(compact) < 6 or _looks_like_formula_block(text):
        return False
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    continuation_markers = [
        "때의",
        "변화율",
        "이므로",
        "으므로",
        "따라서",
        "이고",
        "이며",
        "이다",
        "한다",
        "된다",
        "같다",
        "있다",
        "없다",
        "수있다",
        "알아보자",
        "나타낸다",
    ]
    return korean_count >= 5 and any(marker in compact for marker in continuation_markers)


def refine_blocks_after_ocr(
    blocks: List[Dict],
    ocr_lines: Optional[List[Dict]] = None,
    correction_profile: Optional[str] = None,
) -> List[Dict]:
    lines = ocr_lines or []
    processed = _postprocess_blocks(blocks, None, None)
    processed = _expand_paragraphs_with_nearby_ocr_lines(processed, lines)
    processed = _supplement_and_expand_paragraphs(processed, lines)
    split = _split_mixed_role_blocks(processed, lines)
    normalized = _normalize_content_blocks(_postprocess_blocks(split, None, None))
    recovered = _supplement_and_expand_paragraphs(normalized, lines)
    recovered = _split_answer_lines_from_paragraphs(recovered, lines)
    recovered = _merge_numbered_items_with_parallel_explanations(recovered)
    recovered = _merge_side_badges_into_paragraphs(recovered)
    recovered = _normalize_paragraph_text_order(recovered)
    if correction_profile == "unit3":
        recovered = _apply_unit3_postprocess(recovered, lines)
    return _supplement_nested_formula_lines(recovered, lines)


def _apply_unit3_postprocess(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    """Extra postprocessing for the coordinate-plane/graph textbook unit.

    The unit has dense worksheet-like boxes where one paragraph is often split
    into small left/right fragments. Keep the existing type vocabulary, but make
    paragraph grouping less brittle for this profile only.
    """
    processed = _unit3_reclassify_text_fragments(blocks)
    processed = _unit3_reclassify_choice_tables(processed)
    processed = _unit3_reclassify_variable_tables(processed, ocr_lines)
    processed = _unit3_reclassify_side_roman_markers(processed)
    processed = _merge_unit3_paragraph_fragments(processed)
    processed = _attach_unit3_short_labels_to_paragraphs(processed)
    processed = _merge_unit3_paragraph_fragments(processed)
    processed = _split_unit3_figure_captions_from_paragraphs(processed, ocr_lines)
    processed = _split_unit3_multi_figure_descriptions(processed, ocr_lines)
    processed = _clean_unit3_axis_ticks_from_text_blocks(processed)
    processed = _drop_unit3_decorative_figures(processed)
    return _merge_and_filter(processed)


def _unit3_reclassify_text_fragments(blocks: List[Dict]) -> List[Dict]:
    result: List[Dict] = []
    for block in blocks:
        block = dict(block)
        text = block.get("text", "").strip()
        if not text:
            result.append(block)
            continue
        compact = text.replace(" ", "")
        width = block["bbox"][2] - block["bbox"][0]
        height = block["bbox"][3] - block["bbox"][1]
        is_text_fragment = (
            block["type"] in {"caption", "title", "section_title"}
            and len(compact) >= 6
            and not _looks_like_unit3_figure_caption(text)
            and not _looks_like_formula_block(text)
            and not _looks_like_table_text(text)
            and (height <= 70 or width >= 120)
        )
        if is_text_fragment:
            block["type"] = "paragraph"
            block["detector"] = f"{block.get('detector', 'layout')}_unit3_text"
            block["score"] = min(float(block.get("score", 0.55)), 0.72)
        result.append(block)
    return result


def _unit3_reclassify_choice_tables(blocks: List[Dict]) -> List[Dict]:
    result: List[Dict] = []
    for block in blocks:
        block = dict(block)
        if block["type"] == "table" and _looks_like_unit3_choice_or_solution_text(block.get("text", "")):
            block["type"] = "paragraph"
            block["detector"] = f"{block.get('detector', 'layout')}_unit3_choice_text"
            block["score"] = min(float(block.get("score", 0.55)), 0.72)
        result.append(block)
    return result


def _looks_like_unit3_choice_or_solution_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "".join(lines)
    if not compact:
        return False

    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    numbered_items = sum(
        bool(re.match(r"^(?:[\(（]?\d{1,2}[\)）.]|[①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽])", line.replace(" ", "")))
        for line in lines
    )
    korean_choice_items = sum(bool(re.match(r"^[ㄱ-ㅎ가-힣][\.\)]", line.replace(" ", ""))) for line in lines)
    answer_or_solution = any(marker in compact for marker in ["정답", "답", "풀이", "과정"])
    prose_markers = any(marker in compact for marker in ["입니다", "이다", "된다", "높은", "낮은", "구하", "찾", "포함"])
    cell_like_headers = any(keyword in compact for keyword in ["연도", "구분", "합계", "비중", "증감", "국가", "예산"])

    return (
        not cell_like_headers
        and korean_count >= 8
        and (numbered_items >= 2 or korean_choice_items >= 2 or answer_or_solution)
        and prose_markers
    )


def _unit3_reclassify_variable_tables(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    result: List[Dict] = []
    for block in blocks:
        block = dict(block)
        if block["type"] != "formula":
            result.append(block)
            continue

        text = block.get("text", "").strip()
        if _looks_like_unit3_variable_table_text(text):
            block["type"] = "table"
            block["text"] = text or block.get("text", "")
            block["detector"] = f"{block.get('detector', 'layout')}_unit3_variable_table"
            block["score"] = min(float(block.get("score", 0.55)), 0.72)
        result.append(block)
    return result


def _looks_like_unit3_variable_table_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "".join(lines).replace(" ", "")
    if len(compact) < 8:
        return False
    if _looks_like_long_sentence(text) or _looks_like_prose_paragraph(text):
        return False

    has_variable_header = bool(re.search(r"[xy]\s*[\(（][^)）]+[\)）]", text, re.IGNORECASE))
    has_x_y_rows = bool(re.search(r"(?:^|\n)\s*x\s*(?:\n|$)", text, re.IGNORECASE)) and bool(
        re.search(r"(?:^|\n)\s*y\s*(?:\n|$)", text, re.IGNORECASE)
    )
    has_unit_header = any(unit in compact for unit in ["kWh", "kwh", "원", "km", "mL", "L"]) and any(
        axis in compact for axis in ["x", "y"]
    )
    numeric_tokens = re.findall(r"[−-]?\d+(?:\.\d+)?", text)
    ellipsis_or_sequence = "…" in compact or "..." in compact or len(numeric_tokens) >= 4

    return (has_variable_header or has_x_y_rows or has_unit_header) and ellipsis_or_sequence


def _drop_unit3_decorative_figures(blocks: List[Dict]) -> List[Dict]:
    """Discard tiny low-confidence icons before figure semantics run.

    The unit uses small badges such as character heads and UI-like markers near
    paragraphs. DocLayout sometimes calls them figures, but treating them as
    figures later creates noisy graph/image analysis targets.
    """
    result: List[Dict] = []
    for block in blocks:
        if block["type"] != "figure":
            result.append(block)
            continue

        x1, y1, x2, y2 = block["bbox"]
        width = x2 - x1
        height = y2 - y1
        score = float(block.get("score", 0.0))
        if width <= 70 and height <= 70 and score < 0.55:
            continue
        result.append(block)
    return result


def _unit3_reclassify_side_roman_markers(blocks: List[Dict]) -> List[Dict]:
    result: List[Dict] = []
    for block in blocks:
        block = dict(block)
        if block["type"] in {"figure", "title", "section_title", "caption", "paragraph"} and _looks_like_side_roman_marker(block):
            block["type"] = "footer"
            block["detector"] = f"{block.get('detector', 'layout')}_unit3_side_marker"
            block["score"] = min(float(block.get("score", 0.55)), 0.72)
        result.append(block)
    return result


def _looks_like_side_roman_marker(block: Dict) -> bool:
    text = block.get("text", "").strip().upper()
    x1, y1, x2, y2 = block["bbox"]
    width = x2 - x1
    height = y2 - y1
    if text in {"I", "II", "III", "IV", "V", "VI"}:
        return width <= 100 and height <= 140 and x1 >= 680
    return block["type"] == "figure" and not text and width <= 90 and height <= 120 and x1 >= 760


def _clean_unit3_axis_ticks_from_text_blocks(blocks: List[Dict]) -> List[Dict]:
    result: List[Dict] = []
    for block in blocks:
        if block["type"] not in {"paragraph", "table"}:
            result.append(block)
            continue

        lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
        if len(lines) < 3:
            result.append(block)
            continue

        tick_lines = [line for line in lines if _looks_like_axis_tick_text(line)]
        content_lines = [line for line in lines if not _looks_like_axis_tick_text(line)]
        if len(tick_lines) < 2 or not content_lines:
            result.append(block)
            continue

        cleaned = dict(block)
        cleaned["text"] = "\n".join(content_lines)
        cleaned["detector"] = f"{block.get('detector', 'layout')}_axis_tick_clean"
        if block["type"] == "table" and any(marker in cleaned["text"] for marker in ["구하시오", "답하시오", "그래프"]):
            cleaned["type"] = "paragraph"
            cleaned["score"] = min(float(block.get("score", 0.55)), 0.72)
        result.append(cleaned)
    return result


def _merge_unit3_paragraph_fragments(blocks: List[Dict]) -> List[Dict]:
    result = [dict(block) for block in blocks]
    consumed = set()

    for index, block in enumerate(result):
        if index in consumed or block["type"] != "paragraph":
            continue

        group = [index]
        changed = True
        while changed:
            changed = False
            group_bbox = _bbox_for_blocks([result[item] for item in group])
            for other_index, other in enumerate(result):
                if other_index in consumed or other_index in group or other["type"] != "paragraph":
                    continue
                if _should_unit3_merge_paragraph(group_bbox, other["bbox"], result, group):
                    group.append(other_index)
                    changed = True

        if len(group) <= 1:
            continue

        ordered = sorted(group, key=lambda item: (result[item]["bbox"][1], result[item]["bbox"][0]))
        merged_blocks = [result[item] for item in ordered]
        block["bbox"] = _bbox_for_blocks(merged_blocks)
        block["text"] = _join_paragraph_fragments(
            [item.get("text", "").strip() for item in merged_blocks if item.get("text", "").strip()]
        )
        block["score"] = max(float(item.get("score", 0.0)) for item in merged_blocks)
        block["detector"] = "unit3_paragraph_merge"
        contexts = [item.get("context") for item in merged_blocks if item.get("context")]
        if contexts:
            merged_context = {}
            for context in contexts:
                merged_context.update(context)
            block["context"] = merged_context
        consumed.update(item for item in group if item != index)

    return [block for item, block in enumerate(result) if item not in consumed]


def _should_unit3_merge_paragraph(group_bbox: List[int], candidate_bbox: List[int], blocks: List[Dict], group: List[int]) -> bool:
    gx1, gy1, gx2, gy2 = group_bbox
    cx1, cy1, cx2, cy2 = candidate_bbox
    g_width = max(1, gx2 - gx1)
    c_width = max(1, cx2 - cx1)
    vertical_gap = cy1 - gy2 if cy1 >= gy2 else gy1 - cy2 if gy1 >= cy2 else 0
    if vertical_gap < -18 or vertical_gap > 42:
        return False

    horizontal_overlap = min(gx2, cx2) - max(gx1, cx1)
    overlap_ratio = horizontal_overlap / max(1, min(g_width, c_width))
    same_column = overlap_ratio >= 0.28 or abs(cx1 - gx1) <= 70
    wrapped_right_column = 0 <= cx1 - gx2 <= max(80, g_width * 0.45) and abs(cy1 - gy1) <= 40
    wrapped_left_column = 0 <= gx1 - cx2 <= max(80, c_width * 0.45) and abs(cy1 - gy1) <= 40
    if not (same_column or wrapped_right_column or wrapped_left_column):
        return False

    bridge = [min(gx1, cx1), min(gy1, cy1), max(gx2, cx2), max(gy2, cy2)]
    for other_index, other in enumerate(blocks):
        if other_index in group or other["type"] not in {"figure", "table", "formula"}:
            continue
        if _intersection_over_area(other["bbox"], bridge) >= 0.25:
            return False
    return True


def _attach_unit3_short_labels_to_paragraphs(blocks: List[Dict]) -> List[Dict]:
    result = [dict(block) for block in blocks]
    consumed = set()

    for label_index, label in enumerate(result):
        if label["type"] not in {"title", "section_title", "caption"}:
            continue
        text = label.get("text", "").strip()
        if not text or len(text.replace(" ", "")) > 18:
            continue
        if _looks_like_unit3_figure_caption(text):
            continue

        lx1, ly1, lx2, ly2 = label["bbox"]
        candidates = []
        for paragraph_index, paragraph in enumerate(result):
            if paragraph_index == label_index or paragraph["type"] != "paragraph":
                continue
            px1, py1, px2, py2 = paragraph["bbox"]
            vertical_gap = py1 - ly2
            horizontal_gap = px1 - lx2
            horizontal_overlap = min(lx2, px2) - max(lx1, px1)
            near_above = 0 <= vertical_gap <= 42 and horizontal_overlap >= -30
            near_left = -30 <= horizontal_gap <= 75 and min(ly2, py2) - max(ly1, py1) > 0
            if near_above or near_left:
                candidates.append((abs(vertical_gap) + max(horizontal_gap, 0) * 0.25, paragraph_index))

        if not candidates:
            continue

        _, paragraph_index = min(candidates)
        paragraph = result[paragraph_index]
        paragraph["bbox"] = _bbox_for_blocks([label, paragraph])
        if text and text not in paragraph.get("text", ""):
            paragraph["text"] = _join_paragraph_fragments([text, paragraph.get("text", "")])
        paragraph["detector"] = "unit3_label_attach"
        consumed.add(label_index)

    return [block for index, block in enumerate(result) if index not in consumed]


def _split_unit3_figure_captions_from_paragraphs(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    figures = [block for block in blocks if block["type"] == "figure"]
    if not figures:
        return blocks

    result: List[Dict] = []
    for block in blocks:
        if block["type"] != "paragraph":
            result.append(block)
            continue

        inner_lines = _lines_inside(ocr_lines, block["bbox"]) if ocr_lines else []
        if len(inner_lines) < 2:
            split = _split_unit3_caption_from_block_text(block, figures)
            if split:
                result.extend(split)
            else:
                result.append(block)
            continue

        caption_lines = []
        content_lines = []
        for line in inner_lines:
            text = line.get("text", "").strip()
            if _looks_like_unit3_figure_caption(text) and _is_unit3_caption_near_figure(line["bbox"], figures):
                caption_lines.append(line)
            else:
                content_lines.append(line)

        if not caption_lines or not content_lines:
            split = _split_unit3_caption_from_block_text(block, figures)
            if split:
                result.extend(split)
            else:
                result.append(block)
            continue

        content_block = dict(block)
        ordered_content = _sort_ocr_lines_for_text(content_lines)
        content_block["bbox"] = _bbox_for_lines(ordered_content)
        content_block["text"] = "\n".join(
            line.get("text", "").strip() for line in ordered_content if line.get("text", "").strip()
        )

        result.append(content_block)
        for caption_group in _unit3_group_unassigned_lines(caption_lines):
            ordered_caption = _sort_ocr_lines_for_text(caption_group)
            caption_block = {
                "type": "caption",
                "bbox": _bbox_for_lines(ordered_caption),
                "text": "\n".join(
                    line.get("text", "").strip() for line in ordered_caption if line.get("text", "").strip()
                ),
                "score": min(max(float(line.get("score", 0.45)) for line in ordered_caption), 0.72),
                "detector": "unit3_figure_caption_split",
                "context": {"semantic_role": "figure_caption"},
            }
            result.append(caption_block)

    return result


def _split_unit3_caption_from_block_text(block: Dict, figures: List[Dict]) -> Optional[List[Dict]]:
    source_lines = [line.strip() for line in block.get("text", "").splitlines() if line.strip()]
    if len(source_lines) < 2:
        return None

    caption_indexes = [index for index, line in enumerate(source_lines) if _looks_like_unit3_figure_caption(line)]
    if not caption_indexes:
        return None

    bx1, by1, bx2, by2 = block["bbox"]
    line_height = max(10, (by2 - by1) / max(len(source_lines), 1))
    content_lines = [line for index, line in enumerate(source_lines) if index not in caption_indexes]
    caption_lines = [line for index, line in enumerate(source_lines) if index in caption_indexes]
    if not content_lines or not caption_lines:
        return None

    caption_start = min(caption_indexes)
    caption_end = max(caption_indexes)
    caption_bbox = [
        bx1,
        int(by1 + caption_start * line_height),
        bx2,
        int(by1 + (caption_end + 1) * line_height),
    ]
    if not _is_unit3_caption_near_figure(caption_bbox, figures):
        return None
    caption_text = "\n".join(caption_lines)
    caption_bbox = _tighten_unit3_caption_bbox(caption_bbox, caption_text, figures)

    content_block = dict(block)
    content_block["text"] = "\n".join(content_lines)
    if caption_start == 0:
        content_block["bbox"] = [bx1, int(by1 + (caption_end + 1) * line_height), bx2, by2]
    elif caption_end == len(source_lines) - 1:
        content_block["bbox"] = [bx1, by1, bx2, int(by1 + caption_start * line_height)]

    caption_block = {
        "type": "caption",
        "bbox": caption_bbox,
        "text": caption_text,
        "score": min(float(block.get("score", 0.50)), 0.62),
        "detector": "unit3_figure_caption_text_split",
        "context": {"semantic_role": "figure_caption"},
    }
    return [content_block, caption_block]


def _looks_like_unit3_figure_caption(text: str) -> bool:
    return _looks_like_source_caption_text(text)


def _looks_like_source_caption_text(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if not compact or len(compact) > 80:
        return False
    source_markers = ["출처", "자료", "출전", "출처:", "자료:"]
    return any(marker in compact for marker in source_markers)


def _is_unit3_caption_near_figure(bbox: List[int], figures: List[Dict]) -> bool:
    return _nearest_unit3_caption_figure(bbox, figures) is not None


def _nearest_unit3_caption_figure(bbox: List[int], figures: List[Dict]) -> Optional[Dict]:
    x1, y1, x2, y2 = bbox
    line_cx = (x1 + x2) / 2
    best: Optional[tuple[float, Dict]] = None
    for figure in figures:
        fx1, fy1, fx2, fy2 = figure["bbox"]
        horizontal_margin = max(35, (fx2 - fx1) * 0.25)
        below_figure = fy2 - 12 <= y1 <= fy2 + 58
        inside_bottom = fy1 <= y1 <= fy2 + 10 and y1 >= fy2 - max(45, (fy2 - fy1) * 0.18)
        horizontally_related = fx1 - horizontal_margin <= line_cx <= fx2 + horizontal_margin
        if horizontally_related and (below_figure or inside_bottom):
            figure_cx = (fx1 + fx2) / 2
            distance = abs(line_cx - figure_cx) + abs(y1 - fy2) * 0.35
            if best is None or distance < best[0]:
                best = (distance, figure)
    return None if best is None else best[1]


def _tighten_unit3_caption_bbox(bbox: List[int], text: str, figures: List[Dict]) -> List[int]:
    figure = _nearest_unit3_caption_figure(bbox, figures)
    if figure is None:
        return bbox

    x1, y1, x2, y2 = bbox
    fx1, _, fx2, _ = figure["bbox"]
    estimated_width = max(90, min(x2 - x1, int(len(text.replace("\n", "")) * 7.5) + 24))
    tightened_x2 = min(x2, fx2 + 24)
    tightened_x1 = max(x1, tightened_x2 - estimated_width)
    if tightened_x2 - tightened_x1 < 50:
        return bbox
    return [int(tightened_x1), y1, int(tightened_x2), y2]


def _split_unit3_multi_figure_descriptions(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    if not ocr_lines:
        return blocks

    figures = [block for block in blocks if block["type"] == "figure"]
    if len(figures) < 2:
        return blocks

    result: List[Dict] = []
    for block in blocks:
        if block["type"] != "paragraph":
            result.append(block)
            continue

        nearby_figures = _unit3_description_figures_for_paragraph(block, figures)
        if len(nearby_figures) < 2:
            result.append(block)
            continue

        inner_lines = _lines_inside(ocr_lines, block["bbox"])
        if len(inner_lines) < 2:
            result.append(block)
            continue

        assigned: Dict[int, List[Dict]] = {index: [] for index in range(len(nearby_figures))}
        unassigned: List[Dict] = []
        for line in inner_lines:
            target_index = _unit3_nearest_described_figure_index(line, nearby_figures, block["bbox"])
            if target_index is None:
                unassigned.append(line)
            else:
                assigned[target_index].append(line)

        assigned_groups = [lines for lines in assigned.values() if lines]
        if len(assigned_groups) < 2:
            result.append(block)
            continue

        for lines in assigned_groups:
            description = _unit3_block_from_description_lines(lines, block)
            if description:
                result.append(description)
        for lines in _unit3_group_unassigned_lines(unassigned):
            description = _unit3_block_from_description_lines(lines, block, detector="unit3_unassigned_paragraph")
            if description:
                result.append(description)

    return result


def _unit3_description_figures_for_paragraph(paragraph: Dict, figures: List[Dict]) -> List[Dict]:
    px1, py1, px2, py2 = paragraph["bbox"]
    paragraph_width = max(1, px2 - px1)
    candidates = []
    for figure in figures:
        fx1, fy1, fx2, fy2 = figure["bbox"]
        horizontal_overlap = min(px2, fx2) - max(px1, fx1)
        if horizontal_overlap / max(1, min(paragraph_width, fx2 - fx1)) < 0.15:
            continue
        figure_above_or_inside = fy1 <= py2 and fy2 <= py2 + 12 and fy2 >= py1 - 90
        if not figure_above_or_inside:
            continue
        candidates.append(figure)
    return sorted(candidates, key=lambda item: item["bbox"][0])


def _unit3_nearest_described_figure_index(line: Dict, figures: List[Dict], paragraph_bbox: List[int]) -> Optional[int]:
    lx1, ly1, lx2, ly2 = line["bbox"]
    line_text = line.get("text", "").strip()
    if _is_noise_text(line_text):
        return None

    paragraph_width = max(1, paragraph_bbox[2] - paragraph_bbox[0])
    line_width = max(1, lx2 - lx1)
    if line_width / paragraph_width >= 0.62:
        return None

    line_cx = (lx1 + lx2) / 2
    line_cy = (ly1 + ly2) / 2
    best: Optional[tuple[float, int]] = None
    for index, figure in enumerate(figures):
        fx1, fy1, fx2, fy2 = figure["bbox"]
        if line_cy < fy2 - 8:
            continue
        vertical_gap = ly1 - fy2
        if vertical_gap > 95:
            continue
        horizontal_margin = max(45, (fx2 - fx1) * 0.35)
        if not (fx1 - horizontal_margin <= line_cx <= fx2 + horizontal_margin):
            continue
        figure_cx = (fx1 + fx2) / 2
        distance = abs(line_cx - figure_cx) + max(vertical_gap, 0) * 0.35
        if best is None or distance < best[0]:
            best = (distance, index)
    return None if best is None else best[1]


def _unit3_group_unassigned_lines(lines: List[Dict]) -> List[List[Dict]]:
    if not lines:
        return []
    groups: List[List[Dict]] = []
    for line in _sort_ocr_lines_for_text(lines):
        if not groups:
            groups.append([line])
            continue
        previous = groups[-1][-1]
        vertical_gap = line["bbox"][1] - previous["bbox"][3]
        horizontal_overlap = min(line["bbox"][2], previous["bbox"][2]) - max(line["bbox"][0], previous["bbox"][0])
        same_flow = vertical_gap <= 36 and (horizontal_overlap > 0 or abs(line["bbox"][0] - previous["bbox"][0]) <= 90)
        if same_flow:
            groups[-1].append(line)
        else:
            groups.append([line])
    return groups


def _unit3_block_from_description_lines(
    lines: List[Dict],
    source: Dict,
    detector: str = "unit3_figure_description_split",
) -> Optional[Dict]:
    ordered = _sort_ocr_lines_for_text(lines)
    content_lines = [
        line
        for line in ordered
        if not _looks_like_axis_tick_text(line.get("text", ""))
        and not _looks_like_axis_label_text(line.get("text", ""))
        and not _looks_like_source_caption_text(line.get("text", ""))
    ]
    if not content_lines:
        return None
    if content_lines:
        ordered = content_lines
    block = dict(source)
    block["bbox"] = _bbox_for_lines(ordered)
    block["text"] = "\n".join(line.get("text", "").strip() for line in ordered if line.get("text", "").strip())
    block["type"] = "paragraph"
    block["score"] = min(max(float(line.get("score", 0.45)) for line in ordered), 0.72)
    block["detector"] = detector
    context = dict(block.get("context") or {})
    context["semantic_role"] = context.get("semantic_role", "figure_description")
    block["context"] = context
    return block


def _bbox_for_blocks(blocks: List[Dict]) -> List[int]:
    return [
        min(block["bbox"][0] for block in blocks),
        min(block["bbox"][1] for block in blocks),
        max(block["bbox"][2] for block in blocks),
        max(block["bbox"][3] for block in blocks),
    ]


def _supplement_nested_formula_lines(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    """Keep standalone formula rows even when a larger paragraph contains them."""
    if not ocr_lines:
        return blocks

    result = list(blocks)
    existing_formulas = [block for block in blocks if block["type"] == "formula"]
    paragraphs = [block for block in blocks if block["type"] == "paragraph"]

    for paragraph_index, paragraph in enumerate(paragraphs, start=1):
        px1, py1, px2, py2 = paragraph["bbox"]
        candidates = []
        for line in ocr_lines:
            text = line.get("text", "").strip()
            if not _looks_like_standalone_formula_line(text):
                continue
            lx1, ly1, lx2, ly2 = line["bbox"]
            cx = (lx1 + lx2) / 2
            cy = (ly1 + ly2) / 2
            inside_with_tolerance = px1 - 18 <= cx <= px2 + 18 and py1 - 8 <= cy <= py2 + 22
            if not inside_with_tolerance:
                continue
            if _is_overlapping_any(line["bbox"], existing_formulas, threshold=0.45):
                continue
            candidates.append(line)

        if not candidates:
            continue

        row_groups: List[List[Dict]] = []
        for line in sorted(candidates, key=lambda item: (item["bbox"][1], item["bbox"][0])):
            cy = (line["bbox"][1] + line["bbox"][3]) / 2
            if row_groups:
                previous_cy = sum(
                    (item["bbox"][1] + item["bbox"][3]) / 2 for item in row_groups[-1]
                ) / len(row_groups[-1])
                if abs(cy - previous_cy) <= 18:
                    row_groups[-1].append(line)
                    continue
            row_groups.append([line])

        for row in row_groups:
            ordered = sorted(row, key=lambda item: item["bbox"][0])
            bbox = [
                min(item["bbox"][0] for item in ordered),
                min(item["bbox"][1] for item in ordered),
                max(item["bbox"][2] for item in ordered),
                max(item["bbox"][3] for item in ordered),
            ]
            text = "\n".join(item.get("text", "").strip() for item in ordered if item.get("text", "").strip())
            context = dict(paragraph.get("context") or {})
            context["embedded_in"] = "paragraph"
            result.append(
                {
                    "type": "formula",
                    "bbox": bbox,
                    "text": text,
                    "score": min(max(float(item.get("score", 0.45)) for item in ordered), 0.60),
                    "detector": "ocr_formula_recovery",
                    "container_id": paragraph.get("container_id", f"paragraph_{paragraph_index}"),
                    "context": context,
                }
            )
            existing_formulas.append(result[-1])

    return sorted(result, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _looks_like_standalone_formula_line(text: str) -> bool:
    compact = text.replace(" ", "")
    if (
        len(compact) < 5
        or _looks_like_sentence_with_math(text)
        or _looks_like_long_sentence(text)
        or _looks_like_axis_or_legend_text(text)
        or _looks_like_axis_label_text(text)
        or _looks_like_unit_label_text(text)
    ):
        return False
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    return korean_count <= 6 and (_looks_like_formula(text) or _looks_like_formula_block(text))


def _looks_like_axis_tick_text(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if not compact:
        return False
    number = r"[−-]?\d+(?:\.\d+)?"
    return bool(re.fullmatch(rf"{number}(?:[,，\s]+{number})*", compact))


def _looks_like_axis_label_text(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if not compact or len(compact) > 24:
        return False
    axis_words = ["거리", "시간", "높이", "속력", "온도", "기온", "전압", "전류", "무게", "개수", "가격"]
    axis_units = ["km", "m", "cm", "초", "분", "시간", "원", "%", "℃"]
    return any(word in compact for word in axis_words) and (
        "(" in compact or ")" in compact or any(unit in compact for unit in axis_units)
    )


def _looks_like_unit_label_text(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    if not compact or len(compact) > 16:
        return False
    units = ["kWh", "kwh", "km", "m", "cm", "kg", "g", "원", "초", "분", "시간", "%", "℃"]
    return any(unit in compact for unit in units) and not re.search(r"\d", compact)


def _detect_with_yolo(
    image_path: str | Path,
    model_path: str | Path,
    recover_low_conf_figures: bool = False,
) -> List[Dict]:
    model_ref = str(model_path)
    if model_ref.startswith("hf:"):
        return _detect_with_doclayout_yolo(
            image_path,
            model_ref[3:],
            recover_low_conf_figures=recover_low_conf_figures,
        )

    from ultralytics import YOLO

    model = YOLO(model_ref)
    result = model(str(image_path), verbose=False)[0]
    names = result.names
    blocks: List[Dict] = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = _map_external_label(names.get(cls_id, str(cls_id)))
        blocks.append(
            {
                "type": label,
                "bbox": [int(v) for v in box.xyxy[0].tolist()],
                "score": float(box.conf[0]),
            }
        )
    return blocks


def _detect_with_doclayout_yolo(
    image_path: str | Path,
    repo_id: str,
    recover_low_conf_figures: bool = False,
) -> List[Dict]:
    cache_key = f"doclayout-yolo:{repo_id}"
    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _load_doclayout_yolo_model(repo_id)

    model = _MODEL_CACHE[cache_key]
    result = model.predict(
        str(image_path),
        imgsz=1024,
        conf=0.10 if recover_low_conf_figures else 0.20,
        device="cpu",
        verbose=False,
    )[0]
    names = result.names
    image_height, image_width = result.orig_shape
    blocks: List[Dict] = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        label = _map_external_label(names.get(cls_id, str(cls_id)))
        score = float(box.conf[0])
        bbox = [int(v) for v in box.xyxy[0].tolist()]
        low_conf_figure = score < 0.20
        if low_conf_figure and not _is_recoverable_low_conf_figure(
            label,
            score,
            bbox,
            image_width,
            image_height,
        ):
            continue
        blocks.append(
            {
                "type": label,
                "bbox": bbox,
                "score": score,
                "detector": "doclayout_yolo_low_conf_figure" if low_conf_figure else "doclayout_yolo",
            }
        )
    if recover_low_conf_figures:
        blocks = _drop_broad_figures_covering_content(blocks, image_width, image_height)
    return blocks


def _is_recoverable_low_conf_figure(
    label: str,
    score: float,
    bbox: List[int],
    image_width: int,
    image_height: int,
) -> bool:
    if label != "figure" or score < 0.12:
        return False
    width = max(0, bbox[2] - bbox[0])
    height = max(0, bbox[3] - bbox[1])
    page_area = max(1, image_width * image_height)
    area_ratio = width * height / page_area
    return (
        0.006 <= area_ratio <= 0.12
        and width / max(image_width, 1) <= 0.48
        and height / max(image_height, 1) <= 0.40
    )


def _drop_broad_figures_covering_content(
    blocks: List[Dict],
    image_width: int,
    image_height: int,
) -> List[Dict]:
    page_area = max(1, image_width * image_height)
    result = []
    for block in blocks:
        if block["type"] != "figure" or _area(block["bbox"]) / page_area < 0.20:
            result.append(block)
            continue

        covered_content = sum(
            other is not block
            and other["type"] in {"title", "section_title", "paragraph", "formula", "table", "caption"}
            and _intersection_over_area(other["bbox"], block["bbox"]) >= 0.65
            for other in blocks
        )
        if covered_content < 3:
            result.append(block)
    return result


def _load_doclayout_yolo_model(repo_id: str):
    import os

    cache_dir = PROJECT_ROOT / ".cache" / "huggingface"
    matplotlib_cache_dir = PROJECT_ROOT / ".cache" / "matplotlib"
    yolo_config_dir = PROJECT_ROOT / ".cache" / "ultralytics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
    yolo_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_dir))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache_dir))
    os.environ.setdefault("YOLO_CONFIG_DIR", str(yolo_config_dir))

    try:
        from doclayout_yolo import YOLOv10
    except ImportError as exc:
        raise RuntimeError("DocLayout-YOLO is not installed. Run: pip install doclayout-yolo huggingface_hub") from exc

    try:
        return YOLOv10.from_pretrained(repo_id)
    except Exception:
        from huggingface_hub import hf_hub_download

        filepath = hf_hub_download(
            repo_id=repo_id,
            filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            cache_dir=cache_dir,
        )
        return YOLOv10(filepath)


def _map_external_label(label: str) -> str:
    normalized = label.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "text": "paragraph",
        "plain_text": "paragraph",
        "plain_text_region": "paragraph",
        "text_region": "paragraph",
        "body": "paragraph",
        "body_text": "paragraph",
        "header": "title",
        "heading": "section_title",
        "section": "section_title",
        "section_header": "section_title",
        "figure": "figure",
        "picture": "figure",
        "image": "figure",
        "graph": "figure",
        "graph_or_figure": "figure",
        "equation": "formula",
        "isolate_formula": "formula",
        "isolated_formula": "formula",
        "formula": "formula",
        "formula_box": "formula",
        "table_caption": "caption",
        "figure_caption": "caption",
        "formula_caption": "caption",
        "caption_or_legend": "caption",
        "table_footnote": "caption",
        "abandon": "footer",
        "page_footer": "footer",
        "example_box": "paragraph",
        "problem_box": "paragraph",
        "solution_box": "paragraph",
    }
    return aliases.get(normalized, normalized if normalized in TARGET_CLASSES else "paragraph")


def _detect_with_heuristics(image_path: str | Path, ocr_lines: List[Dict]) -> List[Dict]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    height, width = image.shape[:2]
    blocks: List[Dict] = []
    blocks.extend(_detect_box_regions(image, ocr_lines))
    blocks.extend(_detect_colored_panel_regions(image, ocr_lines))
    blocks.extend(_detect_visual_regions(image, ocr_lines))
    blocks.extend(_classify_text_regions(ocr_lines, width, height, blocks))
    return _postprocess_blocks(_merge_and_filter(blocks), width, height)


def _detect_box_regions(image: np.ndarray, ocr_lines: List[Dict]) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = image.shape[0] * image.shape[1]
    boxes: List[Dict] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < page_area * 0.008 or area > page_area * 0.72:
            continue
        if w < 100 or h < 40:
            continue

        bbox = [x, y, x + w, y + h]
        text = _collect_text(ocr_lines, bbox)
        boxes.append({"type": _classify_box_text(text), "bbox": bbox, "text": text, "score": 0.55})
    return boxes


def _detect_visual_regions(image: np.ndarray, ocr_lines: List[Dict]) -> List[Dict]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    merged = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = image.shape[0] * image.shape[1]
    regions: List[Dict] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < page_area * 0.015 or area > page_area * 0.55:
            continue
        bbox = [x, y, x + w, y + h]
        text = _collect_text(ocr_lines, bbox)

        roi = binary[y : y + h, x : x + w]
        dark_ratio = float(np.count_nonzero(roi)) / max(area, 1)
        if dark_ratio < 0.08:
            continue

        aspect = w / max(h, 1)
        label = "table" if _looks_like_table(roi) else "figure"
        if label == "figure" and _looks_like_paragraph_text(text):
            continue
        regions.append({"type": label, "bbox": bbox, "score": 0.40})
    return regions


def _detect_colored_panel_regions(image: np.ndarray, ocr_lines: List[Dict]) -> List[Dict]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    mask = np.where(((saturation > 18) & (value > 80)) | (value < 235), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    merged = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    page_area = image.shape[0] * image.shape[1]
    regions: List[Dict] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < page_area * 0.02 or area > page_area * 0.65:
            continue
        if w < 140 or h < 70:
            continue
        bbox = [x, y, x + w, y + h]
        text = _collect_text(ocr_lines, bbox)
        if not text.strip():
            continue
        regions.append({"type": _classify_box_text(text), "bbox": bbox, "text": text, "score": 0.45})
    return regions


def _classify_text_regions(
    ocr_lines: List[Dict], page_width: int, page_height: int, existing_blocks: List[Dict]
) -> List[Dict]:
    text_blocks: List[Dict] = []
    for line in ocr_lines:
        bbox = line["bbox"]
        if _covered_by_existing(bbox, existing_blocks):
            continue

        text = line["text"]
        x1, y1, x2, y2 = bbox
        h = y2 - y1
        w = x2 - x1

        if y2 > page_height * 0.93:
            label = "page_number" if text.strip().isdigit() or w < 80 else "footer"
        elif y1 < page_height * 0.16 and h > page_height * 0.018:
            label = "paragraph" if _looks_like_long_sentence(text) else "title" if w > page_width * 0.35 else "section_title"
        elif _looks_like_section_title(text, bbox, page_width):
            label = "section_title"
        elif _looks_like_unit_note(text) or _looks_like_caption(text):
            label = "caption"
        elif _looks_like_formula(text):
            label = "formula"
        else:
            label = "paragraph"

        text_blocks.append({"type": label, "bbox": bbox, "text": text, "score": line.get("score", 0.0)})

    return _group_paragraph_lines(text_blocks)


def _classify_box_text(text: str) -> str:
    compact = text.replace(" ", "")
    if _looks_like_header_box(text):
        return "section_title"
    if _looks_like_paragraph_text(text):
        return "paragraph"
    if _looks_like_table_text(text):
        return "table"
    if _looks_like_formula(text):
        return "formula"
    return "paragraph"


def _looks_like_section_title(text: str, bbox: List[int], page_width: int) -> bool:
    compact = text.strip()
    x1, _, x2, _ = bbox
    return (
        len(compact) <= 30
        and (compact[:2].isdigit() or compact.startswith(("단원", "학습", "탐구", "생각", "활동")))
        and (x2 - x1) < page_width * 0.8
    )


def _looks_like_header_box(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "".join(lines)
    if not compact:
        return False
    has_unit_number = any(line.isdigit() and len(line) <= 2 for line in lines)
    short_title = len(compact) <= 16 and any(char.isalpha() or "\uac00" <= char <= "\ud7a3" for char in compact)
    return has_unit_number and short_title


def _looks_like_caption(text: str) -> bool:
    compact = text.replace(" ", "")
    return compact.startswith(("그림", "표", "[그림", "[표")) or "자료:" in compact


def _looks_like_unit_note(text: str) -> bool:
    compact = text.replace(" ", "")
    return compact.startswith("(단위:") or compact.startswith("단위:") or compact.startswith("(단,") or compact.startswith("단,")


def _looks_like_formula(text: str) -> bool:
    if _looks_like_choice_or_answer_list(text):
        return False
    if _looks_like_unit_note(text):
        return False
    if _looks_like_sentence_with_math(text):
        return False
    formula_chars = ["=", "+", "-", "−", "–", "×", "÷", "∑", "√", "≤", "≥", "(", ")", "^", "lim", "log", "∆", "Δ"]
    math_hits = sum(1 for char in formula_chars if char in text)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return math_hits >= 2 or (math_hits >= 1 and digit_ratio > 0.20)


def _looks_like_formula_block(text: str) -> bool:
    if _looks_like_choice_or_answer_list(text):
        return False
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "".join(lines)
    if not compact:
        return False
    if _looks_like_long_sentence(text):
        return False
    math_chars = sum(char in "=+-−–×÷/%()[]{}^∆Δ√∑" for char in compact)
    alpha_math = sum(token in compact for token in ["f(", "lim", "log", "sin", "cos", "tan"])
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    table_words = any(keyword in compact for keyword in ["연도", "예산", "비중", "증감", "국가", "구분", "합계", "총지출"])
    if korean_count >= 8 and len(lines) <= 3:
        return False
    fraction_like = len(lines) >= 3 and math_chars >= 3 and korean_count <= 6
    equation_like = "=" in compact and math_chars >= 4 and korean_count <= 6
    return not table_words and (fraction_like or equation_like)


def _looks_like_choice_or_answer_list(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    numbered = sum(line.startswith(("(", "（")) and len(line) <= 30 for line in lines)
    compact = "".join(lines)
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    has_equal = "=" in compact
    if numbered >= 2 and not has_equal:
        return True
    if numbered >= 1 and korean_count >= 3 and not has_equal:
        return True
    return False


def _looks_like_long_sentence(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 30:
        return False
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in stripped)
    return korean_count >= 12 and any(mark in stripped for mark in ["다", "며", "고", "는", "은", "을", "를", "의"])


def _looks_like_sentence_with_math(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 35:
        return False
    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in stripped)
    math_count = sum(char in "=+-×÷()^" for char in stripped)
    sentence_hint = any(marker in stripped for marker in ["이다", "한다", "있다", "없다", "예를 들면", "따라"])
    return sentence_hint and korean_count >= 10 and math_count <= 8


def _looks_like_paragraph_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    lines = [line for line in stripped.splitlines() if line.strip()]
    sentence_marks = sum(stripped.count(mark) for mark in [".", "다", "며", "고", "은", "는"])
    return len(stripped) >= 45 and len(lines) >= 2 and sentence_marks >= 2


def _looks_like_explanatory_math_text(text: str) -> bool:
    """Identify prose that explains a calculation instead of presenting a formula alone."""
    compact = "".join(line.strip() for line in text.splitlines() if line.strip())
    if len(compact) < 45:
        return False

    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    explanation_markers = ["이므로", "이기 때문에", "따라서", "이고", "이며", "이다", "된다", "구하면", "계산하면", "약"]
    return (
        korean_count >= 15
        and korean_count / len(compact) >= 0.18
        and any(marker in compact for marker in explanation_markers)
    )


def _looks_like_prose_paragraph(text: str) -> bool:
    """Distinguish wrapped prose containing numbers from short table cells."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    compact = "".join(lines)
    if len(lines) < 2 or len(compact) < 70:
        return _looks_like_explanatory_math_text(text)
    if _looks_like_explanatory_math_text(text):
        return True
    if _looks_like_formula_block(text):
        return False

    korean_count = sum("\uac00" <= char <= "\ud7a3" for char in compact)
    long_lines = sum(len(line) >= 20 for line in lines)
    average_line_length = len(compact) / len(lines)
    return (
        korean_count >= 30
        and korean_count / len(compact) >= 0.35
        and average_line_length >= 22
        and long_lines / len(lines) >= 0.5
    )


def _looks_like_table_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
        return False
    if _looks_like_formula_block(text):
        return False
    if _looks_like_prose_paragraph(text):
        return False
    numeric_lines = sum(any(char.isdigit() for char in line) for line in lines)
    has_table_header = any(keyword in text for keyword in ["연도", "예산", "비중", "증감", "국가", "구분", "합계", "총"])
    return numeric_lines / max(len(lines), 1) >= 0.45 or (has_table_header and numeric_lines >= 3)


def _looks_like_table(binary_roi: np.ndarray) -> bool:
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
    horizontal = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, h_kernel)
    vertical = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, v_kernel)
    line_pixels = np.count_nonzero(horizontal) + np.count_nonzero(vertical)
    return line_pixels / max(binary_roi.size, 1) > 0.025


def _group_paragraph_lines(blocks: List[Dict]) -> List[Dict]:
    grouped: List[Dict] = []
    paragraph_buffer: List[Dict] = []

    def flush():
        if not paragraph_buffer:
            return
        if len(paragraph_buffer) == 1:
            grouped.append(paragraph_buffer[0])
        else:
            x1 = min(item["bbox"][0] for item in paragraph_buffer)
            y1 = min(item["bbox"][1] for item in paragraph_buffer)
            x2 = max(item["bbox"][2] for item in paragraph_buffer)
            y2 = max(item["bbox"][3] for item in paragraph_buffer)
            text = "\n".join(item.get("text", "") for item in paragraph_buffer)
            grouped.append({"type": "paragraph", "bbox": [x1, y1, x2, y2], "text": text, "score": 0.50})
        paragraph_buffer.clear()

    for block in sorted(blocks, key=lambda item: (item["bbox"][1], item["bbox"][0])):
        if block["type"] != "paragraph":
            flush()
            grouped.append(block)
            continue
        if not paragraph_buffer:
            paragraph_buffer.append(block)
            continue
        prev = paragraph_buffer[-1]
        same_column = abs(block["bbox"][0] - prev["bbox"][0]) < 80
        close_y = block["bbox"][1] - prev["bbox"][3] < 38
        if same_column and close_y:
            paragraph_buffer.append(block)
        else:
            flush()
            paragraph_buffer.append(block)
    flush()
    return grouped


def _postprocess_blocks(blocks: List[Dict], page_width: Optional[int], page_height: Optional[int]) -> List[Dict]:
    processed = []
    for block in blocks:
        text = block.get("text", "")
        if block.get("detector") == "role_region_supplement":
            block["type"] = "formula" if _looks_like_role_formula_region(text) else "paragraph"
            role = _infer_role(text)
            if role != "normal":
                block["role"] = role
            processed.append(block)
            continue
        role = _infer_role(text)
        if role != "normal":
            block["role"] = role
        if _looks_like_choice_or_answer_list(text):
            block["type"] = "paragraph"
        elif _looks_like_header_box(text):
            block["type"] = "section_title"
        elif block["type"] == "table" and _looks_like_prose_paragraph(text):
            block["type"] = "paragraph"
        elif block["type"] in {"table", "paragraph", "formula"} and _looks_like_formula_block(text):
            block["type"] = "formula"
        elif block["type"] in {"title", "section_title"} and _looks_like_formula(text):
            block["type"] = "formula"
        elif block["type"] in {"title", "section_title", "formula"} and (
            _looks_like_long_sentence(text) or _looks_like_paragraph_continuation(text)
        ):
            block["type"] = "paragraph"
        elif block["type"] == "formula" and (_looks_like_unit_note(text) or _looks_like_sentence_with_math(text)):
            block["type"] = "caption" if _looks_like_unit_note(text) else "paragraph"
        elif block["type"] == "paragraph" and _looks_like_formula(text) and not _looks_like_long_sentence(text):
            block["type"] = "formula"
        elif block["type"] in {"formula", "paragraph"} and _looks_like_table_text(text):
            block["type"] = "table"
        elif block["type"] in {"caption", "figure", "image", "example_box"} and _looks_like_paragraph_text(text):
            block["type"] = "paragraph"
        processed.append(block)
    return _drop_redundant_text_lines(_merge_and_filter(processed))


def _infer_role(text: str) -> str:
    compact = text.replace(" ", "")
    if any(keyword in compact for keyword in ["예제", "보기", "따라하기"]):
        return "example"
    if any(keyword in compact for keyword in ["풀이", "해설", "정답"]):
        return "solution"
    if any(keyword in compact for keyword in ["문제", "확인문제", "연습문제", "스스로", "생각열기", "탐구활동", "활동하기", "확인하기"]):
        return "problem"
    return "normal"


def _split_mixed_role_blocks(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    split_blocks: List[Dict] = []
    for block in blocks:
        if block.get("detector") == "role_region_supplement":
            split_blocks.append(block)
            continue
        role = block.get("role")
        if role not in {"example", "problem", "solution"}:
            split_blocks.append(block)
            continue
        if block["type"] == "formula":
            split_blocks.append(block)
            continue
        if block["type"] == "table" and not _looks_like_paragraph_text(block.get("text", "")):
            split_blocks.append(block)
            continue

        inner_lines = _lines_inside(ocr_lines, block["bbox"])
        if len(inner_lines) < 2:
            split_blocks.append(block)
            continue

        groups = _group_lines_by_content(inner_lines)
        if len(groups) <= 1:
            split_blocks.append(block)
            continue

        container_id = block.get("block_id") or f"container_{len(split_blocks) + 1}"
        for group_index, group in enumerate(groups, start=1):
            grouped_block = _block_from_lines(group, role, container_id, group_index, block["bbox"])
            if grouped_block:
                split_blocks.append(grouped_block)
    return _merge_and_filter(split_blocks)


def _lines_inside(ocr_lines: List[Dict], bbox: List[int]) -> List[Dict]:
    x1, y1, x2, y2 = bbox
    lines = []
    for line in ocr_lines:
        lx1, ly1, lx2, ly2 = line["bbox"]
        cx = (lx1 + lx2) / 2
        cy = (ly1 + ly2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            lines.append(line)
    return sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _group_lines_by_content(lines: List[Dict]) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    current: List[Dict] = []
    current_type = ""

    for line in lines:
        line_type = _line_content_type(line.get("text", ""))
        if not current:
            current = [line]
            current_type = line_type
            continue
        prev = current[-1]
        gap = line["bbox"][1] - prev["bbox"][3]
        if line_type == current_type and gap < 36:
            current.append(line)
        elif current_type == "table" and line_type in {"table", "paragraph"} and gap < 28:
            current.append(line)
        else:
            groups.append(current)
            current = [line]
            current_type = line_type
    if current:
        groups.append(current)

    merged: List[List[Dict]] = []
    for group in groups:
        if merged and _should_merge_paragraph_groups(merged[-1], group):
            merged[-1].extend(group)
        elif len(group) == 1 and merged and _line_content_type(group[0].get("text", "")) == "caption":
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


def _should_merge_paragraph_groups(previous: List[Dict], current: List[Dict]) -> bool:
    previous_text = "\n".join(item.get("text", "") for item in previous)
    current_text = "\n".join(item.get("text", "") for item in current)
    if _classify_content_text(previous_text) != "paragraph" or _classify_content_text(current_text) != "paragraph":
        return False

    previous_bottom = max(item["bbox"][3] for item in previous)
    current_top = min(item["bbox"][1] for item in current)
    return current_top - previous_bottom < 36


def _line_content_type(text: str) -> str:
    compact = text.replace(" ", "")
    if any(keyword in compact for keyword in ["예제", "보기", "따라하기", "풀이", "해설", "생각열기"]):
        return "paragraph"
    if _looks_like_long_sentence(text) or _looks_like_sentence_with_math(text):
        return "paragraph"
    if _looks_like_unit_note(text) or _looks_like_caption(text):
        return "caption"
    if _looks_like_formula(text):
        return "formula"
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if digit_ratio > 0.25 or any(keyword in text for keyword in ["연도", "예산", "비중", "증감", "국가"]):
        return "table"
    return "paragraph"


def _block_from_lines(
    lines: List[Dict], role: str, container_id: str, group_index: int, container_bbox: List[int]
) -> Optional[Dict]:
    lines = _sort_ocr_lines_for_text(lines)
    text = "\n".join(line.get("text", "") for line in lines).strip()
    if not text:
        return None
    if text.isdigit() and len(text) <= 2:
        return None
    x1 = min(line["bbox"][0] for line in lines)
    y1 = min(line["bbox"][1] for line in lines)
    x2 = max(line["bbox"][2] for line in lines)
    y2 = max(line["bbox"][3] for line in lines)
    block_type = _classify_content_text(text)
    local_role = _infer_role(text)
    compact = text.replace(" ", "")
    if compact in {"풀이", "해설", "정답"}:
        x1 = min(x1, container_bbox[0] + 8)
        x2 = max(x2, container_bbox[2] - 8)
        y2 = max(y2, container_bbox[3] - 8)
        block_type = "formula"
        local_role = "solution"
    return {
        "type": block_type,
        "role": local_role if local_role != "normal" else role,
        "container_id": container_id,
        "bbox": [x1, y1, x2, y2],
        "text": text,
        "score": 0.50,
        "group_index": group_index,
    }


def _is_short_role_title(text: str) -> bool:
    compact = text.replace(" ", "")
    if len(compact) > 16:
        return False
    role_titles = ["예제", "보기", "따라하기", "풀이", "해설", "정답", "문제", "확인문제", "연습문제", "생각열기"]
    return any(compact.startswith(title) for title in role_titles)


def _promote_role_title_text(text: str) -> str:
    text_lines = [line.strip() for line in text.splitlines() if line.strip()]
    title_lines = [line for line in text_lines if _is_short_role_title(line)]
    if not title_lines:
        return text
    return "\n".join(title_lines + [line for line in text_lines if line not in title_lines])


def _normalize_paragraph_text_order(blocks: List[Dict]) -> List[Dict]:
    for block in blocks:
        if block["type"] == "paragraph" and block.get("text"):
            block["text"] = _promote_role_title_text(block["text"])
    return blocks


def _split_answer_lines_from_paragraphs(blocks: List[Dict], ocr_lines: List[Dict]) -> List[Dict]:
    """Separate a final answer line from the explanation above it."""
    result: List[Dict] = []
    for block in blocks:
        if block["type"] != "paragraph":
            result.append(block)
            continue

        text_parts = _split_answer_text(block.get("text", ""))
        if text_parts is None:
            result.append(block)
            continue

        inner_lines = _lines_inside(ocr_lines, block["bbox"])
        answer_lines = [line for line in inner_lines if _is_answer_line(line.get("text", ""))]
        content_lines = [line for line in inner_lines if line not in answer_lines]
        content_block = dict(block)
        answer_block = dict(block)
        if answer_lines and content_lines:
            content_block["bbox"] = _bbox_for_lines(content_lines)
            content_block["text"] = "\n".join(
                line.get("text", "").strip()
                for line in _sort_ocr_lines_for_text(content_lines)
                if line.get("text", "").strip()
            )
            answer_block["bbox"] = _bbox_for_lines(answer_lines)
            answer_block["text"] = "\n".join(
                line.get("text", "").strip()
                for line in _sort_ocr_lines_for_text(answer_lines)
                if line.get("text", "").strip()
            )
            answer_score = min(max(float(line.get("score", 0.45)) for line in answer_lines), 0.60)
        else:
            content_text, answer_text = text_parts
            x1, y1, x2, y2 = block["bbox"]
            source_lines = [line for line in block.get("text", "").splitlines() if line.strip()]
            estimated_line_height = max(12, (y2 - y1) // max(len(source_lines), 2))
            split_y = max(y1 + 1, y2 - estimated_line_height)
            content_block["bbox"] = [x1, y1, x2, split_y]
            content_block["text"] = content_text
            answer_block["bbox"] = [x1, split_y, x2, y2]
            answer_block["text"] = answer_text
            answer_score = min(float(block.get("score", 0.45)), 0.60)
        result.append(content_block)

        answer_context = dict(block.get("context") or {})
        answer_context["semantic_role"] = "answer"
        answer_block["type"] = "paragraph"
        answer_block["score"] = answer_score
        answer_block["detector"] = "ocr_answer_recovery"
        answer_block["context"] = answer_context
        result.append(answer_block)
    return result


def _is_answer_line(text: str) -> bool:
    compact = text.strip().replace(" ", "")
    return bool(re.match(r"^(?:정답|답)(?:[\(（①-⑳⑴-⒇]|:|：)", compact))


def _split_answer_text(text: str) -> Optional[tuple[str, str]]:
    match = re.search(
        r"(?:^|\n)\s*((?:정답|답)\s*(?:[\(（①-⑳⑴-⒇]|:|：).*)$",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return None
    content = text[: match.start(1)].strip()
    answer = match.group(1).strip()
    if not content or not answer:
        return None
    return content, answer


def _bbox_for_lines(lines: List[Dict]) -> List[int]:
    return [
        min(line["bbox"][0] for line in lines),
        min(line["bbox"][1] for line in lines),
        max(line["bbox"][2] for line in lines),
        max(line["bbox"][3] for line in lines),
    ]


def _merge_numbered_items_with_parallel_explanations(blocks: List[Dict]) -> List[Dict]:
    """Merge a numbered prompt with its explanation in the parallel right column."""
    result = [dict(block) for block in blocks]
    consumed = set()

    for left_index, left in enumerate(result):
        if left_index in consumed or left["type"] != "paragraph" or not _starts_with_numbered_item(left.get("text", "")):
            continue
        if (left.get("context") or {}).get("semantic_role") == "answer":
            continue

        lx1, ly1, lx2, ly2 = left["bbox"]
        left_width = max(1, lx2 - lx1)
        left_height = max(1, ly2 - ly1)
        primary_candidates = []
        for right_index, right in enumerate(result):
            if right_index == left_index or right_index in consumed or right["type"] not in {
                "paragraph",
                "title",
                "section_title",
            }:
                continue
            if (right.get("context") or {}).get("semantic_role") == "answer":
                continue
            if _starts_with_numbered_item(right.get("text", "")):
                continue
            if _is_short_role_title(right.get("text", "")):
                continue
            rx1, ry1, rx2, ry2 = right["bbox"]
            horizontal_gap = rx1 - lx2
            vertical_overlap = min(ly2, ry2) - max(ly1, ry1)
            overlap_ratio = vertical_overlap / max(1, min(left_height, ry2 - ry1))
            if 0 <= horizontal_gap <= left_width * 1.5 and overlap_ratio >= 0.25:
                primary_candidates.append((horizontal_gap + abs(ly1 - ry1) * 0.25, right_index))

        if not primary_candidates:
            continue

        _, primary_index = min(primary_candidates)
        primary = result[primary_index]
        merged_indices = [left_index, primary_index]
        px1, py1, px2, py2 = primary["bbox"]

        for continuation_index, continuation in enumerate(result):
            if continuation_index in merged_indices or continuation_index in consumed:
                continue
            if continuation["type"] not in {"paragraph", "title", "section_title"} or _starts_with_numbered_item(
                continuation.get("text", "")
            ):
                continue
            if _is_short_role_title(continuation.get("text", "")):
                continue
            if (continuation.get("context") or {}).get("semantic_role") == "answer":
                continue
            cx1, cy1, _, _ = continuation["bbox"]
            vertical_gap = cy1 - py2
            within_row_band = cy1 <= ly2 + max(left_height, 40)
            if abs(cx1 - px1) <= 80 and -8 <= vertical_gap <= 24 and within_row_band:
                merged_indices.append(continuation_index)

        ordered_right = sorted(
            [index for index in merged_indices if index != left_index],
            key=lambda index: (result[index]["bbox"][1], result[index]["bbox"][0]),
        )
        merged_blocks = [left] + [result[index] for index in ordered_right]
        left["bbox"] = [
            min(block["bbox"][0] for block in merged_blocks),
            min(block["bbox"][1] for block in merged_blocks),
            max(block["bbox"][2] for block in merged_blocks),
            max(block["bbox"][3] for block in merged_blocks),
        ]
        left["text"] = _join_paragraph_fragments(
            [block.get("text", "").strip() for block in merged_blocks if block.get("text", "").strip()]
        )
        left["detector"] = "parallel_paragraph_merge"
        consumed.update(index for index in merged_indices if index != left_index)

    return [block for index, block in enumerate(result) if index not in consumed]


def _join_paragraph_fragments(parts: List[str]) -> str:
    if not parts:
        return ""
    merged = parts[0]
    for part in parts[1:]:
        previous_char = merged.rstrip()[-1:]
        next_char = part.lstrip()[:1]
        wrapped_korean_word = (
            "\uac00" <= previous_char <= "\ud7a3"
            and "\uac00" <= next_char <= "\ud7a3"
            and not merged.rstrip().endswith((".", "!", "?", "。", ":"))
        )
        merged = f"{merged.rstrip()}{part.lstrip()}" if wrapped_korean_word else f"{merged.rstrip()}\n{part.lstrip()}"
    return merged


def _merge_side_badges_into_paragraphs(blocks: List[Dict]) -> List[Dict]:
    """Attach small textbook role badges to the paragraph beside them."""
    result = [dict(block) for block in blocks]
    removed_ids = set()

    for badge_index, badge in enumerate(result):
        if badge["type"] not in {"title", "section_title", "caption", "footer"}:
            continue
        badge_text = badge.get("text", "").strip()
        if len(badge_text.replace(" ", "")) > 16:
            continue
        if _looks_like_source_caption_text(badge_text):
            continue

        bx1, by1, bx2, by2 = badge["bbox"]
        badge_width = bx2 - bx1
        badge_height = by2 - by1
        badge_area = max(1, badge_width * badge_height)
        candidates = []

        for paragraph_index, paragraph in enumerate(result):
            if paragraph_index == badge_index or paragraph["type"] != "paragraph":
                continue
            px1, py1, px2, py2 = paragraph["bbox"]
            paragraph_width = px2 - px1
            paragraph_height = py2 - py1
            paragraph_area = max(1, paragraph_width * paragraph_height)
            horizontal_gap = px1 - bx2
            vertical_overlap = min(by2, py2) - max(by1, py1)
            same_row = vertical_overlap / max(1, min(badge_height, paragraph_height)) >= 0.25
            vertical_gap = py1 - by2
            horizontal_overlap = min(bx2, px2) - max(bx1, px1)
            directly_above = (
                0 <= vertical_gap <= 40
                and horizontal_overlap / max(1, badge_width) >= 0.25
            )
            contained_at_top = (
                px1 - 24 <= bx1
                and bx2 <= px2 + 24
                and py1 - 20 <= by1
                and by2 <= py1 + min(paragraph_height * 0.60, 120)
            )
            small_relative_to_paragraph = (
                badge_area / paragraph_area <= 0.22
                and badge_width <= paragraph_width * 0.40
                and badge_height <= paragraph_height * 1.35
            )
            negative_gap_limit = -max(24, badge_width * 1.25)
            beside_paragraph = same_row and negative_gap_limit <= horizontal_gap <= 90
            if not small_relative_to_paragraph or not (beside_paragraph or directly_above or contained_at_top):
                continue
            distance = 0 if contained_at_top else vertical_gap if directly_above else abs(horizontal_gap) + abs(by1 - py1) * 0.25
            candidates.append((distance, paragraph_index))

        if not candidates:
            continue

        _, paragraph_index = min(candidates)
        paragraph = result[paragraph_index]
        px1, py1, px2, py2 = paragraph["bbox"]
        paragraph["bbox"] = [min(bx1, px1), min(by1, py1), max(bx2, px2), max(by2, py2)]
        if badge_text and badge_text not in paragraph.get("text", ""):
            paragraph["text"] = "\n".join(part for part in [badge_text, paragraph.get("text", "")] if part).strip()

        role_hint = _infer_role(badge_text)
        context = dict(paragraph.get("context") or {})
        context["label_source"] = "side_badge"
        if badge_text:
            context["label_text"] = badge_text
        if role_hint != "normal":
            context["role_hint"] = role_hint
        paragraph["context"] = context
        removed_ids.add(badge_index)

    return [block for index, block in enumerate(result) if index not in removed_ids]


def _classify_content_text(text: str) -> str:
    if _looks_like_choice_or_answer_list(text):
        return "paragraph"
    if _looks_like_formula_block(text):
        return "formula"
    if _looks_like_table_text(text):
        return "table"
    if _looks_like_formula(text):
        return "formula"
    if _looks_like_unit_note(text) or _looks_like_caption(text):
        return "caption"
    if _looks_like_table_title(text):
        return "caption"
    return "paragraph"


def _normalize_content_blocks(blocks: List[Dict]) -> List[Dict]:
    normalized: List[Dict] = []
    for block in blocks:
        block = dict(block)
        block_type = block.get("type", "paragraph")
        role_hint = block.pop("role", None)

        if block_type in ROLE_TYPES:
            role_hint = role_hint or ROLE_TYPES[block_type]
            block_type = _classify_content_text(block.get("text", ""))

        if block_type in {"graph", "image"}:
            block_type = "figure"

        if block_type not in TARGET_CLASSES:
            block_type = _classify_content_text(block.get("text", ""))

        inferred_role = _infer_role(block.get("text", ""))
        if inferred_role != "normal":
            role_hint = inferred_role

        block["type"] = block_type
        if role_hint and role_hint != "normal":
            context = dict(block.get("context") or {})
            context["role_hint"] = role_hint
            block["context"] = context

        normalized.append(block)
    return _merge_and_filter(normalized)


def _looks_like_table_title(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        return False
    compact = lines[0]
    if len(compact) > 80:
        return False
    return any(keyword in compact for keyword in ["비교", "표", "현황", "자료", "배분", "성장률"]) and any(
        char.isdigit() for char in compact
    )


def _drop_redundant_text_lines(blocks: List[Dict]) -> List[Dict]:
    result = []
    for block in blocks:
        if block["type"] == "caption":
            redundant = any(
                other is not block
                and other["type"] == "paragraph"
                and _intersection_over_area(block["bbox"], other["bbox"]) >= 0.65
                and block.get("text", "")
                and block.get("text", "")[:20] in other.get("text", "")
                for other in blocks
            )
            if redundant:
                continue
        result.append(block)
    return result


def _collect_text(ocr_lines: List[Dict], bbox: List[int]) -> str:
    x1, y1, x2, y2 = bbox
    lines = []
    for line in ocr_lines:
        lx1, ly1, lx2, ly2 = line["bbox"]
        cx = (lx1 + lx2) / 2
        cy = (ly1 + ly2) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            lines.append(line)
    lines.sort(key=lambda line: (line["bbox"][1], line["bbox"][0]))
    return "\n".join(line["text"] for line in lines)


def _covered_by_existing(bbox: List[int], blocks: List[Dict]) -> bool:
    for block in blocks:
        if _iou(bbox, block["bbox"]) > 0.70 or _inside(bbox, block["bbox"]):
            return True
    return False


def _inside(inner: List[int], outer: List[int]) -> bool:
    return inner[0] >= outer[0] and inner[1] >= outer[1] and inner[2] <= outer[2] and inner[3] <= outer[3]


def _merge_and_filter(blocks: List[Dict]) -> List[Dict]:
    cleaned = []
    for block in blocks:
        x1, y1, x2, y2 = [int(v) for v in block["bbox"]]
        if x2 <= x1 or y2 <= y1:
            continue
        block["bbox"] = [x1, y1, x2, y2]
        if block["type"] in {"graph", "image"}:
            block["type"] = "figure"
        elif block["type"] in ROLE_TYPES:
            block["role"] = block.get("role") or ROLE_TYPES[block["type"]]
            block["type"] = _classify_content_text(block.get("text", ""))
        if block["type"] not in TARGET_CLASSES:
            block["type"] = "paragraph"
        cleaned.append(block)

    cleaned.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], -(item["bbox"][2] - item["bbox"][0])))
    result: List[Dict] = []
    for block in cleaned:
        duplicate = False
        replaced = False
        for index, kept in enumerate(result):
            overlap = _iou(block["bbox"], kept["bbox"])
            if _should_drop_nested_block(block, kept):
                duplicate = True
                break
            if _should_drop_nested_block(kept, block):
                result[index] = block
                replaced = True
                break
            if overlap > 0.80 or _mostly_inside(block["bbox"], kept["bbox"]) or _mostly_inside(kept["bbox"], block["bbox"]):
                if _prefer_block(block, kept):
                    result[index] = block
                    replaced = True
                else:
                    duplicate = True
                break
        if not duplicate and not replaced:
            result.append(block)
    return result


def _should_drop_nested_block(inner: Dict, outer: Dict) -> bool:
    if outer["type"] != "paragraph":
        return False
    if inner["type"] not in {"title", "section_title", "formula", "caption"}:
        return False
    if not _mostly_inside(inner["bbox"], outer["bbox"], threshold=0.62):
        return False

    area_ratio = _area(inner["bbox"]) / max(_area(outer["bbox"]), 1)
    outer_text = outer.get("text", "")
    role_hint = (outer.get("context") or {}).get("role_hint")
    mixed_explanation = len(outer_text.replace(" ", "")) >= 25 or role_hint in {"example", "solution", "problem"}
    return mixed_explanation and area_ratio < 0.55


def _prefer_block(candidate: Dict, current: Dict) -> bool:
    if _is_role_region_paragraph(candidate) and current["type"] in {"title", "section_title", "formula", "caption"}:
        return True
    if _is_role_region_paragraph(current) and candidate["type"] in {"title", "section_title", "formula", "caption"}:
        return False

    candidate_type = _resolve_overlap_type(candidate, current)
    current_type = _resolve_overlap_type(current, candidate)
    if candidate_type != candidate["type"] and current_type == current["type"]:
        return False
    if current_type != current["type"] and candidate_type == candidate["type"]:
        return True
    return _type_priority(candidate_type) + float(candidate.get("score", 0.0)) > _type_priority(
        current_type
    ) + float(current.get("score", 0.0))


def _is_role_region_paragraph(block: Dict) -> bool:
    return block.get("detector") == "role_region_supplement" and block.get("type") == "paragraph"


def _resolve_overlap_type(block: Dict, other: Dict) -> str:
    types = {block["type"], other["type"]}
    text = block.get("text") or other.get("text") or ""
    if types == {"table", "formula"}:
        return "table" if _looks_like_table_text(text) or _area(block["bbox"]) > 250000 else "formula"
    if types == {"paragraph", "figure"}:
        return "paragraph" if _looks_like_paragraph_text(text) else block["type"]
    if types == {"paragraph", "caption"}:
        return "paragraph" if _looks_like_paragraph_text(text) else block["type"]
    return block["type"]


def _type_priority(block_type: str) -> float:
    priorities = {
        "table": 90,
        "formula": 80,
        "figure": 72,
        "title": 65,
        "section_title": 64,
        "paragraph": 62,
        "caption": 60,
        "footer": 20,
        "page_number": 10,
    }
    return priorities.get(block_type, 0)


def _mostly_inside(inner: List[int], outer: List[int], threshold: float = 0.85) -> bool:
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(_area(inner), 1) >= threshold


def _intersection_over_area(inner: List[int], outer: List[int]) -> float:
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(_area(inner), 1)


def _area(bbox: List[int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _iou(a: List[int], b: List[int]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / max(_area(a) + _area(b) - inter, 1)
