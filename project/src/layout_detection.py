from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np


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
) -> List[Dict]:
    if yolo_model_path:
        try:
            return _postprocess_blocks(_detect_with_yolo(image_path, yolo_model_path), None, None)
        except Exception:
            pass

    return _detect_with_heuristics(image_path, ocr_lines or [])


def refine_blocks_after_ocr(blocks: List[Dict], ocr_lines: Optional[List[Dict]] = None) -> List[Dict]:
    processed = _postprocess_blocks(blocks, None, None)
    split = _split_mixed_role_blocks(processed, ocr_lines or [])
    return _normalize_content_blocks(_postprocess_blocks(split, None, None))


def _detect_with_yolo(image_path: str | Path, model_path: str | Path) -> List[Dict]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
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


def _map_external_label(label: str) -> str:
    normalized = label.lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "text": "paragraph",
        "plain_text": "paragraph",
        "header": "title",
        "figure": "figure",
        "picture": "figure",
        "image": "figure",
        "graph": "figure",
        "equation": "formula",
        "formula_box": "formula",
        "table_caption": "caption",
        "figure_caption": "caption",
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
    if _looks_like_unit_note(text):
        return False
    if _looks_like_sentence_with_math(text):
        return False
    formula_chars = ["=", "+", "-", "×", "÷", "∑", "√", "≤", "≥", "(", ")", "^", "lim", "log"]
    math_hits = sum(1 for char in formula_chars if char in text)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    return math_hits >= 2 or (math_hits >= 1 and digit_ratio > 0.20)


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


def _looks_like_table_text(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) < 5:
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


def _looks_like_graph(binary_roi: np.ndarray, aspect: float) -> bool:
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 45))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 1))
    vertical = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, v_kernel)
    horizontal = cv2.morphologyEx(binary_roi, cv2.MORPH_OPEN, h_kernel)
    has_axes = np.count_nonzero(vertical) > 80 and np.count_nonzero(horizontal) > 80
    return has_axes and 0.5 <= aspect <= 2.5


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
        role = _infer_role(text)
        if role != "normal":
            block["role"] = role
        compact = text.replace(" ", "")
        if _looks_like_header_box(text):
            block["type"] = "section_title"
        elif block["type"] in {"title", "section_title", "formula"} and _looks_like_long_sentence(text):
            block["type"] = "paragraph"
        elif block["type"] == "formula" and (_looks_like_unit_note(text) or _looks_like_sentence_with_math(text)):
            block["type"] = "caption" if _looks_like_unit_note(text) else "paragraph"
        elif block["type"] == "formula" and _looks_like_unit_note(text):
            block["type"] = "caption"
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
        if len(inner_lines) < 4:
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
        if len(group) == 1 and merged and _line_content_type(group[0].get("text", "")) == "caption":
            merged[-1].extend(group)
        else:
            merged.append(group)
    return merged


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


def _classify_content_text(text: str) -> str:
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


def _prefer_block(candidate: Dict, current: Dict) -> bool:
    candidate_type = _resolve_overlap_type(candidate, current)
    current_type = _resolve_overlap_type(current, candidate)
    if candidate_type != candidate["type"] and current_type == current["type"]:
        return False
    if current_type != current["type"] and candidate_type == candidate["type"]:
        return True
    return _type_priority(candidate_type) + float(candidate.get("score", 0.0)) > _type_priority(
        current_type
    ) + float(current.get("score", 0.0))


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
