from __future__ import annotations

from typing import Dict, List


def _bbox_center_y(block: Dict) -> float:
    x1, y1, x2, y2 = block["bbox"]
    return (y1 + y2) / 2


def _bbox_x1(block: Dict) -> float:
    return block["bbox"][0]


def sort_reading_order(blocks: List[Dict], y_tolerance: int = 24) -> List[Dict]:
    """Sort blocks top-to-bottom, then left-to-right within the same visual row."""
    indexed = list(enumerate(blocks))
    indexed.sort(key=lambda item: (_bbox_center_y(item[1]), _bbox_x1(item[1])))

    rows: List[List[tuple[int, Dict]]] = []
    for item in indexed:
        block = item[1]
        if not rows:
            rows.append([item])
            continue

        row_y = sum(_bbox_center_y(row_item[1]) for row_item in rows[-1]) / len(rows[-1])
        if abs(_bbox_center_y(block) - row_y) <= y_tolerance:
            rows[-1].append(item)
        else:
            rows.append([item])

    ordered: List[Dict] = []
    for row in rows:
        row.sort(key=lambda item: _bbox_x1(item[1]))
        ordered.extend(block for _, block in row)

    for order, block in enumerate(ordered, start=1):
        block["reading_order"] = order
    return ordered
