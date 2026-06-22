from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def build_page_result(page_id: int, blocks: List[Dict]) -> Dict:
    for index, block in enumerate(blocks, start=1):
        block.setdefault("block_id", f"p{page_id}_b{index}")
    return {"page_id": page_id, "blocks": blocks}


def export_pages(pages: List[Dict], output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump({"pages": pages}, file, ensure_ascii=False, indent=2)
    return output_path
