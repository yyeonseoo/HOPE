from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from src.analysis.formula.formula_recognizer import contains_formula_signal

from .crop import crop_and_save_table_block, crop_table_image
from .description import generate_table_description
from .engine import MODEL_NAME, MODEL_VERSION, _load_table_engine, run_table_engine
from .normalize import build_table_analysis

# Layout blocks the layout model classified as something other than
# "table" but which are worth re-checking with the table-structure engine
# anyway, in case the layout pipeline misclassified an actual table. Only
# `formula` is reclassifiable: `figure` was tried too, but a full 17-page
# real-PDF batch review (see /outputs review artifact, 2026-07-07) showed
# it kept promoting coordinate-plane graphs to "table" -- a distance-time
# graph's axis labels, then a y=a/x graph's equation/coordinate annotation,
# then a CO2-concentration line chart's legend/axis text each slipped past
# a different round of content-signal checks below. Graphs in this
# textbook are varied enough that no fixed heuristic reliably tells them
# apart from a real table, so figure reclassification was turned off
# entirely rather than keep chasing one more false-positive pattern.
# `formula` blocks are small equation crops -- a large, dense, real-looking
# grid coming out of one is much rarer, so it's kept.
RECLASSIFIABLE_TYPES = {"formula"}
_MIN_RECLASSIFIED_ROWS = 2
_MIN_RECLASSIFIED_COLUMNS = 2
# Real tables have most of their cells filled in. Charts/graphs with axis
# gridlines can get misread by the table-structure engine as a sparse grid
# with only a couple of axis-tick numbers as "cell text" -- requiring most
# cells to have text rejects that false-positive pattern while still
# allowing a real table with a handful of blank/partial cells.
_MIN_RECLASSIFIED_FILLED_RATIO = 0.6

# This textbook leans heavily on distance-time/speed-time coordinate-plane
# graphs, whose axis labels and origin marker get OCR'd as short standalone
# tokens ("거리", "시간", "O") that can end up looking like a dense, clean
# grid once several such mini-graphs sit side by side (see p2_b3/p2_b6/
# p2_b13 in real output: cells like "2 거리↑ 0", "시간", "(3) 거리↑"). A
# real data table essentially never has a cell that's just one of these bare
# axis words, so any hit here is treated as graph-annotation noise.
_GRAPH_AXIS_LABEL_WORDS = ("거리", "시간", "속력", "속도", "높이", "온도", "무게", "넓이", "부피")


def analyze_table_blocks(
    page: Dict[str, Any],
    page_image_path: Optional[str] = None,
    lang: str = "korean",
    engine=None,
) -> List[Dict[str, Any]]:
    """Take a full page result (as produced by the layout pipeline: `{"page_id":
    ..., "blocks": [...]}`) and return a list of complete, schema-conformant
    `semantic_analyses` records — one per `table`-type block, plus any
    `figure`/`formula` block that re-checks as a real table (see
    RECLASSIFIABLE_TYPES) — ready to drop into the API response as-is.

    This mirrors feature/formula-analysis's `analyze_formula_blocks(page,
    page_image_path)` interface exactly, so the integration owner can call
    all three analyzers the same way instead of writing different glue code
    per module. Each record includes `detection` (from the block's own
    `score`/`detector`), `context` (previous/next block id by list-index
    adjacency in `page["blocks"]`, same convention as formula), and
    `crop_path` (saved via crop_and_save_table_block), in addition to the
    `analysis` result itself.
    """
    page_id = page.get("page_id")
    blocks = page.get("blocks", [])

    results = []
    for index, block in enumerate(blocks):
        block_type = block.get("type")
        if block_type == "table":
            results.append(
                _analyze_single_table_block(page_id, block, blocks, index, page_image_path, lang, engine)
            )
        elif block_type in RECLASSIFIABLE_TYPES:
            reclassified = _try_reclassify_as_table(
                page_id, block, blocks, index, page_image_path, lang, engine
            )
            if reclassified is not None:
                results.append(reclassified)

    return results


def _analyze_single_table_block(
    page_id: Optional[int],
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    page_image_path: Optional[str],
    lang: str,
    engine,
) -> Dict[str, Any]:
    bbox = block.get("bbox")

    if page_image_path and bbox:
        output = analyze_table_block(page_image_path, bbox, lang=lang, engine=engine)
    else:
        model = {"name": MODEL_NAME, "version": MODEL_VERSION}
        output = {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": ["페이지 이미지 또는 bbox가 없어 표 구조 인식을 실행하지 못했습니다."],
        }

    output = _flag_if_not_a_real_table(output)

    return _assemble_table_record(page_id, block, blocks, block_index, bbox, page_image_path, output)


def _flag_if_not_a_real_table(output: Dict[str, Any]) -> Dict[str, Any]:
    """For blocks the layout model already tagged `table` (unlike the
    figure/formula reclassification path, we can't just drop these -- the
    schema requires a `table`-type record for them), downgrade to a
    `failed` result with no cells when the recognized structure doesn't
    clear the same real-table bar used for reclassification (see
    `_looks_like_real_table`). We already don't trust this read enough to
    reclassify a figure/formula block on it -- presenting its cells here
    anyway (just with an extra warning bolted on) would show fabricated-
    looking table content the reviewer has no reason to trust either. This
    surfaces layout misclassifications (e.g. a fill-in-the-blank paragraph
    about "2배, 3배, 1/2배..." getting boxed as a table by the upstream
    layout model) as a clear failure with a warning explaining why, not a
    table that merely looks suspicious.
    """
    if output["analysis"].get("status") == "failed":
        return output
    if _looks_like_real_table(output["analysis"]):
        return output

    model = output["analysis"]["model"]
    return {
        "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
        "warnings": output["warnings"]
        + ["표 구조 인식 결과가 실제 표 형태로 보이지 않습니다. 레이아웃 분류가 잘못됐을 수 있어 검수가 필요합니다."],
    }


def _try_reclassify_as_table(
    page_id: Optional[int],
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    page_image_path: Optional[str],
    lang: str,
    engine,
) -> Optional[Dict[str, Any]]:
    """Re-run table-structure recognition on a block the layout pipeline
    labeled `formula` (see RECLASSIFIABLE_TYPES for why `figure` isn't
    included here anymore). Returns a full schema-conformant record (with
    `type` forced to `"table"`) if the result looks like a genuine table,
    or None if the caller should leave the block under its original type --
    e.g. it really was a formula, or the crop/engine failed.

    This is a table-branch-only prototype (see OWNERSHIP.md): it does not
    touch the shared layout postprocessing in page_pipeline.py, so
    reclassified blocks only ever show up here, in `semantic_analyses` --
    `page["blocks"]` itself is untouched and still says `formula`.
    """
    bbox = block.get("bbox")
    if not page_image_path or not bbox:
        return None

    output = analyze_table_block(page_image_path, bbox, lang=lang, engine=engine)

    if not _looks_like_real_table(output["analysis"]):
        return None

    original_type = block.get("type")
    output = {
        **output,
        "warnings": output["warnings"]
        + [
            f"레이아웃 모델이 이 블록을 '{original_type}'(으)로 분류했지만, "
            "표 구조 인식 결과 실제 표로 보여 table 후보로 재분류했습니다."
        ],
    }

    record = _assemble_table_record(page_id, block, blocks, block_index, bbox, page_image_path, output)
    # A reclassification is always a candidate, never a confirmed fact --
    # even the strict checks above are heuristics that a novel graph/figure
    # layout could still slip past. Force review regardless of what
    # generate_table_description's own (independent) review_status logic
    # concluded, so a human always signs off before this is trusted.
    record["description"]["review_status"] = "needs_review"
    return record


def _looks_like_real_table(
    analysis: Dict[str, Any], min_filled_ratio: float = _MIN_RECLASSIFIED_FILLED_RATIO
) -> bool:
    """Confidence bar for reclassifying a non-table block as a table: a
    non-failed status, at least a 2x2 grid, and most cells actually filled
    in. A 1x1 grid, an all-blank grid, or a sparse grid with only a couple
    of filled cells is more likely to be the table engine misreading a
    chart's axis gridlines/tick labels as a table than a real missed table,
    so those are rejected rather than reclassified.

    Shared by both the formula-reclassification path and
    `_flag_if_not_a_real_table`'s symmetric check on blocks already tagged
    `table` by the layout model.
    """
    if analysis.get("status") == "failed":
        return False
    result = analysis.get("result")
    if not result:
        return False
    if result.get("row_count", 0) < _MIN_RECLASSIFIED_ROWS:
        return False
    if result.get("column_count", 0) < _MIN_RECLASSIFIED_COLUMNS:
        return False

    cells = result.get("cells", [])
    if not cells:
        return False

    filled_cells = [cell for cell in cells if cell.get("text")]
    filled_ratio = len(filled_cells) / len(cells)
    if filled_ratio < min_filled_ratio:
        return False

    # A coordinate-plane graph can pack several text annotations (axis
    # labels, an equation like "y=ax", a labeled point like "(1, a)") into
    # a small area that happens to line up into a dense-enough grid to pass
    # the filled-ratio check above. Real data-table cells essentially never
    # contain equation/coordinate-pair syntax, so any hit here means this is
    # much more likely the table engine misreading a figure's annotations
    # than an actual missed table -- reuses feature/formula-analysis's own
    # formula-signal detector rather than re-deriving the same patterns.
    if any(contains_formula_signal(cell["text"]) for cell in filled_cells):
        return False

    if any(_contains_graph_axis_label(cell["text"]) for cell in filled_cells):
        return False

    return True


def _contains_graph_axis_label(text: str) -> bool:
    return any(word in text for word in _GRAPH_AXIS_LABEL_WORDS)


def _assemble_table_record(
    page_id: Optional[int],
    block: Dict[str, Any],
    blocks: List[Dict[str, Any]],
    block_index: int,
    bbox: Optional[List[int]],
    page_image_path: Optional[str],
    output: Dict[str, Any],
) -> Dict[str, Any]:
    crop_path = crop_and_save_table_block(page_image_path, block, page_id)
    previous_block_id = get_neighbor_block_id(blocks, block_index - 1)
    next_block_id = get_neighbor_block_id(blocks, block_index + 1)

    return {
        "schema_version": "1.0.0",
        "page_id": page_id,
        "block_id": block.get("block_id"),
        "type": "table",
        "bbox": bbox,
        "crop_path": crop_path,
        "detection": {
            "model": {"name": block.get("detector", "model-a"), "version": None},
            "confidence": block.get("score"),
        },
        "analysis": output["analysis"],
        "description": generate_table_description(output["analysis"]),
        "context": {
            "previous_block_id": previous_block_id,
            "next_block_id": next_block_id,
            "caption_block_id": None,
            "nearby_block_ids": [
                block_id for block_id in [previous_block_id, next_block_id] if block_id is not None
            ],
        },
        "warnings": output["warnings"],
    }


def get_neighbor_block_id(blocks: List[Dict[str, Any]], index: int) -> Optional[str]:
    """Return the block_id at `index` in `blocks`, or None if out of range.
    Same convention as feature/formula-analysis's `get_neighbor_block_id`."""
    if index < 0 or index >= len(blocks):
        return None
    return blocks[index].get("block_id")


def analyze_table_block(
    image_path: str | Path,
    bbox: List[int],
    lang: str = "korean",
    engine=None,
) -> Dict:
    """Crop `bbox` out of the page image, run table-structure recognition,
    and return `{"analysis": {...}, "warnings": [...]}` matching the
    `analysis`/`warnings` portions of schemas/block_analysis.schema.json for
    a `table`-type block.

    Lower-level primitive used internally by `analyze_table_blocks` (and
    still usable directly/in tests). It never raises: any failure (bad crop,
    engine error, unparseable output) degrades to `analysis.status ==
    "failed"` with `result: None` and a `warnings` entry explaining why, so
    one broken table can't take down a whole page's processing.
    """
    model = {"name": MODEL_NAME, "version": MODEL_VERSION}

    crop = crop_table_image(image_path, bbox)
    if crop is None:
        return {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": ["표 영역 crop에 실패했습니다 (bbox가 이미지 범위를 벗어났거나 비어 있음)."],
        }

    try:
        active_engine = engine or _load_table_engine(lang=lang)
        raw_result = run_table_engine(active_engine, crop)
    except Exception as exc:  # noqa: BLE001 - never let one block kill the page
        return {
            "analysis": {"status": "failed", "model": model, "confidence": None, "result": None},
            "warnings": [f"표 구조 인식 중 오류가 발생했습니다: {exc}"],
        }

    return build_table_analysis(
        raw_result, model_name=MODEL_NAME, model_version=MODEL_VERSION, table_crop=crop
    )
