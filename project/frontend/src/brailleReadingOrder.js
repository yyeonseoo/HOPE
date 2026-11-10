const BLOCK_TYPE_LABELS = {
  title: "제목",
  section_title: "절 제목",
  section_header: "절 제목",
  paragraph: "본문",
  formula: "수식",
  table: "표",
  figure: "그림",
  caption: "원문 캡션",
  footer: "쪽 정보",
  page_number: "원본 쪽수",
};

const FIGURE_TYPE_LABELS = {
  graph: "그래프",
  line_chart: "선그래프",
  bar_chart: "막대그래프",
  pie_chart: "원그래프",
  scatter_plot: "산점도",
  table: "표",
  mathematical_diagram: "수학 도식",
  diagram: "도식",
  illustration: "삽화",
  photo: "사진",
  icon: "아이콘",
  other: "그림",
  unknown: "그림",
};

export function blockTypeLabel(type) {
  return BLOCK_TYPE_LABELS[type] || "내용";
}

export function figureTypeLabel(analysis) {
  const figureType = analysis?.figure_type || analysis?.analysis?.result?.figure_type;
  return FIGURE_TYPE_LABELS[figureType] || "그림";
}

// Human names for the review UI: "그림 3 (그래프)" in braille reading order,
// instead of internal ids like p6_b8.
export function figureDisplayNames(blocks = [], analyses = []) {
  const analysisByBlock = new Map(analyses.map((item) => [item.block_id, item]));
  const names = new Map();
  sortBlocksForBrailleReading(blocks)
    .filter((block) => block.type === "figure")
    .forEach((block, index) => {
      names.set(block.block_id, `그림 ${index + 1} (${figureTypeLabel(analysisByBlock.get(block.block_id))})`);
    });
  return names;
}

function numericReadingOrder(block) {
  if (block?.reading_order === null || block?.reading_order === undefined) return null;
  const value = Number(block?.reading_order);
  return Number.isFinite(value) ? value : null;
}

export function sortBlocksForBrailleReading(blocks = []) {
  return blocks
    .map((block, sourceIndex) => ({ block, sourceIndex }))
    .sort((left, right) => {
      const leftOrder = numericReadingOrder(left.block);
      const rightOrder = numericReadingOrder(right.block);

      if (leftOrder !== null && rightOrder !== null && leftOrder !== rightOrder) {
        return leftOrder - rightOrder;
      }
      if (leftOrder !== null && rightOrder === null) return -1;
      if (leftOrder === null && rightOrder !== null) return 1;
      return left.sourceIndex - right.sourceIndex;
    })
    .map(({ block }) => block);
}

// Reviewer choice wins; otherwise the backend's rule-based suggestion
// (src/analysis/figure/treatment.py); otherwise a transcriber's note.
export function figureVisualTreatment(analysis) {
  const review = analysis?.braille_review;
  return review?.visual_treatment
    || review?.production_mode
    || review?.suggested_visual_treatment
    || "transcriber_note";
}

export function tableOmitted(analysis) {
  const review = analysis?.braille_review;
  return (review?.visual_treatment || review?.suggested_visual_treatment) === "omit";
}

export function linkedFigureCaptions(blocks = [], analyses = []) {
  const blockById = new Map(blocks.map((block) => [block.block_id, block]));
  const captionByFigureId = new Map();
  const linkedCaptionIds = new Set();

  analyses.forEach((analysis) => {
    if (analysis?.type !== "figure") return;
    const captionId = analysis?.context_source?.caption_block_id
      || analysis?.context?.caption_block_id;
    const caption = blockById.get(captionId);
    if (!caption || caption.type !== "caption") return;

    captionByFigureId.set(analysis.block_id, caption);
    linkedCaptionIds.add(captionId);
  });

  return { captionByFigureId, linkedCaptionIds };
}

function readingGroups(blocks = [], analyses = []) {
  const orderedBlocks = sortBlocksForBrailleReading(blocks);
  const { captionByFigureId, linkedCaptionIds } = linkedFigureCaptions(
    orderedBlocks,
    analyses,
  );

  return orderedBlocks
    .filter((block) => !linkedCaptionIds.has(block.block_id))
    .map((block) => {
      const caption = captionByFigureId.get(block.block_id);
      return caption ? [block, caption] : [block];
    });
}

function readingGroupsForInteraction(blocks = [], analyses = [], ...blockIds) {
  const groups = readingGroups(blocks, analyses);
  const requiresIndependentBlocks = blockIds.some((blockId) => (
    blockId
    && groups.some((group) => (
      group.some((block) => block.block_id === blockId)
      && group[0]?.block_id !== blockId
    ))
  ));
  return requiresIndependentBlocks
    ? sortBlocksForBrailleReading(blocks).map((block) => [block])
    : groups;
}

export function moveBrailleReadingGroup(blocks = [], analyses = [], blockId, direction) {
  const groups = readingGroupsForInteraction(blocks, analyses, blockId);
  const currentIndex = groups.findIndex((group) => group[0]?.block_id === blockId);
  if (currentIndex < 0) return blocks;

  const step = direction === "up" ? -1 : direction === "down" ? 1 : 0;
  if (!step) return blocks;

  let targetIndex = currentIndex + step;
  while (
    targetIndex >= 0
    && targetIndex < groups.length
    && groups[targetIndex][0]?.type === "page_number"
  ) {
    targetIndex += step;
  }
  if (targetIndex < 0 || targetIndex >= groups.length) return blocks;

  [groups[currentIndex], groups[targetIndex]] = [groups[targetIndex], groups[currentIndex]];
  const nextOrderById = new Map();
  groups.flat().forEach((block, index) => nextOrderById.set(block.block_id, index + 1));
  return blocks.map((block) => ({
    ...block,
    reading_order: nextOrderById.get(block.block_id) ?? block.reading_order,
  }));
}

export function placeBrailleReadingGroup(
  blocks = [],
  analyses = [],
  draggedBlockId,
  targetBlockId,
  placement = "before",
) {
  if (!draggedBlockId || !targetBlockId || draggedBlockId === targetBlockId) return blocks;
  const groups = readingGroupsForInteraction(
    blocks,
    analyses,
    draggedBlockId,
    targetBlockId,
  );
  const draggedIndex = groups.findIndex((group) => group[0]?.block_id === draggedBlockId);
  const targetIndex = groups.findIndex((group) => group[0]?.block_id === targetBlockId);
  if (draggedIndex < 0 || targetIndex < 0) return blocks;
  if (groups[draggedIndex][0]?.type === "page_number") return blocks;

  const [draggedGroup] = groups.splice(draggedIndex, 1);
  const insertionIndex = groups.findIndex((group) => group[0]?.block_id === targetBlockId);
  const targetInsertionIndex = insertionIndex < 0
    ? groups.length
    : insertionIndex + (placement === "after" ? 1 : 0);
  groups.splice(targetInsertionIndex, 0, draggedGroup);

  const nextOrderById = new Map();
  groups.flat().forEach((block, index) => nextOrderById.set(block.block_id, index + 1));
  return blocks.map((block) => ({
    ...block,
    reading_order: nextOrderById.get(block.block_id) ?? block.reading_order,
  }));
}

export function normalizeTaggedPageDescription(text = "") {
  const lines = String(text || "").split(/\r?\n/);
  const normalized = [];
  let taggedEntry = null;

  const flushTaggedEntry = () => {
    if (!taggedEntry) return;
    const content = taggedEntry.parts
      .map((part) => String(part || "").trim())
      .filter(Boolean)
      .join(" ")
      .replace(/\s+/g, " ")
      .trim();
    normalized.push(content
      ? `[${taggedEntry.type}] ${content}`
      : `[${taggedEntry.type}]`);
    taggedEntry = null;
  };

  lines.forEach((line) => {
    const match = line.match(/^\s*\[([a-z_]+)\]\s*(.*)$/i);
    if (match) {
      flushTaggedEntry();
      taggedEntry = {
        type: match[1].toLowerCase(),
        parts: [match[2]],
      };
      return;
    }

    if (taggedEntry) {
      if (line.trim()) taggedEntry.parts.push(line);
      return;
    }
    normalized.push(line);
  });
  flushTaggedEntry();
  return normalized.join("\n");
}

export function reorderTaggedPageDescription(text, previousBlocks = [], nextBlocks = []) {
  const lines = normalizeTaggedPageDescription(text).split(/\r?\n/);
  const previousByType = new Map();
  sortBlocksForBrailleReading(previousBlocks).forEach((block) => {
    if (!previousByType.has(block.type)) previousByType.set(block.type, []);
    previousByType.get(block.type).push(block.block_id);
  });

  const typeOffsets = new Map();
  const mappedLines = [];
  const mappedSlots = [];
  lines.forEach((line, lineIndex) => {
    const match = line.match(/^\s*\[([a-z_]+)\]\s*/i);
    if (!match) return;
    const type = match[1].toLowerCase();
    const offset = typeOffsets.get(type) || 0;
    const blockId = previousByType.get(type)?.[offset];
    typeOffsets.set(type, offset + 1);
    if (!blockId) return;
    mappedSlots.push(lineIndex);
    mappedLines.push({ blockId, line });
  });

  if (mappedLines.length < 2) return String(text || "");
  const nextOrder = new Map(
    sortBlocksForBrailleReading(nextBlocks)
      .map((block, index) => [block.block_id, index]),
  );
  mappedLines.sort(
    (left, right) => (
      (nextOrder.get(left.blockId) ?? Number.MAX_SAFE_INTEGER)
      - (nextOrder.get(right.blockId) ?? Number.MAX_SAFE_INTEGER)
    ),
  );
  const reorderedLines = [...lines];
  mappedSlots.forEach((slot, index) => {
    reorderedLines[slot] = mappedLines[index].line;
  });
  return reorderedLines.join("\n");
}

// The text block a figure followed, skipping captions and other figures
// (they travel with figures), so a block shifted by a moved figure does not
// count as moved itself.
function precedingBlock(ordered, blockId) {
  const index = ordered.findIndex((block) => block.block_id === blockId);
  for (let i = index - 1; i >= 0; i -= 1) {
    if (ordered[i].type !== "caption" && ordered[i].type !== "figure") return ordered[i];
  }
  return null;
}

// When a figure no longer follows the block it followed in print, braille
// notes its original position (NLD2022 3장 1절 1. 7)). Needs
// original_reading_order from the backend; older saved pages get no note.
export function figureRelocationNote(blocks = [], blockId) {
  if (!blocks.every((block) => Number.isFinite(Number(block.original_reading_order)))) return null;
  const original = [...blocks].sort(
    (a, b) => Number(a.original_reading_order) - Number(b.original_reading_order),
  );
  const before = precedingBlock(original, blockId);
  const now = precedingBlock(sortBlocksForBrailleReading(blocks), blockId);
  if ((before?.block_id ?? null) === (now?.block_id ?? null)) return null;
  if (!before) return "원래 이 그림은 이 쪽의 맨 앞에 있다.";
  const text = String(before.text || "").replace(/\s+/g, " ").trim();
  const excerpt = text.length > 20 ? `${text.slice(0, 20)}…` : text;
  return `원래 이 그림은 "${excerpt || before.block_id}" 다음에 있다.`;
}

const TREATMENT_LABELS = {
  tactile_graphic: "촉도",
  transcriber_note: "점역자 주",
  omit: "생략",
};

// Review checklist: what the reviewer still has to look at on this page.
// Each item carries the block id so the UI can jump to it.
export function brailleReadingWarnings(page, analyses = []) {
  const blocks = page?.blocks || [];
  const items = [];
  const orders = blocks
    .map(numericReadingOrder)
    .filter((order) => order !== null);
  if (orders.length !== blocks.length) {
    items.push({ blockId: null, kind: "order", message: "읽기 순서가 지정되지 않은 블록이 있습니다." });
  }
  if (new Set(orders).size !== orders.length) {
    items.push({ blockId: null, kind: "order", message: "같은 읽기 순서 번호를 가진 블록이 있습니다." });
  }

  const analysisByBlock = new Map(
    analyses.map((analysis) => [analysis.block_id, analysis]),
  );
  sortBlocksForBrailleReading(blocks)
    .filter((block) => block.type === "figure")
    .forEach((block) => {
      const analysis = analysisByBlock.get(block.block_id);
      const review = analysis?.braille_review || {};
      const treatment = figureVisualTreatment(analysis);
      const description = String(
        analysis?.description?.long_text || analysis?.description?.short_text || "",
      );
      if (!review.visual_treatment) {
        const reason = review.treatment_reason ? ` (${review.treatment_reason})` : "";
        items.push({
          blockId: block.block_id,
          kind: "treatment",
          message: `처리 방식 "${TREATMENT_LABELS[treatment]}" 제안을 확정하세요${reason}`,
        });
      }
      if (treatment === "transcriber_note" && !description.trim()) {
        items.push({ blockId: block.block_id, kind: "note", message: "점역자 주 내용이 없습니다." });
      }
      if (figureRelocationNote(blocks, block.block_id)) {
        items.push({
          blockId: block.block_id,
          kind: "moved",
          message: "원본과 다른 위치로 옮겨져 원래 위치를 알리는 점역자 주가 붙습니다. 문구를 확인하세요.",
        });
      }
      const unverified = (description.match(/\[확인 필요\]/g) || []).length;
      if (treatment !== "omit" && unverified) {
        items.push({
          blockId: block.block_id,
          kind: "verify",
          message: `원본과 대조할 값이 ${unverified}개 있습니다([확인 필요]).`,
        });
      }
    });

  blocks
    .filter((block) => block.type === "table")
    .forEach((block) => {
      const analysis = analysisByBlock.get(block.block_id);
      const cells = analysis?.analysis?.result?.cells;
      if (Array.isArray(cells) && cells.length && !cells.some((cell) => cell.is_header)) {
        items.push({ blockId: block.block_id, kind: "table", message: "행·열 머리글을 확인하세요." });
      }
      // Many unreadable cells usually means a screenshot, not a real table.
      const unreadable = (analysis?.warnings || [])
        .map((warning) => Number(String(warning).match(/(\d+)개 셀의 텍스트를 인식하지 못/)?.[1] || 0))
        .reduce((sum, count) => sum + count, 0);
      if (unreadable >= 10 && !analysis?.braille_review?.visual_treatment) {
        items.push({
          blockId: block.block_id,
          kind: tableOmitted(analysis) ? "treatment" : "table",
          message: tableOmitted(analysis)
            ? `셀 ${unreadable}개를 읽지 못해 "생략"을 제안했습니다. 진짜 표라면 "표로 되돌리기"를, 맞으면 그대로 확정하세요.`
            : `셀 ${unreadable}개를 읽지 못했습니다. 화면 캡처 같은 그림이면 "표 아님·생략"을 누르세요.`,
        });
      }
    });

  const { linkedCaptionIds } = linkedFigureCaptions(blocks, analyses);
  blocks
    .filter((block) => block.type === "caption" && !linkedCaptionIds.has(block.block_id))
    .forEach((block) => items.push({
      blockId: block.block_id,
      kind: "caption",
      message: "그림과 연결되지 않은 캡션입니다.",
    }));
  return items;
}

function previewText(block, analysis) {
  const description = analysis?.description?.long_text
    || analysis?.description?.short_text;
  const text = block?.text || block?.content || description || "";
  return String(text).replace(/\s+/g, " ").trim();
}

export function buildBrailleReadingPlan(page, analyses = []) {
  const blocks = sortBlocksForBrailleReading(page?.blocks || []);
  const analysisByBlock = new Map(
    analyses.map((analysis) => [analysis.block_id, analysis]),
  );
  const { captionByFigureId, linkedCaptionIds } = linkedFigureCaptions(
    blocks,
    analyses,
  );
  const plan = [];

  blocks.forEach((block) => {
    if (block.type === "page_number" || linkedCaptionIds.has(block.block_id)) return;

    const analysis = analysisByBlock.get(block.block_id);
    // Braille order for a figure group: source caption, then the note/graphic.
    const caption = captionByFigureId.get(block.block_id);
    if (caption) {
      plan.push({
        blockId: caption.block_id,
        type: "caption",
        label: BLOCK_TYPE_LABELS.caption,
        preview: previewText(caption),
        isFigureCaption: true,
      });
    }
    plan.push({
      blockId: block.block_id,
      type: block.type,
      label: BLOCK_TYPE_LABELS[block.type] || block.type || "내용",
      preview: previewText(block, analysis),
      isFigureGroup: block.type === "figure",
    });
  });

  const movableItems = plan.filter((item) => !item.isFigureCaption);
  return plan.map((item, index) => {
    const movableIndex = movableItems.findIndex(
      (candidate) => candidate.blockId === item.blockId,
    );
    return {
      ...item,
      order: index + 1,
      canMoveUp: movableIndex > 0,
      canMoveDown: movableIndex >= 0 && movableIndex < movableItems.length - 1,
    };
  });
}
