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

export function moveBrailleReadingGroup(blocks = [], analyses = [], blockId, direction) {
  const groups = readingGroups(blocks, analyses);
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
  const groups = readingGroups(blocks, analyses);
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

export function reorderTaggedPageDescription(text, previousBlocks = [], nextBlocks = []) {
  const lines = String(text || "").split(/\r?\n/);
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

export function brailleReadingWarnings(page, analyses = []) {
  const blocks = page?.blocks || [];
  const warnings = [];
  const orders = blocks
    .map(numericReadingOrder)
    .filter((order) => order !== null);
  const uniqueOrders = new Set(orders);
  if (orders.length !== blocks.length) {
    warnings.push("읽기 순서가 지정되지 않은 블록이 있습니다.");
  }
  if (uniqueOrders.size !== orders.length) {
    warnings.push("같은 읽기 순서 번호를 가진 블록이 있습니다.");
  }

  const analysisByBlock = new Map(
    analyses.map((analysis) => [analysis.block_id, analysis]),
  );
  blocks
    .filter((block) => block.type === "figure")
    .forEach((block) => {
      const analysis = analysisByBlock.get(block.block_id);
      const description = analysis?.description?.long_text
        || analysis?.description?.short_text;
      if (!String(description || "").trim()) {
        warnings.push(`${block.block_id} 그림에 점역 참고 설명이 없습니다.`);
      }
      if (analysis?.braille_review?.reviewed !== true) {
        warnings.push(`${block.block_id} 시각자료의 점역 처리 방식이 검수되지 않았습니다.`);
      }
    });

  blocks
    .filter((block) => block.type === "table")
    .forEach((block) => {
      const cells = analysisByBlock.get(block.block_id)?.analysis?.result?.cells;
      if (Array.isArray(cells) && cells.length && !cells.some((cell) => cell.is_header)) {
        warnings.push(`${block.block_id} 표의 행·열 머리글 관계를 확인하세요.`);
      }
    });

  const { linkedCaptionIds } = linkedFigureCaptions(blocks, analyses);
  const unlinkedCaptions = blocks.filter(
    (block) => block.type === "caption" && !linkedCaptionIds.has(block.block_id),
  );
  if (unlinkedCaptions.length) {
    warnings.push(`그림과 연결되지 않은 원문 캡션이 ${unlinkedCaptions.length}개 있습니다.`);
  }
  return warnings;
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
    plan.push({
      blockId: block.block_id,
      type: block.type,
      label: BLOCK_TYPE_LABELS[block.type] || block.type || "내용",
      preview: previewText(block, analysis),
      isFigureGroup: block.type === "figure",
    });

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
