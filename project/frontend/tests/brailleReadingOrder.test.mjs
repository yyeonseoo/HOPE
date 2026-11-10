import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const moduleSource = await readFile(
  new URL("../src/brailleReadingOrder.js", import.meta.url),
  "utf8",
);
const {
  buildBrailleReadingPlan,
  figureDisplayNames,
  figureRelocationNote,
  figureVisualTreatment,
  brailleReadingWarnings,
  linkedFigureCaptions,
  moveBrailleReadingGroup,
  normalizeTaggedPageDescription,
  placeBrailleReadingGroup,
  reorderTaggedPageDescription,
  sortBlocksForBrailleReading,
} = await import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString("base64")}`);

test("uses explicit reading_order without mutating source blocks", () => {
  const blocks = [
    { block_id: "p", type: "paragraph", reading_order: 2 },
    { block_id: "t", type: "title", reading_order: 1 },
  ];

  assert.deepEqual(
    sortBlocksForBrailleReading(blocks).map((block) => block.block_id),
    ["t", "p"],
  );
  assert.equal(blocks[0].block_id, "p");
});

test("places a dragged reading group before the drop target", () => {
  const blocks = [
    { block_id: "a", type: "paragraph", reading_order: 1 },
    { block_id: "b", type: "figure", reading_order: 2 },
    { block_id: "c", type: "paragraph", reading_order: 3 },
  ];
  const moved = placeBrailleReadingGroup(blocks, [], "c", "a");
  assert.deepEqual(
    sortBlocksForBrailleReading(moved).map((block) => block.block_id),
    ["c", "a", "b"],
  );
});

test("places a dragged reading group after the drop target", () => {
  const blocks = [
    { block_id: "a", type: "paragraph", reading_order: 1 },
    { block_id: "b", type: "figure", reading_order: 2 },
    { block_id: "c", type: "paragraph", reading_order: 3 },
  ];
  const moved = placeBrailleReadingGroup(blocks, [], "a", "b", "after");
  assert.deepEqual(
    sortBlocksForBrailleReading(moved).map((block) => block.block_id),
    ["b", "a", "c"],
  );
});

test("reorders tagged accessibility-page lines while preserving untagged text", () => {
  const previousBlocks = [
    { block_id: "p", type: "paragraph", reading_order: 1 },
    { block_id: "f", type: "figure", reading_order: 2 },
  ];
  const nextBlocks = [
    { ...previousBlocks[0], reading_order: 2 },
    { ...previousBlocks[1], reading_order: 1 },
  ];
  const reordered = reorderTaggedPageDescription(
    "페이지 안내\n[paragraph] 본문\n[figure] 그림 설명",
    previousBlocks,
    nextBlocks,
  );
  assert.equal(reordered, "페이지 안내\n[figure] 그림 설명\n[paragraph] 본문");
});

test("collapses wrapped caption OCR into one tagged block", () => {
  assert.equal(
    normalizeTaggedPageDescription(
      "[caption]\n[출\n처:\n한국\n거래\n소,\n2016]\n[paragraph] 다음 문장",
    ),
    "[caption] [출 처: 한국 거래 소, 2016]\n[paragraph] 다음 문장",
  );
});

test("moves a figure and its linked caption as one reading group", () => {
  const blocks = [
    { block_id: "paragraph", type: "paragraph", reading_order: 1 },
    { block_id: "figure", type: "figure", reading_order: 2 },
    { block_id: "caption", type: "caption", reading_order: 3 },
  ];
  const analyses = [{
    block_id: "figure",
    type: "figure",
    description: { long_text: "description" },
    context: { caption_block_id: "caption" },
  }];

  const moved = moveBrailleReadingGroup(blocks, analyses, "figure", "up");
  assert.deepEqual(
    sortBlocksForBrailleReading(moved).map((block) => block.block_id),
    ["figure", "caption", "paragraph"],
  );
});

test("allows a linked caption to move independently when selected directly", () => {
  const blocks = [
    { block_id: "paragraph", type: "paragraph", reading_order: 1 },
    { block_id: "figure", type: "figure", reading_order: 2 },
    { block_id: "caption", type: "caption", reading_order: 3 },
  ];
  const analyses = [{
    block_id: "figure",
    type: "figure",
    context: { caption_block_id: "caption" },
  }];

  const moved = moveBrailleReadingGroup(blocks, analyses, "caption", "up");
  assert.deepEqual(
    sortBlocksForBrailleReading(moved).map((block) => block.block_id),
    ["paragraph", "caption", "figure"],
  );
});

test("review checklist points at each block that still needs a decision", () => {
  const items = brailleReadingWarnings({
    blocks: [
      { block_id: "figure", type: "figure", reading_order: 1 },
      { block_id: "caption", type: "caption", reading_order: 2 },
      { block_id: "graph", type: "figure", reading_order: 3 },
      { block_id: "shot", type: "table", reading_order: 4 },
    ],
  }, [
    {
      block_id: "graph",
      description: { long_text: "눈금 30[확인 필요], 180[확인 필요]" },
      braille_review: { suggested_visual_treatment: "tactile_graphic", treatment_reason: "그래프는 촉도 우선" },
    },
    { block_id: "shot", type: "table", warnings: ["35개 셀의 텍스트를 인식하지 못했습니다."] },
  ]);

  const kinds = items.map((item) => `${item.blockId}:${item.kind}`);
  assert.deepEqual(kinds, [
    "figure:treatment", "figure:note",
    "graph:treatment", "graph:verify",
    "shot:table",
    "caption:caption",
  ]);
  assert.match(items[2].message, /"촉도" 제안을 확정하세요 \(그래프는 촉도 우선\)/);
  assert.match(items[3].message, /2개/);
});

test("a confirmed treatment clears the treatment item", () => {
  const items = brailleReadingWarnings(
    { blocks: [{ block_id: "graph", type: "figure", reading_order: 1 }] },
    [{ block_id: "graph", braille_review: { visual_treatment: "tactile_graphic" } }],
  );
  assert.deepEqual(items, []);
});

test("groups an explicitly linked source caption before its figure", () => {
  const blocks = [
    { block_id: "caption", type: "caption", text: "source caption", reading_order: 1 },
    { block_id: "figure", type: "figure", reading_order: 2 },
  ];
  const analyses = [{
    block_id: "figure",
    type: "figure",
    description: { long_text: "accessible description" },
    context_source: { caption_block_id: "caption" },
  }];

  const links = linkedFigureCaptions(blocks, analyses);
  assert.equal(links.captionByFigureId.get("figure").block_id, "caption");

  const plan = buildBrailleReadingPlan({ blocks }, analyses);
  assert.deepEqual(
    plan.map((item) => [item.type, item.order]),
    [["caption", 1], ["figure", 2]],
  );
});

test("uses the rule-based treatment suggestion until the reviewer chooses", () => {
  assert.equal(figureVisualTreatment({}), "transcriber_note");
  assert.equal(
    figureVisualTreatment({ braille_review: { suggested_visual_treatment: "tactile_graphic" } }),
    "tactile_graphic",
  );
  assert.equal(
    figureVisualTreatment({
      braille_review: { suggested_visual_treatment: "tactile_graphic", visual_treatment: "omit" },
    }),
    "omit",
  );
});

test("a figure moved away from its print position gets an original-position note", () => {
  const blocks = [
    { block_id: "stem", type: "paragraph", text: "다음 물음에 답하시오.", reading_order: 1, original_reading_order: 1 },
    { block_id: "fig", type: "figure", reading_order: 2, original_reading_order: 4 },
    { block_id: "q1", type: "paragraph", text: "⑴ 높이를 구하시오.", reading_order: 3, original_reading_order: 2 },
    { block_id: "q2", type: "paragraph", text: "⑵ 대관람차가 한 바퀴 회전하는 데 걸린 시간을 구하시오.", reading_order: 4, original_reading_order: 3 },
    { block_id: "q3", type: "paragraph", text: "⑶ 횟수를 구하시오.", reading_order: 5, original_reading_order: 5 },
  ];

  assert.equal(
    figureRelocationNote(blocks, "fig"),
    '원래 이 그림은 "⑵ 대관람차가 한 바퀴 회전하는 데 …" 다음에 있다.',
  );
  assert.equal(figureRelocationNote(blocks, "q1"), null);
  const unmoved = blocks.map((block) => ({ ...block, reading_order: block.original_reading_order }));
  assert.equal(figureRelocationNote(unmoved, "fig"), null);
  assert.equal(figureRelocationNote(blocks.map(({ original_reading_order, ...rest }) => rest), "fig"), null);
});

test("figures get human names in reading order for the review UI", () => {
  const names = figureDisplayNames(
    [
      { block_id: "p6_b8", type: "figure", reading_order: 2 },
      { block_id: "p6_b4", type: "figure", reading_order: 1 },
    ],
    [{ block_id: "p6_b8", figure_type: "graph" }, { block_id: "p6_b4", figure_type: "photo" }],
  );
  assert.equal(names.get("p6_b4"), "그림 1 (사진)");
  assert.equal(names.get("p6_b8"), "그림 2 (그래프)");
});

test("an unreadable table suggested for omission is a decision to confirm", () => {
  const items = brailleReadingWarnings(
    { blocks: [{ block_id: "shot", type: "table", reading_order: 1 }] },
    [{
      block_id: "shot",
      type: "table",
      warnings: ["35개 셀의 텍스트를 인식하지 못했습니다."],
      braille_review: { suggested_visual_treatment: "omit" },
    }],
  );
  assert.deepEqual(items.map((item) => item.kind), ["treatment"]);
  assert.match(items[0].message, /"생략"을 제안/);
});
