import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const moduleSource = await readFile(
  new URL("../src/brailleReadingOrder.js", import.meta.url),
  "utf8",
);
const {
  buildBrailleReadingPlan,
  brailleReadingWarnings,
  linkedFigureCaptions,
  moveBrailleReadingGroup,
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

test("reports missing descriptions and unlinked captions without changing content", () => {
  const warnings = brailleReadingWarnings({
    blocks: [
      { block_id: "figure", type: "figure", reading_order: 1 },
      { block_id: "caption", type: "caption", reading_order: 2 },
    ],
  }, []);

  assert.equal(warnings.length, 3);
  assert.ok(warnings.some((warning) => /figure.*설명/.test(warning)));
  assert.ok(warnings.some((warning) => /figure.*검수/.test(warning)));
  assert.ok(warnings.some((warning) => /1개/.test(warning)));
});

test("groups an explicitly linked source caption after its figure", () => {
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
    [["figure", 1], ["caption", 2]],
  );
});
