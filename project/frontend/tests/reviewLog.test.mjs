import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const moduleSource = await readFile(new URL("../src/reviewLog.js", import.meta.url), "utf8");
const { buildReviewLogExport, withReviewLog } = await import(
  `data:text/javascript;base64,${Buffer.from(moduleSource).toString("base64")}`
);

test("summarizes how a reviewer used the AI suggestions", () => {
  let page = {
    page: { page_id: 6 },
    semantic_analyses: [
      { block_id: "a", braille_review: { suggested_visual_treatment: "tactile_graphic", visual_treatment: "tactile_graphic" } },
      { block_id: "b", braille_review: { suggested_visual_treatment: "omit", visual_treatment: "transcriber_note" } },
      { block_id: "c", braille_review: { suggested_visual_treatment: "transcriber_note" } },
      { block_id: "d" },
    ],
  };
  page = withReviewLog(page, { action: "confirm_suggestions", count: 1 }, "2026-11-11T10:00:00Z");
  page = withReviewLog(page, { action: "treatment", block_id: "b", to: "transcriber_note" }, "2026-11-11T10:01:00Z");
  page = withReviewLog(page, { action: "reorder", block_id: "a" }, "2026-11-11T10:02:00Z");
  page = withReviewLog(page, { action: "review_status", to: "reviewed" }, "2026-11-11T10:05:00Z");

  const [exported] = buildReviewLogExport({ title: "교과서", pages: [page] }).pages;

  assert.equal(exported.page, 6);
  assert.deepEqual(exported.summary, {
    ai_suggestions: 3,
    accepted: 1,
    overridden: 1,
    unconfirmed: 1,
    bulk_confirms: 1,
    treatment_changes: 1,
    reorders: 1,
    order_restores: 0,
    description_edits: 0,
    block_edits: 0,
    seconds_to_reviewed: 300,
    actions: 4,
  });
  assert.equal(exported.log.length, 4);
});
