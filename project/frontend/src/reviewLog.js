// Review interaction log, kept on each page result and saved with it, so a
// study can report how reviewers used the AI suggestions (acceptance,
// overrides, reordering, edits, time to review) without extra tooling.

export function withReviewLog(result, entry, at = new Date().toISOString()) {
  if (!result) return result;
  return { ...result, review_log: [...(result.review_log || []), { at, ...entry }] };
}

function pageSummary(result) {
  const log = result?.review_log || [];
  const analyses = (result?.semantic_analyses || []).filter(
    (analysis) => analysis?.braille_review?.suggested_visual_treatment,
  );
  const decided = analyses.filter((analysis) => analysis.braille_review.visual_treatment);
  const accepted = decided.filter(
    (analysis) => analysis.braille_review.visual_treatment === analysis.braille_review.suggested_visual_treatment,
  );
  const count = (action) => log.filter((entry) => entry.action === action).length;
  const reviewed = log.filter((entry) => entry.action === "review_status" && entry.to === "reviewed").at(-1);
  const seconds = log.length && reviewed
    ? Math.round((Date.parse(reviewed.at) - Date.parse(log[0].at)) / 1000)
    : null;
  return {
    ai_suggestions: analyses.length,
    accepted: accepted.length,
    overridden: decided.length - accepted.length,
    unconfirmed: analyses.length - decided.length,
    bulk_confirms: count("confirm_suggestions"),
    treatment_changes: count("treatment"),
    reorders: count("reorder"),
    order_restores: count("restore_order"),
    description_edits: count("edit_description"),
    block_edits: count("edit_block") + count("add_figure") + count("delete_block"),
    seconds_to_reviewed: seconds,
    actions: log.length,
  };
}

export function buildReviewLogExport({ title, pages }) {
  return {
    title,
    exported_at: new Date().toISOString(),
    pages: (pages || [])
      .filter((result) => result?.page)
      .map((result) => ({
        page: result.page.page_id,
        summary: pageSummary(result),
        // Per-block suggestions next to the reviewer's final choice: the
        // reviewer's choices double as labels for evaluating the triage
        // (eval/evaluate_treatments.py).
        decisions: (result.semantic_analyses || [])
          .filter((analysis) => analysis?.braille_review?.suggested_visual_treatment)
          .map((analysis) => ({
            block_id: analysis.block_id,
            rule: analysis.braille_review.rule_suggestion || null,
            tree: analysis.braille_review.tree_suggestion || null,
            suggested: analysis.braille_review.suggested_visual_treatment,
            reviewer: analysis.braille_review.visual_treatment || null,
          })),
        log: result.review_log || [],
      })),
  };
}
