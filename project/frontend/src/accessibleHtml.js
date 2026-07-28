import {
  linkedFigureCaptions,
  sortBlocksForBrailleReading,
} from "./brailleReadingOrder.js";

const BLOCK_HEADINGS = {
  title: 4,
  section_title: 5,
  section_header: 5,
};

const SKIPPED_BLOCK_TYPES = new Set(["page_number"]);
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
const DESCRIPTION_LABELS = {
  title: "제목",
  paragraph: null,
  formula: "수식 설명",
  table: "표 설명",
  figure: "그림 설명",
  caption: "캡션",
};

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function blockText(block) {
  return String(
    block?.text
      ?? block?.content
      ?? block?.raw_text
      ?? "",
  ).trim();
}

function descriptionText(analysis) {
  return String(
    analysis?.description?.long_text
      || analysis?.description?.short_text
      || "",
  ).trim();
}

function isOmittedFigure(analysis) {
  return (
    analysis?.braille_review?.visual_treatment
    || analysis?.braille_review?.production_mode
  ) === "omit";
}

function safeMathMl(value) {
  const mathml = String(value || "").trim();
  if (!/^<math(?:\s|>)/i.test(mathml)) return "";
  if (/<(?:script|iframe|object|embed|foreignObject)\b/i.test(mathml)) return "";
  if (/\bon\w+\s*=|javascript\s*:/i.test(mathml)) return "";
  return mathml;
}

function formulaHtml(analysis, fallbackText) {
  const result = analysis?.analysis?.result;
  const explanation = descriptionText(analysis);
  const spoken = String(result?.plain_text || explanation || fallbackText || result?.latex || "수식").trim();
  const mathml = safeMathMl(result?.mathml);
  const visualFormula = mathml
    || `<span class="formula-text" role="math" aria-label="${escapeHtml(spoken)}">${
      escapeHtml(result?.plain_text || result?.latex || fallbackText || spoken)
    }</span>`;
  const explanationHtml = explanation && explanation !== spoken
    ? `<p class="element-description">${escapeHtml(explanation)}</p>`
    : "";
  return `
    <section class="formula-block" aria-label="수식">
      ${visualFormula}
      ${explanationHtml}
    </section>`;
}

function tableHtml(analysis, fallbackText) {
  const result = analysis?.analysis?.result;
  const cells = Array.isArray(result?.cells) ? result.cells : [];
  const explanation = descriptionText(analysis);
  if (!cells.length) {
    const text = explanation || fallbackText || "표의 내용을 인식하지 못했다.";
    return `<section class="table-block"><p>${escapeHtml(text)}</p></section>`;
  }

  const rowCount = Math.max(
    Number(result?.row_count) || 0,
    ...cells.map((cell) => Number(cell.row) + Number(cell.row_span || 1)),
  );
  const cellsByRow = new Map();
  cells.forEach((cell) => {
    const row = Number(cell.row) || 0;
    if (!cellsByRow.has(row)) cellsByRow.set(row, []);
    cellsByRow.get(row).push(cell);
  });

  const rows = Array.from({ length: rowCount }, (_, rowIndex) => {
    const rowCells = (cellsByRow.get(rowIndex) || [])
      .sort((a, b) => Number(a.column) - Number(b.column))
      .map((cell) => {
        const tag = cell.is_header ? "th" : "td";
        const scope = cell.is_header
          ? ` scope="${Number(cell.column) === 0 && Number(cell.row) > 0 ? "row" : "col"}"`
          : "";
        const rowSpan = Number(cell.row_span || 1) > 1
          ? ` rowspan="${Number(cell.row_span)}"`
          : "";
        const columnSpan = Number(cell.column_span || 1) > 1
          ? ` colspan="${Number(cell.column_span)}"`
          : "";
        return `<${tag}${scope}${rowSpan}${columnSpan}>${escapeHtml(cell.text || "")}</${tag}>`;
      })
      .join("");
    return `<tr>${rowCells}</tr>`;
  }).join("");

  const caption = explanation || fallbackText || "교과서 표";
  return `
    <section class="table-block">
      <table>
        <caption>${escapeHtml(caption)}</caption>
        <tbody>${rows}</tbody>
      </table>
    </section>`;
}

function figureHtml(analysis, fallbackText, pageId, index, sourceCaption = "") {
  const explanation = descriptionText(analysis) || fallbackText || "그림 설명이 제공되지 않았다.";
  const figureType = analysis?.figure_type || analysis?.analysis?.result?.figure_type;
  const typeLabel = FIGURE_TYPE_LABELS[figureType] || "그림";
  const captionId = `page-${pageId}-figure-${index}-caption`;
  return `
    <figure aria-labelledby="${captionId}">
      <figcaption id="${captionId}">
        <span class="element-label">${escapeHtml(`그림 ${index}. ${typeLabel}`)}</span>
        <span class="figure-description"><strong>그림 설명</strong> ${escapeHtml(explanation)}</span>
        ${sourceCaption ? `<span class="source-caption"><strong>원문 캡션</strong> ${escapeHtml(sourceCaption)}</span>` : ""}
      </figcaption>
    </figure>`;
}

function genericBlockHtml(block) {
  const text = blockText(block);
  if (!text || SKIPPED_BLOCK_TYPES.has(block?.type)) return "";
  const headingLevel = BLOCK_HEADINGS[block.type];
  if (headingLevel) return `<h${headingLevel}>${escapeHtml(text)}</h${headingLevel}>`;
  if (block.type === "caption") return `<p class="caption">${escapeHtml(text)}</p>`;
  if (block.type === "footer") return `<footer>${escapeHtml(text)}</footer>`;
  return `<p>${escapeHtml(text)}</p>`;
}

function pageDescriptionHtml(result, pageId) {
  const description = String(result?.page_description?.text || "").trim();
  if (!description) return "";
  const analyses = Array.isArray(result?.semantic_analyses) ? result.semantic_analyses : [];
  const analysisByBlock = new Map(analyses.map((item) => [item.block_id, item]));
  const figureBlocks = sortBlocksForBrailleReading(result?.page?.blocks || [])
    .filter((block) => block.type === "figure");
  let figureOffset = 0;
  const lines = description.split(/\r?\n+/).map((line) => line.trim()).filter(Boolean);
  const content = lines.map((line) => {
    const match = line.match(/^\[([a-z_]+)\]\s*/i);
    const type = match?.[1]?.toLowerCase();
    if (type === "figure") {
      const figureBlock = figureBlocks[figureOffset];
      figureOffset += 1;
      if (figureBlock && isOmittedFigure(analysisByBlock.get(figureBlock.block_id))) {
        return "";
      }
    }
    const text = match ? line.slice(match[0].length).trim() : line;
    if (!text) return "";
    const label = DESCRIPTION_LABELS[type];
    const labelHtml = label
      ? `<span class="element-label">${escapeHtml(label)}</span>`
      : "";
    return `<p>${labelHtml}${escapeHtml(text)}</p>`;
  }).filter(Boolean).join("");

  if (!content) return "";
  return `
    <details class="page-summary">
      <summary>페이지 전체 설명 듣기</summary>
      <div aria-label="${pageId}페이지 전체 설명">${content}</div>
    </details>`;
}

function pageHtml(result, includePageImages) {
  const page = result?.page || result;
  const pageId = Number(page?.page_id) || 1;
  const analyses = Array.isArray(result?.semantic_analyses) ? result.semantic_analyses : [];
  const analysisByBlock = new Map(analyses.map((item) => [item.block_id, item]));
  const orderedBlocks = sortBlocksForBrailleReading(page?.blocks || []);
  const { captionByFigureId, linkedCaptionIds } = linkedFigureCaptions(orderedBlocks, analyses);
  let figureIndex = 0;

  const blocks = orderedBlocks.map((block) => {
    if (linkedCaptionIds.has(block.block_id)) return "";
    const analysis = analysisByBlock.get(block.block_id);
    if (block.type === "formula") return formulaHtml(analysis, blockText(block));
    if (block.type === "table") return tableHtml(analysis, blockText(block));
    if (block.type === "figure") {
      if (isOmittedFigure(analysis)) return "";
      figureIndex += 1;
      const sourceCaption = blockText(captionByFigureId.get(block.block_id));
      return figureHtml(analysis, blockText(block), pageId, figureIndex, sourceCaption);
    }
    return genericBlockHtml(block);
  }).filter(Boolean).join("\n");

  const pageImage = String(result?.page_image || "");
  const imageHtml = includePageImages && pageImage.startsWith("data:image/")
    ? `<details class="original-page">
        <summary>원본 교과서 ${pageId}페이지 보기</summary>
        <img src="${pageImage}" alt="원본 교과서 ${pageId}페이지. 구조화된 접근성 내용은 다음 영역에 제공된다.">
      </details>`
    : "";
  const descriptionHtml = pageDescriptionHtml(result, pageId);

  return `
    <article class="textbook-page" id="page-${pageId}" aria-labelledby="page-${pageId}-title">
      <h2 id="page-${pageId}-title">${pageId}페이지</h2>
      <p class="source-page-reference">원본 교과서 ${pageId}페이지</p>
      ${descriptionHtml}
      ${imageHtml}
      <section class="page-content" aria-labelledby="page-${pageId}-content-title">
        <h3 id="page-${pageId}-content-title">구조화된 교과서 내용</h3>
        ${blocks || "<p>구조화된 페이지 내용이 없다.</p>"}
      </section>
    </article>`;
}

function normalizedPages(pages) {
  const unique = new Map();
  (pages || []).forEach((item) => {
    const result = item?.result || item;
    const pageId = Number(result?.page?.page_id ?? result?.page_id);
    if (Number.isFinite(pageId)) unique.set(pageId, result);
  });
  return [...unique.entries()]
    .sort(([left], [right]) => left - right)
    .map(([, result]) => result);
}

export function accessibleHtmlFilename(title, suffix = "접근성자료") {
  const base = String(title || "교과서")
    .replace(/\.pdf$/i, "")
    .replace(/[<>:"/\\|?*\u0000-\u001f]/g, "_")
    .trim() || "교과서";
  return `${base}_${suffix}.html`;
}

export function buildAccessibleTextbookHtml({ title, pages, includePageImages = true }) {
  const safeTitle = String(title || "교과서 접근성 자료").replace(/\.pdf$/i, "");
  const orderedPages = normalizedPages(pages);
  const toc = orderedPages
    .map((result) => {
      const pageId = Number(result?.page?.page_id) || 1;
      return `<li><a href="#page-${pageId}">${pageId}페이지</a></li>`;
    })
    .join("");

  return `<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>${escapeHtml(safeTitle)} 접근성 자료</title>
  <style>
    :root { color-scheme: light; font-family: "Noto Sans KR", "Malgun Gothic", sans-serif; }
    * { box-sizing: border-box; }
    body { background: #f6f8f5; color: #17211d; line-height: 1.85; margin: 0; }
    a { color: #245f45; }
    .skip-link { background: #17211d; color: white; left: 1rem; padding: .65rem 1rem; position: absolute; top: -5rem; z-index: 10; }
    .skip-link:focus { top: 1rem; }
    header, main, nav { margin-inline: auto; max-width: 920px; }
    header { padding: 3rem 1.5rem 1.5rem; }
    header h1 { line-height: 1.35; margin: 0 0 .5rem; }
    .notice { background: #fff8dc; border-left: 5px solid #b58a1d; margin-top: 1.5rem; padding: 1rem 1.2rem; }
    nav { padding: 0 1.5rem 1.5rem; }
    nav ul { display: flex; flex-wrap: wrap; gap: .6rem 1.2rem; padding-left: 1.25rem; }
    main { padding: 0 1.5rem 4rem; }
    .textbook-page { background: white; border: 1px solid #dce5df; border-radius: 16px; margin: 0 0 2rem; padding: 1.5rem clamp(1.2rem, 4vw, 3rem); }
    .textbook-page > h2 { border-bottom: 2px solid #6c9d51; padding-bottom: .6rem; }
    h2, h3, h4, h5 { line-height: 1.45; }
    p { margin: .8rem 0; }
    .caption { color: #44534d; font-size: .95rem; }
    .source-caption { display: block; margin-top: .65rem; }
    .transcriber-note { display: block; }
    .source-page-reference { color: #53645e; font-size: .9rem; margin-top: -.35rem; }
    .formula-block, .table-block, figure { background: #f4f8f5; border-left: 4px solid #6c9d51; margin: 1.3rem 0; overflow-x: auto; padding: 1rem 1.2rem; }
    .formula-text { font-family: "Cambria Math", serif; font-size: 1.2rem; }
    .element-label { display: block; font-size: .85rem; font-weight: 700; margin-bottom: .35rem; }
    figure { margin-inline: 0; }
    figcaption { margin: 0; }
    table { border-collapse: collapse; min-width: 60%; width: 100%; }
    caption { font-weight: 700; padding: 0 0 .7rem; text-align: left; }
    th, td { border: 1px solid #7b8b84; padding: .55rem .7rem; text-align: left; vertical-align: top; }
    th { background: #e5efe8; }
    .original-page { margin: 1rem 0 1.5rem; }
    .original-page summary { cursor: pointer; font-weight: 700; }
    .original-page img { border: 1px solid #ccd7d0; display: block; height: auto; margin-top: 1rem; max-width: 100%; }
    .page-summary { background: #f8faf5; border: 1px solid #d8e4d3; border-radius: 10px; margin: 1rem 0 1.5rem; padding: .85rem 1rem; }
    .page-summary summary { cursor: pointer; font-weight: 700; }
    .page-summary > div { border-top: 1px solid #dce5d7; margin-top: .8rem; padding-top: .45rem; }
    .page-summary p { margin: .65rem 0; }
    footer { border-top: 1px solid #dce5df; color: #596760; font-size: .9rem; margin-top: 1.5rem; padding-top: .75rem; }
    @media print {
      body { background: white; }
      nav, .skip-link, .original-page { display: none; }
      .textbook-page { border: 0; break-after: page; padding: 0; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#content">본문 바로가기</a>
  <header>
    <h1>${escapeHtml(safeTitle)}</h1>
    <p>교과서 구조 분석 결과를 바탕으로 생성한 접근성 자료이다.</p>
    <aside class="notice" role="note">AI가 생성한 설명이 포함되어 있으므로 교육 현장에서 사용하기 전에 점역사 또는 특수교사의 검수를 권장한다.</aside>
  </header>
  <nav aria-label="페이지 목차">
    <h2>페이지 목차</h2>
    <ul>${toc || "<li>저장된 페이지가 없다.</li>"}</ul>
  </nav>
  <main id="content">
    ${orderedPages.map((result) => pageHtml(result, includePageImages)).join("\n")}
  </main>
</body>
</html>`;
}

export function downloadAccessibleHtml(html, filename) {
  const blob = new Blob([html], { type: "text/html;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
