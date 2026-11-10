import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const moduleSource = await readFile(
  new URL("../src/accessibleHtml.js", import.meta.url),
  "utf8",
);
const readingOrderSource = await readFile(
  new URL("../src/brailleReadingOrder.js", import.meta.url),
  "utf8",
);
const bundledModuleSource = moduleSource.replace(
  /import\s*\{[\s\S]*?\}\s*from\s*["']\.\/brailleReadingOrder\.js["'];/,
  readingOrderSource.replaceAll("export function", "function"),
);
const {
  accessibleHtmlFilename,
  buildAccessibleTextbookHtml,
  buildBrailleEditText,
  buildTactileWorklistCsv,
} = await import(`data:text/javascript;base64,${Buffer.from(bundledModuleSource).toString("base64")}`);

function samplePage(pageId = 2, paragraph = "관계를 살펴본다.") {
  return {
    page: {
      page_id: pageId,
      blocks: [
        { block_id: "title", type: "title", text: "함수" },
        { block_id: "paragraph", type: "paragraph", text: paragraph },
        { block_id: "formula", type: "formula", text: "y=2x" },
        { block_id: "table", type: "table" },
        { block_id: "figure", type: "figure" },
      ],
    },
    semantic_analyses: [
      {
        block_id: "formula",
        analysis: {
          result: {
            latex: "y=2x",
            mathml: "<math><mi>y</mi><mo>=</mo><mn>2</mn><mi>x</mi></math>",
            plain_text: "y는 2 곱하기 x",
          },
        },
      },
      {
        block_id: "table",
        analysis: {
          result: {
            row_count: 2,
            cells: [
              {
                row: 0,
                column: 0,
                row_span: 1,
                column_span: 1,
                text: "x",
                is_header: true,
              },
              {
                row: 1,
                column: 0,
                row_span: 1,
                column_span: 1,
                text: "1",
                is_header: false,
              },
            ],
          },
        },
        description: { long_text: "x 값 표" },
      },
      {
        block_id: "figure",
        figure_type: "graph",
        description: { long_text: "오른쪽 위로 향하는 직선 그래프이다." },
      },
    ],
  };
}

test("builds Korean semantic HTML for textbook blocks", () => {
  const html = buildAccessibleTextbookHtml({
    title: "교과서.pdf",
    pages: [samplePage()],
  });

  assert.match(html, /<html lang="ko">/);
  assert.match(html, /<h4>함수<\/h4>/);
  assert.match(html, /<math>/);
  assert.match(html, /<table>/);
  assert.match(html, /<figure(?:\s|>)/);
  assert.match(html, /점역자 주.*오른쪽 위로 향하는 직선 그래프이다/);
  assert.match(html, /href="#page-2"/);
});

test("uses the last version when the same saved page appears twice", () => {
  const html = buildAccessibleTextbookHtml({
    title: "교과서",
    pages: [
      samplePage(2, "저장 전 문장"),
      samplePage(2, "화면에서 수정한 최신 문장"),
    ],
  });

  assert.doesNotMatch(html, /저장 전 문장/);
  assert.match(html, /화면에서 수정한 최신 문장/);
  assert.equal((html.match(/id="page-2"/g) || []).length, 1);
});

test("can omit embedded page images from the readable source view", () => {
  const page = {
    ...samplePage(),
    page_image: "data:image/png;base64,AAA",
  };
  const html = buildAccessibleTextbookHtml({
    title: "교과서",
    pages: [page],
    includePageImages: false,
  });

  assert.doesNotMatch(html, /data:image\/png/);
  assert.doesNotMatch(html, /원본 교과서 2페이지 보기/);
});

test("preserves the page description without exposing internal block markers", () => {
  const page = {
    ...samplePage(),
    page_description: {
      text: [
        "[title] 그래프",
        "[paragraph] 두 값의 관계를 살펴본다.",
        "[formula] y는 x에 비례한다.",
        "[figure] 오른쪽 위로 향하는 직선 그래프이다.",
      ].join("\n"),
    },
  };
  const html = buildAccessibleTextbookHtml({
    title: "교과서",
    pages: [page],
  });

  assert.match(html, /페이지 전체 설명 듣기/);
  assert.match(html, /두 값의 관계를 살펴본다/);
  assert.match(html, /수식 설명/);
  assert.match(html, /그림 설명/);
  assert.doesNotMatch(html, /\[paragraph\]|\[formula\]|\[figure\]/);
});

test("exports blocks in reading order and keeps a linked caption inside its figure", () => {
  const page = samplePage(4);
  page.page.blocks = [
    { block_id: "caption", type: "caption", text: "그림 1 원문", reading_order: 3 },
    { block_id: "figure", type: "figure", reading_order: 2 },
    { block_id: "paragraph", type: "paragraph", text: "먼저 읽는 본문", reading_order: 1 },
  ];
  page.semantic_analyses = [{
    block_id: "figure",
    type: "figure",
    figure_type: "graph",
    description: { long_text: "그래프 접근성 설명" },
    context_source: { caption_block_id: "caption" },
  }];

  const html = buildAccessibleTextbookHtml({ title: "교과서", pages: [page] });
  // Braille order: source caption before the transcriber's note.
  assert.ok(html.indexOf("먼저 읽는 본문") < html.indexOf("그림 1 원문"));
  assert.ok(html.indexOf("그림 1 원문") < html.indexOf("그래프 접근성 설명"));
  assert.equal((html.match(/그림 1 원문/g) || []).length, 1);
});

test("keeps braille production review metadata out of the student HTML", () => {
  const page = samplePage(5);
  page.semantic_analyses[2].braille_review = {
    visual_strategy: "description",
    visual_treatment: "tactile_graphic",
    transcriber_note: "복잡한 시각 요소를 설명문으로 대체함.",
    reviewed: true,
  };

  const html = buildAccessibleTextbookHtml({ title: "교과서", pages: [page] });
  // A tactile graphic replaces the AI description with a placeholder.
  assert.match(html, /\[촉도 1\]/);
  assert.doesNotMatch(html, /오른쪽 위로 향하는 직선 그래프이다/);
  assert.doesNotMatch(html, /점역자 검수 설명/);
  assert.doesNotMatch(html, /점역 변경·검수 기록/);
  assert.doesNotMatch(html, /복잡한 시각 요소를 설명문으로 대체함/);
  assert.doesNotMatch(html, /tactile_graphic/);
  assert.doesNotMatch(html, /점역 참고 설명 · 검수 필요/);
});

test("replaces an omitted figure with an omission marker instead of dropping it", () => {
  const page = samplePage(6);
  page.page_description = {
    text: [
      "[title] 함수",
      "[paragraph] 두 값의 관계를 살펴본다.",
      "[figure] 오른쪽 위로 향하는 직선 그래프이다.",
    ].join("\n"),
  };
  page.semantic_analyses[2].braille_review = {
    visual_treatment: "omit",
  };

  const html = buildAccessibleTextbookHtml({ title: "교과서", pages: [page] });
  assert.match(html, /\(그림 생략\)/);
  assert.doesNotMatch(html, /오른쪽 위로 향하는 직선 그래프이다/);
  assert.match(html, /두 값의 관계를 살펴본다/);
});

test("escapes textbook text and creates a safe filename", () => {
  const html = buildAccessibleTextbookHtml({
    title: "수학 <교과서>",
    pages: [samplePage(1, "<script>alert(1)</script>")],
  });

  assert.doesNotMatch(html, /<script>alert/);
  assert.match(html, /&lt;script&gt;alert\(1\)&lt;\/script&gt;/);
  assert.equal(
    accessibleHtmlFilename('수학:"교과서".pdf'),
    "수학__교과서__접근성자료.html",
  );
});

test("labels the printed page number from the footer, not the PDF sequence", () => {
  const page = samplePage(3);
  page.page.blocks.push({ block_id: "footer", type: "footer", text: "120  Ⅲ. 좌표평면과 그래프" });

  const html = buildAccessibleTextbookHtml({ title: "교과서", pages: [page] });
  assert.match(html, /\[원본 쪽 120\] \(PDF 3쪽\)/);
});

test("braille edit text follows guideline markers for each treatment", () => {
  const page = samplePage(3);
  page.page.blocks.push(
    { block_id: "footer", type: "footer", text: "120  Ⅲ. 좌표평면과 그래프" },
    { block_id: "icon", type: "figure" },
    { block_id: "photo", type: "figure" },
    { block_id: "tab", type: "title", text: "Ⅲ" },
  );
  page.semantic_analyses[1].analysis.result.cells.push({ row: 1, column: 1, text: "" });
  page.semantic_analyses[2].braille_review = { suggested_visual_treatment: "tactile_graphic" };
  page.semantic_analyses.push(
    { block_id: "icon", type: "figure", braille_review: { suggested_visual_treatment: "omit" } },
    { block_id: "photo", type: "figure", description: { long_text: "대관람차 사진이다." } },
  );

  const text = buildBrailleEditText({ pages: [page] });
  assert.match(text, /^\[이 책의 점역자 주\]/);
  assert.match(text, /\[원본 쪽 120\]/);
  assert.match(text, /촉각 그래픽이 들어갈 자리/);
  assert.match(text, /\[수식\] y=2x/);
  assert.match(text, /x \| ==/);
  assert.match(text, /\[촉도 1\]/);
  assert.match(text, /,'그림 생략,'/);
  assert.match(text, /,'대관람차 사진이다\.,'/);
  assert.doesNotMatch(text, /오른쪽 위로 향하는 직선 그래프이다/);
  assert.doesNotMatch(text, /\[제목\] Ⅲ/);
});

test("tactile worklist lists only tactile figures with their rationale", () => {
  const page = samplePage(3);
  page.semantic_analyses[2].braille_review = {
    suggested_visual_treatment: "tactile_graphic",
    treatment_reason: "그래프·도식은 촉도 제작을 우선한다.",
    treatment_rationale: ["TG2022-1.7"],
  };

  const csv = buildTactileWorklistCsv({ pages: [page] });
  const lines = csv.trim().split("\n");
  assert.equal(lines.length, 2);
  assert.match(lines[1], /^1,,3,figure,graph,,그래프·도식은 촉도 제작을 우선한다\.,TG2022-1\.7,아니오\(제안값\),/);
});

test("a table marked as not-a-table is exported as an omitted image", () => {
  const page = samplePage(9);
  page.semantic_analyses[1].braille_review = { visual_treatment: "omit" };

  const text = buildBrailleEditText({ pages: [page] });
  const html = buildAccessibleTextbookHtml({ title: "교과서", pages: [page] });
  assert.doesNotMatch(text, /\[표\]/);
  assert.match(text, /,'그림 생략,'/);
  assert.doesNotMatch(html, /<table>/);
});

test("exports drop invisible PDF control characters", () => {
  const page = samplePage(3, "⑴ \u200c\u0007자동차의 속력을 구하시오.");
  const text = buildBrailleEditText({ pages: [page] });
  assert.match(text, /⑴ 자동차의 속력을 구하시오\./);
});
