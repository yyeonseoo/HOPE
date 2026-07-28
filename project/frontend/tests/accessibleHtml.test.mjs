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
  assert.match(html, /<figcaption(?:\s|>)/);
  assert.match(html, /오른쪽 위로 향하는 직선 그래프이다/);
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
  assert.ok(html.indexOf("먼저 읽는 본문") < html.indexOf("그래프 접근성 설명"));
  assert.ok(html.indexOf("그래프 접근성 설명") < html.indexOf("그림 1 원문"));
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
  assert.match(html, /그림 설명/);
  assert.match(html, /오른쪽 위로 향하는 직선 그래프이다/);
  assert.doesNotMatch(html, /점역자 검수 설명/);
  assert.doesNotMatch(html, /점역 변경·검수 기록/);
  assert.doesNotMatch(html, /복잡한 시각 요소를 설명문으로 대체함/);
  assert.doesNotMatch(html, /tactile_graphic/);
  assert.doesNotMatch(html, /점역 참고 설명 · 검수 필요/);
});

test("omits a figure and its page-summary sentence from student HTML", () => {
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
  assert.doesNotMatch(html, /<figure(?:\s|>)/);
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
