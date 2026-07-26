import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const moduleSource = await readFile(
  new URL("../src/accessibleHtml.js", import.meta.url),
  "utf8",
);
const {
  accessibleHtmlFilename,
  buildAccessibleTextbookHtml,
} = await import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString("base64")}`);

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
