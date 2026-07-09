import { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";

const API_BASE = "http://127.0.0.1:8000";
const REVIEW_TYPES = ["formula", "table", "figure"];

function fileSizeLabel(size) {
  if (!size) return "";
  if (size > 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  return `${(size / 1024).toFixed(1)} KB`;
}

async function parseError(response) {
  try {
    const payload = await response.json();
    return payload.detail || "요청 처리 중 오류가 발생했습니다.";
  } catch {
    return "요청 처리 중 오류가 발생했습니다.";
  }
}

function analysisEntries(result) {
  if (!result) return [];
  const completed = result.semantic_analyses || [];
  const byBlockId = new Map(completed.map((item) => [item.block_id, item]));

  return (result.page?.blocks || [])
    .filter((block) => REVIEW_TYPES.includes(block.type))
    .map((block) => {
      const analysis = byBlockId.get(block.block_id);
      return analysis || {
        page_id: result.page.page_id,
        block_id: block.block_id,
        type: block.type,
        bbox: block.bbox,
        detection: {
          model: { name: block.detector || "layout detector", version: null },
          confidence: block.score ?? null,
        },
        analysis: null,
        description: null,
        warnings: [],
      };
    });
}

function BlockCrop({ imageUrl, bbox, alt }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!imageUrl || !bbox) return undefined;
    const image = new Image();
    image.onload = () => {
      const [x1, y1, x2, y2] = bbox;
      const width = Math.max(1, x2 - x1);
      const height = Math.max(1, y2 - y1);
      const canvas = canvasRef.current;
      if (!canvas) return;
      canvas.width = width;
      canvas.height = height;
      canvas.getContext("2d").drawImage(image, x1, y1, width, height, 0, 0, width, height);
    };
    image.src = imageUrl;
    return () => { image.onload = null; };
  }, [imageUrl, bbox]);

  return <canvas ref={canvasRef} className="block-crop" role="img" aria-label={alt} />;
}

function Confidence({ value }) {
  return <span>{typeof value === "number" ? value.toFixed(3) : "미제공"}</span>;
}

function TableResult({ result }) {
  if (!result?.cells?.length) return <p className="muted">복원된 셀이 없습니다.</p>;
  const rows = Array.from({ length: result.row_count }, () => []);
  result.cells.forEach((cell) => {
    if (rows[cell.row]) rows[cell.row].push(cell);
  });
  rows.forEach((row) => row.sort((a, b) => a.column - b.column));

  return (
    <div className="table-scroll">
      <table className="reconstructed-table">
        <tbody>
          {rows.map((row, rowIndex) => (
            <tr key={rowIndex}>
              {row.map((cell) => {
                const Tag = cell.is_header ? "th" : "td";
                return <Tag key={`${cell.row}-${cell.column}`} rowSpan={cell.row_span} colSpan={cell.column_span}>{cell.text ?? ""}</Tag>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SemanticResult({ entry }) {
  const result = entry.analysis?.result;
  if (!entry.analysis) return <div className="pending-box">담당 분석 모듈이 아직 연결되지 않았습니다.</div>;
  if (!result) return <div className="pending-box">분석 결과가 없습니다.</div>;

  if (entry.type === "formula") {
    return <pre className="formula-output">{result.latex || result.mathml || result.plain_text || "인식 결과 없음"}</pre>;
  }
  if (entry.type === "table") return <TableResult result={result} />;

  return (
    <dl className="result-fields">
      <div><dt>유형</dt><dd>{result.figure_type}</dd></div>
      <div><dt>제목</dt><dd>{result.title || "미인식"}</dd></div>
      <div><dt>X축</dt><dd>{[result.x_axis?.label, result.x_axis?.unit].filter(Boolean).join(" · ") || "없음"}</dd></div>
      <div><dt>Y축</dt><dd>{[result.y_axis?.label, result.y_axis?.unit].filter(Boolean).join(" · ") || "없음"}</dd></div>
      <div><dt>계열</dt><dd>{result.series?.length ?? 0}개</dd></div>
    </dl>
  );
}

function DescriptionResult({ description }) {
  if (!description || description.status === "not_started") {
    return <div className="pending-box">설명 생성 전입니다.</div>;
  }
  return (
    <div className="description-output">
      <div><strong>짧은 설명</strong><p>{description.short_text || "없음"}</p></div>
      <div><strong>상세 설명</strong><p>{description.long_text || "없음"}</p></div>
      <div><strong>점역 참고</strong><p>{description.transcription_notes || "없음"}</p></div>
      <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
    </div>
  );
}

function AnalysisInspector({ result, type }) {
  const entries = useMemo(() => analysisEntries(result).filter((item) => item.type === type), [result, type]);
  const [selectedId, setSelectedId] = useState(null);

  useEffect(() => {
    setSelectedId(entries[0]?.block_id || null);
  }, [type, result]);

  const selected = entries.find((item) => item.block_id === selectedId) || entries[0];
  if (!entries.length) return <div className="empty-state compact">이 페이지에서 {type} 블록을 찾지 못했습니다.</div>;

  return (
    <div className="analysis-review">
      <div className="block-list" aria-label={`${type} 블록 목록`}>
        {entries.map((entry) => (
          <button key={entry.block_id} className={entry.block_id === selected?.block_id ? "active" : ""} onClick={() => setSelectedId(entry.block_id)}>
            <span>{entry.block_id}</span>
            <small>{entry.analysis?.status || "분석 전"}</small>
          </button>
        ))}
      </div>
      <div className="review-detail">
        <section className="review-section">
          <h3>원본 영역</h3>
          <BlockCrop imageUrl={result.page_image} bbox={selected.bbox} alt={`${selected.block_id} 원본 영역`} />
          <div className="metadata-row">
            <span>탐지 신뢰도</span><Confidence value={selected.detection?.confidence} />
            <span>분석 신뢰도</span><Confidence value={selected.analysis?.confidence} />
          </div>
        </section>
        <section className="review-section">
          <h3>구조화 결과</h3>
          <SemanticResult entry={selected} />
        </section>
        <section className="review-section">
          <h3>접근성 설명</h3>
          <DescriptionResult description={selected.description} />
        </section>
        {!!selected.warnings?.length && <div className="warning-box">{selected.warnings.join(" · ")}</div>}
      </div>
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [pageCount, setPageCount] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [dpi, setDpi] = useState(120);
  const [layoutModel, setLayoutModel] = useState("doclayout_yolo");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("layout");

  const blockStats = useMemo(() => {
    const blocks = result?.page?.blocks || [];
    return blocks.reduce((acc, block) => {
      acc[block.type] = (acc[block.type] || 0) + 1;
      return acc;
    }, {});
  }, [result]);

  async function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile); setPageCount(null); setResult(null); setError(""); setPageNumber(1);
    if (!nextFile) return;
    setStatus("counting");
    const formData = new FormData(); formData.append("file", nextFile);
    try {
      const response = await fetch(`${API_BASE}/api/page-count`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await parseError(response));
      setPageCount((await response.json()).page_count);
    } catch (err) { setError(err.message); } finally { setStatus("idle"); }
  }

  async function analyzePage() {
    if (!file) { setError("먼저 PDF를 업로드하세요."); return; }
    setStatus("analyzing"); setError(""); setResult(null);
    const formData = new FormData();
    formData.append("file", file); formData.append("page_number", String(pageNumber));
    formData.append("dpi", String(dpi)); formData.append("lang", "korean"); formData.append("layout_model", layoutModel);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await parseError(response));
      const payload = await response.json(); setResult(payload); setPageCount(payload.page_count); setActiveView("layout");
    } catch (err) { setError(err.message); } finally { setStatus("idle"); }
  }

  function downloadJson() {
    if (!result) return;
    const payload = { ...result.page, semantic_analyses: result.semantic_analyses || [] };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `page_${String(result.page.page_id).padStart(4, "0")}_analysis.json`;
    anchor.click(); URL.revokeObjectURL(url);
  }

  const busy = status === "counting" || status === "analyzing";
  const tabs = [{ id: "layout", label: "Layout" }, { id: "formula", label: "Formula" }, { id: "table", label: "Table" }, { id: "figure", label: "Figure" }, { id: "json", label: "JSON" }];

  return (
    <main className="app-shell">
      <section className="toolbar">
        <div><h1>경제수학 교과서 구조 분석</h1><p>레이아웃과 의미 분석 결과를 블록별로 검수합니다.</p></div>
        <div className="status-pill">{busy ? "처리 중" : "대기"}</div>
      </section>
      <section className="workspace">
        <aside className="control-panel">
          <label className="file-drop"><input type="file" accept="application/pdf" onChange={handleFileChange} /><span className="file-title">{file ? file.name : "PDF 선택"}</span><span className="file-meta">{file ? fileSizeLabel(file.size) : "교과서 PDF를 업로드하세요"}</span></label>
          <div className="field-row">
            <label>페이지<input type="number" min="1" max={pageCount || 1} value={pageNumber} onChange={(event) => setPageNumber(Number(event.target.value))} /></label>
            <label>DPI<input type="number" min="120" max="300" step="20" value={dpi} onChange={(event) => setDpi(Number(event.target.value))} /></label>
          </div>
          <label className="model-field">Layout model<select value={layoutModel} onChange={(event) => setLayoutModel(event.target.value)}><option value="doclayout_yolo">DocLayout-YOLO + 보정 규칙</option><option value="doclayout_yolo_unit3">DocLayout-YOLO + 3단원 보정 규칙</option><option value="doclayout_yolo_raw">DocLayout-YOLO 원본</option></select></label>
          <div className="page-count"><span>전체 페이지</span><strong>{pageCount ?? "-"}</strong></div>
          <button className="primary-button" disabled={busy || !file} onClick={analyzePage}>{status === "analyzing" ? "분석 중..." : "페이지 분석"}</button>
          {error && <div className="error-box">{error}</div>}
          {result && <div className="stats"><div className="stats-header"><span>탐지 블록</span><strong>{result.page.blocks.length}</strong></div><div className="type-list">{Object.entries(blockStats).map(([type, count]) => <div key={type}><span>{type}</span><strong>{count}</strong></div>)}</div><button className="secondary-button" onClick={downloadJson}>JSON 다운로드</button></div>}
        </aside>
        <section className="result-workspace">
          <nav className="view-tabs" aria-label="결과 보기">{tabs.map((tab) => <button key={tab.id} className={activeView === tab.id ? "active" : ""} onClick={() => setActiveView(tab.id)}>{tab.label}</button>)}</nav>
          {!result ? <div className="empty-state result-empty">분석할 PDF와 페이지를 선택하세요.</div> : activeView === "layout" ? (
            <div className="layout-view"><div className="pane-header"><h2>레이아웃 시각화</h2><span>{result.page.page_id}페이지</span></div><img src={result.visualization_image} alt="레이아웃 분석 시각화" /></div>
          ) : REVIEW_TYPES.includes(activeView) ? <AnalysisInspector result={result} type={activeView} /> : (
            <div className="json-view"><pre>{JSON.stringify({ ...result.page, semantic_analyses: result.semantic_analyses || [] }, null, 2)}</pre></div>
          )}
        </section>
      </section>
    </main>
  );
}
