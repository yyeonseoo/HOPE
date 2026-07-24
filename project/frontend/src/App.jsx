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

function Seconds({ value }) {
  return <span>{typeof value === "number" ? `${value.toFixed(2)}초` : "미제공"}</span>;
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

function DescriptionResult({ description, captioningEnabled }) {
  if (!description || description.status === "not_started") {
    return (
      <div className="pending-box">
        {captioningEnabled ? "설명 생성 결과가 없습니다. 경고와 백엔드 로그를 확인하세요." : "왼쪽에서 Figure 설명 생성을 활성화한 뒤 다시 분석하세요."}
      </div>
    );
  }
  return (
    <div className="description-output">
      <dl className="description-metrics">
        <div><dt>생성 모델</dt><dd>{description.model?.name || "미제공"}</dd></div>
        <div><dt>생성 신뢰도</dt><dd><Confidence value={description.confidence} /></dd></div>
        <div><dt>생성 시간</dt><dd><Seconds value={description.generation_time_seconds} /></dd></div>
      </dl>
      <div><strong>짧은 설명</strong><p>{description.short_text || "없음"}</p></div>
      <div><strong>상세 설명</strong><p>{description.long_text || "없음"}</p></div>
      <div><strong>점역 참고</strong><p>{description.transcription_notes || "없음"}</p></div>
      <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
    </div>
  );
}

function PageDescriptionView({ result }) {
  const description = result.page_description;
  if (!description || description.status === "failed") {
    return <div className="empty-state result-empty">이 페이지에서 읽을 수 있는 내용을 찾지 못했습니다.</div>;
  }

  return (
    <div className="layout-view">
      <div className="pane-header">
        <h2>페이지 전체 설명</h2>
        <span>{description.was_generated ? "모델 다듬기 적용" : "블록 원문 이어붙임"}</span>
      </div>
      <div className="description-output">
        <p className="page-description-text">{description.text || "없음"}</p>
        <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
        {description.warnings?.length > 0 && (
          <ul className="mt-2 list-disc pl-5 text-sm text-amber-900">
            {description.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
          </ul>
        )}
      </div>
    </div>
  );
}

function formatFormulaWarning(warning) {
  if (!warning) {
    return "";
  }
  
  if (warning.includes("fewer formula parts")) {
    return "pix2tex 이미지 수식 인식 결과가 일부 수식만 포함하여, OCR 기반 보정 결과를 사용했습니다. 원본 수식과 변환 결과를 함께 확인해 주세요.";
  }

  if (warning.includes("rejected as unreliable")) {
    return "pix2tex 이미지 수식 인식 결과가 신뢰도 기준을 통과하지 못해 OCR 기반 보정 결과를 사용했습니다. 점역 전 원본 수식 확인이 필요합니다.";
  }

  if (warning.includes("unavailable or failed")) {
    return "pix2tex 이미지 수식 인식을 사용할 수 없어 OCR 기반 보정 결과를 사용했습니다.";
  }

  if (warning.includes("Formula crop path was not provided")) {
    return "수식 이미지 crop 경로가 제공되지 않아 텍스트 기반으로만 분석했습니다.";
  }

  if (warning.includes("Formula crop file does not exist")) {
    return "수식 이미지 crop 파일을 찾을 수 없어 텍스트 기반으로만 분석했습니다.";
  }

  if (warning.includes("does not contain a formula-like expression")) {
    return "수식 영역으로 감지되었지만 수식 형태가 약해 점역 전 확인이 필요합니다.";
  }

  if (warning.includes("could not be recognized")) {
    return "수식을 자동 인식하지 못했습니다. 원문 수식 확인이 필요합니다.";
  }

  if (warning.includes("Formula text was not available from Model A output")) {
    return "Model A 출력에서 수식 텍스트를 찾지 못했습니다. 원문 수식 확인이 필요합니다.";
  }

  return warning;
}

function FormulaWarningResult({ warnings }) {
  if (!warnings || warnings.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3">
      <div className="text-sm font-semibold text-amber-900">
        점역 검수 참고
      </div>
      <ul className="mt-2 list-disc pl-5 text-sm text-amber-900">
        {warnings.map((warning, index) => (
          <li key={index}>{formatFormulaWarning(warning)}</li>
        ))}
      </ul>
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
          <DescriptionResult description={selected.description} captioningEnabled={result.figure_captioning_enabled} />
        </section>
        <FormulaWarningResult warnings={selected.warnings} type={type} />
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
  const [figureCaptioning, setFigureCaptioning] = useState(false);
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
    formData.append("figure_captioning", String(figureCaptioning));
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await parseError(response));
      const payload = await response.json(); setResult(payload); setPageCount(payload.page_count); setActiveView(figureCaptioning ? "figure" : "layout");
    } catch (err) { setError(err.message); } finally { setStatus("idle"); }
  }

  function downloadJson() {
    if (!result) return;
    const payload = { ...result.page, semantic_analyses: result.semantic_analyses || [], page_description: result.page_description || null };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `page_${String(result.page.page_id).padStart(4, "0")}_analysis.json`;
    anchor.click(); URL.revokeObjectURL(url);
  }

  const busy = status === "counting" || status === "analyzing";
  const tabs = [{ id: "layout", label: "Layout" }, { id: "page", label: "Page" }, { id: "formula", label: "Formula" }, { id: "table", label: "Table" }, { id: "figure", label: "Figure" }, { id: "json", label: "JSON" }];

  return (
    <main className="app-shell">
      <section className="toolbar">
        <div><h1>HOPE</h1><p>레이아웃과 의미 분석 결과를 블록별로 검수합니다.</p></div>
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
          <label className="toggle-field"><input type="checkbox" checked={figureCaptioning} onChange={(event) => setFigureCaptioning(event.target.checked)} /><span>Figure 설명 생성</span></label>
          {figureCaptioning && <div className="model-field"><span>Figure model</span><strong>GPT-4o (OpenAI API)</strong><small>OpenAI API를 호출하여 설명을 생성합니다.</small></div>}
          <div className="page-count"><span>전체 페이지</span><strong>{pageCount ?? "-"}</strong></div>
          <button className="primary-button" disabled={busy || !file} onClick={analyzePage}>{status === "analyzing" ? "분석 중..." : "페이지 분석"}</button>
          {error && <div className="error-box">{error}</div>}
          {result && <div className="stats"><div className="stats-header"><span>탐지 블록</span><strong>{result.page.blocks.length}</strong></div><div className="type-list">{Object.entries(blockStats).map(([type, count]) => <div key={type}><span>{type}</span><strong>{count}</strong></div>)}</div><button className="secondary-button" onClick={downloadJson}>JSON 다운로드</button></div>}
        </aside>
        <section className="result-workspace">
          <nav className="view-tabs" aria-label="결과 보기">{tabs.map((tab) => <button key={tab.id} className={activeView === tab.id ? "active" : ""} onClick={() => setActiveView(tab.id)}>{tab.label}</button>)}</nav>
          {!result ? <div className="empty-state result-empty">분석할 PDF와 페이지를 선택하세요.</div> : activeView === "layout" ? (
            <div className="layout-view"><div className="pane-header"><h2>레이아웃 시각화</h2><span>{result.page.page_id}페이지</span></div><img src={result.visualization_image} alt="레이아웃 분석 시각화" /></div>
          ) : activeView === "page" ? <PageDescriptionView result={result} /> : REVIEW_TYPES.includes(activeView) ? <AnalysisInspector result={result} type={activeView} /> : (
            <div className="json-view"><pre>{JSON.stringify({ ...result.page, semantic_analyses: result.semantic_analyses || [], page_description: result.page_description || null }, null, 2)}</pre></div>
          )}
        </section>
      </section>
    </main>
  );
}
