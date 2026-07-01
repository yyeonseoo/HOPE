import { useMemo, useState } from "react";
import "./styles.css";

const API_BASE = "http://127.0.0.1:8000";

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

export default function App() {
  const [file, setFile] = useState(null);
  const [pageCount, setPageCount] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [dpi, setDpi] = useState(120);
  const [layoutModel, setLayoutModel] = useState("doclayout_yolo");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  const blockStats = useMemo(() => {
    const blocks = result?.page?.blocks || [];
    return blocks.reduce((acc, block) => {
      acc[block.type] = (acc[block.type] || 0) + 1;
      return acc;
    }, {});
  }, [result]);

  async function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    setFile(nextFile);
    setPageCount(null);
    setResult(null);
    setError("");
    setPageNumber(1);

    if (!nextFile) return;
    setStatus("counting");
    const formData = new FormData();
    formData.append("file", nextFile);

    try {
      const response = await fetch(`${API_BASE}/api/page-count`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await parseError(response));
      const payload = await response.json();
      setPageCount(payload.page_count);
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  }

  async function analyzePage() {
    if (!file) {
      setError("먼저 PDF를 업로드하세요.");
      return;
    }

    setStatus("analyzing");
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);
    formData.append("page_number", String(pageNumber));
    formData.append("dpi", String(dpi));
    formData.append("lang", "korean");
    formData.append("layout_model", layoutModel);

    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        body: formData,
      });
      if (!response.ok) throw new Error(await parseError(response));
      const payload = await response.json();
      setResult(payload);
      setPageCount(payload.page_count);
    } catch (err) {
      setError(err.message);
    } finally {
      setStatus("idle");
    }
  }

  function downloadJson() {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result.page, null, 2)], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `page_${String(result.page.page_id).padStart(4, "0")}_layout.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const busy = status === "counting" || status === "analyzing";

  return (
    <main className="app-shell">
      <section className="toolbar">
        <div>
          <h1>경제수학 교과서 구조 분석</h1>
          <p>PDF를 업로드하고 한 페이지씩 레이아웃 탐지 결과를 확인합니다.</p>
        </div>
        <div className="status-pill">{busy ? "처리 중" : "대기"}</div>
      </section>

      <section className="workspace">
        <aside className="control-panel">
          <label className="file-drop">
            <input type="file" accept="application/pdf" onChange={handleFileChange} />
            <span className="file-title">{file ? file.name : "PDF 선택"}</span>
            <span className="file-meta">{file ? fileSizeLabel(file.size) : "교과서 PDF를 업로드하세요"}</span>
          </label>

          <div className="field-row">
            <label>
              페이지
              <input
                type="number"
                min="1"
                max={pageCount || 1}
                value={pageNumber}
                onChange={(event) => setPageNumber(Number(event.target.value))}
              />
            </label>
            <label>
              DPI
              <input
                type="number"
                min="120"
                max="300"
                step="20"
                value={dpi}
                onChange={(event) => setDpi(Number(event.target.value))}
              />
            </label>
          </div>

          <label className="model-field">
            Layout model
            <select value={layoutModel} onChange={(event) => setLayoutModel(event.target.value)}>
              <option value="doclayout_yolo">DocLayout-YOLO + 보정 규칙</option>
              <option value="doclayout_yolo_raw">DocLayout-YOLO 원본</option>
            </select>
          </label>

          <div className="page-count">
            <span>전체 페이지</span>
            <strong>{pageCount ?? "-"}</strong>
          </div>

          <button className="primary-button" disabled={busy || !file} onClick={analyzePage}>
            {status === "analyzing" ? "분석 중..." : "페이지 분석"}
          </button>

          {error && <div className="error-box">{error}</div>}

          {result && (
            <div className="stats">
              <div className="stats-header">
                <span>탐지 블록</span>
                <strong>{result.page.blocks.length}</strong>
              </div>
              <div className="type-list">
                {Object.entries(blockStats).map(([type, count]) => (
                  <div key={type}>
                    <span>{type}</span>
                    <strong>{count}</strong>
                  </div>
                ))}
              </div>
              <button className="secondary-button" onClick={downloadJson}>
                JSON 다운로드
              </button>
            </div>
          )}
        </aside>

        <section className="result-view">
          <div className="preview-pane">
            <div className="pane-header">
              <h2>시각화</h2>
              <span>{result ? `${result.page.page_id}페이지` : "분석 전"}</span>
            </div>
            {result ? (
              <img src={result.visualization_image} alt="레이아웃 분석 시각화" />
            ) : (
              <div className="empty-state">분석할 PDF와 페이지를 선택하세요.</div>
            )}
          </div>

          <div className="json-pane">
            <div className="pane-header">
              <h2>JSON</h2>
              <span>{result ? "생성됨" : "대기"}</span>
            </div>
            <pre>{result ? JSON.stringify(result.page, null, 2) : "{}"}</pre>
          </div>
        </section>
      </section>
    </main>
  );
}
