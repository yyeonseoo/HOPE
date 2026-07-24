import { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";
import {
  createTextbookProject,
  deleteWorkspacePage,
  getTextbookProject,
  listSavedPages,
  listTextbookProjects,
  projectFile,
  saveWorkspacePage,
} from "./workspaceStore";

const API_BASE = "http://127.0.0.1:8000";
const REVIEW_TYPES = ["formula", "table", "figure"];
const LAYOUT_MODEL_OPTIONS = [
  { value: "doclayout_yolo", label: "기본 보정 규칙", description: "일반 교과서에 권장" },
  { value: "doclayout_yolo_unit3", label: "3단원 맞춤 보정 규칙", description: "좌표평면과 그래프 단원" },
  { value: "doclayout_yolo_raw", label: "원본 모델 결과", description: "보정 없이 탐지 결과 확인" },
];

function fileSizeLabel(size) {
  if (!size) return "";
  if (size > 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`;
  return `${(size / 1024).toFixed(1)} KB`;
}

function savedAtLabel(value) {
  if (!value) return "";
  const date = new Date(value);
  return new Intl.DateTimeFormat("ko-KR", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
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

function ProcessingState({ status, elapsedSeconds }) {
  const counting = status === "counting";
  return (
    <div className="processing-state" role="status" aria-live="polite">
      <div className="processing-visual" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <p className="processing-eyebrow">{counting ? "PDF 준비 중" : "페이지 분석 진행 중"}</p>
      <h2>{counting ? "교과서 정보를 확인하고 있습니다" : "교과서의 구조와 의미를 분석하고 있습니다"}</h2>
      <p className="processing-copy">
        {counting
          ? "전체 페이지 수를 확인한 뒤 분석할 페이지를 선택할 수 있습니다."
          : "레이아웃과 본문을 구조화하고, 선택한 경우 Figure 접근성 설명까지 생성합니다."}
      </p>
      <div className="progress-track" aria-hidden="true"><span /></div>
      <div className="processing-meta">
        <span>창을 닫지 않아도 됩니다</span>
        <strong>{elapsedSeconds}초 경과</strong>
      </div>
    </div>
  );
}

function UploadIcon({ uploaded }) {
  if (!uploaded) return <span className="upload-arrow">↑</span>;
  return (
    <svg className="book-icon" viewBox="0 0 32 32" aria-hidden="true">
      <path d="M5 6.5c4.7-.8 8.3.2 11 2.4v17c-2.7-2.2-6.3-3.2-11-2.4v-17Z" />
      <path d="M27 6.5c-4.7-.8-8.3.2-11 2.4v17c2.7-2.2 6.3-3.2 11-2.4v-17Z" />
      <path d="M8.5 11.2c1.7-.1 3.1.2 4.4.8M8.5 15c1.7-.1 3.1.2 4.4.8M23.5 11.2c-1.7-.1-3.1.2-4.4.8M23.5 15c-1.7-.1-3.1.2-4.4.8" />
    </svg>
  );
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
      <div><strong>접근성 설명</strong><p>{description.long_text || description.short_text || "없음"}</p></div>
      <div><strong>점역 참고</strong><p>{description.transcription_notes || "없음"}</p></div>
      <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
    </div>
  );
}

function PageSourceViewer({ result, selectedFigure, onClearFigure }) {
  const [imageSize, setImageSize] = useState(null);
  const [magnifierEnabled, setMagnifierEnabled] = useState(false);
  const [magnifier, setMagnifier] = useState(null);
  const [pageZoom, setPageZoom] = useState(100);
  const bbox = selectedFigure?.bbox;
  const magnifierWidth = 220;
  const magnifierHeight = 150;
  const magnifierZoom = 2.25;
  const overlayStyle = bbox && imageSize ? {
    left: `${(bbox[0] / imageSize.width) * 100}%`,
    top: `${(bbox[1] / imageSize.height) * 100}%`,
    width: `${((bbox[2] - bbox[0]) / imageSize.width) * 100}%`,
    height: `${((bbox[3] - bbox[1]) / imageSize.height) * 100}%`,
  } : null;

  useEffect(() => {
    setMagnifier(null);
    setMagnifierEnabled(false);
    setPageZoom(100);
  }, [result.page_image]);

  function updateMagnifier(event) {
    if (!magnifierEnabled) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const scale = pageZoom / 100;
    const imageWidth = rect.width / scale;
    const imageHeight = rect.height / scale;
    const lensWidth = magnifierWidth / scale;
    const lensHeight = magnifierHeight / scale;
    const x = Math.max(0, Math.min(imageWidth, (event.clientX - rect.left) / scale));
    const y = Math.max(0, Math.min(imageHeight, (event.clientY - rect.top) / scale));
    const left = Math.max(0, Math.min(imageWidth - lensWidth, x - lensWidth / 2));
    const top = Math.max(0, Math.min(imageHeight - lensHeight, y - lensHeight / 2));
    setMagnifier({
      left,
      top,
      width: lensWidth,
      height: lensHeight,
      backgroundSize: `${imageWidth * magnifierZoom}px ${imageHeight * magnifierZoom}px`,
      backgroundPosition: `${x - left - x * magnifierZoom}px ${y - top - y * magnifierZoom}px`,
    });
  }

  function changePageZoom(nextZoom) {
    setPageZoom(Math.max(75, Math.min(200, nextZoom)));
    setMagnifier(null);
  }

  return (
    <section className="page-source-pane">
      <div className="page-review-heading">
        <div><span>원본 교과서</span><h3>{result.page?.page_id}페이지</h3></div>
        <div className="page-actions">
          {selectedFigure && <button className="text-button" onClick={onClearFigure}>전체 페이지 보기</button>}
          <div className="page-zoom-controls" aria-label="교과서 페이지 확대 및 축소">
            <button
              type="button"
              aria-label="페이지 축소"
              disabled={pageZoom <= 75}
              onClick={() => changePageZoom(pageZoom - 25)}
            >
              −
            </button>
            <button
              type="button"
              className="page-zoom-value"
              title="원래 크기로 돌아가기"
              onClick={() => changePageZoom(100)}
            >
              {pageZoom}%
            </button>
            <button
              type="button"
              aria-label="페이지 확대"
              disabled={pageZoom >= 200}
              onClick={() => changePageZoom(pageZoom + 25)}
            >
              +
            </button>
          </div>
          <button
            type="button"
            className={`magnifier-toggle ${magnifierEnabled ? "active" : ""}`}
            aria-pressed={magnifierEnabled}
            onClick={() => {
              setMagnifierEnabled((current) => !current);
              setMagnifier(null);
            }}
          >
            <span className="magnifier-icon" aria-hidden="true" />
            {magnifierEnabled ? "돋보기 끄기" : "돋보기"}
          </button>
        </div>
      </div>
      <div className={`page-image-stage ${pageZoom > 100 ? "page-zoomed" : ""}`}>
        <div
          className={`page-image-wrap ${magnifierEnabled ? "magnifier-active" : ""}`}
          style={{ transform: `scale(${pageZoom / 100})` }}
        >
          <img
            src={result.page_image}
            alt={`${result.page?.page_id}페이지 원본 교과서`}
            onLoad={(event) => setImageSize({
              width: event.currentTarget.naturalWidth,
              height: event.currentTarget.naturalHeight,
            })}
            onMouseEnter={updateMagnifier}
            onMouseMove={updateMagnifier}
            onMouseLeave={() => setMagnifier(null)}
          />
          {overlayStyle && <span className="figure-highlight" style={overlayStyle} aria-hidden="true" />}
          {magnifierEnabled && magnifier && (
            <span
              className="page-magnifier"
              aria-hidden="true"
              style={{
                left: magnifier.left,
                top: magnifier.top,
                width: magnifier.width,
                height: magnifier.height,
                backgroundImage: `url("${result.page_image}")`,
                backgroundSize: magnifier.backgroundSize,
                backgroundPosition: magnifier.backgroundPosition,
              }}
            />
          )}
        </div>
        {selectedFigure && (
          <div className="figure-popover" role="dialog" aria-label="선택한 Figure 원본">
            <div className="figure-popover-header">
              <div><small>선택한 원본 영역</small><strong>{selectedFigure.block_id}</strong></div>
              <button aria-label="Figure 원본 닫기" onClick={onClearFigure}>×</button>
            </div>
            <BlockCrop
              imageUrl={result.page_image}
              bbox={selectedFigure.bbox}
              alt={`${selectedFigure.block_id} 원본 Figure`}
            />
          </div>
        )}
      </div>
    </section>
  );
}

function LinkedPageDescription({ text, figures, onSelectFigure, selectedFigure }) {
  let figureIndex = 0;
  return String(text || "").split(/(\[figure\])/gi).map((segment, index) => {
    if (!/^\[figure\]$/i.test(segment)) return <span key={index}>{segment}</span>;
    const figure = figures[figureIndex++];
    if (!figure) return <span key={index} className="block-token">Figure</span>;
    return (
      <button
        key={index}
        className={`figure-reference ${selectedFigure?.block_id === figure.block_id ? "active" : ""}`}
        onClick={() => onSelectFigure(figure)}
        title={`${figure.block_id} 원본 보기`}
      >
        <span aria-hidden="true">▧</span> Figure 원본 보기
      </button>
    );
  });
}

function PageDescriptionView({ result, onUpdateDescription }) {
  const description = result.page_description;
  const [selectedFigure, setSelectedFigure] = useState(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(description?.text || "");
  const figureBlocks = useMemo(
    () => (result.page?.blocks || []).filter((block) => block.type === "figure"),
    [result]
  );

  useEffect(() => {
    setSelectedFigure(null);
    setEditing(false);
    setDraft(description?.text || "");
  }, [result, description?.text]);

  if (!description || description.status === "failed") {
    return <div className="empty-state result-empty">이 페이지에서 읽을 수 있는 내용을 찾지 못했습니다.</div>;
  }

  function applyDraft() {
    onUpdateDescription(draft);
    setEditing(false);
  }

  return (
    <div className="page-review">
      <PageSourceViewer
        result={result}
        selectedFigure={selectedFigure}
        onClearFigure={() => setSelectedFigure(null)}
      />
      <section className="page-description-pane">
        <div className="page-review-heading">
          <div><span>접근성 자료 초안</span><h3>페이지 전체 설명</h3></div>
          <div className="page-actions">
            {editing ? (
              <>
                <button className="text-button" onClick={() => { setDraft(description.text || ""); setEditing(false); }}>취소</button>
                <button className="save-button" onClick={applyDraft}>수정 적용</button>
              </>
            ) : <button className="edit-button" onClick={() => setEditing(true)}>설명 수정</button>}
          </div>
        </div>
        <div className="page-description-body">
          <div className="review-guide">
            <span aria-hidden="true">i</span>
            <p>설명의 <strong>Figure 원본 보기</strong>를 누르면 왼쪽에서 실제 영역을 바로 확인할 수 있습니다.</p>
          </div>
          {editing ? (
            <textarea
              className="page-description-editor"
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              aria-label="페이지 접근성 설명 수정"
            />
          ) : (
            <p className="page-description-text">
              <LinkedPageDescription
                text={description.text || "없음"}
                figures={figureBlocks}
                selectedFigure={selectedFigure}
                onSelectFigure={setSelectedFigure}
              />
            </p>
          )}
          <div className="review-footer">
            <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
            <span>{description.was_generated ? "모델 다듬기 적용" : "블록 원문 이어붙임"}</span>
          </div>
          {description.warnings?.length > 0 && (
            <ul className="page-warning-list">
              {description.warnings.map((warning, index) => <li key={index}>{warning}</li>)}
            </ul>
          )}
        </div>
      </section>
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
        자동 검수 경고
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

function LayoutModelSelect({ value, onChange }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  const optionRefs = useRef([]);
  const selectedIndex = Math.max(0, LAYOUT_MODEL_OPTIONS.findIndex((option) => option.value === value));
  const selected = LAYOUT_MODEL_OPTIONS[selectedIndex];

  useEffect(() => {
    if (!open) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!rootRef.current?.contains(event.target)) setOpen(false);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") {
        setOpen(false);
        rootRef.current?.querySelector(".layout-select-trigger")?.focus();
      }
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("mousedown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [open]);

  function openAndFocus(index = selectedIndex) {
    setOpen(true);
    window.requestAnimationFrame(() => optionRefs.current[index]?.focus());
  }

  function handleOptionKeyDown(event, index) {
    if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) return;
    event.preventDefault();
    const last = LAYOUT_MODEL_OPTIONS.length - 1;
    const nextIndex = event.key === "Home" ? 0
      : event.key === "End" ? last
        : event.key === "ArrowDown" ? Math.min(last, index + 1)
          : Math.max(0, index - 1);
    optionRefs.current[nextIndex]?.focus();
  }

  return (
    <div className="layout-select-control" ref={rootRef}>
      <button
        type="button"
        className={`layout-select-trigger ${open ? "open" : ""}`}
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => (open ? setOpen(false) : openAndFocus())}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "ArrowUp") {
            event.preventDefault();
            openAndFocus(event.key === "ArrowDown" ? selectedIndex : LAYOUT_MODEL_OPTIONS.length - 1);
          }
        }}
      >
        <span>{selected.label}</span>
        <span className="layout-select-chevron" aria-hidden="true" />
      </button>
      {open && (
        <div className="layout-options" role="listbox" aria-label="Layout 분석 방식">
          {LAYOUT_MODEL_OPTIONS.map((option, index) => (
            <button
              type="button"
              role="option"
              aria-selected={option.value === value}
              className={`layout-option ${option.value === value ? "selected" : ""}`}
              key={option.value}
              ref={(element) => { optionRefs.current[index] = element; }}
              onClick={() => {
                onChange(option.value);
                setOpen(false);
              }}
              onKeyDown={(event) => handleOptionKeyDown(event, index)}
            >
              <span className="layout-option-copy">
                <strong>{option.label}</strong>
                <small>{option.description}</small>
              </span>
              {option.value === value && <span className="layout-option-check" aria-hidden="true">✓</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [projects, setProjects] = useState([]);
  const [projectPageCounts, setProjectPageCounts] = useState({});
  const [activeProjectId, setActiveProjectId] = useState(null);
  const [savedPages, setSavedPages] = useState([]);
  const [resultOwnerId, setResultOwnerId] = useState(null);
  const [screen, setScreen] = useState("library");
  const [pageCount, setPageCount] = useState(null);
  const [pageNumber, setPageNumber] = useState(1);
  const [dpi, setDpi] = useState(120);
  const [layoutModel, setLayoutModel] = useState("doclayout_yolo");
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("layout");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const projectFileInputRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    async function restoreWorkspace() {
      try {
        const storedProjects = await listTextbookProjects();
        if (cancelled) return;
        setProjects(storedProjects);
        const pageLists = await Promise.all(storedProjects.map((project) => listSavedPages(project.id)));
        if (cancelled) return;
        setProjectPageCounts(Object.fromEntries(
          storedProjects.map((project, index) => [project.id, pageLists[index].length]),
        ));
      } catch (err) {
        if (!cancelled) setError(`저장된 작업을 불러오지 못했습니다. ${err.message}`);
      }
    }
    restoreWorkspace();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (status !== "counting" && status !== "analyzing") {
      setElapsedSeconds(0);
      return undefined;
    }
    const startedAt = Date.now();
    setElapsedSeconds(0);
    const timer = window.setInterval(() => {
      setElapsedSeconds(Math.floor((Date.now() - startedAt) / 1000));
    }, 1000);
    return () => window.clearInterval(timer);
  }, [status]);

  const blockStats = useMemo(() => {
    const blocks = result?.page?.blocks || [];
    return blocks.reduce((acc, block) => {
      acc[block.type] = (acc[block.type] || 0) + 1;
      return acc;
    }, {});
  }, [result]);

  async function openProject(projectId, preferredPageNumber = null) {
    setResult(null);
    setResultOwnerId(null);
    setError("");
    try {
      const [project, pages] = await Promise.all([
        getTextbookProject(projectId),
        listSavedPages(projectId),
      ]);
      if (!project) throw new Error("저장된 교과서 정보를 찾을 수 없습니다.");

      setActiveProjectId(projectId);
      setScreen("workspace");
      setFile(projectFile(project));
      setPageCount(project.pageCount);
      setSavedPages(pages);

      const restoredPage = preferredPageNumber
        ? pages.find((page) => page.pageNumber === preferredPageNumber)
        : pages.at(-1);

      if (restoredPage) {
        setPageNumber(restoredPage.pageNumber);
        setDpi(restoredPage.settings?.dpi || 120);
        setLayoutModel(restoredPage.settings?.layoutModel || "doclayout_yolo");
        setResult(restoredPage.result);
        setResultOwnerId(projectId);
        setActiveView("page");
      } else {
        setPageNumber(1);
        setActiveView("layout");
      }
    } catch (err) {
      setError(`교과서 작업을 열지 못했습니다. ${err.message}`);
    }
  }

  function openSavedPage(page) {
    setPageNumber(page.pageNumber);
    setDpi(page.settings?.dpi || 120);
    setLayoutModel(page.settings?.layoutModel || "doclayout_yolo");
    setResult(page.result);
    setResultOwnerId(page.projectId);
    setActiveView("page");
    setError("");
  }

  async function deleteSavedPage(page) {
    const confirmed = window.confirm(`${page.pageNumber}페이지의 저장된 분석 결과를 삭제할까요?`);
    if (!confirmed) return;

    try {
      const visibleResultPage = Number(result?.page?.page_id);
      if (resultOwnerId === page.projectId && visibleResultPage === page.pageNumber) {
        setResult(null);
        setResultOwnerId(null);
        setActiveView("layout");
      }
      await deleteWorkspacePage(page.projectId, page.pageNumber);
      setSavedPages((current) => current.filter((item) => item.pageNumber !== page.pageNumber));
      setProjectPageCounts((current) => ({
        ...current,
        [page.projectId]: Math.max(0, (current[page.projectId] || 0) - 1),
      }));
    } catch (err) {
      setError(`저장된 페이지를 삭제하지 못했습니다. ${err.message}`);
    }
  }

  async function handleFileChange(event) {
    const nextFile = event.target.files?.[0] || null;
    event.target.value = "";
    if (!nextFile) return;

    const existingProject = projects.find((project) => (
      project.fileName === nextFile.name
      && project.fileSize === nextFile.size
      && project.fileLastModified === nextFile.lastModified
    ));
    if (existingProject) {
      await openProject(existingProject.id);
      return;
    }

    setFile(nextFile);
    setPageCount(null);
    setResult(null);
    setResultOwnerId(null);
    setError("");
    setPageNumber(1);
    setStatus("counting");
    const formData = new FormData(); formData.append("file", nextFile);
    try {
      const response = await fetch(`${API_BASE}/api/page-count`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await parseError(response));
      const count = (await response.json()).page_count;
      const project = await createTextbookProject(nextFile, count);
      setProjects((current) => [project, ...current]);
      setProjectPageCounts((current) => ({ ...current, [project.id]: 0 }));
      setActiveProjectId(project.id);
      setScreen("workspace");
      setSavedPages([]);
      setPageCount(count);
    } catch (err) { setError(err.message); } finally { setStatus("idle"); }
  }

  async function analyzePage() {
    if (!file) { setError("먼저 PDF를 업로드하세요."); return; }
    setStatus("analyzing"); setError(""); setResult(null);
    const formData = new FormData();
    formData.append("file", file); formData.append("page_number", String(pageNumber));
    formData.append("dpi", String(dpi)); formData.append("lang", "korean"); formData.append("layout_model", layoutModel);
    formData.append("figure_captioning", "true");
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, { method: "POST", body: formData });
      if (!response.ok) throw new Error(await parseError(response));
      const payload = await response.json();
      setResult(payload);
      setResultOwnerId(activeProjectId);
      setPageCount(payload.page_count);
      setActiveView("layout");

      if (activeProjectId) {
        const savedPage = await saveWorkspacePage(activeProjectId, pageNumber, payload, { dpi, layoutModel });
        setSavedPages((current) => (
          [...current.filter((page) => page.pageNumber !== pageNumber), savedPage]
            .sort((a, b) => a.pageNumber - b.pageNumber)
        ));
        setProjectPageCounts((current) => ({
          ...current,
          [activeProjectId]: new Set([...savedPages.map((page) => page.pageNumber), pageNumber]).size,
        }));
        setProjects((current) => current.map((project) => (
          project.id === activeProjectId ? { ...project, updatedAt: savedPage.savedAt } : project
        )));
      }
    } catch (err) { setError(err.message); } finally { setStatus("idle"); }
  }

  useEffect(() => {
    if (!result || !activeProjectId || resultOwnerId !== activeProjectId) return undefined;
    const analyzedPageNumber = Number(result?.page?.page_id);
    if (!Number.isInteger(analyzedPageNumber) || analyzedPageNumber < 1) return undefined;
    const timer = window.setTimeout(async () => {
      try {
        const savedPage = await saveWorkspacePage(
          activeProjectId,
          analyzedPageNumber,
          result,
          { dpi, layoutModel },
        );
        setSavedPages((current) => (
          [...current.filter((page) => page.pageNumber !== analyzedPageNumber), savedPage]
            .sort((a, b) => a.pageNumber - b.pageNumber)
        ));
      } catch (err) {
        setError(`페이지 결과를 자동 저장하지 못했습니다. ${err.message}`);
      }
    }, 500);
    return () => window.clearTimeout(timer);
  }, [activeProjectId, result, resultOwnerId]);

  function updatePageDescription(text) {
    setResult((current) => {
      if (!current?.page_description) return current;
      return {
        ...current,
        page_description: {
          ...current.page_description,
          text,
          review_status: "needs_review",
        },
      };
    });
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
  const tabs = [{ id: "layout", label: "Layout" }, { id: "formula", label: "Formula" }, { id: "table", label: "Table" }, { id: "figure", label: "Figure" }, { id: "json", label: "JSON" }, { id: "page", label: "Page" }];

  return (
    <main className="app-shell">
      <section className="toolbar">
        <div className="brand">
          <div className="brand-mark" aria-hidden="true">H</div>
          <div>
            <div className="brand-line"><h1>HOPE</h1><span>Textbook Accessibility</span></div>
            <p>교과서를 구조화하고 접근성 자료 제작을 돕습니다.</p>
          </div>
        </div>
        <div className={`status-pill ${busy ? "busy" : ""}`}><span />{busy ? "분석 진행 중" : "분석 준비"}</div>
      </section>
      <input
        ref={projectFileInputRef}
        className="visually-hidden"
        type="file"
        accept="application/pdf"
        onChange={handleFileChange}
      />
      {screen === "library" ? (
        <section className="textbook-library">
          <div className="library-heading">
            <div>
              <span>나의 작업공간</span>
              <h2>교과서 보관함</h2>
              <p>교과서를 선택해 분석을 이어가거나 새로운 교과서를 추가하세요.</p>
            </div>
          </div>
          <div className="textbook-grid">
            {projects.map((project, index) => (
              <button
                key={project.id}
                type="button"
                className="textbook-card"
                onClick={() => openProject(project.id)}
              >
                <span className={`textbook-cover cover-${(index % 4) + 1}`} aria-hidden="true">
                  <span>HOPE</span>
                  <strong>{project.name.replace(/\.pdf$/i, "").slice(0, 18)}</strong>
                  <i />
                </span>
                <span className="textbook-card-copy">
                  <strong>{project.name.replace(/\.pdf$/i, "")}</strong>
                  <small>{project.pageCount}페이지 · {fileSizeLabel(project.fileSize)}</small>
                  <span>
                    <em>{projectPageCounts[project.id] || 0}페이지 저장</em>
                    <time>{savedAtLabel(project.updatedAt)}</time>
                  </span>
                </span>
                <span className="textbook-open" aria-hidden="true">→</span>
              </button>
            ))}
            {projects.length > 0 && (
              <button type="button" className="new-textbook-card" onClick={() => projectFileInputRef.current?.click()}>
                <span aria-hidden="true">＋</span>
                <strong>새 교과서 추가</strong>
                <small>PDF 파일을 불러와 작업공간을 만듭니다.</small>
              </button>
            )}
          </div>
          {projects.length === 0 && (
            <button type="button" className="library-guide" onClick={() => projectFileInputRef.current?.click()}>
              <span aria-hidden="true">▥</span>
              <p>아직 저장된 교과서가 없습니다.<br />첫 교과서 PDF를 추가해 분석을 시작하세요.</p>
            </button>
          )}
        </section>
      ) : (
        <>
          <section className="workspace-heading">
            <button type="button" onClick={() => setScreen("library")}><span aria-hidden="true">←</span> 교과서 보관함</button>
            <div>
              <h2>{file?.name.replace(/\.pdf$/i, "")}</h2>
              <p>{savedPages.length}개 페이지 저장됨</p>
            </div>
          </section>
          <section className="workspace">
        <aside className="control-panel">
          <div className="panel-heading">
            <span>새 분석</span>
            <h2>교과서 PDF 설정</h2>
            <p>분석할 파일과 페이지를 선택하세요.</p>
          </div>
          <div className={`file-drop ${file ? "has-file" : ""}`}>
            <span className="upload-icon" aria-hidden="true"><UploadIcon uploaded={Boolean(file)} /></span>
            <span className="file-title">{file?.name}</span>
            <span className="file-meta">{file ? `${fileSizeLabel(file.size)} · 작업공간에 저장됨` : ""}</span>
          </div>
          <div className="settings-card">
            <div className="settings-title"><strong>분석 범위</strong><span>Layout 분석</span></div>
            <div className="field-row">
              <label>페이지<input type="number" min="1" max={pageCount || 1} value={pageNumber} onChange={(event) => setPageNumber(Number(event.target.value))} /></label>
              <label>해상도(DPI)<input type="number" min="120" max="300" step="20" value={dpi} onChange={(event) => setDpi(Number(event.target.value))} /></label>
            </div>
            <div className="layout-select-field">
              <span>Layout 분석 방식</span>
              <LayoutModelSelect value={layoutModel} onChange={setLayoutModel} />
            </div>
            <div className="page-count"><span>전체 페이지</span><strong>{pageCount ?? "-"}</strong></div>
          </div>
          <button className="primary-button" disabled={busy || !file} onClick={analyzePage}>
            <span>{status === "analyzing" ? "분석 중" : "페이지 분석 시작"}</span>
            <span aria-hidden="true">{status === "analyzing" ? "···" : "→"}</span>
          </button>
          {error && <div className="error-box">{error}</div>}
          {activeProjectId && (
            <section className="saved-pages-card">
              <div className="saved-pages-header">
                <span>저장된 페이지</span>
                <strong>{savedPages.length}</strong>
              </div>
              {savedPages.length > 0 ? (
                <div className="saved-page-list">
                  {savedPages.map((page) => (
                    <div
                      key={page.id}
                      className={`saved-page-item ${
                        resultOwnerId === page.projectId
                        && Number(result?.page?.page_id) === page.pageNumber
                          ? "active"
                          : ""
                      }`}
                    >
                      <button type="button" className="saved-page-open" onClick={() => openSavedPage(page)}>
                        <span><strong>{page.pageNumber}</strong> 페이지</span>
                        <small>{page.reviewStatus === "reviewed" ? "검수 완료" : "자동 저장"}</small>
                      </button>
                      <button
                        type="button"
                        className="saved-page-delete"
                        onClick={() => deleteSavedPage(page)}
                        aria-label={`${page.pageNumber}페이지 저장 결과 삭제`}
                        title="저장 결과 삭제"
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <p>분석을 완료한 페이지가 여기에 자동 저장됩니다.</p>
              )}
            </section>
          )}
          {result && <div className="stats"><div className="stats-header"><span>탐지 블록</span><strong>{result.page.blocks.length}</strong></div><div className="type-list">{Object.entries(blockStats).map(([type, count]) => <div key={type}><span>{type}</span><strong>{count}</strong></div>)}</div><button className="secondary-button" onClick={downloadJson}>JSON 다운로드</button></div>}
        </aside>
        <section className={`result-workspace ${activeView === "page" ? "page-workspace" : ""}`}>
          <nav className="view-tabs" aria-label="결과 보기">{tabs.map((tab) => <button key={tab.id} className={activeView === tab.id ? "active" : ""} onClick={() => setActiveView(tab.id)}>{tab.label}</button>)}</nav>
          {busy ? <ProcessingState status={status} elapsedSeconds={elapsedSeconds} /> : !result ? (
            <div className="empty-state result-empty">
              <div className="empty-illustration" aria-hidden="true"><span /><span /><span /></div>
              <h2>분석 결과가 여기에 표시됩니다</h2>
              <p>왼쪽에서 교과서 PDF와 페이지를 선택한 뒤 분석을 시작하세요.</p>
            </div>
          ) : activeView === "layout" ? (
            <div className="layout-view"><div className="pane-header"><h2>레이아웃 시각화</h2><span>{result.page.page_id}페이지</span></div><img src={result.visualization_image} alt="레이아웃 분석 시각화" /></div>
          ) : activeView === "page" ? <PageDescriptionView result={result} onUpdateDescription={updatePageDescription} /> : REVIEW_TYPES.includes(activeView) ? <AnalysisInspector result={result} type={activeView} /> : (
            <div className="json-view"><pre>{JSON.stringify({ ...result.page, semantic_analyses: result.semantic_analyses || [], page_description: result.page_description || null }, null, 2)}</pre></div>
          )}
        </section>
          </section>
        </>
      )}
    </main>
  );
}
