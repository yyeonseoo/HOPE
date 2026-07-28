import { useEffect, useMemo, useRef, useState } from "react";
import "./styles.css";
import "./libraryEnhancements.css";
import {
  accessibleHtmlFilename,
  buildAccessibleTextbookHtml,
  downloadAccessibleHtml,
} from "./accessibleHtml";
import {
  moveBrailleReadingGroup,
  placeBrailleReadingGroup,
  reorderTaggedPageDescription,
  sortBlocksForBrailleReading,
} from "./brailleReadingOrder";
import {
  createTextbookProject,
  deleteTextbookProject,
  deleteWorkspacePage,
  getTextbookProject,
  listSavedPages,
  listTextbookProjects,
  projectFile,
  saveWorkspacePage,
  updateWorkspacePageReviewStatus,
} from "./workspaceStore";

const API_BASE = "http://127.0.0.1:8000";
const REVIEW_TYPES = ["formula", "table", "figure"];
const BOOK_COLORS = [
  { id: "leaf", label: "새싹", value: "#8bcf64" },
  { id: "sky", label: "하늘", value: "#83c6d2" },
  { id: "sun", label: "햇살", value: "#f2c972" },
  { id: "peach", label: "복숭아", value: "#e6a6a2" },
  { id: "lilac", label: "라일락", value: "#b7a8db" },
];
const LIBRARY_META_KEY = "hope-library-meta-v1";
const LIBRARY_GROUPS_KEY = "hope-library-groups-v1";
const ONBOARDING_KEY = "hope-onboarding-seen-v1";
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

function loadLocalValue(key, fallback) {
  try {
    const value = window.localStorage.getItem(key);
    return value ? JSON.parse(value) : fallback;
  } catch {
    return fallback;
  }
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

function changeTaggedDescriptionLine(text, blocks, blockId, nextType = null) {
  const orderedBlocks = sortBlocksForBrailleReading(blocks || []);
  const block = orderedBlocks.find((item) => item.block_id === blockId);
  if (!block) return text;
  const sameTypeIndex = orderedBlocks
    .filter((item) => item.type === block.type)
    .findIndex((item) => item.block_id === blockId);
  if (sameTypeIndex < 0) return text;

  const lines = String(text || "").split(/\r?\n/);
  let currentIndex = -1;
  const lineIndex = lines.findIndex((line) => {
    const match = line.match(/^\s*\[([a-z_]+)\]/i);
    if (match?.[1]?.toLowerCase() !== block.type) return false;
    currentIndex += 1;
    return currentIndex === sameTypeIndex;
  });
  if (lineIndex < 0) return text;
  if (!nextType) lines.splice(lineIndex, 1);
  else lines[lineIndex] = lines[lineIndex].replace(
    /^(\s*)\[([a-z_]+)\]/i,
    `$1[${nextType}]`,
  );
  return lines.join("\n");
}

function taggedLineSignature(text) {
  return String(text || "")
    .split(/\r?\n/)
    .map((line) => line.match(/^\s*\[([a-z_]+)\]/i)?.[1]?.toLowerCase() || null)
    .filter(Boolean);
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

// Cutoffs come from the actual detector-score distribution across a sample
// textbook chapter (17 pages, 83 figure/table/formula blocks): scores cluster
// into three bands with gaps at ~0.31-0.39 and ~0.72-0.76, so 0.35/0.75 split
// the bands at their midpoints rather than at round arbitrary numbers.
function ConfidenceLevel({ value }) {
  if (typeof value !== "number") return null;
  const level = value >= 0.75
    ? { key: "high", label: "높음" }
    : value >= 0.35
      ? { key: "medium", label: "보통" }
      : { key: "low", label: "낮음" };
  return <span className={`confidence-level ${level.key}`}>{level.label}</span>;
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
    </dl>
  );
}

function DescriptionResult({ description, captioningEnabled, type }) {
  if (!description || description.status === "not_started") {
    return (
      <div className="pending-box">
        {captioningEnabled ? "설명 생성 결과가 없습니다. 경고와 백엔드 로그를 확인하세요." : "왼쪽에서 Figure 설명 생성을 활성화한 뒤 다시 분석하세요."}
      </div>
    );
  }
  return (
    <div className="description-output">
      {type === "figure" && (
        <dl className="description-metrics">
          <div><dt>생성 모델</dt><dd>{description.model?.name || "미제공"}</dd></div>
          <div><dt>생성 시간</dt><dd><Seconds value={description.generation_time_seconds} /></dd></div>
        </dl>
      )}
      <div><strong>접근성 설명</strong><p>{description.long_text || description.short_text || "없음"}</p></div>
      <div><strong>점역 참고</strong><p>{description.transcription_notes || "없음"}</p></div>
      <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
    </div>
  );
}

function PageSourceViewer({
  result,
  selectedFigure,
  onClearFigure,
  selectedBlock,
  onSelectBlock,
  onUpdateBlock,
  onAddFigure,
  onDeleteBlock,
}) {
  const [imageSize, setImageSize] = useState(null);
  const [magnifierEnabled, setMagnifierEnabled] = useState(false);
  const [magnifier, setMagnifier] = useState(null);
  const [pageZoom, setPageZoom] = useState(100);
  const [structureEditing, setStructureEditing] = useState(false);
  const [structureBlock, setStructureBlock] = useState(null);
  const [draftRegion, setDraftRegion] = useState(null);
  const [redrawingBlock, setRedrawingBlock] = useState(false);
  const bbox = selectedFigure?.bbox;
  const magnifierWidth = 300;
  const magnifierHeight = 210;
  const magnifierZoom = 2.25;
  const overlayStyle = bbox && imageSize ? {
    left: `${(bbox[0] / imageSize.width) * 100}%`,
    top: `${(bbox[1] / imageSize.height) * 100}%`,
    width: `${((bbox[2] - bbox[0]) / imageSize.width) * 100}%`,
    height: `${((bbox[3] - bbox[1]) / imageSize.height) * 100}%`,
  } : null;
  const editableBlocks = imageSize ? (result.page?.blocks || []).filter((block) => (
    Array.isArray(block.bbox) && block.bbox.length === 4
  )) : [];

  useEffect(() => {
    setMagnifier(null);
    setMagnifierEnabled(false);
    setPageZoom(100);
    setStructureEditing(false);
    setStructureBlock(null);
    setDraftRegion(null);
    setRedrawingBlock(false);
  }, [result.page_image]);

  useEffect(() => {
    if (!structureBlock) return;
    const refreshed = result.page?.blocks?.find(
      (block) => block.block_id === structureBlock.block_id,
    );
    setStructureBlock(refreshed || null);
  }, [result.page?.blocks, structureBlock?.block_id]);

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

  function imagePoint(event) {
    if (!imageSize) return null;
    const bounds = event.currentTarget.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(
        imageSize.width,
        ((event.clientX - bounds.left) / bounds.width) * imageSize.width,
      )),
      y: Math.max(0, Math.min(
        imageSize.height,
        ((event.clientY - bounds.top) / bounds.height) * imageSize.height,
      )),
    };
  }

  function normalizedDraftBbox() {
    if (!draftRegion) return null;
    const x1 = Math.round(Math.min(draftRegion.start.x, draftRegion.end.x));
    const y1 = Math.round(Math.min(draftRegion.start.y, draftRegion.end.y));
    const x2 = Math.round(Math.max(draftRegion.start.x, draftRegion.end.x));
    const y2 = Math.round(Math.max(draftRegion.start.y, draftRegion.end.y));
    return x2 - x1 >= 8 && y2 - y1 >= 8 ? [x1, y1, x2, y2] : null;
  }

  const draftBbox = normalizedDraftBbox();
  const draftStyle = draftBbox && imageSize ? {
    left: `${(draftBbox[0] / imageSize.width) * 100}%`,
    top: `${(draftBbox[1] / imageSize.height) * 100}%`,
    width: `${((draftBbox[2] - draftBbox[0]) / imageSize.width) * 100}%`,
    height: `${((draftBbox[3] - draftBbox[1]) / imageSize.height) * 100}%`,
  } : null;

  return (
    <section className="page-source-pane">
      <div className="page-review-heading">
        <div><span>원본 교과서</span><h3>{result.page?.page_id}페이지</h3></div>
        <div className="page-actions">
          {selectedFigure && <button className="text-button" onClick={onClearFigure}>전체 페이지 보기</button>}
          <button
            type="button"
            className={`structure-edit-toggle ${structureEditing ? "active" : ""}`}
            aria-pressed={structureEditing}
            onClick={() => {
              setStructureEditing((current) => !current);
              setStructureBlock(null);
              setDraftRegion(null);
              setRedrawingBlock(false);
              setMagnifierEnabled(false);
              setMagnifier(null);
            }}
          >
            {structureEditing ? "구조 수정 종료" : "구조 수정"}
          </button>
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
              {pageZoom === 100 ? "맞춤" : `${pageZoom}%`}
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
          className={[
            "page-image-wrap",
            magnifierEnabled ? "magnifier-active" : "",
            structureEditing ? "structure-editing" : "",
          ].filter(Boolean).join(" ")}
          style={{ transform: `scale(${pageZoom / 100})` }}
          onPointerDown={(event) => {
            if (!structureEditing || event.target.closest(".page-edit-region")) return;
            const point = imagePoint(event);
            if (!point) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            setDraftRegion({ start: point, end: point });
            if (!redrawingBlock) setStructureBlock(null);
          }}
          onPointerMove={(event) => {
            if (!structureEditing || !draftRegion) return;
            const point = imagePoint(event);
            if (point) setDraftRegion((current) => ({ ...current, end: point }));
          }}
          onPointerUp={(event) => {
            if (!structureEditing || !draftRegion) return;
            if (event.currentTarget.hasPointerCapture(event.pointerId)) {
              event.currentTarget.releasePointerCapture(event.pointerId);
            }
          }}
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
          {!magnifierEnabled && editableBlocks.map((block) => (
            <button
              type="button"
              className={[
                "page-edit-region",
                selectedBlock?.block_id === block.block_id ? "active" : "",
                structureEditing ? "structure-visible" : "",
                structureBlock?.block_id === block.block_id ? "structure-selected" : "",
              ].filter(Boolean).join(" ")}
              key={block.block_id}
              style={{
                left: `${(block.bbox[0] / imageSize.width) * 100}%`,
                top: `${(block.bbox[1] / imageSize.height) * 100}%`,
                width: `${((block.bbox[2] - block.bbox[0]) / imageSize.width) * 100}%`,
                height: `${((block.bbox[3] - block.bbox[1]) / imageSize.height) * 100}%`,
              }}
              onClick={(event) => {
                event.stopPropagation();
                if (structureEditing) {
                  setStructureBlock(block);
                  setDraftRegion(null);
                  setRedrawingBlock(false);
                } else {
                  onSelectBlock(block);
                }
              }}
              aria-label={structureEditing ? `${block.type} 구조 선택` : "설명 수정"}
              title={structureEditing ? `${block.type} · ${block.block_id}` : "설명 수정"}
            >
              {structureEditing && <span>{block.type}</span>}
            </button>
          ))}
          {structureEditing && draftStyle && (
            <span className="structure-draft-region" style={draftStyle} aria-hidden="true" />
          )}
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
        {structureEditing && (
          <div className="structure-editor-panel">
            <div>
              <strong>구조 수정</strong>
              <span>
                {structureBlock
                  ? `${structureBlock.block_id} 영역을 선택했습니다.`
                  : draftBbox
                    ? "새 영역을 그렸습니다."
                    : redrawingBlock
                      ? "선택한 블록의 새 범위를 드래그하세요."
                      : "영역을 클릭하거나 빈 곳을 드래그하세요."}
              </span>
            </div>
            {structureBlock && (
              <>
                <label>
                  <span>블록 유형</span>
                  <select
                    value={structureBlock.type}
                    onChange={(event) => onUpdateBlock(
                      structureBlock.block_id,
                      { type: event.target.value },
                    )}
                  >
                    {[
                      "title", "section_title", "paragraph", "formula",
                      "table", "figure", "caption", "footer", "page_number",
                    ].map((type) => <option key={type} value={type}>{type}</option>)}
                  </select>
                </label>
                {!draftBbox && (
                  <button
                    type="button"
                    onClick={() => {
                      setDraftRegion(null);
                      setRedrawingBlock(true);
                    }}
                  >
                    영역 다시 그리기
                  </button>
                )}
                {draftBbox && (
                  <button
                    type="button"
                    onClick={() => {
                      onUpdateBlock(structureBlock.block_id, { bbox: draftBbox });
                      setDraftRegion(null);
                      setRedrawingBlock(false);
                    }}
                  >
                    그린 범위로 영역 변경
                  </button>
                )}
                <button
                  type="button"
                  className="danger"
                  onClick={() => {
                    if (!window.confirm("선택한 블록을 오탐으로 삭제할까요?")) return;
                    onDeleteBlock(structureBlock.block_id);
                    setStructureBlock(null);
                    setRedrawingBlock(false);
                  }}
                >
                  오탐 블록 삭제
                </button>
              </>
            )}
            {draftBbox && !structureBlock && (
              <button
                type="button"
                className="primary"
                onClick={() => {
                  onAddFigure(draftBbox);
                  setDraftRegion(null);
                }}
              >
                누락 Figure 추가
              </button>
            )}
            {draftRegion && (
              <button
                type="button"
                onClick={() => {
                  setDraftRegion(null);
                  setRedrawingBlock(false);
                }}
              >
                선택 취소
              </button>
            )}
          </div>
        )}
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

function LinkedPageDescription({
  text,
  blocks,
  semanticAnalyses,
  onMoveBlock,
  onUpdateFigureTreatment,
  onSelectFigure,
  selectedFigure,
}) {
  const [draggedBlockId, setDraggedBlockId] = useState(null);
  const [dropTarget, setDropTarget] = useState(null);
  const orderedBlocks = useMemo(
    () => sortBlocksForBrailleReading(blocks || []),
    [blocks],
  );
  const analysesByBlockId = useMemo(
    () => new Map((semanticAnalyses || []).map((item) => [item.block_id, item])),
    [semanticAnalyses],
  );
  const blocksByType = new Map();
  orderedBlocks.forEach((block) => {
    if (!blocksByType.has(block.type)) blocksByType.set(block.type, []);
    blocksByType.get(block.type).push(block);
  });
  const typeOffsets = new Map();

  return String(text || "").split(/\r?\n/).map((line, index) => {
    const taggedMatch = line.match(/^\s*\[([a-z_]+)\]\s*(.*)$/i);
    if (!taggedMatch) {
      return <span key={index} className="page-description-line">{line}</span>;
    }
    const type = taggedMatch[1].toLowerCase();
    if (type === "transcriber_note") {
      return (
        <div key={`manual-note-${index}`} className="manual-note-line">
          <span className="inline-transcriber-note-toggle static">
            <span aria-hidden="true">✓</span> 점역자 주
          </span>
          <span>{taggedMatch[2] || "점역자 주 내용을 작성해 주세요."}</span>
        </div>
      );
    }
    const typeOffset = typeOffsets.get(type) || 0;
    const block = blocksByType.get(type)?.[typeOffset];
    typeOffsets.set(type, typeOffset + 1);
    if (!block) {
      return <span key={index} className="page-description-line">{line}</span>;
    }

    const movable = type !== "caption" && type !== "page_number";
    const moveHandle = movable && (
      <button
        type="button"
        className="description-drag-handle"
        draggable
        aria-label={`${type} 블록 순서 이동`}
        title="끌어서 순서 이동 · Alt+위/아래 화살표로 이동"
        onDragStart={(event) => {
          setDraggedBlockId(block.block_id);
          event.dataTransfer.effectAllowed = "move";
          event.dataTransfer.setData("text/plain", block.block_id);
        }}
        onDragEnd={() => {
          setDraggedBlockId(null);
          setDropTarget(null);
        }}
        onKeyDown={(event) => {
          if (!event.altKey || !["ArrowUp", "ArrowDown"].includes(event.key)) return;
          event.preventDefault();
          onMoveBlock(block.block_id, event.key === "ArrowUp" ? "up" : "down");
        }}
      >
        <span aria-hidden="true">⠿</span>
      </button>
    );
    const rowProps = {
      className: [
        "draggable-description-block",
        draggedBlockId === block.block_id ? "dragging" : "",
        dropTarget?.blockId === block.block_id
          ? `drop-${dropTarget.position}`
          : "",
      ].filter(Boolean).join(" "),
      onDragOver: (event) => {
        if (!movable || !draggedBlockId) return;
        event.preventDefault();
        event.dataTransfer.dropEffect = "move";
        const bounds = event.currentTarget.getBoundingClientRect();
        const position = event.clientY < bounds.top + bounds.height / 2
          ? "before"
          : "after";
        if (
          dropTarget?.blockId !== block.block_id
          || dropTarget?.position !== position
        ) {
          setDropTarget({ blockId: block.block_id, position });
        }
      },
      onDragLeave: (event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          setDropTarget((current) => (
            current?.blockId === block.block_id ? null : current
          ));
        }
      },
      onDrop: (event) => {
        event.preventDefault();
        const sourceId = draggedBlockId || event.dataTransfer.getData("text/plain");
        if (sourceId && sourceId !== block.block_id) {
          onMoveBlock(
            sourceId,
            dropTarget?.position === "after" ? "place_after" : "place",
            block.block_id,
          );
        }
        setDraggedBlockId(null);
        setDropTarget(null);
      },
    };

    if (type !== "figure") {
      return (
        <div key={block.block_id || index} {...rowProps}>
          {moveHandle}
          <span className="page-description-line">{line}</span>
        </div>
      );
    }

    const generatedText = taggedMatch[2] || "";
    const noteText = String(generatedText).trim();
    const figureAnalysis = analysesByBlockId.get(block.block_id);
    const visualTreatment = (
      figureAnalysis?.braille_review?.visual_treatment
      || figureAnalysis?.braille_review?.production_mode
      || "transcriber_note"
    );
    const isTactileGraphic = visualTreatment === "tactile_graphic";
    return (
      <div key={block.block_id || index} {...rowProps}>
        {moveHandle}
        <div className={`figure-description-inline ${isTactileGraphic ? "is-tactile-graphic" : "is-transcriber-note"}`}>
          <div className="figure-description-meta">
            <button
              type="button"
              className={`inline-visual-treatment-toggle ${isTactileGraphic ? "tactile" : "note"}`}
              aria-pressed={isTactileGraphic}
              title={isTactileGraphic
                ? "점역자 주 방식으로 변경"
                : "촉각 그래픽 제작 대상으로 변경"}
              onClick={() => onUpdateFigureTreatment(
                block.block_id,
                isTactileGraphic ? "transcriber_note" : "tactile_graphic",
              )}
            >
              <span aria-hidden="true">{isTactileGraphic ? "▧" : "✓"}</span>
              {isTactileGraphic ? "촉각 그래픽 제작 대상" : "점역자 주"}
            </button>
            <span className="figure-tag">[figure]</span>
          </div>
          <div className="figure-description-content">
            <span className="figure-description-copy">
              {noteText || "Figure 설명을 작성해 주세요."}
            </span>
            <button
              className={`figure-reference inline ${selectedFigure?.block_id === block.block_id ? "active" : ""}`}
              onClick={() => onSelectFigure(block)}
              title={`${block.block_id} 원본 보기`}
            >
              <span aria-hidden="true">▧</span> 원본 보기
            </button>
          </div>
        </div>
      </div>
    );
  });
}

function PageDescriptionView({
  result,
  onUpdateDescription,
  onMoveBlock,
  onUpdateFigureTreatment,
  onUpdateReview,
  onUpdateBlock,
  onAddFigure,
  onDeleteBlock,
}) {
  const description = result.page_description;
  const [selectedFigure, setSelectedFigure] = useState(null);
  const [selectedBlock, setSelectedBlock] = useState(null);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(description?.text || "");
  const [cursorCueVisible, setCursorCueVisible] = useState(false);
  const [cursorCueTop, setCursorCueTop] = useState(0);
  const [tagEditWarning, setTagEditWarning] = useState(false);
  const [reviewerName, setReviewerName] = useState(
    description?.review?.reviewer || loadLocalValue("hope-reviewer-name", ""),
  );
  const editorRef = useRef(null);
  const cursorCueTimerRef = useRef(null);
  useEffect(() => {
    setSelectedFigure(null);
    setSelectedBlock(null);
    setEditing(false);
    setDraft(description?.text || "");
    setTagEditWarning(false);
    setReviewerName(
      description?.review?.reviewer || loadLocalValue("hope-reviewer-name", ""),
    );
  }, [result, description?.text]);

  useEffect(() => () => window.clearTimeout(cursorCueTimerRef.current), []);

  if (!description || description.status === "failed") {
    return <div className="empty-state result-empty">이 페이지에서 읽을 수 있는 내용을 찾지 못했습니다.</div>;
  }

  function applyDraft() {
    onUpdateDescription(draft);
    setEditing(false);
    setSelectedBlock(null);
  }

  function insertTranscriberNote() {
    const editor = editorRef.current;
    const start = editor?.selectionStart ?? draft.length;
    const end = editor?.selectionEnd ?? start;
    const needsLeadingBreak = start > 0 && draft[start - 1] !== "\n";
    const needsTrailingBreak = end < draft.length && draft[end] !== "\n";
    const insertedText = `${needsLeadingBreak ? "\n" : ""}[transcriber_note] 점역자 주 내용을 입력하세요.${needsTrailingBreak ? "\n" : ""}`;
    const nextDraft = `${draft.slice(0, start)}${insertedText}${draft.slice(end)}`;
    const selectionStart = start
      + insertedText.indexOf("점역자 주 내용을 입력하세요.");
    const selectionEnd = selectionStart + "점역자 주 내용을 입력하세요.".length;
    setDraft(nextDraft);
    window.requestAnimationFrame(() => {
      if (!editorRef.current) return;
      editorRef.current.focus();
      editorRef.current.setSelectionRange(selectionStart, selectionEnd);
    });
  }

  function beginBlockEdit(block) {
    setSelectedBlock(block);
    setEditing(true);
    const sameTypeBlocks = (result.page?.blocks || []).filter((item) => item.type === block.type);
    const blockIndex = Math.max(0, sameTypeBlocks.findIndex((item) => item.block_id === block.block_id));
    const tokenPattern = new RegExp(`\\[${block.type}\\]`, "gi");
    const matches = [...draft.matchAll(tokenPattern)];
    const match = matches[blockIndex] || matches[0];
    const selectionStart = match ? match.index + match[0].length : draft.length;
    const leadingWhitespace = draft.slice(selectionStart).match(/^\s*/)?.[0].length || 0;
    const cursorPosition = selectionStart + leadingWhitespace;
    window.clearTimeout(cursorCueTimerRef.current);
    setCursorCueVisible(false);
    window.requestAnimationFrame(() => {
      const editor = editorRef.current;
      if (!editor) return;
      editor.focus();
      editor.setSelectionRange(cursorPosition, cursorPosition);

      const computedStyle = window.getComputedStyle(editor);
      const mirror = document.createElement("div");
      const caretMarker = document.createElement("span");
      const mirroredProperties = [
        "fontFamily", "fontSize", "fontWeight", "fontStyle", "letterSpacing", "lineHeight",
        "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
        "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
      ];
      mirroredProperties.forEach((property) => {
        mirror.style[property] = computedStyle[property];
      });
      const mirrorBorderWidth = (
        (Number.parseFloat(computedStyle.borderLeftWidth) || 0)
        + (Number.parseFloat(computedStyle.borderRightWidth) || 0)
      );
      mirror.style.borderStyle = "solid";
      mirror.style.boxSizing = "border-box";
      mirror.style.overflowWrap = "break-word";
      mirror.style.position = "fixed";
      mirror.style.visibility = "hidden";
      mirror.style.whiteSpace = "pre-wrap";
      mirror.style.width = `${editor.clientWidth + mirrorBorderWidth}px`;
      mirror.style.wordBreak = "break-word";
      mirror.textContent = draft.slice(0, cursorPosition);
      caretMarker.textContent = "\u200b";
      mirror.appendChild(caretMarker);
      document.body.appendChild(mirror);

      const caretTop = caretMarker.offsetTop;
      const targetScrollTop = Math.max(0, caretTop - editor.clientHeight * 0.3);
      editor.scrollTop = targetScrollTop;
      setCursorCueTop(Math.max(16, caretTop - editor.scrollTop));
      setCursorCueVisible(true);
      cursorCueTimerRef.current = window.setTimeout(() => setCursorCueVisible(false), 2200);
      mirror.remove();
      editor.scrollIntoView({ behavior: "smooth", block: "center" });
    });
  }

  return (
    <div className="page-review">
      <PageSourceViewer
        result={result}
        selectedFigure={selectedFigure}
        onClearFigure={() => setSelectedFigure(null)}
        selectedBlock={selectedBlock}
        onSelectBlock={beginBlockEdit}
        onUpdateBlock={onUpdateBlock}
        onAddFigure={onAddFigure}
        onDeleteBlock={onDeleteBlock}
      />
      <section className="page-description-pane">
        <div className="page-review-heading">
          <div><span>접근성 자료 초안</span><h3>페이지 전체 설명</h3></div>
          <div className="page-actions">
            {editing ? (
              <>
                <button className="text-button add-note-button" onClick={insertTranscriberNote}>
                  + 점역자 주 추가
                </button>
                <button className="text-button" onClick={() => { setDraft(description.text || ""); setEditing(false); setSelectedBlock(null); }}>취소</button>
                <button className="save-button" onClick={applyDraft}>수정 적용</button>
              </>
            ) : <button className="edit-button" onClick={() => setEditing(true)}>설명 수정</button>}
          </div>
        </div>
        <div className="page-description-body">
          <div className="review-guide">
            <span aria-hidden="true">i</span>
            <p>
              각 태그 앞의 <strong>⠿ 손잡이</strong>를 끌어 읽기 순서를 바꿀 수 있습니다.
              Figure 원본 보기를 누르면 왼쪽에서 실제 영역을 확인할 수 있습니다.
              <strong>점역자 주</strong> 표시를 누르면 촉각 그래픽 제작 대상으로 전환됩니다.
            </p>
          </div>
          {editing ? (
            <>
              {selectedBlock && (
                <div className="editing-target" key={selectedBlock.block_id}>
                  <span aria-hidden="true"><img src="/icons/pencil.png" alt="" /></span>
                  <p><strong>{selectedBlock.type}</strong> 영역의 설명을 수정하고 있어요.</p>
                  <button type="button" onClick={() => setSelectedBlock(null)} aria-label="선택 해제">×</button>
                </div>
              )}
              <div className={`editor-shell ${cursorCueVisible ? "show-cursor-cue" : ""}`}>
                <div className={`locked-tag-notice ${tagEditWarning ? "warning" : ""}`}>
                  <span aria-hidden="true">🔒</span>
                  {tagEditWarning
                    ? "블록 태그는 여기서 변경할 수 없습니다. 원본 영역의 ‘구조 수정’을 이용해 주세요."
                    : "[figure], [table], [paragraph] 등의 블록 태그는 구조 수정에서만 변경됩니다."}
                </div>
                {cursorCueVisible && (
                  <span
                    className="cursor-position-cue"
                    style={{ top: `${cursorCueTop}px` }}
                    aria-hidden="true"
                  >
                    <i>여기부터 수정</i><b>→</b>
                  </span>
                )}
                <textarea
                  ref={editorRef}
                  className="page-description-editor"
                  value={draft}
                  onChange={(event) => {
                    const nextDraft = event.target.value;
                    const currentTags = taggedLineSignature(draft);
                    const nextTags = taggedLineSignature(nextDraft);
                    if (
                      currentTags.length !== nextTags.length
                      || currentTags.some((tag, index) => tag !== nextTags[index])
                    ) {
                      setTagEditWarning(true);
                      window.setTimeout(() => setTagEditWarning(false), 2400);
                      return;
                    }
                    setDraft(nextDraft);
                    setTagEditWarning(false);
                    setCursorCueVisible(false);
                  }}
                  onPointerDown={() => setCursorCueVisible(false)}
                  aria-label="페이지 접근성 설명 수정"
                />
              </div>
            </>
          ) : (
            <div className="page-description-text">
              <LinkedPageDescription
                text={description.text || "없음"}
                blocks={result.page?.blocks || []}
                semanticAnalyses={result.semantic_analyses || []}
                onMoveBlock={onMoveBlock}
                onUpdateFigureTreatment={onUpdateFigureTreatment}
                selectedFigure={selectedFigure}
                onSelectFigure={setSelectedFigure}
              />
            </div>
          )}
          <div className="review-footer">
            <div className="review-status-summary">
              <span className={`review-badge ${description.review_status}`}>{description.review_status}</span>
              {description.was_generated && <span>모델 다듬기 적용</span>}
              {description.review?.history?.length > 0 && (
                <span>
                  최근 이력: {description.review.history.at(-1).reviewer}
                  {" · "}
                  {new Date(description.review.history.at(-1).at).toLocaleString("ko-KR")}
                  {" · "}
                  {description.review.history.at(-1).status === "reviewed" ? "검수 완료" : "재검수 시작"}
                </span>
              )}
            </div>
            <div className="page-review-controls">
              <label>
                <span>검수자</span>
                <input
                  value={reviewerName}
                  placeholder="이름 입력"
                  onChange={(event) => setReviewerName(event.target.value)}
                />
              </label>
              <button
                type="button"
                className={description.review_status === "reviewed" ? "reopen" : "complete"}
                onClick={() => {
                  const reviewer = reviewerName.trim();
                  if (!reviewer) {
                    window.alert("검수자 이름을 입력해 주세요.");
                    return;
                  }
                  window.localStorage.setItem(
                    "hope-reviewer-name",
                    JSON.stringify(reviewer),
                  );
                  onUpdateReview(
                    description.review_status === "reviewed" ? "needs_review" : "reviewed",
                    reviewer,
                  );
                }}
              >
                {description.review_status === "reviewed" ? "검수 다시 열기" : "검수 완료"}
              </button>
            </div>
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




const USER_FRIENDLY_FORMULA_WARNING = "\uc218\uc2dd \ud655\uc778\uc774 \ud544\uc694\ud569\ub2c8\ub2e4\n\n\uc774 \uc218\uc2dd\uc740 \uc774\ubbf8\uc9c0\uc5d0\uc11c \uc790\ub3d9\uc73c\ub85c \uc77d\uc5b4 \uc628 \uacb0\uacfc\uc785\ub2c8\ub2e4.\n\uc77c\ubd80 \uae30\ud638\ub098 \uae00\uc790\uac00 \ube60\uc84c\uc744 \uc218 \uc788\uc5b4, \uc8fc\ubcc0 \uae00\uc790 \uc778\uc2dd \uacb0\uacfc\ub97c \ucc38\uace0\ud574 \ubcf4\uc644\ud588\uc2b5\ub2c8\ub2e4.\n\uc810\uc5ed \ub610\ub294 \uc790\ub8cc \uc81c\uc791 \uc804\uc5d0 \uc6d0\ubcf8 \uc218\uc2dd\uacfc \ubcc0\ud658\ub41c \uc218\uc2dd\uc774 \uc77c\uce58\ud558\ub294\uc9c0 \ud655\uc778\ud574 \uc8fc\uc138\uc694.";
const FORMULA_WARNING_HEADING = "\uc790\ub3d9 \uac80\uc218 \uacbd\uace0";

function getUserFriendlyWarning(warning) {
  const warningText = String(warning || "");

  if (
    warningText.includes("pix2tex") ||
    warningText.includes("OCR") ||
    warningText.includes("fewer formula parts") ||
    warningText.includes("rejected as unreliable") ||
    warningText.includes("unavailable or failed") ||
    warningText.includes("Detected formula block does not contain") ||
    warningText.includes("Formula LaTeX could not be recognized") ||
    warningText.includes("Formula text was not available")
  ) {
    return USER_FRIENDLY_FORMULA_WARNING;
  }

  return warningText;
}


function FormulaWarningResult({ warnings, status, type }) {
  const normalizedStatus = String(status || "").toLowerCase();
  const shouldShowWarning = type === "formula" && normalizedStatus && normalizedStatus !== "success";

  if (!shouldShowWarning || !warnings || warnings.length === 0) {
    return null;
  }

  return (
    <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 p-3">
      <div className="text-sm font-semibold text-amber-900">
        {FORMULA_WARNING_HEADING}
      </div>
      <ul className="mt-2 list-disc pl-5 text-sm text-amber-900">
        <li style={{ whiteSpace: "pre-line" }}>
          {USER_FRIENDLY_FORMULA_WARNING}
        </li>
      </ul>
    </div>
  );
}


function AnalysisInspector({ result, type }) {
  const entries = useMemo(() => analysisEntries(result).filter((item) => item.type === type), [result, type]);
  const [selectedId, setSelectedId] = useState(null);

  useEffect(() => {
    setSelectedId((current) => (
      entries.some((entry) => entry.block_id === current)
        ? current
        : entries[0]?.block_id || null
    ));
  }, [type, entries]);

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
        <section className="review-section review-source-section">
          <h3>원본 영역</h3>
          <BlockCrop imageUrl={result.page_image} bbox={selected.bbox} alt={`${selected.block_id} 원본 영역`} />
          <div className="metadata-row">
            <span>탐지 신뢰도</span>
            <span className="confidence-cell">
              <Confidence value={selected.detection?.confidence} />
              <ConfidenceLevel value={selected.detection?.confidence} />
            </span>
          </div>
        </section>
        <section className="review-section review-structure-section">
          <h3>구조화 결과</h3>
          <SemanticResult entry={selected} />
        </section>
        <section className="review-section review-description-section">
          <DescriptionResult description={selected.description} captioningEnabled={result.figure_captioning_enabled} type={type} />
        </section>
        <FormulaWarningResult warnings={selected.warnings} status={selected.analysis?.status} type={type} />
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
        <svg className="layout-select-chevron" viewBox="0 0 16 16" aria-hidden="true">
          <path d="m4 6 4 4 4-4" />
        </svg>
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
  const layoutModel = "doclayout_yolo";
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("layout");
  const [activeElementType, setActiveElementType] = useState("formula");
  const [elapsedSeconds, setElapsedSeconds] = useState(0);
  const [libraryMeta, setLibraryMeta] = useState(() => loadLocalValue(LIBRARY_META_KEY, {}));
  const [libraryGroups, setLibraryGroups] = useState(() => loadLocalValue(LIBRARY_GROUPS_KEY, []));
  const [activeGroup, setActiveGroup] = useState("all");
  const [groupDialogOpen, setGroupDialogOpen] = useState(false);
  const [groupDraft, setGroupDraft] = useState("");
  const [openGroupMenu, setOpenGroupMenu] = useState(null);
  const [libraryQuery, setLibraryQuery] = useState("");
  const [librarySort, setLibrarySort] = useState("recent");
  const [sortMenuOpen, setSortMenuOpen] = useState(false);
  const [renameProject, setRenameProject] = useState(null);
  const [renameDraft, setRenameDraft] = useState("");
  const [savedPagesDialogOpen, setSavedPagesDialogOpen] = useState(false);
  const [savedPageQuery, setSavedPageQuery] = useState("");
  const [savedPageFilter, setSavedPageFilter] = useState("all");
  const [toast, setToast] = useState(null);
  const [groupManagerOpen, setGroupManagerOpen] = useState(false);
  const [editingGroup, setEditingGroup] = useState(null);
  const [editingGroupDraft, setEditingGroupDraft] = useState("");
  const [onboardingOpen, setOnboardingOpen] = useState(() => (
    window.localStorage.getItem(ONBOARDING_KEY) !== "true"
  ));
  const [onboardingStep, setOnboardingStep] = useState(0);
  const [openBookSettings, setOpenBookSettings] = useState(null);
  const [resultFullscreen, setResultFullscreen] = useState(false);
  const toastTimerRef = useRef(null);
  const projectFileInputRef = useRef(null);
  const resultWorkspaceRef = useRef(null);

  useEffect(() => {
    function syncFullscreenState() {
      setResultFullscreen(document.fullscreenElement === resultWorkspaceRef.current);
    }

    document.addEventListener("fullscreenchange", syncFullscreenState);
    return () => document.removeEventListener("fullscreenchange", syncFullscreenState);
  }, []);

  useEffect(() => {
    if (!openBookSettings) return undefined;
    function closeSettingsOutside(event) {
      if (event.target.closest(".book-settings-menu, .book-settings-trigger")) return;
      setOpenBookSettings(null);
    }
    document.addEventListener("pointerdown", closeSettingsOutside);
    return () => document.removeEventListener("pointerdown", closeSettingsOutside);
  }, [openBookSettings]);

  async function toggleResultFullscreen() {
    try {
      if (document.fullscreenElement === resultWorkspaceRef.current) {
        await document.exitFullscreen();
        return;
      }
      await resultWorkspaceRef.current?.requestFullscreen();
    } catch {
      setError("전체화면 보기를 시작하지 못했습니다. 브라우저 설정을 확인해 주세요.");
    }
  }

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

  useEffect(() => {
    window.localStorage.setItem(LIBRARY_META_KEY, JSON.stringify(libraryMeta));
  }, [libraryMeta]);

  useEffect(() => {
    window.localStorage.setItem(LIBRARY_GROUPS_KEY, JSON.stringify(libraryGroups));
  }, [libraryGroups]);

  useEffect(() => () => window.clearTimeout(toastTimerRef.current), []);

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
      showToast(`${page.pageNumber}페이지 저장 결과를 삭제했어요.`);
    } catch (err) {
      setError(`저장된 페이지를 삭제하지 못했습니다. ${err.message}`);
    }
  }

  async function toggleSavedPageReview(page) {
    const nextStatus = page.reviewStatus === "reviewed" ? "needs_review" : "reviewed";
    let reviewer = loadLocalValue("hope-reviewer-name", "");
    if (!reviewer) {
      reviewer = window.prompt("검수자 이름을 입력해 주세요.")?.trim() || "";
      if (!reviewer) return;
      window.localStorage.setItem("hope-reviewer-name", JSON.stringify(reviewer));
    }
    try {
      const updatedPage = await updateWorkspacePageReviewStatus(
        page.projectId,
        page.pageNumber,
        nextStatus,
        reviewer,
      );
      setSavedPages((current) => current.map((item) => (
        item.id === updatedPage.id ? updatedPage : item
      )));
      if (
        resultOwnerId === page.projectId
        && Number(result?.page?.page_id) === page.pageNumber
      ) {
        setResult(updatedPage.result);
      }
      showToast(nextStatus === "reviewed" ? "검토 완료로 표시했어요." : "검토 필요로 되돌렸어요.");
    } catch (err) {
      setError(`검토 상태를 저장하지 못했습니다. ${err.message}`);
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
      showToast("교과서를 보관함에 추가했어요.");
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
    showToast("수정한 설명을 저장했어요.");
  }

  function updateFigureTreatment(blockId, visualTreatment) {
    setResult((current) => {
      if (!current?.page?.blocks) return current;
      const figureBlock = current.page.blocks.find(
        (block) => block.block_id === blockId && block.type === "figure",
      );
      if (!figureBlock) return current;

      const analyses = [...(current.semantic_analyses || [])];
      const analysisIndex = analyses.findIndex((item) => item.block_id === blockId);
      const existingAnalysis = analysisIndex >= 0
        ? analyses[analysisIndex]
        : {
            page_id: current.page.page_id,
            block_id: blockId,
            type: "figure",
            bbox: figureBlock.bbox,
          };
      const updatedAnalysis = {
        ...existingAnalysis,
        braille_review: {
          ...(existingAnalysis.braille_review || {}),
          visual_treatment: visualTreatment,
        },
      };

      if (analysisIndex >= 0) analyses[analysisIndex] = updatedAnalysis;
      else analyses.push(updatedAnalysis);

      return {
        ...current,
        semantic_analyses: analyses,
        page_description: current.page_description
          ? {
              ...current.page_description,
              review_status: "needs_review",
            }
          : current.page_description,
      };
    });
    showToast(
      visualTreatment === "tactile_graphic"
        ? "촉각 그래픽 제작 대상으로 표시했어요."
        : "점역자 주 방식으로 표시했어요.",
    );
  }

  function updatePageReview(reviewStatus, reviewer) {
    const reviewedAt = new Date().toISOString();
    setResult((current) => {
      if (!current?.page_description) return current;
      const previousReview = current.page_description.review || {};
      const history = [
        ...(previousReview.history || []),
        {
          status: reviewStatus,
          reviewer,
          at: reviewedAt,
        },
      ];
      return {
        ...current,
        page_description: {
          ...current.page_description,
          review_status: reviewStatus,
          review: {
            ...previousReview,
            reviewer,
            reviewed_at: reviewStatus === "reviewed"
              ? reviewedAt
              : previousReview.reviewed_at || null,
            reopened_at: reviewStatus === "needs_review" ? reviewedAt : null,
            history,
          },
        },
      };
    });
    showToast(reviewStatus === "reviewed" ? "검수 완료 이력을 저장했어요." : "검수를 다시 열었어요.");
  }

  function updatePageBlock(blockId, changes) {
    setResult((current) => {
      if (!current?.page?.blocks) return current;
      const previousBlocks = current.page.blocks;
      const previousBlock = previousBlocks.find((block) => block.block_id === blockId);
      if (!previousBlock) return current;
      const nextType = changes.type || previousBlock.type;
      const blocks = previousBlocks.map((block) => (
        block.block_id === blockId
          ? {
              ...block,
              ...changes,
              manually_corrected: true,
            }
          : block
      ));
      const typeChanged = nextType !== previousBlock.type;
      const pageDescriptionText = typeChanged && current.page_description?.text
        ? changeTaggedDescriptionLine(
            current.page_description.text,
            previousBlocks,
            blockId,
            nextType,
          )
        : current.page_description?.text;
      const semanticAnalyses = typeChanged
        ? (current.semantic_analyses || []).filter((item) => item.block_id !== blockId)
        : (current.semantic_analyses || []).map((item) => (
            item.block_id === blockId && changes.bbox
              ? { ...item, bbox: changes.bbox }
              : item
          ));
      return {
        ...current,
        page: { ...current.page, blocks },
        semantic_analyses: semanticAnalyses,
        page_description: current.page_description
          ? {
              ...current.page_description,
              text: pageDescriptionText,
              review_status: "needs_review",
            }
          : current.page_description,
      };
    });
    showToast(changes.type ? "블록 유형을 수정했어요." : "블록 영역을 수정했어요.");
  }

  function addMissingFigure(bbox) {
    setResult((current) => {
      if (!current?.page?.blocks) return current;
      const pageId = current.page.page_id;
      const manualFigures = current.page.blocks.filter(
        (block) => String(block.block_id).startsWith(`p${pageId}_manual_figure_`),
      );
      const blockId = `p${pageId}_manual_figure_${manualFigures.length + 1}`;
      const readingOrder = Math.max(
        0,
        ...current.page.blocks.map((block) => Number(block.reading_order) || 0),
      ) + 1;
      const figureBlock = {
        block_id: blockId,
        type: "figure",
        bbox,
        score: null,
        detector: "manual",
        reading_order: readingOrder,
        manually_added: true,
      };
      return {
        ...current,
        page: {
          ...current.page,
          blocks: [...current.page.blocks, figureBlock],
        },
        semantic_analyses: [
          ...(current.semantic_analyses || []),
          {
            page_id: pageId,
            block_id: blockId,
            type: "figure",
            bbox,
            detection: {
              model: { name: "manual", version: null },
              confidence: null,
            },
            analysis: null,
            description: {
              status: "not_started",
              review_status: "needs_review",
            },
            warnings: ["사용자가 누락된 Figure 영역을 추가했습니다."],
          },
        ],
        page_description: current.page_description
          ? {
              ...current.page_description,
              text: `${String(current.page_description.text || "").trimEnd()}\n[figure] 새 Figure 설명을 작성해 주세요.`,
              review_status: "needs_review",
            }
          : current.page_description,
      };
    });
    showToast("누락된 Figure 영역을 추가했어요. 설명을 검수해 주세요.");
  }

  function deletePageBlock(blockId) {
    setResult((current) => {
      if (!current?.page?.blocks) return current;
      const previousBlocks = current.page.blocks;
      if (!previousBlocks.some((block) => block.block_id === blockId)) return current;
      const pageDescriptionText = current.page_description?.text
        ? changeTaggedDescriptionLine(
            current.page_description.text,
            previousBlocks,
            blockId,
            null,
          )
        : current.page_description?.text;
      return {
        ...current,
        page: {
          ...current.page,
          blocks: previousBlocks.filter((block) => block.block_id !== blockId),
        },
        semantic_analyses: (current.semantic_analyses || []).filter(
          (item) => item.block_id !== blockId,
        ),
        page_description: current.page_description
          ? {
              ...current.page_description,
              text: pageDescriptionText,
              review_status: "needs_review",
            }
          : current.page_description,
      };
    });
    showToast("오탐 블록을 삭제했어요.");
  }

  function updateBrailleReadingOrder(blockId, direction, targetBlockId = null) {
    setResult((current) => {
      if (!current?.page?.blocks) return current;
      const previousBlocks = current.page.blocks;
      const blocks = direction === "place" || direction === "place_after"
        ? placeBrailleReadingGroup(
            previousBlocks,
            current.semantic_analyses,
            blockId,
            targetBlockId,
            direction === "place_after" ? "after" : "before",
          )
        : moveBrailleReadingGroup(
            previousBlocks,
            current.semantic_analyses,
            blockId,
            direction,
          );
      const pageDescriptionText = current.page_description?.text
        ? reorderTaggedPageDescription(
            current.page_description.text,
            previousBlocks,
            blocks,
          )
        : current.page_description?.text;
      return {
        ...current,
        page: { ...current.page, blocks },
        page_description: current.page_description
          ? {
              ...current.page_description,
              text: pageDescriptionText,
              review_status: "needs_review",
              warnings: [
                ...(current.page_description.warnings || []).filter(
                  (warning) => (
                    warning !== "읽기 순서가 수동으로 조정되었습니다."
                    && !warning.startsWith("읽기 순서를 조정했습니다.")
                  ),
                ),
                "읽기 순서가 수동으로 조정되었습니다.",
              ],
            }
          : current.page_description,
      };
    });
    showToast("권장 읽기 순서를 조정했어요.");
  }

  function downloadJson() {
    if (!result) return;
    const payload = { ...result.page, semantic_analyses: result.semantic_analyses || [], page_description: result.page_description || null };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json;charset=utf-8" });
    const url = URL.createObjectURL(blob); const anchor = document.createElement("a");
    anchor.href = url; anchor.download = `page_${String(result.page.page_id).padStart(4, "0")}_analysis.json`;
    anchor.click(); URL.revokeObjectURL(url);
  }

  function downloadCurrentPageHtml() {
    if (!result) return;
    const project = projects.find((item) => item.id === activeProjectId);
    const title = project ? libraryProjectName(project) : file?.name || "교과서";
    const html = buildAccessibleTextbookHtml({ title, pages: [result] });
    downloadAccessibleHtml(
      html,
      accessibleHtmlFilename(title, `${String(result.page.page_id).padStart(4, "0")}페이지`),
    );
    showToast("현재 페이지의 접근성 HTML을 만들었어요.");
  }

  function downloadTextbookHtml() {
    const project = projects.find((item) => item.id === activeProjectId);
    const title = project ? libraryProjectName(project) : file?.name || "교과서";
    const pages = savedPages.map((page) => page.result);
    if (result && resultOwnerId === activeProjectId) {
      // The HTML builder keeps the last value for a duplicate page number,
      // so an on-screen edit wins even before the debounced auto-save ends.
      pages.push(result);
    }
    const html = buildAccessibleTextbookHtml({ title, pages });
    downloadAccessibleHtml(html, accessibleHtmlFilename(title));
    showToast(`${pages.length}개 페이지를 접근성 HTML로 만들었어요.`);
  }

  function currentPageHtmlSource() {
    if (!result) return "";
    const project = projects.find((item) => item.id === activeProjectId);
    const title = project ? libraryProjectName(project) : file?.name || "교과서";
    return buildAccessibleTextbookHtml({
      title,
      pages: [result],
      // A base64 page image makes the source tab unreadable. The downloaded
      // HTML still embeds it; only the on-screen source view omits it.
      includePageImages: false,
    });
  }

  const busy = status === "counting" || status === "analyzing";
  const activeProjects = projects.filter((project) => !libraryMeta[project.id]?.trashed);
  const trashedProjects = projects.filter((project) => libraryMeta[project.id]?.trashed);
  const groupedProjects = activeGroup === "trash"
    ? trashedProjects
    : activeGroup === "all"
      ? activeProjects
      : activeProjects.filter((project) => (
      activeGroup === "ungrouped"
        ? !libraryMeta[project.id]?.group
        : libraryMeta[project.id]?.group === activeGroup
      ));
  const filteredProjects = [...groupedProjects]
    .filter((project) => (
      libraryProjectName(project).toLocaleLowerCase("ko")
        .includes(libraryQuery.trim().toLocaleLowerCase("ko"))
    ))
    .sort((a, b) => {
      if (librarySort === "name") return libraryProjectName(a).localeCompare(libraryProjectName(b), "ko");
      if (librarySort === "progress") {
        return (projectPageCounts[b.id] || 0) - (projectPageCounts[a.id] || 0);
      }
      return new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0);
    });
  const totalSavedPages = activeProjects.reduce((sum, project) => sum + (projectPageCounts[project.id] || 0), 0);
  const recentProject = [...activeProjects].sort(
    (a, b) => new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0),
  )[0];
  const visibleSavedPages = savedPages.filter((page) => {
    const matchesQuery = !savedPageQuery.trim()
      || String(page.pageNumber).includes(savedPageQuery.trim());
    const matchesFilter = savedPageFilter === "all"
      || (savedPageFilter === "reviewed" && page.reviewStatus === "reviewed")
      || (savedPageFilter === "pending" && page.reviewStatus !== "reviewed");
    return matchesQuery && matchesFilter;
  });

  function createLibraryGroup(event) {
    event?.preventDefault();
    const trimmedName = groupDraft.trim();
    if (!trimmedName || libraryGroups.includes(trimmedName)) return;
    setLibraryGroups((current) => [...current, trimmedName]);
    setActiveGroup(trimmedName);
    setGroupDraft("");
    setGroupDialogOpen(false);
    showToast(`‘${trimmedName}’ 그룹을 만들었어요.`);
  }

  function showToast(message, tone = "success") {
    window.clearTimeout(toastTimerRef.current);
    setToast({ message, tone });
    toastTimerRef.current = window.setTimeout(() => setToast(null), 2600);
  }

  function updateProjectMeta(projectId, patch) {
    setLibraryMeta((current) => ({
      ...current,
      [projectId]: { ...current[projectId], ...patch },
    }));
  }

  function libraryProjectName(project) {
    return libraryMeta[project.id]?.displayName || project.name.replace(/\.pdf$/i, "");
  }

  function openRenameDialog(project) {
    setRenameProject(project);
    setRenameDraft(libraryProjectName(project));
  }

  function saveProjectName(event) {
    event.preventDefault();
    const nextName = renameDraft.trim();
    if (!renameProject || !nextName) return;
    updateProjectMeta(renameProject.id, { displayName: nextName });
    setRenameProject(null);
    setRenameDraft("");
    showToast("교과서 이름을 수정했어요.");
  }

  function renameLibraryGroup(event) {
    event.preventDefault();
    const nextName = editingGroupDraft.trim();
    if (!editingGroup || !nextName || libraryGroups.some((group) => group !== editingGroup && group === nextName)) return;
    setLibraryGroups((current) => current.map((group) => group === editingGroup ? nextName : group));
    setLibraryMeta((current) => Object.fromEntries(Object.entries(current).map(([id, meta]) => [
      id,
      meta?.group === editingGroup ? { ...meta, group: nextName } : meta,
    ])));
    if (activeGroup === editingGroup) setActiveGroup(nextName);
    setEditingGroup(null);
    setEditingGroupDraft("");
    showToast("그룹 이름을 수정했어요.");
  }

  function deleteLibraryGroup(group) {
    if (!window.confirm(`‘${group}’ 그룹을 삭제할까요? 교재는 삭제되지 않고 미분류로 이동합니다.`)) return;
    setLibraryGroups((current) => current.filter((item) => item !== group));
    setLibraryMeta((current) => Object.fromEntries(Object.entries(current).map(([id, meta]) => [
      id,
      meta?.group === group ? { ...meta, group: "" } : meta,
    ])));
    if (activeGroup === group) setActiveGroup("all");
    showToast("그룹을 삭제하고 교재를 미분류로 이동했어요.");
  }

  function moveProjectToTrash(project) {
    setOpenBookSettings(null);
    updateProjectMeta(project.id, { trashed: true });
    showToast(`‘${libraryProjectName(project)}’을 휴지통으로 이동했어요.`);
  }

  function restoreProject(project) {
    updateProjectMeta(project.id, { trashed: false });
    showToast("교과서를 보관함으로 복원했어요.");
  }

  async function permanentlyDeleteProject(project) {
    if (!window.confirm(`‘${libraryProjectName(project)}’을 완전히 삭제할까요? 원본 PDF와 저장 페이지를 복구할 수 없습니다.`)) return;
    try {
      await deleteTextbookProject(project.id);
      setProjects((current) => current.filter((item) => item.id !== project.id));
      setProjectPageCounts((current) => {
        const next = { ...current };
        delete next[project.id];
        return next;
      });
      setLibraryMeta((current) => {
        const next = { ...current };
        delete next[project.id];
        return next;
      });
      showToast("교과서를 완전히 삭제했어요.");
    } catch (err) {
      showToast(`삭제하지 못했어요. ${err.message}`, "error");
    }
  }

  function closeOnboarding() {
    window.localStorage.setItem(ONBOARDING_KEY, "true");
    setOnboardingOpen(false);
    setOnboardingStep(0);
  }

  const tabs = [
    { id: "layout", label: "레이아웃" },
    { id: "elements", label: "요소 분석" },
    { id: "html", label: "HTML" },
    { id: "page", label: "접근성 페이지" },
  ];

  return (
    <main className="app-shell">
      <section className="toolbar">
        <div className="brand">
          <div className="brand-mark">
            <img src="/edubridge-logo.png" alt="EduBridge 로고" />
          </div>
          <div>
            <div className="brand-line"><h1>EduBridge</h1><span>Textbook Accessibility</span></div>
            <p>교과서를 구조화하고 접근성 자료 제작을 돕습니다.</p>
          </div>
        </div>
        <div className="toolbar-actions">
          {screen !== "library" && (
            <button type="button" className="guide-button" onClick={() => { setOnboardingStep(0); setOnboardingOpen(true); }}>튜토리얼</button>
          )}
          {screen !== "library" && busy && (
            <div className="status-pill busy"><span />분석 진행 중</div>
          )}
        </div>
      </section>
      <input
        ref={projectFileInputRef}
        className="visually-hidden"
        type="file"
        accept="application/pdf"
        onChange={handleFileChange}
      />
      {groupDialogOpen && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={() => setGroupDialogOpen(false)}>
          <section
            className="group-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="group-dialog-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="dialog-close"
              aria-label="닫기"
              onClick={() => setGroupDialogOpen(false)}
            >
              ×
            </button>
            <span className="dialog-icon folder-dialog-icon" aria-hidden="true">
              <svg viewBox="0 0 24 24">
                <path d="M3.5 7.5h6l1.7 2h9.3v8.8a2.2 2.2 0 0 1-2.2 2.2H5.7a2.2 2.2 0 0 1-2.2-2.2V7.5Z" />
                <path d="M3.5 7.5V5.7a2.2 2.2 0 0 1 2.2-2.2h3.5l2 2h7.1a2.2 2.2 0 0 1 2.2 2.2v1.8" />
                <path d="M12 12.5v5M9.5 15h5" />
              </svg>
            </span>
            <p>교재 정리</p>
            <h2 id="group-dialog-title">새 그룹 만들기</h2>
            <small>과목이나 학기별로 교재를 묶어 더 쉽게 찾아보세요.</small>
            <form onSubmit={createLibraryGroup}>
              <label htmlFor="group-name">그룹 이름</label>
              <input
                id="group-name"
                autoFocus
                maxLength={20}
                value={groupDraft}
                onChange={(event) => setGroupDraft(event.target.value)}
                placeholder="예: 수학, 2학기 교재"
              />
              {libraryGroups.includes(groupDraft.trim()) && (
                <span className="dialog-error">이미 같은 이름의 그룹이 있어요.</span>
              )}
              <div>
                <button type="button" onClick={() => setGroupDialogOpen(false)}>취소</button>
                <button
                  type="submit"
                  disabled={!groupDraft.trim() || libraryGroups.includes(groupDraft.trim())}
                >
                  그룹 만들기
                </button>
              </div>
            </form>
          </section>
        </div>
      )}
      {renameProject && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={() => setRenameProject(null)}>
          <section
            className="group-dialog rename-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="rename-dialog-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <button type="button" className="dialog-close" aria-label="닫기" onClick={() => setRenameProject(null)}>
              ×
            </button>
            <span className="dialog-icon" aria-hidden="true">Aa</span>
            <p>교재 관리</p>
            <h2 id="rename-dialog-title">교과서 이름 수정</h2>
            <small>보관함에 표시할 이름만 변경됩니다. 원본 PDF 파일은 그대로 유지돼요.</small>
            <form onSubmit={saveProjectName}>
              <label htmlFor="textbook-name">교과서 이름</label>
              <input
                id="textbook-name"
                autoFocus
                maxLength={60}
                value={renameDraft}
                onChange={(event) => setRenameDraft(event.target.value)}
              />
              <span className="original-file-name">원본 파일 · {renameProject.name}</span>
              <div>
                <button type="button" onClick={() => setRenameProject(null)}>취소</button>
                <button type="submit" disabled={!renameDraft.trim()}>저장하기</button>
              </div>
            </form>
          </section>
        </div>
      )}
      {savedPagesDialogOpen && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={() => setSavedPagesDialogOpen(false)}>
          <section
            className="saved-pages-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="saved-pages-dialog-title"
            onMouseDown={(event) => event.stopPropagation()}
          >
            <header>
              <div>
                <span>페이지 관리</span>
                <h2 id="saved-pages-dialog-title">저장된 페이지</h2>
                <p>분석한 페이지를 검색하고 바로 이동할 수 있어요.</p>
              </div>
              <button type="button" className="dialog-close" aria-label="닫기" onClick={() => setSavedPagesDialogOpen(false)}>×</button>
            </header>
            <div className="saved-pages-toolbar">
              <label>
                <span aria-hidden="true">⌕</span>
                <input
                  type="search"
                  inputMode="numeric"
                  value={savedPageQuery}
                  onChange={(event) => setSavedPageQuery(event.target.value.replace(/\D/g, ""))}
                  placeholder="페이지 번호 검색"
                  aria-label="저장된 페이지 번호 검색"
                />
                {savedPageQuery && <button type="button" onClick={() => setSavedPageQuery("")} aria-label="검색어 지우기">×</button>}
              </label>
              <div className="saved-page-filters" aria-label="페이지 검토 상태">
                {[
                  ["all", "전체", savedPages.length],
                  ["pending", "검토 필요", savedPages.filter((page) => page.reviewStatus !== "reviewed").length],
                  ["reviewed", "검토 완료", savedPages.filter((page) => page.reviewStatus === "reviewed").length],
                ].map(([value, label, count]) => (
                  <button
                    type="button"
                    key={value}
                    className={savedPageFilter === value ? "active" : ""}
                    onClick={() => setSavedPageFilter(value)}
                  >
                    {label}<span>{count}</span>
                  </button>
                ))}
              </div>
            </div>
            <div className="saved-page-grid">
              {visibleSavedPages.map((page) => {
                const isActive = (
                  resultOwnerId === page.projectId
                  && Number(result?.page?.page_id) === page.pageNumber
                );
                return (
                  <article className={`saved-page-card ${isActive ? "active" : ""}`} key={page.id}>
                    <button
                      type="button"
                      className="saved-page-card-main"
                      onClick={() => {
                        openSavedPage(page);
                        setSavedPagesDialogOpen(false);
                      }}
                    >
                      <span><strong>{page.pageNumber}</strong>페이지</span>
                      <small>{page.reviewStatus === "reviewed" ? "검토 완료" : "검토 필요"}</small>
                      {isActive && <em>현재 페이지</em>}
                    </button>
                    <button
                      type="button"
                      className={`saved-page-review-toggle ${page.reviewStatus === "reviewed" ? "reviewed" : ""}`}
                      onClick={() => toggleSavedPageReview(page)}
                    >
                      <span aria-hidden="true">{page.reviewStatus === "reviewed" ? "✓" : "○"}</span>
                      {page.reviewStatus === "reviewed" ? "검토 완료" : "완료로 표시"}
                    </button>
                    <button
                      type="button"
                      className="saved-page-card-delete"
                      onClick={() => deleteSavedPage(page)}
                      aria-label={`${page.pageNumber}페이지 저장 결과 삭제`}
                      title="삭제"
                    >
                      ×
                    </button>
                  </article>
                );
              })}
              {visibleSavedPages.length === 0 && (
                <div className="saved-pages-empty">
                  <span aria-hidden="true">⌕</span>
                  <strong>조건에 맞는 페이지가 없어요</strong>
                  <p>다른 번호를 검색하거나 필터를 변경해보세요.</p>
                </div>
              )}
            </div>
          </section>
        </div>
      )}
      {groupManagerOpen && (
        <div className="dialog-backdrop" role="presentation" onMouseDown={() => setGroupManagerOpen(false)}>
          <section className="group-manager-dialog" role="dialog" aria-modal="true" aria-labelledby="group-manager-title" onMouseDown={(event) => event.stopPropagation()}>
            <header>
              <div><span>교재 정리</span><h2 id="group-manager-title">그룹 관리</h2><p>그룹 이름을 바꾸거나 필요 없는 그룹을 정리하세요.</p></div>
              <button type="button" className="dialog-close" aria-label="닫기" onClick={() => setGroupManagerOpen(false)}>×</button>
            </header>
            <div className="group-manager-list">
              {libraryGroups.map((group) => (
                <div className="group-manager-item" key={group}>
                  {editingGroup === group ? (
                    <form onSubmit={renameLibraryGroup}>
                      <input autoFocus maxLength={20} value={editingGroupDraft} onChange={(event) => setEditingGroupDraft(event.target.value)} aria-label="새 그룹 이름" />
                      <button type="button" onClick={() => setEditingGroup(null)}>취소</button>
                      <button type="submit" disabled={!editingGroupDraft.trim()}>저장</button>
                    </form>
                  ) : (
                    <>
                      <span className="group-manager-dot" />
                      <div><strong>{group}</strong><small>{activeProjects.filter((project) => libraryMeta[project.id]?.group === group).length}권</small></div>
                      <button type="button" onClick={() => { setEditingGroup(group); setEditingGroupDraft(group); }}>이름 수정</button>
                      <button type="button" className="danger-text-button" onClick={() => deleteLibraryGroup(group)}>삭제</button>
                    </>
                  )}
                </div>
              ))}
              {libraryGroups.length === 0 && <p className="group-manager-empty">아직 생성한 그룹이 없어요.</p>}
            </div>
            <button type="button" className="manager-create-button" onClick={() => { setGroupManagerOpen(false); setGroupDialogOpen(true); }}>＋ 새 그룹 만들기</button>
          </section>
        </div>
      )}
      {onboardingOpen && (
        <div className="onboarding-backdrop">
          <section className="onboarding-card" role="dialog" aria-modal="true" aria-labelledby="onboarding-title">
            <button type="button" className="onboarding-skip" onClick={closeOnboarding}>건너뛰기</button>
            <div className="onboarding-brand"><img src="/edubridge-logo.png" alt="" /><strong>EduBridge</strong></div>
            <div key={`visual-${onboardingStep}`} className={`onboarding-visual step-${onboardingStep + 1}`} aria-hidden="true">
              {onboardingStep === 0 && <><span>교과서 PDF</span><i>→</i><strong>접근성 자료</strong></>}
              {onboardingStep === 1 && <><span>1</span><span>2</span><span>3</span></>}
              {onboardingStep === 2 && <><span>⌕</span><i>✎</i><strong>✓</strong></>}
            </div>
            <div className="onboarding-copy">
            <span className="onboarding-eyebrow">점역사를 위한 교과서 접근성 제작 도구 · {onboardingStep + 1}/3</span>
            <h2 key={`title-${onboardingStep}`} id="onboarding-title">
              {[
                <><span>시각장애 학생에게</span><span>교과서를 더 빠르게</span></>,
                <><span>반복 작업은 줄이고</span><span>제작에 집중해요</span></>,
                <><span>자동 초안을 확인하고</span><span>정확한 자료로 완성해요</span></>,
              ][onboardingStep]}
            </h2>
            <p key={`copy-${onboardingStep}`}>
              {[
                "EduBridge는 점역사의 접근성 자료 제작을 도와, 시각장애 학생이 필요한 교과서를 더 빠르게 만날 수 있도록 만든 서비스입니다.",
                "교과서 PDF의 레이아웃과 수식·표·그림을 분석하고 설명 초안을 만듭니다. 페이지별 결과는 자동 저장되어 이어서 작업할 수 있어요.",
                "자동 생성된 내용은 완성본이 아닙니다. 원본과 초안을 함께 확인하고 직접 수정한 뒤 검토 완료로 표시해 정확한 자료를 완성하세요.",
              ][onboardingStep]}
            </p>
            <div className="onboarding-dots">{[0, 1, 2].map((step) => <span className={step === onboardingStep ? "active" : ""} key={step} />)}</div>
            <footer>
              <button type="button" disabled={onboardingStep === 0} onClick={() => setOnboardingStep((step) => step - 1)}>이전</button>
              <button type="button" onClick={() => onboardingStep === 2 ? closeOnboarding() : setOnboardingStep((step) => step + 1)}>
                {onboardingStep === 2 ? "시작하기" : "다음"}
              </button>
            </footer>
            </div>
          </section>
        </div>
      )}
      {toast && (
        <div className={`app-toast ${toast.tone}`} role="status">
          <span aria-hidden="true">{toast.tone === "error" ? "!" : "✓"}</span>{toast.message}
        </div>
      )}
      {screen === "library" ? (
        <section className="textbook-library">
          <div className="library-heading">
            <div>
              <span>나의 작업공간</span>
              <h2>교과서 보관함</h2>
              <p>교과서를 선택해 분석을 이어가거나 새로운 교과서를 추가하세요.</p>
            </div>
            <div className="library-actions">
              <button type="button" className="secondary-button tutorial-replay-button" onClick={() => { setOnboardingStep(0); setOnboardingOpen(true); }}>
                튜토리얼
              </button>
              <button type="button" className="secondary-button" onClick={() => setGroupManagerOpen(true)}>그룹 관리</button>
              <button type="button" className="secondary-button trash-button" onClick={() => setActiveGroup("trash")}>
                휴지통 {trashedProjects.length > 0 && <span>{trashedProjects.length}</span>}
              </button>
              <button type="button" className="add-project-button" onClick={() => projectFileInputRef.current?.click()}>
                <span aria-hidden="true">＋</span> 교과서 추가
              </button>
            </div>
          </div>
          {(activeProjects.length > 0 || trashedProjects.length > 0) && (
            <div className="library-organizer">
              <label className="library-search">
                <span aria-hidden="true">⌕</span>
                <input
                  type="search"
                  value={libraryQuery}
                  onChange={(event) => setLibraryQuery(event.target.value)}
                  placeholder="교재 이름 검색"
                  aria-label="교재 이름 검색"
                />
                {libraryQuery && (
                  <button type="button" onClick={() => setLibraryQuery("")} aria-label="검색어 지우기">×</button>
                )}
              </label>
              <div className="library-sort">
                <span>정렬</span>
                <button
                  type="button"
                  className="sort-trigger"
                  aria-haspopup="listbox"
                  aria-expanded={sortMenuOpen}
                  onClick={() => setSortMenuOpen((current) => !current)}
                >
                  {librarySort === "recent" && "최근 작업순"}
                  {librarySort === "name" && "이름순"}
                  {librarySort === "progress" && "분석 페이지순"}
                  <span aria-hidden="true">⌄</span>
                </button>
                {sortMenuOpen && (
                  <div className="sort-menu" role="listbox" aria-label="교재 정렬 방식">
                    {[
                      ["recent", "최근 작업순", "최근에 열어본 교재부터"],
                      ["name", "이름순", "가나다순으로 정리"],
                      ["progress", "분석 페이지순", "완료한 페이지가 많은 순"],
                    ].map(([value, label, description]) => (
                      <button
                        type="button"
                        role="option"
                        aria-selected={librarySort === value}
                        className={librarySort === value ? "selected" : ""}
                        key={value}
                        onClick={() => {
                          setLibrarySort(value);
                          setSortMenuOpen(false);
                        }}
                      >
                        <span><strong>{label}</strong><small>{description}</small></span>
                        {librarySort === value && <i aria-hidden="true">✓</i>}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
          {(activeProjects.length > 0 || trashedProjects.length > 0) && (
            <div className="library-groups" aria-label="교과서 그룹">
              <button
                type="button"
                className={activeGroup === "all" ? "active" : ""}
                onClick={() => setActiveGroup("all")}
              >
                전체 <span>{activeProjects.length}</span>
              </button>
              {libraryGroups.map((group) => (
                <button
                  type="button"
                  key={group}
                  className={activeGroup === group ? "active" : ""}
                  onClick={() => setActiveGroup(group)}
                >
                  {group}
                  <span>{activeProjects.filter((project) => libraryMeta[project.id]?.group === group).length}</span>
                </button>
              ))}
              <button
                type="button"
                className={activeGroup === "ungrouped" ? "active" : ""}
                onClick={() => setActiveGroup("ungrouped")}
              >
                미분류
                <span>{activeProjects.filter((project) => !libraryMeta[project.id]?.group).length}</span>
              </button>
              {trashedProjects.length > 0 && (
                <button type="button" className={activeGroup === "trash" ? "active" : ""} onClick={() => setActiveGroup("trash")}>
                  휴지통 <span>{trashedProjects.length}</span>
                </button>
              )}
            </div>
          )}
          <div className="textbook-grid">
            {filteredProjects.map((project, index) => {
              const projectMeta = libraryMeta[project.id] || {};
              const projectIndex = projects.findIndex((item) => item.id === project.id);
              const bookColor = BOOK_COLORS.find((color) => color.id === projectMeta.color)
                || BOOK_COLORS[(projectIndex >= 0 ? projectIndex : index) % BOOK_COLORS.length];
              return (
              <article
                key={project.id}
                className="textbook-card"
              >
                {activeGroup !== "trash" && (
                  <>
                    <button
                      type="button"
                      className="book-settings-trigger"
                      aria-label={`${libraryProjectName(project)} 설정`}
                      aria-haspopup="menu"
                      aria-expanded={openBookSettings === project.id}
                      onClick={() => setOpenBookSettings((current) => current === project.id ? null : project.id)}
                    >
                      <span aria-hidden="true">•••</span>
                    </button>
                    {openBookSettings === project.id && (
                      <div className="book-settings-menu" role="menu">
                          <div className="book-settings-heading">
                            <span>교과서 설정</span>
                            <strong>{libraryProjectName(project)}</strong>
                          </div>
                          <button type="button" className="book-setting-action" onClick={() => { setOpenBookSettings(null); openRenameDialog(project); }}>
                            <span aria-hidden="true">✎</span>
                            <span><strong>이름 수정</strong><small>보관함에 표시되는 이름을 바꿔요</small></span>
                          </button>
                          <div className="book-setting-section">
                            <span>그룹</span>
                            <div className="book-setting-groups">
                              <button type="button" className={!projectMeta.group ? "active" : ""} onClick={() => updateProjectMeta(project.id, { group: "" })}>미분류</button>
                              {libraryGroups.map((group) => (
                                <button type="button" key={group} className={projectMeta.group === group ? "active" : ""} onClick={() => updateProjectMeta(project.id, { group })}>{group}</button>
                              ))}
                            </div>
                          </div>
                          <div className="book-setting-section">
                            <span>표지 색상</span>
                            <div className="book-setting-colors">
                              {BOOK_COLORS.map((color) => (
                                <button
                                  type="button"
                                  key={color.id}
                                  className={bookColor.id === color.id ? "active" : ""}
                                  style={{ "--swatch": color.value }}
                                  onClick={() => updateProjectMeta(project.id, { color: color.id })}
                                  aria-label={`${color.label} 표지`}
                                  title={color.label}
                                />
                              ))}
                            </div>
                          </div>
                          <button type="button" className="book-setting-action danger" onClick={() => moveProjectToTrash(project)}>
                            <span aria-hidden="true">🗑️</span>
                            <span><strong>보관함에서 치우기</strong><small>휴지통에서 다시 복원할 수 있어요</small></span>
                          </button>
                      </div>
                    )}
                  </>
                )}
                <button
                  type="button"
                  className="textbook-main"
                  disabled={activeGroup === "trash"}
                  onClick={() => activeGroup !== "trash" && openProject(project.id)}
                  aria-label={`${libraryProjectName(project)} 열기`}
                >
                <span className="textbook-cover" style={{ "--book-color": bookColor.value }} aria-hidden="true">
                  <span className="book-pages"><i /><i /><i /><i /><i /></span>
                  <span className="book-front">
                    <strong>{libraryProjectName(project).slice(0, 18)}</strong>
                    <i />
                  </span>
                </span>
                <span className="textbook-card-copy">
                  <strong>{libraryProjectName(project)}</strong>
                  <small>{project.pageCount}페이지 · {fileSizeLabel(project.fileSize)}</small>
                  <span>
                    <em>{projectPageCounts[project.id] || 0}페이지 저장</em>
                    <time>{savedAtLabel(project.updatedAt)}</time>
                  </span>
                </span>
                <span className="textbook-open" aria-hidden="true">→</span>
                </button>
                {activeGroup === "trash" ? (
                  <div className="textbook-tools trash-tools">
                    <button type="button" onClick={() => restoreProject(project)}>보관함으로 복원</button>
                    <button type="button" className="permanent-delete-button" onClick={() => permanentlyDeleteProject(project)}>완전 삭제</button>
                  </div>
                ) : (
                <div className="textbook-tools">
                  <div className="group-picker">
                    <span>그룹</span>
                    <button
                      type="button"
                      className="group-picker-trigger"
                      aria-haspopup="listbox"
                      aria-expanded={openGroupMenu === project.id}
                      onClick={() => setOpenGroupMenu((current) => current === project.id ? null : project.id)}
                    >
                      {projectMeta.group || "미분류"}
                      <span aria-hidden="true">⌄</span>
                    </button>
                    {openGroupMenu === project.id && (
                      <div className="group-picker-menu" role="listbox" aria-label={`${project.name} 그룹 선택`}>
                        <button
                          type="button"
                          role="option"
                          aria-selected={!projectMeta.group}
                          className={!projectMeta.group ? "selected" : ""}
                          onClick={() => {
                            updateProjectMeta(project.id, { group: "" });
                            setOpenGroupMenu(null);
                          }}
                        >
                          <span className="group-dot ungrouped" />미분류
                          {!projectMeta.group && <i aria-hidden="true">✓</i>}
                        </button>
                        {libraryGroups.map((group, groupIndex) => (
                          <button
                            type="button"
                            role="option"
                            aria-selected={projectMeta.group === group}
                            className={projectMeta.group === group ? "selected" : ""}
                            key={group}
                            onClick={() => {
                              updateProjectMeta(project.id, { group });
                              setOpenGroupMenu(null);
                            }}
                          >
                            <span className={`group-dot tone-${(groupIndex % 4) + 1}`} />{group}
                            {projectMeta.group === group && <i aria-hidden="true">✓</i>}
                          </button>
                        ))}
                        <button
                          type="button"
                          className="create-group-option"
                          onClick={() => {
                            setOpenGroupMenu(null);
                            setGroupDialogOpen(true);
                          }}
                        >
                          <span aria-hidden="true">＋</span> 새 그룹 만들기
                        </button>
                      </div>
                    )}
                  </div>
                  <div className="book-color-picker" aria-label="표지 색상">
                    {BOOK_COLORS.map((color) => (
                      <button
                        type="button"
                        key={color.id}
                        className={bookColor.id === color.id ? "active" : ""}
                        style={{ "--swatch": color.value }}
                        onClick={() => updateProjectMeta(project.id, { color: color.id })}
                        aria-label={`${color.label}색 표지`}
                        title={color.label}
                      />
                    ))}
                  </div>
                  <button
                    type="button"
                    className="rename-book-button"
                    onClick={() => openRenameDialog(project)}
                    aria-label={`${libraryProjectName(project)} 이름 수정`}
                    title="이름 수정"
                  >
                    <span aria-hidden="true">✎</span>
                  </button>
                  <button
                    type="button"
                    className="trash-book-button"
                    onClick={() => moveProjectToTrash(project)}
                    aria-label={`${libraryProjectName(project)} 휴지통으로 이동`}
                    title="휴지통으로 이동"
                  >
                    <span aria-hidden="true">⌫</span>
                  </button>
                </div>
                )}
              </article>
              );
            })}
            {activeGroup !== "trash" && activeProjects.length > 0 && (
              <button type="button" className="new-textbook-card" onClick={() => projectFileInputRef.current?.click()}>
                <span aria-hidden="true">＋</span>
                <strong>새 교과서 추가</strong>
                <small>PDF 파일을 불러와 작업공간을 만듭니다.</small>
              </button>
            )}
          </div>
          {(activeProjects.length > 0 || trashedProjects.length > 0) && filteredProjects.length === 0 && (
            <div className="library-no-results">
              <span aria-hidden="true">⌕</span>
              <strong>조건에 맞는 교재가 없어요</strong>
              <p>검색어를 바꾸거나 다른 그룹을 선택해보세요.</p>
              <button type="button" onClick={() => { setLibraryQuery(""); setActiveGroup("all"); }}>
                전체 교재 보기
              </button>
            </div>
          )}
          {activeProjects.length > 0 && activeGroup === "all" && (
            <div className="library-overview">
              <div className="overview-title">
                <span aria-hidden="true">🌱</span>
                <div><strong>학습 자료 현황</strong><small>차근차근 접근성 자료를 완성하고 있어요.</small></div>
              </div>
              <dl>
                <div><dt>등록 교재</dt><dd>{activeProjects.length}<small>권</small></dd></div>
                <div><dt>분석 완료</dt><dd>{totalSavedPages}<small>페이지</small></dd></div>
                <div><dt>생성한 그룹</dt><dd>{libraryGroups.length}<small>개</small></dd></div>
              </dl>
              {recentProject && (
                <button type="button" onClick={() => openProject(recentProject.id)}>
                  <span>최근 작업</span>
                  <strong>{libraryProjectName(recentProject)}</strong>
                  <i aria-hidden="true">이어하기 →</i>
                </button>
              )}
            </div>
          )}
          {activeProjects.length === 0 && trashedProjects.length === 0 && (
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
              <h2>{projects.find((project) => project.id === activeProjectId)
                ? libraryProjectName(projects.find((project) => project.id === activeProjectId))
                : file?.name.replace(/\.pdf$/i, "")}</h2>
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
            <div className="page-count"><span>전체 페이지</span><strong>{pageCount ?? "-"}</strong></div>
          </div>
          <button className="primary-button" disabled={busy || !file} onClick={analyzePage}>
            <span>{status === "analyzing" ? "분석 중" : "페이지 분석 시작"}</span>
            <span aria-hidden="true">{status === "analyzing" ? "···" : "→"}</span>
          </button>
          {error && <div className="error-box">{error}</div>}
          {activeProjectId && (
            <>
            <button
              type="button"
              className="saved-pages-summary-card"
              onClick={() => setSavedPagesDialogOpen(true)}
            >
              <span className="saved-summary-icon" aria-hidden="true">
                <svg viewBox="0 0 24 24">
                  <path d="M6 4.5h12v15H6z" />
                  <path d="M9 8h6M9 11.5h6M9 15h4" />
                </svg>
              </span>
              <span className="saved-summary-copy">
                <strong>저장된 페이지</strong>
                <small>
                  {savedPages.length > 0
                    ? `최근 ${savedPages.at(-1)?.pageNumber}페이지 · 눌러서 검색 및 이동`
                    : "분석한 페이지가 여기에 자동 저장돼요"}
                </small>
              </span>
              <span className="saved-summary-count"><strong>{savedPages.length}</strong><small>페이지</small></span>
              <span className="saved-summary-arrow" aria-hidden="true">→</span>
            </button>
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
              {savedPages.length > 0 && (
                <button
                  type="button"
                  className="secondary-button textbook-html-download"
                  onClick={downloadTextbookHtml}
                >
                  교과서 HTML 다운로드
                </button>
              )}
            </section>
            </>
          )}
          {result && (
            <details className="stats analysis-details">
              <summary>
                <span>분석 상세</span>
                <strong>{result.page.blocks.length}개 블록</strong>
                <svg className="analysis-details-chevron" viewBox="0 0 16 16" aria-hidden="true">
                  <path d="m4 6 4 4 4-4" />
                </svg>
              </summary>
              <div className="type-list">
                {Object.entries(blockStats).map(([type, count]) => (
                  <div key={type}><span>{type}</span><strong>{count}</strong></div>
                ))}
              </div>
              <div className="download-actions">
                <button className="secondary-button" onClick={downloadCurrentPageHtml}>
                  접근성 HTML 다운로드
                </button>
                <button className="secondary-button" onClick={downloadJson}>JSON 다운로드</button>
              </div>
            </details>
          )}
        </aside>
        <section
          ref={resultWorkspaceRef}
          className={`result-workspace ${activeView === "page" ? "page-workspace" : ""}`}
        >
          <nav className="view-tabs" aria-label="결과 보기">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                className={`${activeView === tab.id ? "active" : ""} ${tab.id === "page" ? "page-view-tab" : ""}`.trim()}
                onClick={() => setActiveView(tab.id)}
              >
                {tab.label}
              </button>
            ))}
            {result && (
              <button
                type="button"
                className="fullscreen-toggle"
                onClick={toggleResultFullscreen}
                aria-pressed={resultFullscreen}
                title={resultFullscreen ? "전체화면 종료" : "전체화면 보기"}
              >
                <svg viewBox="0 0 24 24" aria-hidden="true">
                  {resultFullscreen ? (
                    <path d="M9 4v5H4M15 4v5h5M9 20v-5H4M15 20v-5h5" />
                  ) : (
                    <path d="M9 4H4v5M15 4h5v5M9 20H4v-5M15 20h5v-5" />
                  )}
                </svg>
                <span>{resultFullscreen ? "전체화면 종료" : "전체화면"}</span>
              </button>
            )}
          </nav>
          {busy ? <ProcessingState status={status} elapsedSeconds={elapsedSeconds} /> : !result ? (
            <div className="empty-state result-empty">
              <div className="empty-illustration" aria-hidden="true"><span /><span /><span /></div>
              <h2>분석 결과가 여기에 표시됩니다</h2>
              <p>왼쪽에서 교과서 PDF와 페이지를 선택한 뒤 분석을 시작하세요.</p>
            </div>
          ) : activeView === "layout" ? (
            <div className="layout-view">
              <div className="pane-header"><h2>레이아웃 시각화</h2><span>{result.page.page_id}페이지</span></div>
              <div className="layout-image-pane">
                <img src={result.visualization_image} alt="레이아웃 분석 시각화" />
              </div>
            </div>
          ) : activeView === "page" ? (
            <PageDescriptionView
              result={result}
              onUpdateDescription={updatePageDescription}
              onMoveBlock={updateBrailleReadingOrder}
              onUpdateFigureTreatment={updateFigureTreatment}
              onUpdateReview={updatePageReview}
              onUpdateBlock={updatePageBlock}
              onAddFigure={addMissingFigure}
              onDeleteBlock={deletePageBlock}
            />
          ) : activeView === "elements" ? (
            <div className="element-analysis-view">
              <nav className="element-tabs" aria-label="요소 분석 종류">
                {[
                  ["formula", "수식"],
                  ["table", "표"],
                  ["figure", "그림"],
                ].map(([type, label]) => (
                  <button
                    type="button"
                    key={type}
                    className={activeElementType === type ? "active" : ""}
                    onClick={() => setActiveElementType(type)}
                  >
                    <span>{label}</span>
                    <strong>{blockStats[type] || 0}</strong>
                  </button>
                ))}
              </nav>
              <AnalysisInspector result={result} type={activeElementType} />
            </div>
          ) : (
            <div className="html-view">
              <div className="pane-header">
                <h2>접근성 HTML</h2>
                <button type="button" className="html-source-download" onClick={downloadCurrentPageHtml}>
                  HTML 다운로드
                </button>
              </div>
              <pre><code>{currentPageHtmlSource()}</code></pre>
            </div>
          )}
        </section>
          </section>
        </>
      )}
    </main>
  );
}
