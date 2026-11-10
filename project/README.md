# HOPE: 시각장애학생을 위한 접근성 교과서 제작 도구

경제수학 교과서 PDF를 분석해 점자교과서 제작을 돕는 **점역 전처리 도구**입니다. 페이지의 제목, 본문, 수식, 표, 그림을 찾아 점자 읽기 순서로 정리하고, 그림마다 점역 지침에 따른 처리 방식(촉각 그래픽 / 점역자 주 / 생략)을 근거와 함께 제안합니다. 검수자는 웹 화면에서 제안을 확정·수정한 뒤 점역 편집용 텍스트, 촉도 작업 목록, 자동 점자 초안, 접근성 HTML로 내보냅니다.

HOPE는 점자교과서를 완성하지 않습니다. 수학 점자 변환, 촉각 그래픽 제작, 최종 교정은 점역사가 합니다.

## 주요 기능

- **레이아웃 분석**: DocLayout-YOLO, OCR, 교과서 맞춤 보정 규칙으로 페이지 블록과 읽기 순서를 찾습니다.
- **요소별 의미 분석** (`src/analysis/`)
  - 수식: 수식 crop에서 LaTeX를 인식합니다.
  - 표: 행·열·셀 구조를 복원하고 설명문을 만듭니다.
  - 그림: OpenCLIP으로 그림 유형을 분류하고 GPT-5로 대체 설명을 생성합니다. 이 기능은 선택 사항입니다.
- **페이지 접근성 설명** (`src/page_description.py`): 블록 결과를 읽기 순서대로 이어 붙여 페이지 전체 설명 초안을 만듭니다. 초안에 없는 사실이나 숫자는 추가하지 않습니다.
- **점자 읽기 순서 검수** (`frontend/src/brailleReadingOrder.js`): 블록 순서를 조정하고, 그림과 캡션을 연결하고, 순서 경고를 표시합니다. 페이지마다 "검수 완료"로 표시할 수 있습니다.
- **블록 재분석** (`/api/analyze-block`): 블록 유형을 수동으로 바꾸면 페이지 전체를 다시 돌리지 않고 그 블록만 다시 분석합니다.
- **접근성 HTML 내보내기** (`frontend/src/accessibleHtml.js`): 검수한 페이지들을 제목 구조와 그림 설명이 들어간 HTML 한 파일로 저장합니다.

### 점역 지침 대응 (`reports/HOPE 점역 지침 대조 평가.md` 기준)

- **처리 방식 판정** (`src/analysis/figure/treatment.py`): 설명을 만들기 전에 그림마다 촉도 / 점역자 주 / 생략을 제안하고 근거 조항을 저장합니다. 판정이 생성 방식을 정합니다(촉도는 제작 사양 목록, 점역자 주는 한 문장 묘사, 생략은 모델 호출 없음). 수치를 해석하지 않고 보이는 것만 묘사하며, 근거 없는 수치는 `[확인 필요]`로 표시합니다.
- **점자 읽기 구조 복원** (`src/braille_structure.py`, `src/vector_table.py`): 캡션·출처 분리와 그림 연결, 2단(사고 과정/풀이 과정) 분리, 물음 사이 그림 이동과 원래 위치 점역자 주, 날개 위치 조정, 여백 배지를 유형어 "(함께 풀기)"로 변환, 원본 쪽수, 학습지형 빈칸 표 복원, OCR 줄바꿈 띄어쓰기 교정(kiwipiepy).
- **검수 화면**: 확인할 항목 체크리스트, AI 제안 일괄 확정, 촉도·점역자 주·생략 선택, AI가 자동으로 바꾼 것 목록과 원본 순서 되돌리기, 분석 진행·완료 표시.
- **내보내기**: 점역 편집용 텍스트(점역자 주표 `,'…,'`, 빈 셀 `==`, `[촉도 n]`, 이 책의 점역자 주), 촉도 작업 목록 CSV, 자동 점자 초안(braillify, 검수용), 검수 기록 JSON(AI 제안 수용·변경, 수정 횟수, 검수 시간).

## Current Pipeline

```text
PDF upload
  -> selected page rendering (PyMuPDF)
  -> embedded PDF text or PaddleOCR
  -> DocLayout-YOLO layout detection
  -> OCR/OpenCV supplementary detection
  -> textbook-aware postprocessing
  -> reading order
  -> braille structure (captions, columns, sidebars, badges, page number, tables)
  -> formula / table / figure analysis (figure: treatment first, then generation)
  -> page description + review UI + exports
```

웹에서는 같은 DocLayout-YOLO 모델을 세 가지 방식으로 비교할 수 있습니다. 웹 요청은 선택한 페이지만 임시 폴더에서 처리하므로 페이지 PNG가 프로젝트에 계속 쌓이지 않습니다.

- `DocLayout-YOLO + 보정 규칙`: 모델 탐지 후 OCR/OpenCV 보충, 교과서 박스 분리, 누락 문장·수식 복구, 오분류 교정을 적용합니다.
- `DocLayout-YOLO + 3단원 보정 규칙`: `3단원 좌표평면과 그래프`처럼 문제·그래프·풀이가 촘촘히 섞인 페이지를 확인하기 위한 실험 모드입니다. 기존 클래스 체계는 유지하되, 작은 제목/캡션 조각을 주변 paragraph에 붙이고 가까운 paragraph 조각을 더 적극적으로 병합합니다.
- `DocLayout-YOLO 원본`: 모델이 반환한 bounding box와 클래스에 OCR 텍스트와 reading order만 추가합니다. 프로젝트의 보충·후처리 규칙은 적용하지 않습니다.

세 모드는 같은 렌더링 DPI, OCR, DocLayout-YOLO 가중치를 사용하므로 보정 규칙 적용 전후를 비교하기 위한 옵션입니다. `3단원 보정 규칙`은 아직 범용 정책이 아니라 구조화 안정화 실험용 프로필입니다.

## Output Classes

최종 JSON의 `type`은 다음 9개 클래스로 정규화됩니다.

- `title`
- `section_title`
- `paragraph`
- `formula`
- `table`
- `figure`
- `caption`
- `footer`
- `page_number`

`graph`와 `image`는 최종적으로 `figure`에 통합합니다. 예제·문제·풀이 박스는 별도 레이아웃 클래스로 강제하지 않고 내부 콘텐츠를 `paragraph`, `formula`, `table`, `figure` 등으로 나눕니다. 박스의 역할은 필요한 경우 다음처럼 보존합니다.

```json
"context": {
  "role_hint": "example"
}
```

가능한 `role_hint` 값은 `example`, `problem`, `solution`입니다.

## Project Structure

```text
project/
├── schemas/
│   └── block_analysis.schema.json
├── backend/
│   └── app.py
├── data/
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── accessibleHtml.js
│   │   └── brailleReadingOrder.js
│   └── tests/
├── outputs/
├── src/
│   ├── analysis/
│   │   ├── formula/
│   │   ├── table/
│   │   └── figure/
│   ├── export_json.py
│   ├── layout_detection.py
│   ├── main.py
│   ├── ocr.py
│   ├── page_description.py
│   ├── page_pipeline.py
│   ├── pdf_text.py
│   ├── pdf_to_image.py
│   └── reading_order.py
├── tests/
│   ├── formula/
│   ├── table/
│   ├── figure/
│   ├── test_analysis_schema.py
│   └── test_layout_postprocessing.py
├── OWNERSHIP.md
├── requirements.txt
├── run_backend.ps1
├── run_frontend.ps1
└── README.md
```

## Installation

Windows PowerShell 기준입니다.

```powershell
cd C:\Users\USER\HOPE\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

cd .\frontend
npm.cmd install
```

DocLayout-YOLO와 PaddleOCR 모델은 처음 사용할 때 내려받기 때문에 첫 분석이 이후 요청보다 오래 걸릴 수 있습니다.

그림 분석(OpenCLIP, GPT) 의존성은 `src/analysis/figure/requirements.txt`에 있습니다. 이 중 `hanja`는 `pyyaml==6.0.1`을 요구하지만 PaddleOCR은 6.0.2가 필요합니다. 설치 중 `pyyaml` 충돌이 나면 `hanja`만 의존성 없이 따로 설치하세요(한자 사전은 HOPE가 UTF-8로 직접 읽으므로 `pyyaml` 버전과 무관합니다).

```powershell
python -m pip install --no-deps hanja==0.15.1
```

수식 인식(pix2tex)은 별도 가상환경 `.venv-pix2tex`에서 돌아갑니다(`requirements-pix2tex.txt`). 백엔드는 pix2tex 프로세스를 한 번 띄워 재사용하므로 첫 수식만 모델 로딩 시간이 걸립니다.

## Run Web App

PowerShell 터미널 두 개를 사용합니다.

Backend:

```powershell
cd C:\Users\USER\HOPE\project
powershell -ExecutionPolicy Bypass -File .\run_backend.ps1
```

Frontend:

```powershell
cd C:\Users\USER\HOPE\project
powershell -ExecutionPolicy Bypass -File .\run_frontend.ps1
```

브라우저에서 `http://127.0.0.1:5174`를 엽니다. PDF를 업로드하고 페이지 번호와 DPI를 입력한 뒤, `Layout model`에서 보정 규칙 적용 여부를 선택하고 `페이지 분석`을 누릅니다.

그림 설명을 확인하려면 `Figure 설명 생성`을 체크합니다. OpenCLIP이 figure를 graph, table, mathematical diagram, illustration, photo로 분류하고 OpenAI GPT-5가 유형별 지시문으로 설명을 생성합니다. 분석이 끝나면 Figure 탭에서 figure crop, 설명, 모델명, confidence와 생성 시간을 확인할 수 있습니다. 사용하려면 `OPENAI_API_KEY` 환경 변수가 설정되어 있어야 합니다.

결과 화면은 다음 탭으로 구성됩니다.

- `Layout`: 전체 페이지의 레이아웃 탐지 시각화
- `Formula`: 수식 crop, LaTeX/MathML 결과, confidence, 설명과 warning
- `Table`: 표 crop, 복원된 셀 구조, confidence, 설명과 warning
- `Figure`: 그림 crop, 유형, 설명과 warning
- `JSON`: 레이아웃과 의미 분석 결과 원문

`/api/analyze` 응답의 `semantic_analyses`에는 수식·표·그림 분석 결과가 `schemas/block_analysis.schema.json` 형식으로 들어갑니다. `page_description`에는 페이지 접근성 설명이 들어갑니다. 결과 화면에서 점자 읽기 순서와 페이지 설명을 검수한 뒤 `HTML` 버튼으로 접근성 자료를 내려받습니다.

교과서 PDF는 웹에서 직접 업로드하므로 반드시 `data/`에 넣을 필요는 없습니다. CLI로 전체 PDF를 처리할 때는 `project/data/` 사용을 권장합니다.

## JSON Format

```json
{
  "page_id": 15,
  "blocks": [
    {
      "block_id": "p15_b1",
      "type": "paragraph",
      "bbox": [105, 95, 745, 233],
      "text": "경제 성장률에 대한 설명...",
      "score": 0.95,
      "detector": "doclayout_yolo",
      "context": {
        "role_hint": "example"
      },
      "reading_order": 1
    }
  ]
}
```

`bbox`는 렌더링된 페이지 이미지 기준의 `[x1, y1, x2, y2]` 좌표입니다. `detector`는 해당 블록의 출처를 나타내며 모델 탐지 외에도 OCR 복구 단계가 기록될 수 있습니다.

## Full PDF CLI

전체 페이지를 파일로 저장하며 처리하려면 다음 명령을 사용합니다.

```powershell
cd C:\Users\USER\HOPE\project
.\.venv\Scripts\python.exe .\src\main.py `
  --pdf .\data\economics_math_textbook.pdf `
  --output .\outputs `
  --dpi 200 `
  --yolo-model "hf:juliozhao/DocLayout-YOLO-DocStructBench"
```

CLI 출력 구조:

```text
outputs/
├── pages/
├── visualizations/
└── layout.json
```

## Postprocessing Policy

- 숫자가 많더라도 긴 한국어 설명문이면 `paragraph`를 우선합니다.
- 짧은 셀 값과 반복 행·열 구조가 나타나면 `table`로 유지합니다.
- 설명과 계산이 섞인 풀이 영역은 `paragraph`로 두고, 독립 수식 행은 `formula` 하위 블록으로 복구합니다.
- 같은 행에 나란히 있는 여러 식은 하나의 formula 블록으로 묶을 수 있습니다.
- 모델이 놓친 짧은 문장 조각은 가장 가까운 paragraph에 연결합니다.
- paragraph 옆의 작은 `생각열기`·`문제`·`예제` 배지는 독립 제목으로 남기지 않고 paragraph의 `context`에 역할 정보로 보존합니다.
- 보정 모드에서는 모델 기준점 바로 아래의 figure 후보도 페이지 상대 크기가 합리적이면 복구하며, 과도하게 크거나 아이콘처럼 작은 후보는 제외합니다.
- 모델 탐지를 우선하며 OCR/OpenCV 규칙은 누락 복구와 클래스 교정에 사용합니다.

## Tests

```powershell
cd C:\Users\USER\HOPE\project
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

현재 테스트는 숫자가 많은 문단, 설명형 계산, 실제 표, 박스 내부 수식, 짧은 문장 이어쓰기 등 주요 회귀 사례를 검사합니다.

점자 읽기 순서와 접근성 HTML 테스트:

```powershell
cd C:\Users\USER\HOPE\project\frontend
npm.cmd test
```

## Current Limitations

- `formula` 탐지는 수식 영역의 위치를 찾는 단계입니다. 현재 일반 OCR의 `text`는 분수, 위첨자, 아래첨자 구조를 정확히 보존하지 못할 수 있습니다. 수식을 의미 구조로 사용하려면 formula crop에 수식 전용 OCR을 적용해 LaTeX 또는 MathML 필드를 추가해야 합니다.
- `figure` 설명 생성은 선택 기능입니다. OpenCLIP으로 5개 유형을 분류한 뒤 OpenAI GPT-5 API로 설명을 생성하며, 기본값은 비활성화이므로 필요할 때만 API를 호출합니다.
- 어떤 범용 모델도 모든 교과서에서 누락 0건을 보장하지는 않습니다. 다른 출판사와 과목으로 일반화하려면 다양한 페이지의 정답 라벨과 정량 평가가 필요합니다.
- 레이아웃 모델이 그림·표를 놓치면 처리 방식 판정도 할 수 없습니다. 검수 체크리스트가 축 눈금 조각이나 떨어진 캡션 같은 흔적을 "그림 누락 의심"으로 알려 주며, 검수자가 "그림 추가"로 넣어야 합니다.
- 처리 방식 판정은 그림 유형, 주변 발문, 크기, 그림 안 숫자만 보는 규칙입니다. 분류기가 큰 장식 아이콘을 도식으로 보는 등 오판이 있으므로 모든 제안은 검수자가 확정합니다.
- 배지 사전, 기호 글꼴 대응, 학습지형 표 규칙, 캡션 검색 범위, 작은 그림 기준은 이 교과서에 맞춘 값입니다(코드의 `ponytail:` 주석). 다른 교과서에는 조정이 필요합니다.
- 한국 수학 점자 변환은 구현하지 않았습니다. 점자 초안에서 LaTeX 수식은 "수식 점역 필요"로 남습니다. 점역자 주표·빈 셀 점형은 북미 점자 ASCII 대응을 따른 것으로, 점역사 확인이 필요합니다.

## Semantic Analysis Contract

역할별 파일 소유권과 병렬 작업 규칙은 `OWNERSHIP.md`를 따릅니다.

수식, 표, 그림 분석기는 반드시 `schemas/block_analysis.schema.json`을 준수해야 합니다. 이 스키마는 기존 레이아웃 JSON을 대체하지 않으며, 탐지된 `formula`, `table`, `figure` 블록에 대한 후속 의미 분석 결과를 정의합니다.

공통 규칙:

- `schema_version`은 현재 `1.0.0`입니다. 호환되지 않는 변경은 버전을 올리고 팀 합의를 거쳐야 합니다.
- `bbox`는 렌더링된 페이지의 픽셀 좌표 `[x1, y1, x2, y2]`입니다.
- `type`은 `formula`, `table`, `figure` 중 하나입니다.
- `analysis.status`는 `success`, `partial`, `failed` 중 하나입니다.
- confidence는 `0.0` 이상 `1.0` 이하이며, 모델이 제공하지 않으면 `null`입니다.
- 탐지 confidence와 의미 분석 confidence는 서로 다른 값이며 합산하지 않습니다.
- 알 수 없는 값은 빈 문자열이나 추정값 대신 `null`로 기록합니다.
- 모델 이름과 버전을 기록하며, 버전을 알 수 없으면 `null`을 사용합니다.
- 일부 값만 인식했으면 `partial`과 `warnings`를 사용합니다. 값을 추측해서 채우지 않습니다.
- `crop_path`는 선택 사항입니다. `page_id`와 `bbox`로 crop을 재생성할 수 있어야 합니다.
- 앞뒤 본문과 캡션은 `context`에 블록 ID로 연결합니다.
- 유형별 결과는 `analysis.result`에 저장하며 `kind`는 바깥쪽 `type`과 같아야 합니다.

유형별 결과:

- `formula`: LaTeX, MathML, 일반 텍스트를 구분해 저장합니다.
- `table`: 행·열 수, 셀 위치, 헤더 여부, `row_span`, `column_span`을 저장합니다.
- `figure`: 그림 유형, 제목, 축, 단위, 계열과 데이터 점을 저장합니다. 읽지 못한 수치를 만들어 내지 않습니다.
- `description`: 의미 분석과 분리된 선택 영역입니다. 짧은 설명, 상세 설명, 점역 참고, 문맥 사용 여부와 검수 상태를 저장합니다.

분석 결과 예시:

```json
{
  "schema_version": "1.0.0",
  "page_id": 15,
  "block_id": "p15_b4",
  "type": "formula",
  "bbox": [105, 329, 745, 426],
  "crop_path": "crops/p15_b4.png",
  "detection": {
    "model": {"name": "DocLayout-YOLO", "version": null},
    "confidence": 0.94
  },
  "analysis": {
    "status": "success",
    "model": {"name": "PP-FormulaNet", "version": "server"},
    "confidence": 0.88,
    "result": {
      "kind": "formula",
      "latex": "\\frac{f(b)-f(a)}{b-a}",
      "mathml": null,
      "plain_text": null
    }
  },
  "context": {
    "previous_block_id": "p15_b3",
    "next_block_id": "p15_b5",
    "caption_block_id": null,
    "nearby_block_ids": ["p15_b3", "p15_b5"]
  },
  "warnings": []
}
```

스키마 검증:

```powershell
cd C:\Users\USER\HOPE\project
.\.venv\Scripts\python.exe -m unittest tests.test_analysis_schema -v
```

에이전트와 작업할 때는 이 README와 스키마를 먼저 읽도록 요청합니다. 공통 계약 변경은 기능 구현과 분리된 PR에서 세 명의 합의를 거쳐 진행합니다.
