# Economics Math Textbook Layout Parser

경제수학 교과서 PDF에서 원하는 페이지를 분석하고, 문서 요소를 bounding box 단위의 JSON으로 반환하는 MVP입니다. React 웹 화면에서 PDF와 페이지 번호를 선택하면 원본 페이지, 탐지 시각화, JSON을 함께 확인할 수 있습니다.

## Current Pipeline

```text
PDF upload
  -> selected page rendering (PyMuPDF)
  -> embedded PDF text or PaddleOCR
  -> DocLayout-YOLO layout detection
  -> OCR/OpenCV supplementary detection
  -> textbook-aware postprocessing
  -> reading order
  -> JSON + visualization
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
│   └── src/
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

그림 설명을 확인하려면 `Figure 설명 생성`을 체크합니다. OpenCLIP이 figure를 graph, table, mathematical diagram, illustration, photo로 분류하고 Qwen3-VL-2B-Instruct가 유형별 지시문으로 설명을 생성합니다. 분석이 끝나면 Figure 탭에서 figure crop, 설명, 모델명, confidence와 생성 시간을 확인할 수 있습니다. 최초 실행은 Hugging Face 모델 다운로드 때문에 오래 걸릴 수 있습니다.

결과 화면은 다음 탭으로 구성됩니다.

- `Layout`: 전체 페이지의 레이아웃 탐지 시각화
- `Formula`: 수식 crop, LaTeX/MathML 결과, confidence, 설명과 warning
- `Table`: 표 crop, 복원된 셀 구조, confidence, 설명과 warning
- `Figure`: 그림 crop, 유형·축·계열·데이터, 설명과 warning
- `JSON`: 레이아웃과 의미 분석 결과 원문

현재 API의 `semantic_analyses`는 빈 배열입니다. 각 담당 분석기가 구현되면 통합 담당자가 이 배열에 `schemas/block_analysis.schema.json` 형식의 결과를 연결합니다. 담당자는 프론트엔드를 직접 수정하지 않습니다.

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

## Current Limitations

- `formula` 탐지는 수식 영역의 위치를 찾는 단계입니다. 현재 일반 OCR의 `text`는 분수, 위첨자, 아래첨자 구조를 정확히 보존하지 못할 수 있습니다. 수식을 의미 구조로 사용하려면 formula crop에 수식 전용 OCR을 적용해 LaTeX 또는 MathML 필드를 추가해야 합니다.
- `figure` 설명 생성은 선택 기능입니다. OpenCLIP으로 5개 유형을 분류한 뒤 Qwen3-VL-2B-Instruct로 설명을 생성하며, 기본값은 비활성화이므로 필요할 때만 모델을 내려받습니다.
- 어떤 범용 모델도 모든 교과서에서 누락 0건을 보장하지는 않습니다. 다른 출판사와 과목으로 일반화하려면 다양한 페이지의 정답 라벨과 정량 평가가 필요합니다.

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
