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

웹에서 DocLayout-YOLO를 선택하면 모델이 기본 레이아웃을 먼저 탐지하며, 규칙은 모델이 놓친 텍스트와 수식을 복구하고 오분류를 교정하는 보조 단계로 사용됩니다. Heuristic을 선택하면 OCR과 OpenCV 기반 탐지만 실행합니다. 웹 요청은 선택한 페이지만 임시 폴더에서 처리하므로 페이지 PNG가 프로젝트에 계속 쌓이지 않습니다.

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
├── backend/
│   └── app.py
├── data/
├── frontend/
│   └── src/
├── outputs/
├── src/
│   ├── export_json.py
│   ├── layout_detection.py
│   ├── main.py
│   ├── ocr.py
│   ├── page_pipeline.py
│   ├── pdf_text.py
│   ├── pdf_to_image.py
│   └── reading_order.py
├── tests/
│   └── test_layout_postprocessing.py
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

브라우저에서 `http://127.0.0.1:5174`를 엽니다. PDF를 업로드하고 페이지 번호와 DPI를 입력한 뒤, `Layout model`에서 `DocLayout-YOLO`를 선택하고 `페이지 분석`을 누릅니다.

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
- 모델 탐지를 우선하며 OCR/OpenCV 규칙은 누락 복구와 클래스 교정에 사용합니다.

## Tests

```powershell
cd C:\Users\USER\HOPE\project
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

현재 테스트는 숫자가 많은 문단, 설명형 계산, 실제 표, 박스 내부 수식, 짧은 문장 이어쓰기 등 주요 회귀 사례를 검사합니다.

## Current Limitations

- `formula` 탐지는 수식 영역의 위치를 찾는 단계입니다. 현재 일반 OCR의 `text`는 분수, 위첨자, 아래첨자 구조를 정확히 보존하지 못할 수 있습니다. 수식을 의미 구조로 사용하려면 formula crop에 수식 전용 OCR을 적용해 LaTeX 또는 MathML 필드를 추가해야 합니다.
- `figure`는 현재 영역만 탐지하며 그래프나 그림에 대한 자연어 설명은 생성하지 않습니다. 이후 figure crop, 주변 caption/paragraph, 비전 모델을 결합하는 단계가 필요합니다.
- 어떤 범용 모델도 모든 교과서에서 누락 0건을 보장하지는 않습니다. 다른 출판사와 과목으로 일반화하려면 다양한 페이지의 정답 라벨과 정량 평가가 필요합니다.
