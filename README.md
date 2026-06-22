# Economics Math Textbook Layout Parser

경제수학 교과서 PDF를 페이지 이미지로 변환하고, 교과서형 레이아웃 요소를 bounding box 단위로 분리해 JSON으로 저장하는 MVP입니다.

## Architecture

1. `pdf_to_image.py`: PDF 각 페이지를 PNG로 렌더링합니다.
2. `ocr.py`: PaddleOCR로 텍스트 라인을 추출하고, 탐지된 블록에 텍스트를 붙입니다.
3. `layout_detection.py`: DocLayout-YOLO 모델이 있으면 우선 사용하고, 없으면 OCR + OpenCV 휴리스틱으로 레이아웃을 탐지합니다.
4. `reading_order.py`: 사람이 읽는 순서에 가깝게 위에서 아래, 같은 행은 왼쪽에서 오른쪽으로 정렬합니다.
5. `export_json.py`: 페이지별 블록을 JSON으로 저장합니다.
6. `main.py`: 전체 파이프라인 실행 및 시각화 이미지 저장을 담당합니다.

## Detect Classes

- `title`
- `section_title`
- `paragraph`
- `formula`
- `table`
- `graph`
- `image`
- `example_box`
- `problem_box`
- `solution_box`
- `caption`
- `footer`
- `page_number`

## Folder Structure

```text
project/
├── data/
├── outputs/
├── src/
│   ├── pdf_to_image.py
│   ├── layout_detection.py
│   ├── ocr.py
│   ├── reading_order.py
│   ├── export_json.py
│   └── main.py
├── requirements.txt
└── README.md
```

## PDF 넣는 위치

교과서 PDF 파일은 `project/data/` 폴더에 넣으면 됩니다.

예시:

```text
project/data/economics_math_textbook.pdf
```

현재 사용자가 열어둔 PDF가 `C:\Users\USER\Desktop\HOPE\2015개정경제수학-광주교육청.pdf`에 있다면, 파일을 아래 위치로 복사해 넣으세요.

```text
C:\Users\USER\HOPE\project\data\2015개정경제수학-광주교육청.pdf
```

## Install

Windows PowerShell 기준:

```powershell
cd C:\Users\USER\HOPE\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

PaddlePaddle은 환경에 따라 CPU/GPU 설치 명령이 달라질 수 있습니다. GPU를 쓰려면 PaddlePaddle 공식 설치 명령에 맞게 `paddlepaddle-gpu`를 설치하세요.

## Run

```powershell
cd C:\Users\USER\HOPE\project
.\.venv\Scripts\Activate.ps1
python .\src\main.py --pdf .\data\2015개정경제수학-광주교육청.pdf --output .\outputs
```

## React Web App

전체 PDF를 한 번에 이미지로 저장하지 않고, PDF를 업로드한 뒤 원하는 페이지 번호만 분석하려면 React + FastAPI 웹 앱을 사용하세요.

### 1. Backend 실행

```powershell
cd C:\Users\USER\HOPE\project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn backend.app:app --reload --host 127.0.0.1 --port 8000
```

### 2. Frontend 실행

새 PowerShell 터미널을 열고 실행하세요. PowerShell 실행 정책 때문에 `npm`이 막히면 `npm.cmd`를 사용하면 됩니다.

```powershell
cd C:\Users\USER\HOPE\project\frontend
npm.cmd install
npm.cmd run dev
```

브라우저에서 `http://127.0.0.1:5173`을 열면 됩니다. PDF를 업로드하면 전체 페이지 수가 표시되고, 분석할 페이지 번호를 입력한 뒤 `페이지 분석`을 누르면 해당 페이지만 렌더링해서 구조화 시각화와 JSON 결과를 보여줍니다.

DocLayout-YOLO로 학습한 모델 파일이 있으면 다음처럼 추가합니다.

```powershell
python .\src\main.py --pdf .\data\2015개정경제수학-광주교육청.pdf --output .\outputs --yolo-model .\models\doclayout_yolo.pt
```

## Outputs

```text
outputs/
├── pages/
│   ├── page_0001.png
│   └── ...
├── visualizations/
│   ├── page_0001_layout.png
│   └── ...
└── layout.json
```

## JSON Example

```json
{
  "pages": [
    {
      "page_id": 15,
      "blocks": [
        {
          "type": "section_title",
          "bbox": [120, 80, 950, 180],
          "text": "경제 성장률",
          "score": 0.98,
          "reading_order": 1,
          "block_id": "p15_b1"
        }
      ]
    }
  ]
}
```

## Textbook Box Strategy

경제수학 교과서의 박스형 요소는 다음 순서로 구분합니다.

1. DocLayout-YOLO 모델이 있으면 모델의 `example_box`, `problem_box`, `solution_box`, `formula` 예측을 우선합니다.
2. 모델이 없으면 OpenCV로 테두리/배경이 있는 박스 후보를 찾습니다.
3. 박스 내부 OCR 텍스트에 `예제`, `보기`, `따라하기`가 있으면 `example_box`로 분류합니다.
4. `문제`, `확인문제`, `연습문제`, `스스로`가 있으면 `problem_box`로 분류합니다.
5. `풀이`, `해설`, `정답`이 있으면 `solution_box`로 분류합니다.
6. 수학 기호와 숫자 비율이 높으면 `formula`로 분류합니다.

MVP 휴리스틱은 빠른 시작용입니다. 정확도를 높이려면 실제 교과서 페이지 일부를 라벨링해서 DocLayout-YOLO를 위 클래스 체계로 fine-tuning하는 것을 권장합니다.
