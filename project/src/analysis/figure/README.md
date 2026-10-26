# Figure Analysis

이 폴더는 graph/figure 담당자 전용 작업 영역입니다.

- figure crop 입력
- 그래프·도형·삽화·사진 분류
- 그래프 제목·축·단위·범례·계열·데이터 추출
- confidence와 warning 생성
- `figureResult` 스키마 출력

수식·표 구현이나 공통 스키마는 이 폴더의 작업과 함께 수정하지 않습니다.

## Public Interface

```python
from src.analysis.figure import analyze_figure_blocks

results = analyze_figure_blocks(
    page=page_result,
    page_image_path="page_0001.png",
    engine=figure_model_adapter,
)
```

`analyze_figure_blocks`는 `type == "figure"`인 블록만 처리하고 공통 스키마를 만족하는 레코드 목록을 반환합니다. 모델 하나가 실패해도 다른 블록과 페이지 처리는 계속됩니다.

모델 adapter는 다음 인터페이스를 구현합니다.

```python
class FigureModelAdapter:
    model_name = "model-name"
    model_version = "model-version"

    def analyze(self, image_path):
        return {
            "confidence": 0.9,
            "figure_type": "line_chart",
            "title": "연도별 매출",
            "x_axis": {"label": "연도", "unit": "년"},
            "y_axis": {"label": "매출", "unit": "억원"},
            "series": [
                {"name": "매출", "points": [{"x": "2024", "y": 100}]}
            ],
            "warnings": [],
        }
```

모델이 연결되지 않았거나 유형을 판단하지 못하면 `unknown`과 `partial`을 반환합니다. 이미지 모양이나 파일명만으로 유형과 수치를 추측하는 fallback은 두지 않습니다.

## Caption Pipeline

기본 선택형 파이프라인은 다음 구성입니다.

1. OpenCLIP(로컬)으로 `graph`, `table`, `mathematical_diagram`, `illustration`, `photo` 분류
2. OpenAI Chat Completions API(`gpt-5`)에 유형별 한국어 Prompt와 이미지를 전달
3. 생성 결과를 공통 레코드의 `description`에 저장

분류·grounding용 패키지는 선택 의존성입니다. 사용하는 CPU/CUDA 환경에 맞는 PyTorch를 먼저 설치한 뒤 다음을 실행합니다.

```powershell
python -m pip install -r .\src\analysis\figure\requirements.txt
```

Python에서 직접 구성할 수 있습니다.

```python
from src.analysis.figure import analyze_figure_blocks, create_openai_figure_engine

engine = create_openai_figure_engine(device="auto")  # OPENAI_API_KEY 환경 변수 사용
semantic_analyses = analyze_figure_blocks(page, page_image_path, engine=engine)
```

OpenCLIP 분류기와 grounding scorer는 첫 추론 때 lazy load됩니다. GPU가 없으면 `device="cpu"`로 실행할 수 있습니다. 캡션 생성 자체는 OpenAI API 호출이므로 로컬 GPU/CPU와 무관합니다.

웹 화면에서는 왼쪽의 `Figure 설명 생성`을 체크한 뒤 `페이지 분석`을 누릅니다. 완료되면 Figure 탭으로 자동 이동하며 crop, 생성 설명, 모델명, 분류·생성 신뢰도와 생성 시간을 함께 표시합니다.

백엔드 연결은 기본적으로 꺼져 있습니다. 모든 웹 요청에서 항상 활성화하려면 백엔드 실행 전에 환경 변수를 설정합니다(`OPENAI_API_KEY`도 함께 설정해야 합니다).

```powershell
$env:HOPE_FIGURE_CAPTIONING="1"
$env:HOPE_FIGURE_DEVICE="auto"          # auto, cpu, cuda, cuda:0 (OpenCLIP/grounding 전용)
$env:OPENAI_API_KEY="sk-..."
$env:HOPE_FIGURE_GPT_MODEL="gpt-5"                # 생략 시 gpt-5
$env:HOPE_FIGURE_GPT_TIMEOUT_SECONDS="60"         # 생략 시 60초
$env:HOPE_FIGURE_GPT_MAX_RETRIES="3"              # 생략 시 3회(OpenAI SDK가 429/5xx/연결 오류에 지수 백오프로 재시도)
```

`analysis.confidence`에는 OpenCLIP의 선택 route 확률을, `description.confidence`는 GPT API가 확률을 주지 않으므로 `null`입니다. 실제 생성 시간은 `description.generation_time_seconds`에 초 단위로 저장됩니다.

## Files

- `analyzer.py`: 페이지 단위 공개 인터페이스와 공통 레코드 생성
- `crop.py`: bbox 검증, 페이지 경계 보정, crop 저장
- `engine.py`: 교체 가능한 모델 adapter 규약과 오류 격리
- `classifier.py`: 모델 분류명을 공통 figure 유형으로 정규화
- `normalize.py`: 축·계열·데이터와 confidence를 공통 스키마로 정규화

기본 분석기는 특정 모델 없이도 동작하며 이 경우 `unknown/partial`을 반환합니다.
