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

## Files

- `analyzer.py`: 페이지 단위 공개 인터페이스와 공통 레코드 생성
- `crop.py`: bbox 검증, 페이지 경계 보정, crop 저장
- `engine.py`: 교체 가능한 모델 adapter 규약과 오류 격리
- `classifier.py`: 모델 분류명을 공통 figure 유형으로 정규화
- `normalize.py`: 축·계열·데이터와 confidence를 공통 스키마로 정규화

기본 분석기는 특정 모델 없이도 동작하며 이 경우 `unknown/partial`을 반환합니다. 실제 모델은 아래 adapter처럼 선택적으로 연결하고, 동일한 정답 crop 평가셋에서 후보별 성능을 비교합니다.

## PP-Chart2Table Baseline

`PPChart2TableEngine`은 PaddleOCR의 `PP-Chart2Table` 모델을 사용해 그래프 crop을 표 형태의 데이터로 변환합니다.

```python
from src.analysis.figure import PPChart2TableEngine, analyze_figure_blocks

results = analyze_figure_blocks(page, page_image_path, engine=PPChart2TableEngine())
```

이 모델은 그래프 데이터 추출 전용입니다. 사진과 일반 삽화 설명에는 사용하지 않으며, 시각적 그래프 유형을 신뢰성 있게 반환하지 않으므로 `figure_type`은 `other`로 보존합니다. 최초 실행 시 약 1.4GB 모델을 내려받고 CPU에서는 오래 걸릴 수 있습니다.

로컬 모델 경로:

```text
project/.cache/models/PP-Chart2Table/
```

추가 실행 의존성은 `tiktoken`입니다. 공용 `requirements.txt` 반영은 통합 담당자가 결정합니다. 2026-07-05 CPU 환경에서 216x200 픽셀 그래프 한 장에 1,255초가 걸렸고 축과 데이터 계열을 추출하지 못했습니다. 따라서 이 샘플 기준으로는 채택하지 않으며 비교 실험 기록으로만 남깁니다.
