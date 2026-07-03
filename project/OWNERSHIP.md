# Parallel Work Ownership

세 담당자는 각자 feature 브랜치에서 작업하고 자기 소유 폴더만 수정합니다.

| 역할 | 구현 소유 폴더 | 테스트 소유 폴더 | 권장 브랜치 |
|---|---|---|---|
| Formula | `src/analysis/formula/` | `tests/formula/` | `feature/formula-analysis` |
| Table | `src/analysis/table/` | `tests/table/` | `feature/table-analysis` |
| Graph/Figure | `src/analysis/figure/` | `tests/figure/` | `feature/figure-analysis` |

## Shared Files

다음 파일은 통합 담당자 한 명만 수정합니다.

- `schemas/block_analysis.schema.json`
- `src/analysis/__init__.py`
- `backend/app.py`
- `frontend/src/App.jsx`
- `requirements.txt`
- `README.md`

새 라이브러리가 필요하면 담당자는 자기 PR 설명에 패키지명과 버전, 사용 이유를 적습니다. 통합 담당자가 `requirements.txt`에 반영합니다.

공통 스키마 변경이 필요하면 기능 PR에 섞지 않고 별도 이슈 또는 별도 PR로 제안합니다. 세 명이 합의하기 전에는 변경하지 않습니다.

## Merge Order

1. 각 담당자가 자기 feature 브랜치에 구현과 테스트를 push합니다.
2. 통합 담당자가 PR을 하나씩 검토하고 `main`에 병합합니다.
3. 각 병합 후 나머지 담당자는 `origin/main`을 자기 브랜치에 반영합니다.
4. 세 분석기가 병합된 뒤 통합 담당자가 각 결과를 API의 `semantic_analyses`에 연결합니다.

서로의 feature 브랜치를 직접 병합하지 않습니다. `backend/app.py`에 각자 연결 코드를 동시에 추가하지 않습니다.
