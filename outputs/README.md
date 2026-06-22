# Outputs

이 폴더는 Model A 실행 결과물을 관리하기 위한 폴더이다.

## pages

`outputs/pages` 폴더에는 PDF 전체 페이지를 PNG 이미지로 변환한 결과가 저장된다.

- 원본 PDF 전체 페이지 수: 198페이지
- 생성 결과: page_001.png ~ page_198.png
- 생성 방법: `notebooks/modelA_02_pdf_split_preprocessing.ipynb` 실행

전체 페이지 이미지는 용량 문제로 GitHub에 모두 업로드하지 않는다.
필요한 경우 각 팀원이 노트북을 실행하여 동일한 결과를 생성할 수 있다.

## preprocessed_pages

`outputs/preprocessed_pages` 폴더에는 OCR 성능 향상을 위해 그레이스케일로 변환한 전처리 이미지가 저장된다.

- 생성 결과: page_001.png ~ page_198.png
- 생성 방법: `notebooks/modelA_02_pdf_split_preprocessing.ipynb` 실행

## pages_sample

`outputs/pages_sample` 폴더에는 결과 확인용 샘플 이미지를 업로드한다.
