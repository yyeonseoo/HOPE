#!/bin/bash
cd /home/yj/HOPE/project
source .venv/bin/activate
python3 - <<'PYEOF' > /tmp/all_pages_figures.log 2>&1
import sys, time
from pathlib import Path
sys.path.insert(0, 'src')
from page_pipeline import process_single_page
from analysis.formula.formula_analyzer import analyze_formula_blocks
from analysis.table import analyze_table_blocks
from analysis.figure import analyze_figure_blocks, create_openai_figure_engine
from page_description import build_page_description

pdf_path = Path('data/3단원 좌표평면과 그래프 2.pdf')
import fitz
doc = fitz.open(pdf_path)
n_pages = doc.page_count
doc.close()

figure_engine = create_openai_figure_engine()

for page_num in range(1, n_pages + 1):
    started = time.perf_counter()
    try:
        result = process_single_page(
            pdf_path, page_num, Path('/tmp/pd_all_fig'),
            dpi=120, yolo_model_path='hf:juliozhao/DocLayout-YOLO-DocStructBench',
        )
        semantic_analyses = analyze_formula_blocks(result['page'], page_image_path=result['page_image_path'])
        semantic_analyses.extend(analyze_table_blocks(result['page'], str(result['page_image_path'])))
        figure_analyses = analyze_figure_blocks(
            result['page'],
            result['page_image_path'],
            figure_engine,
            ocr_lines=result.get('ocr_lines'),
            semantic_analyses=semantic_analyses,
            pdf_path=pdf_path,
            source_dpi=120,
        )
        semantic_analyses.extend(figure_analyses)
        desc = build_page_description(result['page'], semantic_analyses)
        elapsed = time.perf_counter() - started
        n_figures = len(figure_analyses)
        print(f'=== PAGE {page_num}/{n_pages} (status={desc["status"]}, figures={n_figures}, {elapsed:.1f}s) ===', flush=True)
        print(desc['text'] or '(내용 없음)', flush=True)
        print(flush=True)
    except Exception as exc:
        print(f'=== PAGE {page_num}/{n_pages} FAILED: {exc} ===', flush=True)
        print(flush=True)
print('ALL_DONE', flush=True)
PYEOF
echo SCRIPT_EXIT_$?
