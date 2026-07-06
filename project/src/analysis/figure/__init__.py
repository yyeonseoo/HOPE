"""Figure and chart understanding adapters and normalization."""

from .analyzer import analyze_figure_block, analyze_figure_blocks
from .description import build_context_free_description
from .pdf_vector import analyze_pdf_vector_figure, extract_vector_evidence
from .pp_chart2table import PPChart2TableEngine

__all__ = [
    "PPChart2TableEngine",
    "analyze_figure_block",
    "analyze_figure_blocks",
    "build_context_free_description",
    "analyze_pdf_vector_figure",
    "extract_vector_evidence",
]
