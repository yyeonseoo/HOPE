"""Figure and chart understanding adapters and normalization."""

from .analyzer import analyze_figure_block, analyze_figure_blocks
from .captioners import CaptionOutput, ChartGemmaCaptioner, ChatGPTCaptioner, Florence2ImageCaptioner, GraphCaptioner, ImageCaptioner
from .description import build_context_free_description
from .hf_pipeline import HuggingFaceFigureCaptionEngine, create_openai_figure_engine
from .openclip_classifier import OpenCLIPFigureTypeClassifier, OpenCLIPGraphImageClassifier, RoutePrediction
from .pdf_vector import analyze_pdf_vector_figure, extract_vector_evidence
from .pp_chart2table import PPChart2TableEngine
from .router import classify_figure_route

__all__ = [
    "PPChart2TableEngine",
    "CaptionOutput",
    "ChartGemmaCaptioner",
    "ChatGPTCaptioner",
    "Florence2ImageCaptioner",
    "GraphCaptioner",
    "ImageCaptioner",
    "HuggingFaceFigureCaptionEngine",
    "OpenCLIPGraphImageClassifier",
    "OpenCLIPFigureTypeClassifier",
    "RoutePrediction",
    "analyze_figure_block",
    "analyze_figure_blocks",
    "build_context_free_description",
    "analyze_pdf_vector_figure",
    "extract_vector_evidence",
    "classify_figure_route",
    "create_openai_figure_engine",
]
