"""Figure and chart understanding adapters and normalization."""

from .analyzer import analyze_figure_block, analyze_figure_blocks
from .captioners import CaptionOutput, ChatGPTCaptioner
from .hf_pipeline import HuggingFaceFigureCaptionEngine, create_openai_figure_engine
from .openclip_classifier import OpenCLIPFigureTypeClassifier, RoutePrediction

__all__ = [
    "CaptionOutput",
    "ChatGPTCaptioner",
    "HuggingFaceFigureCaptionEngine",
    "OpenCLIPFigureTypeClassifier",
    "RoutePrediction",
    "analyze_figure_block",
    "analyze_figure_blocks",
    "create_openai_figure_engine",
]
