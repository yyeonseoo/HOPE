from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from PIL import Image


TYPE_PROMPTS = {
    "graph": ("an x y graph with coordinate axes and plotted data",),
    "table": ("a data table with rows columns and numbers",),
    "mathematical_diagram": ("a geometry diagram showing shapes solids lines or angles",),
    "illustration": ("a colorful educational drawing or illustration",),
    "photo": ("a real world photograph taken by a camera",),
}


@dataclass(frozen=True)
class RoutePrediction:
    route: str
    confidence: float
    scores: Mapping[str, float]
    elapsed_seconds: float


@runtime_checkable
class FigureRouteClassifier(Protocol):
    model_name: str
    model_version: str | None

    def classify(self, image_path: str | Path) -> RoutePrediction:
        """Classify a figure crop into one of the supported figure types."""


class OpenCLIPFigureTypeClassifier:
    """Zero-shot five-way figure classifier using OpenCLIP weights."""

    model_name = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"

    def __init__(
        self,
        model_id: str = model_name,
        *,
        device: str = "auto",
        revision: str | None = None,
        type_prompts: Mapping[str, tuple[str, ...]] = TYPE_PROMPTS,
    ) -> None:
        self.model_name = model_id
        self.model_version = revision
        self.device_request = device
        self.type_prompts = dict(type_prompts)
        self._model: Any = None
        self._preprocess: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    def classify(self, image_path: str | Path) -> RoutePrediction:
        torch = _import_torch()
        self._load(torch)
        started = time.perf_counter()
        with Image.open(image_path) as source:
            image_tensor = self._preprocess(source.convert("RGB")).unsqueeze(0).to(self._device)
        labels = list(self.type_prompts)
        prompts = [prompt for label in labels for prompt in self.type_prompts[label]]
        text_tensor = self._tokenizer(prompts).to(self._device)
        with torch.inference_mode():
            image_features = self._model.encode_image(image_tensor)
            text_features = self._model.encode_text(text_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            prompt_probabilities = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]
        scores = {}
        offset = 0
        for label in labels:
            count = len(self.type_prompts[label])
            scores[label] = float(prompt_probabilities[offset : offset + count].sum().item())
            offset += count
        route = max(scores, key=scores.get)
        return RoutePrediction(
            route=route,
            confidence=scores[route],
            scores=scores,
            elapsed_seconds=time.perf_counter() - started,
        )

    def _load(self, torch: Any) -> None:
        if self._model is not None:
            return
        try:
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP routing requires the optional figure dependencies: "
                "pip install -r src/analysis/figure/requirements.txt"
            ) from exc
        self._device = _resolve_device(torch, self.device_request)
        model_ref = f"hf-hub:{self.model_name}"
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(model_ref, device=self._device)
        self._model.eval()
        self._tokenizer = open_clip.get_tokenizer(model_ref)


# Backwards-compatible import name for callers created before five-way routing.
OpenCLIPGraphImageClassifier = OpenCLIPFigureTypeClassifier


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("OpenCLIP routing requires PyTorch.") from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is not available.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'.")
    return requested
