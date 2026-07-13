from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from PIL import Image


@dataclass(frozen=True)
class CaptionOutput:
    text: str
    confidence: float | None
    generation_time_seconds: float
    model_name: str
    model_version: str | None = None
    warnings: list[str] = field(default_factory=list)


@runtime_checkable
class GraphCaptioner(Protocol):
    model_name: str
    model_version: str | None

    def caption(self, image_path: str | Path) -> CaptionOutput:
        """Generate a factual description for a graph crop."""


@runtime_checkable
class ImageCaptioner(Protocol):
    model_name: str
    model_version: str | None

    def caption(self, image_path: str | Path) -> CaptionOutput:
        """Generate a description for a photo, illustration, or diagram crop."""


class ChartGemmaCaptioner:
    """Lazy Hugging Face adapter for the MIT-licensed ChartGemma checkpoint."""

    model_name = "ahmed-masry/chartgemma"

    def __init__(
        self,
        model_id: str = model_name,
        *,
        device: str = "auto",
        revision: str | None = None,
        max_new_tokens: int = 256,
        num_beams: int | None = None,
    ) -> None:
        self.model_name = model_id
        self.model_version = revision
        self.device_request = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None

    def caption(self, image_path: str | Path) -> CaptionOutput:
        torch = _import_torch()
        self._load(torch)
        started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            prompt = (
                "chart summary: Describe this chart factually. State the chart type, title, axes, "
                "units, series, and visible trends. Do not invent unreadable values."
            )
            inputs = self._processor(text=prompt, images=image, return_tensors="pt")
        prompt_length = inputs["input_ids"].shape[1]
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        beams = self.num_beams if self.num_beams is not None else (1 if self._device == "cpu" else 4)
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                num_beams=beams,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - started
        text = self._processor.batch_decode(
            generated.sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return CaptionOutput(
            text=text,
            confidence=_sequence_confidence(self._model, generated),
            generation_time_seconds=elapsed,
            model_name=self.model_name,
            model_version=self.model_version,
            warnings=[] if text else ["ChartGemma returned an empty caption."],
        )

    def _load(self, torch: Any) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "ChartGemma requires the optional figure dependencies: "
                "pip install -r src/analysis/figure/requirements.txt"
            ) from exc
        self._device = _resolve_device(torch, self.device_request)
        dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        kwargs = {"torch_dtype": dtype}
        if self.model_version:
            kwargs["revision"] = self.model_version
        self._model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name, **kwargs).to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            revision=self.model_version,
        )


class Florence2ImageCaptioner:
    """Lazy native Transformers Florence-2 adapter with automatic CPU mode."""

    model_name = "florence-community/Florence-2-base"

    def __init__(
        self,
        model_id: str = model_name,
        *,
        device: str = "auto",
        revision: str | None = None,
        max_new_tokens: int = 256,
        task_prompt: str = "<MORE_DETAILED_CAPTION>",
        num_beams: int | None = None,
    ) -> None:
        self.model_name = model_id
        self.model_version = revision
        self.device_request = device
        self.max_new_tokens = max_new_tokens
        self.task_prompt = task_prompt
        self.num_beams = num_beams
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: Any = None

    def caption(self, image_path: str | Path) -> CaptionOutput:
        torch = _import_torch()
        self._load(torch)
        started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            inputs = self._processor(text=self.task_prompt, images=image, return_tensors="pt")
            image_size = image.size
        inputs = {
            name: value.to(self._device, self._dtype) if name == "pixel_values" else value.to(self._device)
            for name, value in inputs.items()
        }
        beams = self.num_beams if self.num_beams is not None else (1 if self._device == "cpu" else 3)
        with torch.inference_mode():
            generated = self._model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=self.max_new_tokens,
                num_beams=beams,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - started
        generated_text = self._processor.batch_decode(generated.sequences, skip_special_tokens=False)[0]
        parsed = self._processor.post_process_generation(
            generated_text,
            task=self.task_prompt,
            image_size=image_size,
        )
        text = _extract_florence_text(parsed, self.task_prompt)
        return CaptionOutput(
            text=text,
            confidence=_sequence_confidence(self._model, generated),
            generation_time_seconds=elapsed,
            model_name=self.model_name,
            model_version=self.model_version,
            warnings=[] if text else ["Florence-2 returned an empty caption."],
        )

    def _load(self, torch: Any) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, Florence2ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Florence-2 requires the optional figure dependencies: "
                "pip install -r src/analysis/figure/requirements.txt"
            ) from exc
        self._device = _resolve_device(torch, self.device_request)
        self._dtype = torch.float16 if self._device.startswith("cuda") else torch.float32
        kwargs: dict[str, Any] = {"dtype": self._dtype}
        if self.model_version:
            kwargs["revision"] = self.model_version
        self._model = Florence2ForConditionalGeneration.from_pretrained(self.model_name, **kwargs).to(self._device)
        self._model.eval()
        self._processor = AutoProcessor.from_pretrained(
            self.model_name,
            revision=self.model_version,
        )


QWEN_TYPE_PROMPTS = {
    "graph": (
        "이 그래프가 전달하는 수학적 의미를 한국어로 설명하세요. 축 이름과 단위, 점·선·곡선의 관계, "
        "증가·감소 경향을 확인하세요. 단위는 이미지에 표시된 경우에만 쓰고 직선과 곡선을 혼동하지 마세요. "
        "읽을 수 없는 값은 추측하지 말고 단순한 외형 묘사보다 학습 내용을 우선하세요. "
        "제목, 목록, 단계 구분 없이 2~4문장으로만 답하세요."
    ),
    "table": (
        "이 표의 행과 열, 대응하는 값, 배수나 비례 관계를 한국어로 설명하세요. 보이는 한국어와 수식을 정확히 읽고, "
        "행·열 이름은 이미지에 적힌 표현만 사용하세요. 읽을 수 없는 값이나 이름은 추측하지 말고 표가 전달하는 핵심 관계를 우선하세요. "
        "제목이나 목록 없이 2~4문장으로만 답하세요."
    ),
    "mathematical_diagram": (
        "이 수학 도식의 구성 요소와 위치 관계, 표시된 기호와 수학적 의미를 한국어로 설명하세요. "
        "단순한 모양 묘사에 그치지 말고 도식이 설명하는 개념을 말하되, 보이지 않는 내용은 만들지 마세요. "
        "제목이나 목록 없이 2~4문장으로만 답하세요."
    ),
    "illustration": (
        "이 교과서 삽화에서 관찰되는 대상과 전후 변화를 한국어로 설명하세요. 과학 도식이면 용기, 추, 화살표, "
        "입자의 수·간격·분포처럼 직접 보이는 변화를 우선하세요. 물질의 종류나 원인은 근거가 없으면 추측하지 말고, "
        "읽을 수 없는 글자도 추측하지 마세요. "
        "제목이나 목록 없이 2~4문장으로만 답하세요."
    ),
    "photo": (
        "이 교과서 사진에 실제로 보이는 대상과 상황을 한국어로 간결하게 설명하세요. "
        "사진의 교육적 맥락을 추측하지 말고 확인 가능한 사실만 제목이나 목록 없이 2~4문장으로 작성하세요."
    ),
}


class Qwen3VLCaptioner:
    """Instruction captioner for Korean textbook figures."""

    model_name = "Qwen/Qwen3-VL-2B-Instruct"

    def __init__(
        self,
        model_id: str = model_name,
        *,
        device: str = "auto",
        revision: str | None = None,
        max_new_tokens: int = 256,
    ) -> None:
        self.model_name = model_id
        self.model_version = revision
        self.device_request = device
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: Any = None

    def caption(self, image_path: str | Path, figure_type: str) -> CaptionOutput:
        torch = _import_torch()
        self._load(torch)
        prompt = QWEN_TYPE_PROMPTS.get(figure_type, QWEN_TYPE_PROMPTS["illustration"])
        started = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }]
            inputs = self._processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        prompt_length = inputs["input_ids"].shape[1]
        inputs = {
            name: value.to(self._device, self._dtype) if value.is_floating_point() else value.to(self._device)
            for name, value in inputs.items()
        }
        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - started
        text = self._processor.batch_decode(
            generated.sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        return CaptionOutput(
            text=text,
            confidence=_sequence_confidence(self._model, generated),
            generation_time_seconds=elapsed,
            model_name=self.model_name,
            model_version=self.model_version,
            warnings=[] if text else ["Qwen3-VL returned an empty caption."],
        )

    def _load(self, torch: Any) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL requires the optional figure dependencies: "
                "pip install -r src/analysis/figure/requirements.txt"
            ) from exc
        self._device = _resolve_device(torch, self.device_request)
        kwargs: dict[str, Any] = {"dtype": "auto"}
        if self.model_version:
            kwargs["revision"] = self.model_version
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_name, **kwargs).to(self._device)
        self._model.eval()
        self._dtype = next(self._model.parameters()).dtype
        self._processor = AutoProcessor.from_pretrained(self.model_name, revision=self.model_version)


def _import_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Figure captioning requires PyTorch.") from exc
    return torch


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {requested!r} was requested but CUDA is not available.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError("device must be 'auto', 'cpu', or a CUDA device such as 'cuda:0'.")
    return requested


def _sequence_confidence(model: Any, generated: Any) -> float | None:
    """Return geometric mean token probability; this is not a calibrated score."""
    scores = getattr(generated, "scores", None)
    sequences = getattr(generated, "sequences", None)
    if not scores or sequences is None:
        return None
    try:
        transition = model.compute_transition_scores(
            sequences,
            scores,
            beam_indices=getattr(generated, "beam_indices", None),
            normalize_logits=True,
        )
        values = transition[0]
        values = values[values != 0]
        if values.numel() == 0:
            return None
        return min(1.0, max(0.0, math.exp(float(values.float().mean().item()))))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


def _extract_florence_text(parsed: Any, task_prompt: str) -> str:
    if isinstance(parsed, str):
        return parsed.strip()
    if isinstance(parsed, dict):
        value = parsed.get(task_prompt)
        if isinstance(value, str):
            return value.strip()
        if value is not None:
            return str(value).strip()
    return ""
