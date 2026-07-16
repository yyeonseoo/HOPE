from __future__ import annotations

import math
import re
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


QWEN_ACCESSIBILITY_PROMPT = (
    "한국 교과서 Figure를 스크린리더 사용자가 이해할 수 있도록 설명하세요. "
    "보이는 주요 요소와 읽을 수 있는 문자·수식·좌표·값·단위를 바탕으로, 요소 사이의 위치와 관계가 자연스럽게 이어지도록 설명하세요. "
    "문자, 기호, 화살표, 2배·3배·4배 같은 배수 표시와 범례가 명확히 보이면 중요한 시각 정보로 자연스럽게 포함하세요. "
    "이미지에서 높은 확신으로 확인되는 정보는 적극적으로 활용하되, 이미지에 없는 객체·값·관계·수식이나 추가적인 계산·일반화·해설은 만들지 마세요. "
    "같은 의미를 다른 표현으로 반복하지 말고, 하나의 사실이나 관계는 한 번만 설명하며 새로운 정보를 추가할 수 있을 때만 다음 문장을 이어가세요. "
    "OCR 결과가 깨졌거나 다른 언어 문자와 혼합된 문자열은 그대로 출력하지 말고 문맥에 맞는 자연스러운 한국어로 작성하세요. "
    "Figure와 Table 모두 항목 제목, 번호 목록, 체크리스트와 Markdown 없이 자연스럽고 읽기 쉬운 하나의 서술형 문단으로 작성하세요. "
    "모든 문장을 끝까지 완성하고 접근성 설명만 출력하세요. "
)


QWEN_TYPE_PROMPTS = {
    "graph": (
        "직선의 방향, 점의 위치, 증가·감소 경향, 축과 데이터의 관계처럼 시각적으로 확인되는 특징은 구체적으로 설명하세요. "
        "정확한 함수식은 이미지 안에 식 전체가 문자로 직접 적혀 있고 명확히 읽힐 때만 언급하세요. "
        "선의 모양이나 좌표·눈금으로 식을 계산하거나 추정하지 말고, 식이 직접 적혀 있지 않으면 직선의 방향이나 원점 통과 여부처럼 보이는 특징만 설명하세요."
    ),
    "table": (
        "표의 행·열 이름과 읽을 수 있는 값, 화살표와 대응 관계를 항목식으로 나열하지 말고 하나의 문단으로 연결하여 설명하세요."
    ),
    "mathematical_diagram": (
        "도형, 기호, 수식과 표시된 위치 관계를 설명하세요."
    ),
    "illustration": (
        "보이는 대상, 강조 표시, 화살표와 전후 관계를 설명하세요."
    ),
    "photo": (
        "사진에 보이는 대상과 상황을 간결하게 설명하세요."
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
        max_new_tokens: int = 192,
        repetition_penalty: float = 1.05,
        no_repeat_ngram_size: int = 4,
    ) -> None:
        self.model_name = model_id
        self.model_version = revision
        self.device_request = device
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self._model: Any = None
        self._processor: Any = None
        self._device: str | None = None
        self._dtype: Any = None

    def caption(self, image_path: str | Path, figure_type: str) -> CaptionOutput:
        torch = _import_torch()
        self._load(torch)
        prompt = QWEN_ACCESSIBILITY_PROMPT + QWEN_TYPE_PROMPTS.get(
            figure_type, QWEN_TYPE_PROMPTS["illustration"]
        )
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
            special_token_ids = _generation_special_token_ids(self._processor, self._model)
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                do_sample=False,
                **special_token_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - started
        text = self._processor.batch_decode(
            generated.sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        text = _postprocess_qwen_caption(text)
        warnings = [] if text else ["Qwen3-VL returned an empty caption."]
        warnings += _find_suspicious_caption_content(text)
        return CaptionOutput(
            text=text,
            confidence=_sequence_confidence(self._model, generated),
            generation_time_seconds=elapsed,
            model_name=self.model_name,
            model_version=self.model_version,
            warnings=warnings,
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


def _substitute_stray_hanja(text: str) -> str:
    """Convert stray Han characters (e.g. '\uacfc\u7a0b') back to their Korean reading.

    Qwen3-VL is trained on plenty of Chinese text and occasionally emits the
    Han form of a syllable instead of the intended Hangul one -- the same
    morpheme, wrong script. Unlike the other checks in this module this is
    safe to auto-correct rather than just warn about, since the character's
    Sino-Korean reading is a fixed, unambiguous lookup, not a guess.
    """
    try:
        import hanja
    except ImportError:
        return text
    return hanja.translate(text, "substitution")


def _collapse_decimal_point_spacing(text: str) -> str:
    """Remove the stray space generation sometimes inserts after a decimal
    point (e.g. '0. 6' -> '0.6'), a detokenization artifact rather than two
    separate numbers."""
    return re.sub(r"(?<=\d)\.\s+(?=\d)", ".", text)


def _postprocess_qwen_caption(text: str) -> str:
    """Remove obvious generation artifacts without rewriting valid OCR or math."""
    text = _substitute_stray_hanja(text)
    text = text.replace("\ufffd", "").replace("```", "").replace("`", "")
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*(?:[-*•]\s+|\d+[.)]\s+|#{1,6}\s*)", "", text)
    text = re.sub(r"^\s*[^.!?。！？\n:]{1,20}:\s*", "", text)
    text = text.replace("#(", "(")
    parts = [part.strip() for part in re.findall(r".+?(?:[.!?。！？]+|$)", text, flags=re.DOTALL) if part.strip()]
    kept: list[str] = []
    seen: set[str] = set()
    duplicate_run = 0
    for index, sentence in enumerate(parts):
        is_trailing_fragment = index == len(parts) - 1 and not re.search(r"[.!?。！？]$", sentence)
        if is_trailing_fragment and kept:
            break
        sentence = _collapse_adjacent_repeated_phrases(sentence)
        key = re.sub(r"\s+", " ", sentence).strip()
        if key in seen:
            duplicate_run += 1
            if duplicate_run >= 3:
                break
            continue
        duplicate_run = 0
        seen.add(key)
        kept.append(sentence)
    return _collapse_decimal_point_spacing(" ".join(kept).strip())


def _find_invalid_month_mentions(text: str) -> list[str]:
    """Flag Korean month mentions like '27월' that cannot exist on any calendar.

    A model that cannot actually read a fine-grained axis tick sometimes
    invents a precise-looking value instead of leaving it unread. An
    out-of-range month is an unambiguous, checkable sign of that.
    """
    warnings = []
    for match in re.finditer(r"(\d{1,3})월", text):
        if not 1 <= int(match.group(1)) <= 12:
            warnings.append(f"Caption references an impossible month value: {match.group(0)!r}.")
    return list(dict.fromkeys(warnings))


def _find_incomplete_numbered_list(text: str) -> list[str]:
    """Flag captions that promise a numbered range like '(1)부터 (4)까지' but
    then never actually describe every item in that range."""
    match = re.search(r"\(1\)\s*부터\s*\((\d+)\)\s*까지", text)
    if not match:
        return []
    declared_count = int(match.group(1))
    if not 1 <= declared_count <= 20:
        return []
    # Only count markers after the declaration itself, so "(1)부터 (4)까지" doesn't
    # count as having already described item 4.
    present = {int(marker) for marker in re.findall(r"\((\d+)\)", text[match.end():])}
    missing = sorted(number for number in range(1, declared_count + 1) if number not in present)
    if missing:
        return [f"Caption declares {declared_count} numbered items but never describes {missing}."]
    return []


def _find_suspicious_caption_content(text: str) -> list[str]:
    """Best-effort detectors for fabricated or self-contradictory generated content."""
    return _find_invalid_month_mentions(text) + _find_incomplete_numbered_list(text)


def _collapse_adjacent_repeated_phrases(text: str) -> str:
    """Collapse only verbatim, immediately adjacent phrase loops."""
    previous = None
    while text != previous:
        previous = text
        text = re.sub(r"(?<!\S)((?:\S+\s+){1,5}\S+)(?:\s+\1){1,}", r"\1", text)
        text = re.sub(r"(?<!\S)(\S+)(?:\s+\1){2,}", r"\1", text)
    return text


def _generation_special_token_ids(processor: Any, model: Any) -> dict[str, Any]:
    tokenizer = getattr(processor, "tokenizer", processor)
    generation_config = getattr(model, "generation_config", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if eos_token_id is None and generation_config is not None:
        eos_token_id = getattr(generation_config, "eos_token_id", None)
    if pad_token_id is None and generation_config is not None:
        pad_token_id = getattr(generation_config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id[0] if isinstance(eos_token_id, (list, tuple)) else eos_token_id
    return {
        key: value
        for key, value in {"eos_token_id": eos_token_id, "pad_token_id": pad_token_id}.items()
        if value is not None
    }


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
