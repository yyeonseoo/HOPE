from __future__ import annotations

import base64
import io
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from PIL import Image

# GPT captioning is configured via environment variables (same convention as
# HOPE_FIGURE_CAPTIONING/HOPE_FIGURE_DEVICE in backend/app.py), read fresh on
# each ChatGPTCaptioner construction so the model, request timeout, and retry
# budget can change without a code edit.
_FALLBACK_GPT_MODEL = "gpt-5"
_FALLBACK_GPT_TIMEOUT_SECONDS = 60.0
_FALLBACK_GPT_MAX_RETRIES = 3

# Reasoning-family models only accept the default temperature (1) and reject
# any other value the caller passes, unlike gpt-4o/gpt-4.1 which accept a
# custom temperature -- so this captioner omits temperature entirely for them.
_TEMPERATURE_UNSUPPORTED_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def _supports_custom_temperature(model_name: str) -> bool:
    return not model_name.startswith(_TEMPERATURE_UNSUPPORTED_MODEL_PREFIXES)


@dataclass(frozen=True)
class CaptionOutput:
    text: str
    confidence: float | None
    generation_time_seconds: float
    model_name: str
    model_version: str | None = None
    warnings: list[str] = field(default_factory=list)
    context_block_ids: tuple[str, ...] = ()


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


class ChatGPTCaptioner:
    """Adapter that generates figure captions via the OpenAI Chat Completions
    vision API (e.g. gpt-4o).

    Only implements ``caption_with_prompt`` -- the entrypoint the
    context-aware pipeline (generator.py) calls with an externally built
    prompt.
    """

    model_name = _FALLBACK_GPT_MODEL

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        timeout: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self.model_name = model or os.environ.get("HOPE_FIGURE_GPT_MODEL", _FALLBACK_GPT_MODEL)
        self.model_version: str | None = None
        self.api_key = api_key
        # Reasoning models spend part of max_completion_tokens on a hidden
        # reasoning trace before any visible text -- 400 tokens (enough for
        # gpt-4o) leaves nothing left over and silently yields an empty
        # caption, so they get a larger budget unless the caller overrides it.
        self.max_tokens = (
            max_tokens if max_tokens is not None
            else (400 if _supports_custom_temperature(self.model_name) else 2000)
        )
        self.temperature = temperature
        self.timeout = (
            timeout if timeout is not None
            else float(os.environ.get("HOPE_FIGURE_GPT_TIMEOUT_SECONDS", _FALLBACK_GPT_TIMEOUT_SECONDS))
        )
        self.max_retries = (
            max_retries if max_retries is not None
            else int(os.environ.get("HOPE_FIGURE_GPT_MAX_RETRIES", _FALLBACK_GPT_MAX_RETRIES))
        )
        self._client: Any = None

    def caption_with_prompt(
        self,
        image_path: str | Path,
        prompt: str,
        evidence: Sequence[Any] | None = None,
    ) -> CaptionOutput:
        client = self._load_client()
        started = time.perf_counter()
        request: dict[str, Any] = {
            "model": self.model_name,
            "max_completion_tokens": self.max_tokens,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                ],
            }],
        }
        if _supports_custom_temperature(self.model_name):
            request["temperature"] = self.temperature
        response = client.chat.completions.create(**request)
        elapsed = time.perf_counter() - started
        raw_text = (response.choices[0].message.content or "").strip()
        text, claim_warnings = _remove_unsupported_exact_claims(raw_text, evidence)
        text = _postprocess_caption_text(text)
        warnings = [] if text else ["ChatGPT returned an empty caption."]
        warnings += claim_warnings
        warnings += _find_suspicious_caption_content(text)
        return CaptionOutput(
            text=text,
            confidence=None,
            generation_time_seconds=elapsed,
            model_name=f"{self.model_name}-context-aware",
            model_version=self.model_version,
            warnings=warnings,
        )

    def _load_client(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "ChatGPT captioning requires the optional 'openai' package: pip install openai"
            ) from exc
        # timeout/max_retries make the SDK's own retry logic (exponential
        # backoff on connection errors, 429 rate limits, and 5xx responses)
        # apply here instead of failing the figure block on the first hiccup.
        client_kwargs: dict[str, Any] = {"timeout": self.timeout, "max_retries": self.max_retries}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        self._client = OpenAI(**client_kwargs)
        return self._client


def _image_data_url(image_path: str | Path) -> str:
    with Image.open(image_path) as source:
        buffer = io.BytesIO()
        source.convert("RGB").save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


_EXACT_COORDINATE_PATTERN = re.compile(
    r"\(\s*[+-]?(?:\d+(?:\.\d+)?|[A-Za-z])\s*,\s*[+-]?(?:\d+(?:\.\d+)?|[A-Za-z])\s*\)"
)
_EXACT_EQUATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z](?:\s*\(\s*[A-Za-z]\s*\))?\s*)="
    r"\s*[+-]?[A-Za-z0-9\\](?:[A-Za-z0-9\\{}\s+\-*/^().]{0,24})"
)
_EXACT_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")
_EVIDENCE_EQUATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z](?:\s*\(\s*[A-Za-z]\s*\))?)\s*=\s*[+-]?"
    r"(?:\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}\s*[A-Za-z]?|"
    r"[A-Za-z0-9]+\s*/\s*[A-Za-z0-9]+|"
    r"(?:\d+(?:\.\d+)?\s*)?[A-Za-z]{1,3}(?:\s*[+\-]\s*\d+(?:\.\d+)?)?|"
    r"\d+(?:\.\d+)?)"
)


def _compact_grounding_text(text: str) -> str:
    return re.sub(r"\s+", "", text).lower().replace("×", "*").replace("−", "-")


def _remove_unsupported_exact_claims(
    text: str,
    evidence: Sequence[Any] | None,
) -> tuple[str, list[str]]:
    """Drop sentences containing exact values that are absent from OCR/PDF evidence."""
    evidence_text = " ".join(item["text"] for item in _normalized_evidence(evidence))
    supported = _compact_grounding_text(evidence_text)
    supported_numbers = set(_EXACT_NUMBER_PATTERN.findall(evidence_text))
    sentences = [
        part.strip()
        for part in re.findall(r".+?(?:[.!?。！？]+|$)", text, flags=re.DOTALL)
        if part.strip()
    ]
    kept: list[str] = []
    removed_claims: list[str] = []
    for sentence in sentences:
        structural_claims = (
            _EXACT_COORDINATE_PATTERN.findall(sentence)
            + _EXACT_EQUATION_PATTERN.findall(sentence)
        )
        unsupported = [
            claim for claim in structural_claims
            if _compact_grounding_text(claim) not in supported
        ]
        unsupported.extend(
            claim for claim in _EXACT_NUMBER_PATTERN.findall(sentence)
            if (
                claim not in supported_numbers
                and not _is_panel_number(sentence, claim)
                and not _is_axis_origin_reference(sentence, claim)
            )
        )
        if unsupported:
            removed_claims.extend(unsupported)
            continue
        kept.append(sentence)
    warnings = []
    if removed_claims:
        unique = list(dict.fromkeys(removed_claims))
        warnings.append(
            "Removed caption sentence(s) containing exact claims absent from current image/OCR evidence: "
            + ", ".join(repr(item) for item in unique)
        )
    return " ".join(kept).strip(), warnings


def _is_panel_number(sentence: str, claim: str) -> bool:
    """Treat visible subfigure labels as identifiers, not measured values."""
    number = re.escape(claim.lstrip("+"))
    patterns = (
        rf"\(\s*{number}\s*\)",
        rf"^\s*{number}\s*[.)](?:\s+|$)",
        rf"(?<!\d){number}\s*(?:번|번째)(?!\d)",
        rf"(?:도형|그래프|그림|패널|보기)\s*\(?\s*{number}\s*\)?",
    )
    return any(re.search(pattern, sentence) for pattern in patterns)


def _is_axis_origin_reference(sentence: str, claim: str) -> bool:
    """Keep zero when it merely names the graph origin/start, not a data value."""
    normalized = claim.lstrip("+")
    return normalized in {"0", "-0", "0.0", "-0.0"} and bool(
        re.search(r"(?:원점|시작점|출발점|축의\s*교점)", sentence)
    )


def _normalized_evidence(evidence: Sequence[Any] | None) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(evidence or [], start=1):
        if isinstance(item, Mapping):
            text = str(item.get("text") or "").strip()
            candidate_id = str(item.get("id") or f"t{index}")
            bbox = item.get("bbox")
            relative_bbox = item.get("relative_bbox")
        else:
            text = str(item).strip()
            candidate_id = f"t{index}"
            bbox = None
            relative_bbox = None
        if not text:
            continue
        evidence_id = candidate_id if candidate_id not in used_ids else f"t{index}"
        used_ids.add(evidence_id)
        normalized.append({
            "id": evidence_id,
            "text": text,
            "bbox": bbox,
            "relative_bbox": relative_bbox,
        })
    return normalized


def _trusted_axis_labels(evidence: Sequence[Any] | None) -> tuple[str | None, str | None]:
    """Return only labels located in the conventional axis-label zones."""
    x_candidates: list[tuple[float, str]] = []
    y_candidates: list[tuple[float, str]] = []
    for item in _normalized_evidence(evidence):
        text = _safe_axis_label_text(item["text"])
        bbox = item.get("relative_bbox")
        if text is None or not _relative_bbox(bbox):
            continue
        center_x = (float(bbox[0]) + float(bbox[2])) / 2
        center_y = (float(bbox[1]) + float(bbox[3])) / 2
        if center_y >= 0.62 and center_x >= 0.42:
            x_candidates.append((abs(center_y - 0.82) + abs(center_x - 0.78) * 0.35, text))
        if center_x <= 0.38 and 0.08 <= center_y <= 0.56:
            y_candidates.append((abs(center_x - 0.14) + abs(center_y - 0.22) * 0.35, text))
    x_axis = min(x_candidates, default=(0.0, None))[1]
    y_axis = min(y_candidates, default=(0.0, None))[1]
    return x_axis, y_axis


def _relative_bbox(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
        and all(-0.05 <= float(item) <= 1.05 for item in value)
    )


def _safe_axis_label_text(value: str) -> str | None:
    text = " ".join(value.strip().split())
    compact = re.sub(r"\s+", "", text)
    if not compact or len(compact) > 16:
        return None
    if re.fullmatch(r"[+\-]?\d+(?:\.\d+)?|\(?\d+\)?|O", compact, flags=re.IGNORECASE):
        return None
    if _EVIDENCE_EQUATION_PATTERN.search(text) or _EXACT_COORDINATE_PATTERN.search(text):
        return None
    ui_words = {
        "파일", "편집", "보기", "도구", "도움말", "입력", "그래픽", "스프레드시트",
    }
    if compact in ui_words:
        return None
    if not re.search(r"[가-힣A-Za-z]", text):
        return None
    return text


def _substitute_stray_hanja(text: str) -> str:
    """Convert stray Han characters (e.g. '과程') back to their Korean reading.

    A model trained on plenty of Chinese text can occasionally emit the Han
    form of a syllable instead of the intended Hangul one -- the same
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


def _postprocess_caption_text(text: str) -> str:
    """Remove obvious generation artifacts without rewriting valid OCR or math."""
    text = _substitute_stray_hanja(text)
    text = re.sub(r"(?i)\bcoordinate\s*축", "좌표축", text)
    text = text.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    text = text.replace("�", "").replace("```", "").replace("`", "")
    text = text.replace("**", "")
    text = re.sub(r"(?m)^\s*(?:[-*•]\s+|\d+[.)]\s+|#{1,6}\s*)", "", text)
    text = re.sub(r"^\s*[^.!?。！？\n:]{1,20}:\s*", "", text)
    text = text.replace("#(", "(")
    parts = [part.strip() for part in re.findall(r".+?(?:[.!?。！？]+|$)", text, flags=re.DOTALL) if part.strip()]
    kept: list[str] = []
    seen: set[str] = set()
    duplicate_run = 0
    for index, sentence in enumerate(parts):
        if re.fullmatch(r"\s*(?:[ㄱ-ㅎ]|\(\s*\d+\s*\))\s*[.!?。！？]+\s*", sentence):
            continue
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
