from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

from PIL import Image

from .graph_visual import GraphVisualCue, analyze_graph_visual


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
        "축 이름이나 의미는 OCR 근거에 명확한 이름이 있을 때만 언급하고, 없으면 좌측·우측 축의 의미를 만들지 말고 점이나 선의 분포와 경향을 설명하세요. "
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

QWEN_COMPLEX_GRAPH_PROMPT = (
    "이 단계에서는 교과서 해설이나 함수의 성질을 추론하지 말고 그래프에서 보이는 시각 구조만 설명하세요. "
    "서로 분리된 좌표계·패널·곡선 가지·계열의 개수를 먼저 확인하고, 여러 개면 각각 구분하세요. "
    "그래프와 번호가 붙은 도형이 함께 있으면 도형도 왼쪽부터 하나씩 따로 관찰하세요. 각 도형은 상단·중앙·하단의 "
    "너비와 형태, 서로 분리되거나 겹쳐 보이는 구성 부분의 실제 개수를 확인하여 설명하고, 위아래 방향을 바꾸어 말하지 마세요. "
    "도형의 정확한 명칭이 확실하지 않으면 이름을 추정하지 말고 보이는 형태로 설명하며, 아래 그래프와 도형의 특징을 섞지 마세요. "
    "연결되지 않은 점, 이어진 직선과 꺾은선, 매끄러운 곡선을 구별하고 각 선의 상승·하강·수평 구간과 꺾이는 순서를 설명하세요. "
    "정확한 식·좌표·값은 이미지 내부 OCR 근거에 있을 때만 사용하세요. 자연스러운 한국어 2~4문장만 출력하세요. "
)

QWEN_GRAPH_DESCRIPTION_PROMPT = (
    "한국 교과서의 그래프 이미지를 보고 스크린리더용 설명을 작성하세요. 이미지에서 보이는 그래프를 가장 중요한 근거로 삼으세요. "
    "주변 문맥에 그래프의 대상이나 정확한 수식이 명시되어 있으면 도입에 간단히 반영하고, 라벨의 의미는 해당 계열을 설명할 때 정확히 사용하세요. "
    "축 이름이 이미지나 주변 문맥에 명시되어 있으면 그대로 사용하고, 없으면 수평축(x축)과 수직축(y축)으로만 표현하세요. "
    "그래프 프로그램의 도구 모음, 입력 표, 창 테두리는 그래프 계열로 세지 말고 실제 좌표 영역에 표시된 점과 선만 관찰하세요. "
    "점 사이에 실제 선이 보이지 않으면 점들이 연결되어 있다고 설명하지 마세요. "
    "화살표나 시간 순서가 표시되지 않았다면 점들이 이동한다고 표현하지 마세요. "
    "좌표계가 여러 개이거나 한 좌표계에 서로 다른 선·곡선·점 계열이 여러 개 있으면 각각을 라벨·색·형태로 구분하여 따로 설명하세요. "
    "각 계열이 왼쪽에서 오른쪽으로 상승·하강·수평 유지·방향 전환하는 순서를 보이는 그대로 설명하고 하나의 경향으로 합치지 마세요. "
    "서로 연결되지 않은 곡선 가지는 하나의 연속 구간처럼 합치지 말고, 각 가지의 위치와 형태를 나누어 설명하세요. "
    "이미지나 제공된 정확 정보에 좌표값이 명시되어 있으면 필요한 값만 언급하되, 함수의 정의역·부호·극한·수렴·무한대·기울기·절편은 해설하지 마세요. "
    "주변 문맥은 그래프가 나타내는 대상과 라벨의 의미를 파악하는 데만 참고하고, 이미지에 보이는 계열의 형태를 바꾸지 마세요. "
    "같은 내용을 반복하지 말고 자연스러운 한국어 2~4문장으로 접근성 설명만 출력하세요. "
)


class Qwen3VLCaptioner:
    """Instruction captioner for Korean textbook figures."""

    model_name = "Qwen/Qwen3-VL-2B-Instruct"
    handles_context_inline = True

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

    def caption(
        self,
        image_path: str | Path,
        figure_type: str,
        evidence: Sequence[Any] | None = None,
        context: Sequence[Mapping[str, Any]] | None = None,
    ) -> CaptionOutput:
        detected_at = time.perf_counter()
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        visual_cue = analyze_graph_visual(image) if figure_type == "graph" else None
        context_candidates = _caption_context_candidates(context, figure_type)
        visible_panel_count = _visible_panel_count(evidence) if figure_type == "graph" else 0
        complex_graph = bool(
            visual_cue is not None and _graph_needs_structured_review(visual_cue, evidence)
        )
        if visual_cue is not None and visual_cue.state == "empty":
            text = _grounded_coordinate_graph_caption(visual_cue, evidence)
            return CaptionOutput(
                text=text,
                confidence=visual_cue.confidence,
                generation_time_seconds=time.perf_counter() - detected_at,
                model_name="opencv-ocr-grounded-graph-captioner",
                model_version="1.0",
            )
        if visual_cue is not None and figure_type == "graph":
            if visual_cue.coordinate_plane:
                if not complex_graph:
                    text = _grounded_coordinate_graph_caption(visual_cue, evidence)
                    return CaptionOutput(
                        text=text,
                        confidence=visual_cue.confidence,
                        generation_time_seconds=time.perf_counter() - detected_at,
                        model_name="opencv-ocr-grounded-graph-captioner",
                        model_version="1.0",
                    )

        torch = _import_torch()
        self._load(torch)
        if figure_type == "graph":
            prompt = (
                QWEN_GRAPH_DESCRIPTION_PROMPT
                + _graph_context_prompt(context_candidates)
                + _verified_graph_facts_prompt(evidence, context_candidates)
                + _graph_visual_prompt(visual_cue)
                + _graph_component_prompt(visual_cue)
                + _panel_completeness_prompt(visible_panel_count)
            )
        else:
            prompt = QWEN_ACCESSIBILITY_PROMPT + QWEN_TYPE_PROMPTS.get(
                figure_type, QWEN_TYPE_PROMPTS["illustration"]
            ) + (QWEN_COMPLEX_GRAPH_PROMPT if figure_type == "graph" else "") + _graph_visual_prompt(
                visual_cue
            ) + _grounding_evidence_prompt(evidence)
        started = time.perf_counter()
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
                max_new_tokens=min(
                    self.max_new_tokens,
                    96 if figure_type in {"photo", "illustration"}
                    else 192 if visible_panel_count >= 3 else 128,
                ),
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                do_sample=False,
                **special_token_ids,
                return_dict_in_generate=True,
                output_scores=True,
            )
        elapsed = time.perf_counter() - started
        raw_text = self._processor.batch_decode(
            generated.sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        selected_context_ids: tuple[str, ...] = ()
        grounding_evidence: list[Any] = list(evidence or [])
        grounding_warnings: list[str] = []
        if context_candidates:
            # Context-grounded generation is intentionally permissive here:
            # Qwen sees the crop and ranked nearby textbook blocks together and
            # returns the final prose directly. Strict intermediate schemas and
            # image-only exact-claim filters previously discarded useful output.
            text = _postprocess_qwen_caption(raw_text)
            text = _restore_trusted_context_terms(text, context_candidates)
            text = _restore_truncated_context_equation(text, context_candidates)
            text, claim_warnings = _filter_unsupported_context_graph_claims(
                text, evidence, context_candidates
            )
            text = _ensure_context_graph_visual_anchor(
                text, visual_cue, evidence, context_candidates
            )
            grounding_warnings.extend(claim_warnings)
            selected_context_ids = tuple(item["block_id"] for item in context_candidates)
        else:
            text = raw_text
            text, removed_claim_warnings = _remove_unsupported_exact_claims(text, grounding_evidence)
            grounding_warnings.extend(removed_claim_warnings)
        text, used_visual_fallback = _fallback_after_exact_claim_filter(text, visual_cue, evidence)
        if used_visual_fallback:
            grounding_warnings.append(
                "All image-only sentences with unsupported exact claims were removed; "
                "a qualitative visual fallback was used."
            )
        if not text and context_candidates:
            text = _minimal_figure_fallback(figure_type)
            grounding_warnings.append(
                "Qwen returned no usable prose; a minimal figure-type fallback was used."
            )
        if figure_type != "graph" or not complex_graph:
            text = _apply_graph_trend_grounding(text, visual_cue)
        if figure_type == "graph":
            text = _anchor_trusted_axis_labels(text, evidence)
            text = _anchor_visible_direction_sequence(text, visual_cue)
        text = _postprocess_qwen_caption(text)
        warnings = [] if text else ["Qwen3-VL returned an empty caption."]
        warnings += grounding_warnings
        warnings += _find_suspicious_caption_content(text)
        return CaptionOutput(
            text=text,
            confidence=_sequence_confidence(self._model, generated),
            generation_time_seconds=elapsed,
            model_name=(
                f"{self.model_name}-context-grounded"
                if context_candidates
                else f"{self.model_name}-visual-graph" if figure_type == "graph" else self.model_name
            ),
            model_version=self.model_version,
            warnings=warnings,
            context_block_ids=selected_context_ids,
        )

    def fuse_with_context(
        self,
        base: CaptionOutput,
        figure_type: str,
        context: Sequence[Mapping[str, Any]] | None,
    ) -> CaptionOutput:
        candidates = [
            {
                "block_id": str(item.get("block_id") or ""),
                "type": str(item.get("type") or ""),
                "position": str(item.get("relative_position") or ""),
                "text": str(item.get("text") or "")[:600],
            }
            for item in context or []
            if str(item.get("block_id") or "").strip() and str(item.get("text") or "").strip()
        ]
        if not candidates:
            return base
        torch = _import_torch()
        self._load(torch)
        prompt = (
            "아래의 이미지 전용 설명과 주변 교과서 문맥을 의미적으로 연결하세요. 단어가 같지 않아도 앞뒤 문장이 현재 Figure를 "
            "가리키거나 그 의미를 설명하면 사용할 수 있습니다. 이미지 설명을 가장 중요한 근거로 삼되, 관련 문맥을 이용해 Figure의 "
            "교과서상 의미를 자연스럽게 보완하거나 고쳐 쓸 수 있습니다. 문맥과 시각 설명이 충돌하면 해당 문맥을 선택하지 마세요. "
            "시각 설명에 없는 객체·위치·방향은 새로 만들지 말고, 수식·값·개념은 선택한 문맥에 실제로 있을 때만 사용하세요. "
            "관련 문맥이 없으면 relevant_context_ids를 빈 배열로 하고 caption은 이미지 전용 설명을 그대로 복사하세요. "
            'JSON 하나만 출력하세요. 형식: {"relevant_context_ids":[],"caption":"완성된 접근성 설명"} '
            f"Figure 유형: {figure_type} 이미지 전용 설명: {base.text} 주변 문맥: "
            + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
        )
        started = time.perf_counter()
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
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
                max_new_tokens=96,
                repetition_penalty=1.05,
                no_repeat_ngram_size=3,
                do_sample=False,
                **_generation_special_token_ids(self._processor, self._model),
                return_dict_in_generate=True,
                output_scores=True,
            )
        raw = self._processor.batch_decode(
            generated.sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        parsed = _parse_context_fusion_response(raw, candidates)
        elapsed = time.perf_counter() - started
        if parsed is None or not parsed["relevant_context_ids"]:
            return replace_caption_time(base, elapsed)
        selected = [item for item in candidates if item["block_id"] in parsed["relevant_context_ids"]]
        fused, grounding_warnings = _remove_unsupported_exact_claims(
            parsed["caption"], [{"text": base.text}, *selected]
        )
        fused = _postprocess_qwen_caption(fused)
        if not fused or not _preserves_visual_anchor(base.text, fused):
            return replace_caption_time(
                base,
                elapsed,
                warning="Context fusion was rejected because it did not preserve the visual description.",
            )
        confidence = _sequence_confidence(self._model, generated)
        if base.confidence is not None and confidence is not None:
            confidence = min(base.confidence, confidence)
        return CaptionOutput(
            text=fused,
            confidence=confidence if confidence is not None else base.confidence,
            generation_time_seconds=base.generation_time_seconds + elapsed,
            model_name=f"{base.model_name}+context-fusion",
            model_version=base.model_version,
            warnings=[*base.warnings, *grounding_warnings],
            context_block_ids=tuple(parsed["relevant_context_ids"]),
        )

    def _caption_structured_graph(
        self,
        image: Image.Image,
        visual_cue: GraphVisualCue,
        evidence: Sequence[Any] | None,
    ) -> CaptionOutput:
        torch = _import_torch()
        self._load(torch)
        started = time.perf_counter()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": _structured_graph_prompt(evidence)},
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
                # A multi-panel response can exceed 128 tokens even though it
                # contains JSON only.  Truncating it makes parsing fail and
                # silently falls back to a single OpenCV summary.
                max_new_tokens=max(256, self.max_new_tokens),
                do_sample=False,
                **_generation_special_token_ids(self._processor, self._model),
                return_dict_in_generate=True,
                output_scores=True,
            )
        raw_text = self._processor.batch_decode(
            generated.sequences[:, prompt_length:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        structure = _parse_structured_graph_response(raw_text, evidence)
        structure = _reconcile_structured_graph(visual_cue, structure)
        warnings = []
        if structure is None:
            warnings.append("Qwen structured graph response was invalid; OpenCV fallback was used.")
        text = _grounded_coordinate_graph_caption(visual_cue, evidence, structure)
        return CaptionOutput(
            text=text,
            confidence=_sequence_confidence(self._model, generated),
            generation_time_seconds=time.perf_counter() - started,
            model_name=f"{self.model_name}-structured-graph",
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


def _graph_visual_prompt(cue: GraphVisualCue | None) -> str:
    if cue is None or cue.state != "plotted" or cue.trend is None:
        return ""
    if cue.mark_type == "multiple" or (cue.series_count or 0) >= 2:
        return ""
    if len(cue.direction_sequence) >= 2:
        names = {
            "increasing": "상승",
            "decreasing": "하강",
            "horizontal": "수평 유지",
        }
        sequence = " → ".join(names[item] for item in cue.direction_sequence)
        return (
            f" 저비용 시각 검출에서 왼쪽부터 {sequence} 순서의 변화가 확인되었습니다. "
            "이 순서는 그래프 모양의 근거로 사용하되 원인이나 수학적 의미를 덧붙이지 마세요."
        )
    return (
        f" 시각 검출 결과 다음 형태가 확인되었습니다: {_graph_trend_lead(cue)} "
        "이 경향을 설명의 앞부분에 분명히 쓰고, 선을 시작점이나 끝점이 있는 것처럼 표현하지 마세요."
    )


def _context_candidates(
    context: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    return [
        {
            "block_id": str(item.get("block_id") or ""),
            "type": str(item.get("type") or ""),
            "position": str(item.get("relative_position") or ""),
            "score": float(item.get("score") or 0.0),
            "text": str(item.get("text") or "")[:500],
        }
        for item in context or []
        if str(item.get("block_id") or "").strip() and str(item.get("text") or "").strip()
    ][:3]


def _caption_context_candidates(
    context: Sequence[Mapping[str, Any]] | None,
    figure_type: str,
) -> list[dict[str, Any]]:
    # Nearby graph lessons must not turn photos or illustrations into graphs.
    return _context_candidates(context) if figure_type == "graph" else []


def _minimal_figure_fallback(figure_type: str) -> str:
    kind = {
        "graph": "그래프",
        "table": "표",
        "mathematical_diagram": "수학 도식",
        "illustration": "삽화",
        "photo": "사진",
    }.get(figure_type, "시각 자료")
    return f"주변 교과서 문맥에서 다루는 내용을 보여 주는 {kind}이다."


def _graph_context_prompt(candidates: Sequence[Mapping[str, Any]]) -> str:
    if not candidates:
        return ""
    equations = _context_equations(candidates)
    equation_note = (
        f" 문맥에서 정확히 확인된 수식은 {', '.join(equations)}입니다. 수식을 설명에 사용할 때는 문자나 분모를 생략하지 말고 그대로 옮기세요."
        if equations
        else ""
    )
    return (
        " 다음 주변 문맥은 축 이름, 계열 라벨의 대상, 그래프가 나타내는 상황을 확인하는 보조 자료입니다. 이미지의 선·곡선·점 개수와 "
        "형태는 문맥보다 이미지를 우선하세요. 문맥의 문장을 복사하거나 일반적인 그래프 해설을 추가하지 마세요. 주변 문맥: "
        + json.dumps(list(candidates), ensure_ascii=False, separators=(",", ":"))
        + equation_note
    )


def _verified_graph_facts_prompt(
    evidence: Sequence[Any] | None,
    context: Sequence[Mapping[str, Any]] | None,
) -> str:
    grounding_text = _graph_grounding_text(evidence, context)
    equations = _deduplicated_matches(_EVIDENCE_EQUATION_PATTERN, grounding_text)
    coordinates = _deduplicated_matches(_EXACT_COORDINATE_PATTERN, grounding_text)
    facts: list[str] = []
    x_axis, y_axis = _trusted_axis_labels(evidence)
    if x_axis and y_axis:
        facts.append(f"이미지에서 확인된 축: x축={x_axis}, y축={y_axis}")
    if equations:
        facts.append(f"허용된 정확한 수식: {', '.join(equations)}")
    if coordinates:
        facts.append(f"허용된 정확한 좌표: {', '.join(coordinates[:16])}")
    if not facts:
        return " 정확한 수식이나 좌표 근거가 없으므로 이를 새로 만들지 마세요."
    return (
        " 다음은 이미지 내부 OCR 또는 주변 문맥에서 직접 확인된 정확 정보입니다. "
        + " / ".join(facts)
        + ". 목록에 없는 수식이나 좌표는 출력하지 말고, 모든 값을 나열할 필요도 없습니다."
    )


def _visible_panel_count(evidence: Sequence[Any] | None) -> int:
    markers: set[int] = set()
    for item in _normalized_evidence(evidence):
        text = item["text"]
        markers.update(int(value) for value in re.findall(r"\(\s*(\d{1,2})\s*\)", text))
        if re.fullmatch(r"\s*\d{1,2}[.)]\s*", text):
            markers.add(int(re.search(r"\d+", text).group()))
    if not markers or 1 not in markers:
        return 0
    count = 1
    while count + 1 in markers:
        count += 1
    return count if count >= 2 else 0


def _panel_completeness_prompt(panel_count: int) -> str:
    if panel_count < 2:
        return ""
    return (
        f" 이미지에서 (1)부터 ({panel_count})까지 {panel_count}개의 패널 번호가 확인되었습니다. "
        "각 번호를 빠짐없이 한 번씩, 짧은 절이나 문장으로 구분해 설명하세요."
    )


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


def _anchor_trusted_axis_labels(text: str, evidence: Sequence[Any] | None) -> str:
    x_axis, y_axis = _trusted_axis_labels(evidence)
    if not x_axis or not y_axis:
        return text
    sentences = [
        part.strip()
        for part in re.findall(r".+?(?:[.!?。！？]+|$)", text, flags=re.DOTALL)
        if part.strip()
    ]
    kept: list[str] = []
    for sentence in sentences:
        mentions_axes = any(marker in sentence for marker in ("x축", "y축", "수평축", "수직축"))
        if mentions_axes and not (x_axis in sentence and y_axis in sentence):
            continue
        kept.append(sentence)
    body = " ".join(kept).strip()
    if x_axis in body and y_axis in body and "따른" in body[:160]:
        return body
    lead = f"{x_axis}에 따른 {y_axis}의 변화를 나타낸 그래프이다."
    return f"{lead} {body}".strip()


def _anchor_visible_direction_sequence(text: str, cue: GraphVisualCue | None) -> str:
    if (
        cue is None
        or cue.state != "plotted"
        or cue.confidence < 0.90
        or cue.mark_type != "line"
        or (cue.series_count or 1) != 1
        or cue.path_shape != "smooth_curve"
        or not 2 <= len(cue.direction_sequence) <= 5
    ):
        return text
    names = {
        "increasing": "상승",
        "decreasing": "하강",
        "horizontal": "수평 유지",
    }
    sequence = [names[item] for item in cue.direction_sequence]
    if all(item in text for item in set(sequence)):
        return text
    sentence = f"곡선은 왼쪽에서 오른쪽으로 {', '.join(sequence)} 순서로 이어진다."
    return f"{text} {sentence}".strip()


def _restore_trusted_context_terms(
    text: str,
    context: Sequence[Mapping[str, Any]] | None,
) -> str:
    context_text = " ".join(str(item.get("text") or "") for item in context or [])
    trusted: set[str] = set()
    trusted.update(
        cleaned
        for raw in re.findall(r"[A-Z]\s*(?:는|은)\s*([가-힣]{2,8})", context_text)
        if (cleaned := _clean_context_entity_term(raw))
    )
    for first, second in re.findall(r"([가-힣]{2,8})(?:와|과)\s*([가-힣]{2,8})", context_text):
        trusted.update(filter(None, (_clean_context_entity_term(first), _clean_context_entity_term(second))))
    if not trusted:
        return text
    words = {
        (raw, stem, particle)
        for raw in re.findall(r"[가-힣]{2,10}", text)
        for stem, particle in [_split_korean_particle(raw)]
    }
    for expected in trusted:
        if expected in text:
            continue
        close = [
            (raw, particle) for raw, stem, particle in words
            if len(stem) == len(expected)
            and stem[0] == expected[0]
            and sum(left != right for left, right in zip(stem, expected)) == 1
        ]
        if len(close) == 1:
            raw, particle = close[0]
            replacement = expected + _matching_korean_particle(expected, particle)
            text = re.sub(rf"(?<![가-힣]){re.escape(raw)}(?![가-힣])", replacement, text)
    return text


def _clean_context_entity_term(value: str) -> str | None:
    term = value
    for suffix in ("이라고", "라고", "이고", "이며", "이다", "으로", "에서", "에게", "까지", "부터", "의", "을", "를", "은", "는", "이", "가"):
        if term.endswith(suffix) and len(term) - len(suffix) >= 2:
            term = term[:-len(suffix)]
            break
    return term if 2 <= len(term) <= 8 else None


def _split_korean_particle(value: str) -> tuple[str, str]:
    for suffix in ("이라고", "라고", "이고", "이며", "이다", "으로", "에서", "에게", "까지", "부터", "과", "와", "의", "을", "를", "은", "는", "이", "가"):
        if value.endswith(suffix) and len(value) - len(suffix) >= 2:
            return value[:-len(suffix)], suffix
    return value, ""


def _matching_korean_particle(term: str, particle: str) -> str:
    if not term or not 0xAC00 <= ord(term[-1]) <= 0xD7A3:
        return particle
    has_final = (ord(term[-1]) - 0xAC00) % 28 != 0
    pairs = {
        "과": ("과", "와"), "와": ("과", "와"),
        "은": ("은", "는"), "는": ("은", "는"),
        "이": ("이", "가"), "가": ("이", "가"),
        "을": ("을", "를"), "를": ("을", "를"),
    }
    return pairs[particle][0 if has_final else 1] if particle in pairs else particle


def _graph_component_prompt(cue: GraphVisualCue | None) -> str:
    if (
        cue is not None
        and cue.coordinate_plane
        and cue.mark_type == "multiple"
        and (cue.series_count or 0) >= 2
    ):
        return (
            " 저수준 시각 검출에서는 좌표 영역 안에 서로 분리된 그래프 요소가 두 개 이상 확인되었습니다. "
            "이미지에서 이를 다시 확인하고, 하나의 연속 구간으로 합쳐 설명하지 마세요."
        )
    return ""


def _parse_context_fusion_response(
    text: str,
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("caption"), str):
        return None
    caption = value["caption"].strip()
    raw_ids = value.get("relevant_context_ids")
    if not caption or len(caption) > 1000 or not isinstance(raw_ids, list):
        return None
    allowed_ids = {str(item.get("block_id")) for item in candidates}
    ids = list(dict.fromkeys(
        item for item in raw_ids if isinstance(item, str) and item in allowed_ids
    ))
    return {"relevant_context_ids": ids, "caption": caption}


def _preserves_visual_anchor(base: str, fused: str) -> bool:
    anchors = {
        "사진", "도로", "다리", "자동차", "사람", "건물", "산", "바다",
        "그래프", "좌표평면", "직선", "꺾은선", "곡선", "점", "축", "격자",
        "표", "도형", "원", "삼각형", "사각형", "원기둥", "화살표",
    }
    base_anchors = {item for item in anchors if item in base}
    if not base_anchors:
        return True
    retained = base_anchors & {item for item in anchors if item in fused}
    required = 1 if len(base_anchors) <= 2 else 2
    return len(retained) >= required


def replace_caption_time(
    base: CaptionOutput,
    elapsed: float,
    warning: str | None = None,
) -> CaptionOutput:
    warnings = [*base.warnings, *([warning] if warning else [])]
    return CaptionOutput(
        text=base.text,
        confidence=base.confidence,
        generation_time_seconds=base.generation_time_seconds + elapsed,
        model_name=base.model_name,
        model_version=base.model_version,
        warnings=warnings,
        context_block_ids=base.context_block_ids,
    )


def _graph_needs_structured_review(cue: GraphVisualCue, evidence: Sequence[Any] | None = None) -> bool:
    return (
        cue.state == "plotted"
        and (
            cue.confidence < 0.9
            or cue.mark_type in {"multiple", "unknown"}
            or (cue.series_count or 0) >= 2
            or cue.variation in {"turning", "oscillating"}
            or _has_axis_label_candidate(evidence)
        )
    )


def _structured_graph_prompt(evidence: Sequence[Any] | None = None) -> str:
    candidates = [
        {"id": item["id"], "text": item["text"], "bbox": item.get("bbox")}
        for item in _normalized_evidence(evidence)[:24]
    ]
    return (
        "한국 교과서 Figure의 시각 구조를 분석하고 JSON 하나만 출력하세요. 도구 모음과 프로그램 입력표만 제외하되, "
        "그래프와 함께 제시된 도형·삽화는 context에 기록하세요. 먼저 서로 분리된 좌표계의 수를 세고 panel_count에 적으세요. "
        "좌표계가 여러 개면 각 좌표계를 plots에 하나씩 빠짐없이, 겹친 계열은 각 계열을 plots에 각각 기록하세요. "
        "점들이 선으로 연결되어 있으면 points가 아니라 straight_segments입니다. straight_segments는 직선 또는 꺾은선이고, "
        "smooth_curve는 모서리 없이 휘어진 곡선이며, points는 서로 연결되지 않은 점들일 때만 사용하세요. "
        "직선 조각의 진행 방향이 계속 같더라도 기울기가 달라지는 꼭짓점은 bends에 꺾인 횟수로 기록하세요. "
        "dirs에는 왼쪽에서 오른쪽으로 본 방향 변화를 순서대로 "
        "up, down, flat 중에서 기록하고, net은 시작과 끝의 전체 변화 up, down, flat, mixed, unknown 중 하나로 기록하세요. "
        "좌표평면의 곡선이 어느 사분면에 있는지 명확하면 quadrants에 1, 2, 3, 4 중 해당 번호를 기록하고, "
        "명확하지 않거나 좌표평면이 아니면 빈 배열로 두세요. "
        "shape은 points, straight_segments, smooth_curve, bars, unknown 중 하나입니다. "
        "x, y, name에는 아래 OCR 후보와 일치하는 id를 사용하세요. 후보에 축 이름이 없지만 이미지에 명확히 보이면 "
        "x_text, y_text에 축 이름과 단위를 보이는 그대로 짧게 옮기고, 불명확하면 null로 두세요. 수식·좌표는 새로 만들지 마세요. "
        "context는 함께 제시된 비그래프 시각물의 kind=solid_diagrams, diagrams, illustrations, none, "
        "count, position=above, below, left, right, mixed, paired를 기록하세요. 서로 분리된 도형은 items에 하나씩 빠짐없이 넣고 "
        "각 item은 보이는 형태만 짧게 설명하세요. 용도나 수학적 의미는 추론하지 마세요. 없으면 kind는 none입니다. "
        "arrangement는 single, panels, overlaid 중 하나입니다. "
        '형식: {"arrangement":"single","panel_count":1,"context":{"kind":"none","count":0,'
        '"position":"mixed","paired":false,"items":[]},"quadrants":[],"plots":[{"shape":"unknown","dirs":[],"net":"unknown",'
        '"bends":0,"x":null,"y":null,"name":null,"x_text":null,"y_text":null}]} OCR 후보: '
        + json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    )


def _parse_structured_graph_response(
    text: str,
    evidence: Sequence[Any] | None = None,
) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict) or value.get("arrangement") not in {"single", "panels", "overlaid"}:
        return None
    raw_plots = value.get("plots")
    if not isinstance(raw_plots, list) or not 1 <= len(raw_plots) <= 12:
        return None
    allowed_shapes = {"points", "straight_segments", "smooth_curve", "bars", "unknown"}
    allowed_directions = {"up", "down", "flat"}
    allowed_nets = {"up", "down", "flat", "mixed", "unknown"}
    evidence_ids = {item["id"] for item in _normalized_evidence(evidence)}
    plots: list[dict[str, Any]] = []
    for raw in raw_plots:
        if not isinstance(raw, dict) or raw.get("shape") not in allowed_shapes:
            return None
        directions = raw.get("dirs")
        if not isinstance(directions, list) or len(directions) > 10 or any(
            item not in allowed_directions for item in directions
        ):
            return None
        net = raw.get("net")
        if net not in allowed_nets:
            return None
        bends = raw.get("bends", 0)
        if not isinstance(bends, int) or isinstance(bends, bool) or not 0 <= bends <= 12:
            bends = 0
        plot = {"shape": raw["shape"], "dirs": directions, "net": net, "bends": bends}
        for key in ("x", "y", "name"):
            candidate = raw.get(key)
            plot[key] = candidate if isinstance(candidate, str) and candidate in evidence_ids else None
        for key in ("x_text", "y_text"):
            plot[key] = _safe_visible_axis_label(raw.get(key))
        plots.append(plot)
    panel_count = value.get("panel_count", len(plots) if value["arrangement"] == "panels" else 1)
    if not isinstance(panel_count, int) or isinstance(panel_count, bool) or not 1 <= panel_count <= 12:
        panel_count = len(plots) if value["arrangement"] == "panels" else 1
    context = _parse_graph_context(value.get("context"))
    raw_quadrants = value.get("quadrants", [])
    quadrants = [] if not isinstance(raw_quadrants, list) else sorted({
        item for item in raw_quadrants if isinstance(item, int) and not isinstance(item, bool) and 1 <= item <= 4
    })
    return {
        "arrangement": value["arrangement"],
        "panel_count": panel_count,
        "context": context,
        "quadrants": quadrants,
        "plots": plots,
    }


def _safe_visible_axis_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    label = " ".join(value.strip().split())
    if not label or len(label) > 24 or any(character in label for character in "{}[]<>#`="):
        return None
    return label


def _parse_graph_context(value: Any) -> dict[str, Any]:
    empty = {"kind": "none", "count": 0, "position": "mixed", "paired": False, "items": []}
    if not isinstance(value, dict):
        return empty
    kind = value.get("kind")
    count = value.get("count")
    position = value.get("position")
    if kind not in {"solid_diagrams", "diagrams", "illustrations", "none"}:
        return empty
    if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 12:
        return empty
    if position not in {"above", "below", "left", "right", "mixed"}:
        position = "mixed"
    raw_items = value.get("items")
    items = [] if not isinstance(raw_items, list) else [
        description
        for item in raw_items[:12]
        if isinstance(item, dict)
        and (description := _safe_visual_description(item.get("description"))) is not None
    ]
    return {
        "kind": kind,
        "count": max(count, len(items)),
        "position": position,
        "paired": value.get("paired") is True,
        "items": items,
    }


def _safe_visual_description(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    description = " ".join(value.strip().split())
    if not description or len(description) > 40 or any(character in description for character in "{}[]<>#`="):
        return None
    return description.rstrip(".!?")


def _reconcile_structured_graph(
    cue: GraphVisualCue,
    structure: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Resolve one high-confidence conflict without inventing graph content."""
    if structure is None:
        return None
    plots = structure.get("plots", [])
    if (
        len(plots) == 1
        and structure.get("arrangement") == "single"
        and cue.mark_type == "line"
        and cue.confidence >= 0.9
        and (
            plots[0].get("shape") == "points"
            or (plots[0].get("shape") == "smooth_curve" and cue.path_shape == "straight_segments")
        )
    ):
        plots[0] = {**plots[0], "shape": "straight_segments"}
    if (
        len(plots) == 1
        and structure.get("arrangement") == "single"
        and cue.path_shape == "straight_segments"
        and cue.bend_count > plots[0].get("bends", 0)
    ):
        plots[0] = {**plots[0], "bends": cue.bend_count}
    return structure


def _apply_graph_trend_grounding(text: str, cue: GraphVisualCue | None) -> str:
    if (
        cue is None
        or cue.coordinate_plane
        or cue.state != "plotted"
        or cue.trend is None
        or cue.confidence < 0.8
        or cue.mark_type == "multiple"
        or (cue.series_count or 0) >= 2
    ):
        return text
    sentences = [
        part.strip()
        for part in re.findall(r".+?(?:[.!?。！？]+|$)", text, flags=re.DOTALL)
        if part.strip()
    ]
    trend_markers = (
        "우상향", "우하향", "증가", "감소", "상단에서 하단", "하단에서 상단",
        "위로 향", "아래로 향", "오르내", "올라갔다", "내려갔다",
    )
    sentences = [sentence for sentence in sentences if not any(marker in sentence for marker in trend_markers)]
    lead = _graph_trend_lead(cue)
    return " ".join([lead, *sentences]).strip()


def _fallback_after_exact_claim_filter(
    text: str,
    cue: GraphVisualCue | None,
    evidence: Sequence[Any] | None,
) -> tuple[str, bool]:
    """Keep a graph describable when every exact-claim sentence was rejected."""
    if text.strip() or cue is None or cue.state != "plotted":
        return text, False
    return _grounded_coordinate_graph_caption(cue, evidence), True


def _graph_trend_lead(cue: GraphVisualCue) -> str:
    if cue.variation == "oscillating":
        overall = {
            "increasing": "전체적으로는 오른쪽으로 갈수록 높아진다.",
            "decreasing": "전체적으로는 오른쪽으로 갈수록 낮아진다.",
            "horizontal": "전체 높이는 대체로 비슷하게 유지된다.",
        }[cue.trend]
        return f"좌표평면에 위아래로 반복해서 오르내리는 그래프가 표시되어 있으며, {overall}"
    if cue.variation == "turning":
        if cue.initial_direction == "increasing":
            return "좌표평면에 오른쪽으로 가면서 올라갔다가 내려가는 그래프가 표시되어 있다."
        if cue.initial_direction == "decreasing":
            return "좌표평면에 오른쪽으로 가면서 내려갔다가 올라가는 그래프가 표시되어 있다."
        return "좌표평면에 진행 방향이 한 차례 바뀌는 그래프가 표시되어 있다."
    return {
        "increasing": "좌표평면에 왼쪽 아래에서 오른쪽 위로 향하는 우상향 그래프가 표시되어 있다.",
        "decreasing": "좌표평면에 왼쪽 위에서 오른쪽 아래로 향하는 우하향 그래프가 표시되어 있다.",
        "horizontal": "좌표평면에 왼쪽에서 오른쪽으로 높이가 거의 일정한 그래프가 표시되어 있다.",
    }[cue.trend]


def _grounded_coordinate_graph_caption(
    cue: GraphVisualCue,
    evidence: Sequence[Any] | None,
    structure: dict[str, Any] | None = None,
    context: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    """Compose coordinate-graph prose without asking a VLM to invent relations."""
    if cue.state == "empty":
        return "x축과 y축, 격자가 표시된 빈 좌표평면이다."

    evidence_text = " ".join(item["text"] for item in _normalized_evidence(evidence))
    equations = _deduplicated_matches(_EVIDENCE_EQUATION_PATTERN, evidence_text)
    context_equations = [item for item in _context_equations(context) if item not in equations]
    grounded_equations = [*equations, *context_equations]
    coordinates = _deduplicated_matches(_EXACT_COORDINATE_PATTERN, evidence_text)
    multiple_curves = cue.mark_type == "multiple" or (cue.series_count or 0) >= 2
    if structure is not None:
        multiple_curves = multiple_curves or len(structure.get("plots", [])) >= 2
    if multiple_curves and any(_is_reciprocal_equation(item) for item in grounded_equations):
        quadrants = set(structure.get("quadrants", [])) if structure else set()
        if quadrants in ({1, 3}, {2, 4}):
            first, second = sorted(quadrants)
            sentences = [
                f"좌표평면의 제{first}사분면과 제{second}사분면에 서로 마주 보는 두 갈래의 곡선이 표시되어 있다."
            ]
        else:
            sentences = ["좌표평면에 서로 마주 보는 두 갈래의 곡선이 표시되어 있다."]
    else:
        sentences = _structured_graph_sentences(cue, structure, evidence)
    if equations:
        sentences.append(f"그림에는 {', '.join(equations)}가 적혀 있다.")
    if context_equations:
        sentences.append(f"주변 설명에서는 {', '.join(context_equations)}를 다룬다.")
    if coordinates:
        sentences.append(f"또한 {', '.join(coordinates)} 표기가 있다.")
    return " ".join(sentences)


def _context_equations(context: Sequence[Mapping[str, Any]] | None) -> list[str]:
    text = " ".join(str(item.get("text") or "") for item in context or [])
    return _deduplicated_matches(_EVIDENCE_EQUATION_PATTERN, text)


def _graph_grounding_text(
    evidence: Sequence[Any] | None,
    context: Sequence[Mapping[str, Any]] | None,
) -> str:
    evidence_text = " ".join(item["text"] for item in _normalized_evidence(evidence))
    context_text = " ".join(str(item.get("text") or "") for item in context or [])
    return f"{evidence_text} {context_text}".strip()


def _restore_truncated_context_equation(
    text: str,
    context: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Restore only a visibly truncated slash equation backed by one context fact.

    This deliberately does not invent or broadly spell-correct equations.  A
    repair is allowed only when every slash equation found in the selected
    context resolves to the same compact expression and the generated text
    contains that exact left side and numerator followed by a bare slash.
    """
    equations = _context_equations(context)
    compact_equations = list(dict.fromkeys(re.sub(r"\s+", "", item) for item in equations))
    if len(compact_equations) != 1:
        return text
    slash_equations = []
    for equation in compact_equations:
        match = re.fullmatch(
            r"([A-Za-z](?:\([A-Za-z]\))?)=([+\-]?[A-Za-z0-9]+)/([A-Za-z0-9]+)",
            equation,
        )
        if match:
            slash_equations.append((equation, match.groups()))
    if len(slash_equations) != 1:
        return text

    equation, (left, numerator, _denominator) = slash_equations[0]
    incomplete = re.compile(
        rf"{re.escape(left)}\s*=\s*(?:{re.escape(numerator)}\s*/)?(?=[^A-Za-z0-9\\]|$)"
    )
    return incomplete.sub(equation, text)


def _filter_unsupported_context_graph_claims(
    text: str,
    evidence: Sequence[Any] | None,
    context: Sequence[Mapping[str, Any]] | None,
) -> tuple[str, list[str]]:
    """Keep visual prose while rejecting unsupported exact or derived claims.

    The older all-or-nothing grounding filter removed useful descriptions.
    This narrower check only governs equations, coordinate pairs, and
    mathematical interpretation that is not needed to describe the picture.
    """
    grounding_text = _graph_grounding_text(evidence, context)
    trusted_equations = {
        _compact_grounding_text(item)
        for item in _deduplicated_matches(_EVIDENCE_EQUATION_PATTERN, grounding_text)
    }
    trusted_coordinates = {
        _compact_grounding_text(item)
        for item in _deduplicated_matches(_EXACT_COORDINATE_PATTERN, grounding_text)
    }
    forbidden_interpretations = (
        "무작위", "무한대", "수렴", "정의역", "치역", "양수", "음수",
        "양의 값", "음의 값", "절편", "기울기",
    )
    incomplete_equation = re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Za-z](?:\s*\(\s*[A-Za-z]\s*\))?)\s*=\s*"
        r"(?:[+\-]?[A-Za-z0-9]+\s*/\s*)?(?=[^A-Za-z0-9\\]|$)"
    )
    kept: list[str] = []
    removed: list[str] = []
    for sentence in (
        part.strip()
        for part in re.findall(r".+?(?:[.!?。！？]+|$)", text, flags=re.DOTALL)
        if part.strip()
    ):
        equations = _deduplicated_matches(_EVIDENCE_EQUATION_PATTERN, sentence)
        coordinates = _deduplicated_matches(_EXACT_COORDINATE_PATTERN, sentence)
        unsupported_equation = any(
            _compact_grounding_text(item) not in trusted_equations for item in equations
        )
        unsupported_coordinate = any(
            _compact_grounding_text(item) not in trusted_coordinates for item in coordinates
        )
        unsupported_interpretation = "무작위" in sentence or any(
            marker in sentence and marker not in grounding_text
            for marker in forbidden_interpretations
            if marker != "무작위"
        )
        if (
            unsupported_equation
            or unsupported_coordinate
            or incomplete_equation.search(sentence)
            or unsupported_interpretation
        ):
            removed.append(sentence)
            continue
        kept.append(sentence)
    warnings = (
        ["Removed unsupported exact or derived graph claim(s): " + repr(" ".join(removed))]
        if removed
        else []
    )
    return " ".join(kept).strip(), warnings


def _ensure_context_graph_visual_anchor(
    text: str,
    cue: GraphVisualCue | None,
    evidence: Sequence[Any] | None,
    context: Sequence[Mapping[str, Any]] | None,
) -> str:
    """Retain a minimal verified visual statement after claim filtering."""
    if cue is None or cue.state != "plotted" or "곡선" in text:
        return text
    equations = _deduplicated_matches(
        _EVIDENCE_EQUATION_PATTERN, _graph_grounding_text(evidence, context)
    )
    multiple = cue.mark_type == "multiple" and (cue.series_count or 0) >= 2
    if multiple and any(_is_reciprocal_equation(item) for item in equations):
        supplement = "좌표평면에는 서로 분리된 두 갈래의 곡선이 표시되어 있다."
        return f"{text} {supplement}".strip()
    return text


def _structured_graph_lead(cue: GraphVisualCue, structure: dict[str, Any] | None) -> str:
    return " ".join(_structured_graph_sentences(cue, structure, None))


def _structured_graph_sentences(
    cue: GraphVisualCue,
    structure: dict[str, Any] | None,
    evidence: Sequence[Any] | None,
) -> list[str]:
    if structure is None:
        if cue.mark_type == "points":
            return [_point_distribution_lead(cue.trend)]
        return [_graph_trend_lead(cue) if cue.trend is not None else "좌표평면에 그래프가 표시되어 있다."]

    plots = structure["plots"]
    evidence_map = {item["id"]: item["text"] for item in _normalized_evidence(evidence)}
    resolved_axes = [
        (
            evidence_map.get(plot.get("x")) or plot.get("x_text"),
            evidence_map.get(plot.get("y")) or plot.get("y_text"),
        )
        for plot in plots
    ]
    axis_pairs = {(x_label, y_label) for x_label, y_label in resolved_axes if x_label and y_label}
    axis_labels = {label for pair in resolved_axes for label in pair if label}
    sentences: list[str] = []
    context = structure.get("context") or {}
    if context.get("kind") != "none" and context.get("count", 0) > 0:
        kind_name = {
            "solid_diagrams": "입체도형",
            "diagrams": "도형",
            "illustrations": "삽화",
        }.get(context["kind"], "시각 자료")
        position_name = {
            "above": "위쪽에",
            "below": "아래쪽에",
            "left": "왼쪽에",
            "right": "오른쪽에",
            "mixed": "함께",
        }[context.get("position", "mixed")]
        context_items = context.get("items") or []
        if context_items:
            item_prefixes = ["첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", "여섯 번째"]
            for index, description in enumerate(context_items):
                label = item_prefixes[index] if index < len(item_prefixes) else f"{index + 1}번째"
                sentences.append(f"{position_name} 있는 {label} {kind_name}은 {description}이다.")
        else:
            sentences.append(f"{position_name} {context['count']}개의 {kind_name}이 배치되어 있다.")
        if context.get("paired") and structure.get("arrangement") == "panels":
            sentences.append("이 시각 자료들은 여러 그래프와 순서대로 대응한다.")
    if len(axis_pairs) == 1:
        x_label, y_label = next(iter(axis_pairs))
        sentences.append(
            f"x축은 {x_label}이고 y축은 {y_label}이며, {x_label}에 따른 {y_label}의 변화를 나타낸 그래프이다."
        )

    declared_panels = structure.get("panel_count", len(plots))
    if structure["arrangement"] == "panels" and declared_panels > len(plots):
        sentences.append(f"서로 분리된 그래프가 {declared_panels}개 배치되어 있다.")

    prefixes = ["첫 번째", "두 번째", "세 번째", "네 번째", "다섯 번째", "여섯 번째"]
    for index, plot in enumerate(plots):
        name = evidence_map.get(plot.get("name"))
        if name and name not in axis_labels:
            prefix = f"{name} 계열은"
        elif len(plots) > 1:
            label = prefixes[index] if index < len(prefixes) else f"{index + 1}번째"
            prefix = f"{label} {'그래프는' if structure['arrangement'] == 'panels' else '계열은'}"
        else:
            prefix = ""
        description = _describe_structured_plot(plot)
        x_label, y_label = resolved_axes[index]
        if len(axis_pairs) != 1 and x_label and y_label:
            relation = f"x축은 {x_label}이고 y축은 {y_label}이며, {x_label}에 따른 {y_label}의 변화를 나타내고, "
        else:
            relation = ""
        sentences.append(f"{prefix + ' ' if prefix else ''}{relation}{description}")
    return sentences


def _describe_structured_plot(plot: Mapping[str, Any]) -> str:
    directions = _collapse_directions(plot["dirs"])
    if plot["shape"] == "points" and len(directions) <= 1:
        return {
            "up": "여러 점이 오른쪽으로 갈수록 대체로 높게 분포한다.",
            "down": "여러 점이 오른쪽으로 갈수록 대체로 낮게 분포한다.",
            "flat": "여러 점이 비슷한 높이로 분포한다.",
            None: "여러 점이 분포한다.",
        }[directions[0] if directions else None]
    bends = plot.get("bends", 0)
    subject = {
        "points": "여러 점이",
        "straight_segments": (
            f"{bends}번 꺾인 선이" if bends >= 2 else "한 번 꺾인 선이" if bends == 1 else "이어진 직선 조각이"
        ),
        "smooth_curve": "곡선이",
        "bars": "막대들이",
        "unknown": "그래프가",
    }[plot["shape"]]
    if directions:
        movement_names = {"up": "상승", "down": "하강", "flat": "수평 유지"}
        if len(directions) == 1:
            movement = {
                "up": "오른쪽으로 갈수록 높아진다.",
                "down": "오른쪽으로 갈수록 낮아진다.",
                "flat": "비슷한 높이로 이어진다.",
            }[directions[0]]
        else:
            movement = f"{', '.join(movement_names[item] for item in directions)} 순서로 이어진다."
    else:
        movement = "표시되어 있다."
    sentence = f"{subject} {movement}"
    if len(directions) >= 2:
        net = plot["net"]
        net_sentence = {
            "up": "전체적으로는 시작보다 끝의 높이가 높다.",
            "down": "전체적으로는 시작보다 끝의 높이가 낮다.",
            "flat": "전체적으로는 시작과 끝의 높이가 비슷하다.",
            "mixed": "전체 방향은 하나로 정리되지 않는다.",
            "unknown": "",
        }[net]
        if net_sentence:
            sentence += f" {net_sentence}"
    return sentence


def _collapse_directions(values: Sequence[str]) -> list[str]:
    collapsed: list[str] = []
    for value in values:
        if not collapsed or value != collapsed[-1]:
            collapsed.append(value)
    return collapsed


def _point_distribution_lead(trend: str | None) -> str:
    return {
        "increasing": "좌표평면에 여러 점이 표시되어 있으며, 점들은 오른쪽으로 갈수록 대체로 높게 분포한다.",
        "decreasing": "좌표평면에 여러 점이 표시되어 있으며, 점들은 오른쪽으로 갈수록 대체로 낮게 분포한다.",
        "horizontal": "좌표평면에 여러 점이 비슷한 높이로 분포한다.",
        None: "좌표평면에 여러 점이 분포한다.",
    }[trend]


def _deduplicated_matches(pattern: re.Pattern[str], text: str) -> list[str]:
    matches: list[str] = []
    for match in pattern.findall(text):
        cleaned = re.sub(r"\s+", " ", match).strip()
        if cleaned and cleaned not in matches:
            matches.append(cleaned)
    return matches


def _is_reciprocal_equation(text: str) -> bool:
    compact = re.sub(r"\s+", "", text).lower()
    return bool(
        re.fullmatch(r"[a-z](?:\([a-z]\))?=[+\-]?[a-z0-9]+/[a-z0-9]+", compact)
        or re.fullmatch(r"[a-z](?:\([a-z]\))?=[+\-]?\\frac\{[^{}]+\}\{[^{}]+\}[a-z]?", compact)
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


def _has_axis_label_candidate(evidence: Sequence[Any] | None) -> bool:
    for item in _normalized_evidence(evidence):
        text = item["text"].strip()
        compact = re.sub(r"\s+", "", text)
        if _EVIDENCE_EQUATION_PATTERN.search(text) or _EXACT_COORDINATE_PATTERN.search(text):
            continue
        if compact.lower() in {"x", "y", "o"} or re.fullmatch(r"[+\-]?\d+(?:\.\d+)?", compact):
            continue
        if re.search(r"[가-힣]{2,}", text):
            return True
    return False


def _grounding_evidence_prompt(evidence: Sequence[Any] | None) -> str:
    verified = [item["text"] for item in _normalized_evidence(evidence)]
    if not verified:
        return (
            " 이미지 영역에서 별도로 확인된 문자 근거가 없습니다. "
            "따라서 정확한 수식·좌표·절편값·숫자를 새로 만들지 말고 시각적으로 보이는 형태와 관계만 설명하세요."
        )
    joined = " / ".join(verified)[:1200]
    return (
        f" 이미지 영역의 PDF 텍스트 또는 OCR에서 확인된 표기는 다음과 같습니다: [{joined}]. "
        "정확한 수식·좌표·절편값·숫자는 이 표기에 같은 내용이 있을 때만 사용하세요. "
        "표기에 없는 값을 선의 모양이나 눈금으로 계산하지 말고, 표시된 점을 선의 시작점이나 끝점으로 확대 해석하지 마세요."
    )


_EXACT_COORDINATE_PATTERN = re.compile(
    r"\(\s*[+-]?(?:\d+(?:\.\d+)?|[A-Za-z])\s*,\s*[+-]?(?:\d+(?:\.\d+)?|[A-Za-z])\s*\)"
)
_EXACT_EQUATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z](?:\s*\(\s*[A-Za-z]\s*\))?\s*)="
    r"\s*[+-]?[A-Za-z0-9\\](?:[A-Za-z0-9\\{}\s+\-*/^().]{0,24})"
)
_EVIDENCE_EQUATION_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])(?:[A-Za-z](?:\s*\(\s*[A-Za-z]\s*\))?)\s*=\s*[+-]?"
    r"(?:\\frac\s*\{[^{}]+\}\s*\{[^{}]+\}\s*[A-Za-z]?|"
    r"[A-Za-z0-9]+\s*/\s*[A-Za-z0-9]+|"
    r"(?:\d+(?:\.\d+)?\s*)?[A-Za-z]{1,3}(?:\s*[+\-]\s*\d+(?:\.\d+)?)?|"
    r"\d+(?:\.\d+)?)"
)
_EXACT_NUMBER_PATTERN = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?(?![A-Za-z])")


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
            if claim not in supported_numbers and not _is_panel_number(sentence, claim)
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


def _postprocess_qwen_caption(text: str) -> str:
    """Remove obvious generation artifacts without rewriting valid OCR or math."""
    text = _substitute_stray_hanja(text)
    text = re.sub(r"(?i)\bcoordinate\s*축", "좌표축", text)
    text = text.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
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
    return " ".join(kept).strip()


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
