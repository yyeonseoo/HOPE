from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .captioners import _import_torch, _resolve_device
from .context_builder import FigureContext

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Below this similarity, a caption shares essentially no topical relation with
# any of the selected page context -- evaluated against a real garbled-word
# generation (0.616), a well-grounded paraphrase (0.803), and a caption
# describing a completely unrelated image (0.063). Sentence embeddings average
# over the whole caption, so this catches "wrong image entirely" / gross
# hallucination; it is not sensitive to a single fabricated word inside an
# otherwise on-topic sentence (see tests/figure/test_grounding.py for the
# minimal-pair evidence) -- that class of error needs a different technique.
LOW_TOPIC_SIMILARITY_THRESHOLD = 0.25

# Substring identifying this specific warning among a record's `warnings`
# list, so other callers (e.g. page_reliability.py) can recognize and
# exclude it -- it's the same underlying similarity score computed by
# TopicGroundingScorer, not an independent signal.
TOPIC_MISMATCH_WARNING_MARKER = "topical similarity to surrounding page context is low"


class TopicGroundingScorer:
    """Cheap sanity check for gross figure/context topic mismatch."""

    model_name = MODEL_NAME
    model_version: str | None = None

    def __init__(self, device: str = "cpu") -> None:
        self.device_request = device
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: str | None = None

    def score(
        self,
        caption: str,
        context: Sequence[Mapping[str, Any]] | None,
    ) -> float | None:
        """Return the caption's best cosine similarity to any context block's
        text, or None if there's no caption/context to compare."""
        context_texts = [
            text for item in (context or [])
            if (text := str(item.get("text") or "").strip())
        ]
        caption = (caption or "").strip()
        if not caption or not context_texts:
            return None

        torch = _import_torch()
        self._load(torch)
        embeddings = self._embed([caption, *context_texts], torch)
        caption_vector, context_vectors = embeddings[:1], embeddings[1:]
        similarities = caption_vector @ context_vectors.T
        return round(float(similarities.max().item()), 3)

    def _embed(self, texts: list[str], torch: Any) -> Any:
        encoded = self._tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        encoded = {name: value.to(self._device) for name, value in encoded.items()}
        with torch.no_grad():
            output = self._model(**encoded)
        token_embeddings = output.last_hidden_state
        mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return torch.nn.functional.normalize(pooled, p=2, dim=1)

    def _load(self, torch: Any) -> None:
        if self._model is not None:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Topic grounding requires the optional figure dependencies: "
                "pip install -r src/analysis/figure/requirements.txt"
            ) from exc
        self._device = _resolve_device(torch, self.device_request)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()


# Caption outweighs surrounding paragraphs when both are scoreable: a caption
# is written specifically about this figure, while previous/next paragraph
# text is topically related but not necessarily about the figure itself (see
# CAPTION_PRIORITY_INSTRUCTION in prompt_builder.py -- the generation prompt
# applies the same priority, so grounding scoring mirrors it).
CAPTION_SCORE_WEIGHT = 0.6
CONTEXT_SCORE_WEIGHT = 0.4


@dataclass(frozen=True)
class GroundingScores:
    """Caption/context/overall similarity breakdown for one generated description."""

    caption_score: float | None
    context_score: float | None
    overall_score: float | None

    def to_dict(self) -> dict[str, float | None]:
        return {
            "caption_score": self.caption_score,
            "context_score": self.context_score,
            "overall_score": self.overall_score,
        }


def compute_grounding_scores(
    description: str,
    figure_context: FigureContext,
    scorer: TopicGroundingScorer,
) -> GroundingScores:
    """Score a generated description against the figure's caption and its
    surrounding (previous/next) paragraphs separately, then combine them.

    Splitting the two lets a reviewer tell "doesn't match the caption" apart
    from "doesn't match the general topic of the page" -- collapsing them
    into one number (the old `score`/`find_topic_mismatch_warning` behavior)
    hides which source actually disagrees with the generated text.
    """
    caption_score = (
        scorer.score(description, [{"text": figure_context.caption}]) if figure_context.caption else None
    )
    # Prefer the full context window (previous_paragraphs/next_paragraphs) so
    # scoring matches what the prompt actually included (see
    # FigureContextBuilder's window_size); fall back to the single nearest
    # paragraph for a FigureContext built without the window populated.
    window_texts = (*figure_context.previous_paragraphs, *figure_context.next_paragraphs)
    if not window_texts:
        window_texts = (figure_context.previous_paragraph, figure_context.next_paragraph)
    context_texts = [{"text": text} for text in window_texts if text]
    context_score = scorer.score(description, context_texts) if context_texts else None

    if caption_score is not None and context_score is not None:
        overall_score = round(
            CAPTION_SCORE_WEIGHT * caption_score + CONTEXT_SCORE_WEIGHT * context_score, 3
        )
    elif caption_score is not None:
        overall_score = caption_score
    elif context_score is not None:
        overall_score = context_score
    else:
        overall_score = None

    return GroundingScores(caption_score=caption_score, context_score=context_score, overall_score=overall_score)


def find_topic_mismatch_warning(
    caption: str,
    context: Sequence[Mapping[str, Any]] | None,
    scorer: TopicGroundingScorer,
) -> tuple[float | None, list[str]]:
    """Return (similarity, warnings); warnings is non-empty only when the
    caption looks like it may describe a different image than the one
    surrounded by this context."""
    similarity = scorer.score(caption, context)
    if similarity is None or similarity >= LOW_TOPIC_SIMILARITY_THRESHOLD:
        return similarity, []
    return similarity, [
        f"Caption's {TOPIC_MISMATCH_WARNING_MARKER} ({similarity:.2f}); "
        "this description may not match the actual page content. This is a coarse check for "
        "gross mismatch, not a guarantee -- it will not catch a single fabricated word in an "
        "otherwise on-topic caption."
    ]
