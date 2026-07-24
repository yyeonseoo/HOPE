from __future__ import annotations

import re

_SENTENCE_END_PATTERN = re.compile(r"[.!?。！？]")
_DEFAULT_MAX_CHARS = 30

# Deliberately provider-agnostic (pure string in, string out) so formula and
# table analyzers can reuse it once they adopt the same
# description/summary/confidence shape (see request item 11) -- lives here
# only because figure is the first analyzer to need it.


def derive_summary(description: str | None, max_chars: int = _DEFAULT_MAX_CHARS) -> str | None:
    """A one-sentence, screen-reader-friendly headline for a description.

    Derived from the description's own first sentence rather than generated
    independently: the description has already been through grounding
    verification, so trimming it can't introduce a new hallucination the way
    a second free-form generation could.
    """
    text = (description or "").strip()
    if not text:
        return None

    match = _SENTENCE_END_PATTERN.search(text)
    first_sentence = text[: match.start()] if match else text
    first_sentence = first_sentence.strip().rstrip(".!?。！？ ")
    if not first_sentence:
        return None
    if len(first_sentence) <= max_chars:
        return first_sentence

    truncated = first_sentence[:max_chars]
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0]
    truncated = truncated.strip() or first_sentence[:max_chars].strip()
    return f"{truncated}…"
