"""Textbook-specific values (badge dictionary, size thresholds, symbol fonts).

They live in profiles/default.json instead of code so that a new textbook is
a new profile, not a code change. HOPE_TEXTBOOK_PROFILE selects another file.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

_DEFAULT = Path(__file__).resolve().parents[1] / "profiles" / "default.json"


@lru_cache(maxsize=None)
def load_profile(path: str | None = None) -> Dict[str, Any]:
    chosen = Path(path or os.getenv("HOPE_TEXTBOOK_PROFILE") or _DEFAULT)
    return json.loads(chosen.read_text(encoding="utf-8"))
