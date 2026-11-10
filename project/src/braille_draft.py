"""Turn the braille-edit text export into an automatic braille draft.

Korean text is translated with braillify (2024 Korean braille rules). This is
a draft for a transcriber to check, not finished braille:
- structural markers ([원본 쪽 n], [표], [촉도 n], [제목] ...) stay in print so
  the reviewer can see the structure;
- transcriber's-note marks ,'…,' and empty cells == are written as cells via
  North American Braille ASCII (, = dot 6, ' = dot 3, = = dots 1-6).
  ponytail: that the Korean textbook files use exactly these cells is
  unconfirmed with a transcriber;
- Korean math braille is not implemented: formulas with LaTeX commands are
  left in print with a "수식 점역 필요" flag.
"""

from __future__ import annotations

import re
from typing import Callable

NOTE_MARK = chr(0x2820) + chr(0x2804)  # ,' in braille ASCII (dots 6, 3)
EMPTY_CELL = chr(0x283F) * 2  # == in braille ASCII (dots 1-6 twice)
CELL_GAP = chr(0x2800) * 2  # two blank cells between table cells
_STRUCTURE_ONLY = re.compile(r"^\[(?:원본 쪽[^\]]*|표|표 끝|이 책의 점역자 주|촉도 \d+)\]$")
_PREFIXED = re.compile(r"^(\[(?:제목|소제목|날개)\]) (.*)$")
_NOTE = re.compile(r"^,'(.*),'$")
_LATEX_COMMAND = re.compile(r"\\[A-Za-z]+|[_^{}]")

HEADER = "# 자동 점역 초안(braillify). 점역사 검수가 필요하다. 수학 점자와 표 서식은 규정대로 점역되지 않았다."


def edit_text_to_braille_draft(text: str, translate: Callable[[str], str] | None = None) -> str:
    translate = translate or _braillify()
    out = [HEADER]
    in_table = False
    for line in text.splitlines():
        line = line.rstrip()
        if not line or _STRUCTURE_ONLY.match(line):
            in_table = (in_table or line == "[표]") and line != "[표 끝]"
            out.append(line)
            continue
        if in_table:
            cells = [cell.strip() for cell in line.split("|")]
            out.append(CELL_GAP.join(EMPTY_CELL if cell == "==" else translate(cell) for cell in cells))
            continue
        note = _NOTE.match(line)
        if note:
            out.append(NOTE_MARK + translate(note.group(1)) + NOTE_MARK)
            continue
        if line.startswith("[수식] "):
            formula = line[len("[수식] "):]
            out.append(f"[수식 점역 필요] {formula}" if _LATEX_COMMAND.search(formula) else translate(formula))
            continue
        prefixed = _PREFIXED.match(line)
        if prefixed:
            out.append(f"{prefixed.group(1)} {translate(prefixed.group(2))}")
            continue
        out.append(translate(line))
    return "\n".join(out) + "\n"


def _braillify() -> Callable[[str], str]:
    try:
        import braillify
    except ImportError as exc:
        raise RuntimeError("Braille draft requires the 'braillify' package: pip install braillify") from exc
    return lambda text: _safe_translate(braillify.translate_to_unicode, text)


# Filled circled numbers (❶ U+2776 ...) are not in braillify's table; they
# read as the same item numbers as "(1)".
_FILLED_CIRCLED = {chr(0x2776 + n): f"({n + 1})" for n in range(10)}


def _safe_translate(translate: Callable[[str], str], text: str) -> str:
    """One unsupported character must not sink the whole draft: it is kept
    in print as [원문: x] for the transcriber and the rest is translated."""
    text = "".join(_FILLED_CIRCLED.get(ch, ch) for ch in text)
    try:
        return translate(text)
    except ValueError:
        pass
    out, run = [], ""
    for ch in text:
        try:
            translate(ch)
        except ValueError:
            if run:
                out.append(translate(run))
                run = ""
            out.append(f"[원문: {ch}]")
        else:
            run += ch
    if run:
        out.append(translate(run))
    return "".join(out)
