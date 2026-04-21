"""
Shared helpers: token estimates, regex patterns, text utilities.
"""
from __future__ import annotations

import re
from typing import Callable

# --- Section / structure (numbered hierarchies) ---
# Major numbered heading: "1.2 Title" or "4.3.1 Something" at line start
SECTION_HEADING_LINE = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+){1,5})\s+(?P<title>\S.*)$",
    re.MULTILINE,
)
# Single top-level section e.g. "1 Introduction"
TOP_LEVEL_SECTION = re.compile(
    r"^\s*(?P<num>\d+)\s+(?P<title>[A-Za-z].{2,120})$",
    re.MULTILINE,
)

# Procedure / step patterns
STEP_PATTERN = re.compile(
    r"(?im)^\s*(?:step\s*[#.]?\s*\d+|\d+[\).\]]\s+(?=[A-Za-z])|"
    r"\[[A-Z]{1,4}\d*\]|procedure\s+[A-Za-z]?\d*|"
    r"phase\s+\d+|task\s*[#.]?\s*\d+)",
)
PROCEDURE_ID_CAPTURE = re.compile(
    r"\[([A-Z]{1,4}\d*)\]|(?:procedure|proc\.?)\s*[#:]?\s*([A-Za-z0-9\-]+)",
    re.I,
)

# Table of contents: dot leaders + trailing page number
TOC_LINE_PATTERN = re.compile(
    r"^.{8,}\.{3,}\s*\d{1,4}\s*$|^.{8,}\s+\d{1,4}\s*$",
    re.MULTILINE,
)

# Cross-references: one capture group per pattern (page number or section id)
XREF_PAGE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(?i)\b(?:see|refer\s+to|cf\.?)\s+(?:pages?\s+)?(\d{1,4})\b"),
    re.compile(r"(?i)\bp{1,2}\.?\s*(\d{1,4})\b"),
    re.compile(r"(?i)\[\s*▶\s*(\d{1,4})\s*\]"),
    re.compile(r"(?i)\(p{1,2}\.\s*(\d{1,4})\)"),
    re.compile(r"(?i)\bpage\s+(\d{1,4})\b"),
    re.compile(r"(?i)\bpg\.?\s*(\d{1,4})\b"),
]

# Maintenance / periodic frequency hints
FREQUENCY_PATTERN = re.compile(
    r"(?i)\b(?:every|each|at\s+intervals?\s+of|interval)\s+"
    r"(\d+\s*(?:hours?|hrs?|days?|weeks?|months?|years?|min(?:utes)?))"
    r"|(?:daily|weekly|monthly|quarterly|annually|bimonthly)\b"
    r"|(?:\d+\s*[-–]\s*\d+\s*months?)\b",
)

_word_re = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+(?:['’][A-Za-z]+)?")


def word_count(text: str) -> int:
    if not text or not text.strip():
        return 0
    return len(_word_re.findall(text))


def get_token_counter() -> Callable[[str], int]:
    """Prefer tiktoken (cl100k_base); fallback ~4 chars per token."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")

        def count(t: str) -> int:
            if not t:
                return 0
            return len(enc.encode(t))

        return count
    except Exception:

        def count_fallback(t: str) -> int:
            if not t:
                return 0
            return max(1, len(t) // 4)

        return count_fallback


count_tokens: Callable[[str], int] = get_token_counter()


def tokens_to_chars_approx(tokens: int) -> int:
    """Rough upper bound for slicing text by token budget."""
    return max(80, tokens * 5)


def sentence_boundary_rfind(text: str, end: int) -> int:
    """Largest index <= end ending at sentence boundary (. ! ? followed by space/newline)."""
    if end >= len(text):
        return len(text)
    window = text[: end + 1]
    for sep in (". ", ".\n", "! ", "?\n", "? ", "!\n"):
        pos = window.rfind(sep)
        if pos != -1:
            return pos + len(sep.rstrip())
    # fall back to last newline
    nl = window.rfind("\n")
    if nl > end // 2:
        return nl + 1
    return end


def line_boundary_rfind(text: str, end: int) -> int:
    if end >= len(text):
        return len(text)
    window = text[: end + 1]
    nl = window.rfind("\n")
    return nl + 1 if nl != -1 else end


def normalize_whitespace(s: str) -> str:
    return re.sub(r"[ \t]+", " ", re.sub(r"\n{3,}", "\n\n", s.strip()))


def table_cell_empty(cell: object) -> bool:
    if cell is None:
        return True
    s = str(cell).strip()
    return len(s) == 0
