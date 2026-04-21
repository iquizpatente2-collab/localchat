"""
Automatic page type tagging for technical manuals.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from .utils import STEP_PATTERN, TOC_LINE_PATTERN, word_count


def _toc_score(text: str) -> float:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 5:
        return 0.0
    hits = sum(1 for ln in lines if TOC_LINE_PATTERN.match(ln))
    return hits / max(len(lines), 1)


def _mostly_heading_page(text: str) -> bool:
    """Few words, dominant short lines (title page / chapter opener)."""
    wc = word_count(text)
    if wc > 120:
        return False
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return False
    short = sum(1 for ln in lines if len(ln) < 80)
    if short / len(lines) < 0.6:
        return False
    letters = sum(1 for c in text if c.isalpha())
    if letters == 0:
        return False
    upper = sum(1 for c in text if c.isupper())
    return (upper / letters) > 0.35 or wc < 60


def classify_page(page: dict[str, Any], total_pages: int) -> str:
    """
    Return one of: cover, toc, section_header, text, table_heavy, procedure,
    thin, mixed
    """
    pnum = int(page.get("page_number", 0))
    text = page.get("raw_text") or ""
    tables = page.get("tables") or []
    wc = int(page.get("word_count", 0) or word_count(text))
    n_tables = len(tables)

    # thin: image-heavy / sparse
    if wc < 80:
        if n_tables >= 1 and wc > 0:
            return "mixed"
        return "thin"

    # cover: first pages, very little body
    if pnum <= 3 and wc < 150 and n_tables == 0:
        return "cover"

    # TOC
    if _toc_score(text) > 0.25 and wc < 2500:
        return "toc"

    # table heavy
    if n_tables > 1:
        base = "table_heavy"
    else:
        base = "text"

    has_proc = bool(STEP_PATTERN.search(text))
    has_table = n_tables > 0

    if has_proc and has_table:
        return "mixed" if base != "table_heavy" else "procedure"
    if has_proc:
        return "procedure"
    if has_table and wc > 100:
        if n_tables == 1 and base == "text":
            return "mixed"
        return base

    if _mostly_heading_page(text) and pnum > 3:
        return "section_header"

    return "text" if n_tables == 0 else "mixed"


def classify_all(pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(pages)
    out = []
    prev_wc: int | None = None
    for p in pages:
        d = dict(p)
        if d.get("skipped"):
            d["page_type"] = "skipped"
            out.append(d)
            continue
        pt = classify_page(d, total)
        d["page_type"] = pt
        wc = int(d.get("word_count", 0))
        if prev_wc is not None and wc > 0 and prev_wc > 200 and wc < prev_wc // 4:
            logger.warning(
                "Page {} word count dropped sharply ({} → {})",
                d.get("page_number"),
                prev_wc,
                wc,
            )
        prev_wc = wc
        out.append(d)
    return out
