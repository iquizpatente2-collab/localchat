"""
Cross-reference resolution and context header prepending.
"""
from __future__ import annotations

import re
from typing import Any

from loguru import logger

from .utils import FREQUENCY_PATTERN, XREF_PAGE_PATTERNS, normalize_whitespace, word_count


def build_page_snippet_map(pages: list[dict[str, Any]], max_chars: int = 250) -> dict[int, str]:
    """page_number -> first N chars of text_with_tables or raw_text."""
    m: dict[int, str] = {}
    for p in pages:
        if p.get("skipped"):
            continue
        pn = int(p["page_number"])
        txt = (p.get("text_with_tables") or p.get("raw_text") or "").strip()
        if not txt:
            continue
        one_line = normalize_whitespace(txt.replace("\n", " "))
        m[pn] = one_line[:max_chars] + ("…" if len(one_line) > max_chars else "")
    return m


def _extract_page_refs(text: str) -> list[int]:
    pages: list[int] = []
    for pat in XREF_PAGE_PATTERNS:
        for m in pat.finditer(text):
            g = m.group(1)
            if g and g.isdigit():
                pages.append(int(g))
    # de-dupe preserve order
    seen: set[int] = set()
    out: list[int] = []
    for p in pages:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def resolve_xrefs(
    chunk_text: str,
    page_map: dict[int, str],
    *,
    max_refs: int = 3,
) -> tuple[str, int]:
    """Append [REF → page X]: snippet for up to max_refs page xrefs."""
    refs = _extract_page_refs(chunk_text)[:max_refs]
    if not refs:
        return chunk_text, 0
    extras: list[str] = []
    for pn in refs:
        snip = page_map.get(pn)
        if not snip:
            continue
        extras.append(f"[REF → page {pn}]: {snip}")
    if not extras:
        return chunk_text, 0
    return chunk_text.rstrip() + "\n\n" + "\n".join(extras), len(extras)


def _page_range_str(meta: dict[str, Any]) -> str:
    a, b = meta.get("page_start"), meta.get("page_end")
    if a == b:
        return str(a)
    return f"{a}-{b}"


def build_context_header(
    *,
    document_name: str,
    section: str,
    subsection: str,
    page_range: str,
    chunk_type: str,
    procedure_id: str | None,
    frequency: str | None,
) -> str:
    lines = [
        f"Document: {document_name}",
        f"Section: {section or '(none)'}",
        f"Subsection: {subsection or '(none)'}",
        f"Page: {page_range}",
        f"Type: {chunk_type}",
        f"Procedure ID: {procedure_id or 'null'}",
        f"Frequency: {frequency or '(none)'}",
    ]
    return "\n".join(lines) + "\n---\n"


def extract_frequency_hint(text: str) -> str | None:
    m = FREQUENCY_PATTERN.search(text)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g.strip()[:120]
    return m.group(0).strip()[:120] if m.group(0) else None


def enrich_chunks(
    chunks: list[dict[str, Any]],
    page_map: dict[int, str],
    document_name: str,
) -> tuple[list[dict[str, Any]], int]:
    """
    Prepend context header; resolve page xrefs. Returns (chunks, xrefs_resolved_count).
    """
    out: list[dict[str, Any]] = []
    xref_total = 0
    for c in chunks:
        dc = dict(c)
        meta = dict(dc["metadata"])
        body = dc["content"]
        body, nxref = resolve_xrefs(body, page_map, max_refs=3)
        xref_total += nxref

        freq = extract_frequency_hint(body)
        header = build_context_header(
            document_name=document_name,
            section=meta.get("section") or "",
            subsection=meta.get("subsection") or "",
            page_range=_page_range_str(meta),
            chunk_type=str(meta.get("type") or "text"),
            procedure_id=meta.get("procedure_id"),
            frequency=freq,
        )
        full = header + body
        meta["word_count"] = word_count(full)
        dc["content"] = full
        dc["metadata"] = meta
        out.append(dc)
    return out, xref_total
