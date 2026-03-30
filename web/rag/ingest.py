from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def _page_extract_text(page) -> str:
    """Prefer layout mode — often preserves gaps between words on older PDFs vs plain."""
    try:
        t = page.extract_text(extraction_mode="layout") or ""
    except (TypeError, ValueError):
        t = page.extract_text() or ""
    if not t.strip():
        t = page.extract_text() or ""
    return t


def extract_pdf_text(pdf_path: Path) -> tuple[str, list[tuple[int, str]]]:
    """Return full text and per-page segments for citation."""
    reader = PdfReader(str(pdf_path))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        t = _page_extract_text(page)
        pages.append((i + 1, t))
    full = "\n\n".join(f"[Page {p}]\n{text}" for p, text in pages if text.strip())
    return full, pages


def extract_pages_cleaned(pdf_path: Path) -> list[tuple[int, str]]:
    """Non-empty pages as (1-based page number, cleaned text)."""
    reader = PdfReader(str(pdf_path))
    out: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        t = _clean_text(_page_extract_text(page))
        if t:
            out.append((i + 1, t))
    return out


def pages_to_chunks(
    pages: list[tuple[int, str]],
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    """Chunk each page's text; each chunk carries that page number."""
    out: list[dict] = []
    for page_no, t in pages:
        for ch in chunk_text(t, chunk_size=chunk_size, overlap=overlap):
            ch["page"] = page_no
            out.append(ch)
    return out


def ingest_pdf_chunks(
    pdf_path: Path,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    """Extract text per page, chunk with overlap, attach page numbers."""
    return pages_to_chunks(
        extract_pages_cleaned(pdf_path),
        chunk_size=chunk_size,
        overlap=overlap,
    )


def _clean_text(s: str) -> str:
    s = _spacing_heuristic(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _spacing_heuristic(s: str) -> str:
    """
    Mitigate PDF text runs where spaces were lost (common on scanned/older books).
    Does not fix all cases; OCR or a dedicated repair pass may still be needed.
    """
    s = re.sub(r"([a-z])([A-Z][a-z])", r"\1 \2", s)
    s = re.sub(r"([a-zA-Z])([0-9])", r"\1 \2", s)
    s = re.sub(r"([0-9])([a-zA-Z])", r"\1 \2", s)
    return s


def chunk_text(
    text: str,
    chunk_size: int = 900,
    overlap: int = 150,
) -> list[dict]:
    """
    Split manual text into overlapping chunks. Tries paragraph boundaries first,
    then falls back to character windows.
    """
    text = _clean_text(text)
    if not text:
        return []

    chunks: list[dict] = []
    # Split on double newlines / page markers
    parts = re.split(r"\n\s*\n", text)
    buf = ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if len(buf) + len(part) + 1 <= chunk_size:
            buf = f"{buf}\n\n{part}".strip() if buf else part
            continue
        if buf:
            chunks.append({"text": buf})
        if len(part) <= chunk_size:
            buf = part
        else:
            for c in _window_chunks(part, chunk_size, overlap):
                chunks.append({"text": c})
            buf = ""
    if buf:
        chunks.append({"text": buf})

    # Merge tiny tail pieces into previous chunk
    merged: list[dict] = []
    for ch in chunks:
        if merged and len(ch["text"]) < 80:
            merged[-1]["text"] = merged[-1]["text"] + "\n\n" + ch["text"]
        else:
            merged.append(ch)
    return merged


def _window_chunks(s: str, size: int, overlap: int) -> list[str]:
    out: list[str] = []
    start = 0
    step = max(1, size - overlap)
    while start < len(s):
        out.append(s[start : start + size])
        start += step
    return out
