"""
Split merged pages into RAG chunks: respect sections, procedures, tables; token targets and overlap.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from loguru import logger

from .utils import (
    PROCEDURE_ID_CAPTURE,
    SECTION_HEADING_LINE,
    STEP_PATTERN,
    count_tokens,
    line_boundary_rfind,
    sentence_boundary_rfind,
    tokens_to_chars_approx,
    word_count,
)

_TABLE_START = re.compile(r"^\[TABLE:", re.MULTILINE)


@dataclass
class Segment:
    text: str
    page_start: int
    page_end: int
    dominant_type: str
    has_table: bool
    source_pages: list[int] = field(default_factory=list)


def _split_table_and_text_blocks(text: str) -> list[tuple[str, str]]:
    """
    Returns list of ("table"|"text", content).
    """
    if not text:
        return []
    parts: list[tuple[str, str]] = []
    pos = 0
    for m in _TABLE_START.finditer(text):
        if m.start() > pos:
            parts.append(("text", text[pos : m.start()]))
        # extend until next [TABLE: or EOS
        nxt = _TABLE_START.search(text, m.end())
        end = nxt.start() if nxt else len(text)
        parts.append(("table", text[m.start() : end].strip()))
        pos = end
    if pos < len(text):
        parts.append(("text", text[pos:]))
    return [(k, v.strip()) for k, v in parts if v.strip()]


def merge_thin_pages(pages: list[dict[str, Any]]) -> tuple[list[Segment], int]:
    """Merge consecutive thin pages until combined words > 100."""
    segments: list[Segment] = []
    thin_buf: list[dict[str, Any]] = []
    thin_merged_count = 0

    def flush_thin() -> None:
        nonlocal thin_buf, thin_merged_count
        if not thin_buf:
            return
        texts = [(p.get("text_with_tables") or "").strip() for p in thin_buf]
        combined = "\n\n".join(t for t in texts if t)
        pnums = [int(p["page_number"]) for p in thin_buf]
        if len(thin_buf) > 1:
            thin_merged_count += len(thin_buf) - 1
        segments.append(
            Segment(
                text=combined,
                page_start=min(pnums),
                page_end=max(pnums),
                dominant_type="thin",
                has_table=any((p.get("tables") or []) for p in thin_buf),
                source_pages=pnums,
            )
        )
        thin_buf = []

    for p in pages:
        if p.get("skipped"):
            continue
        ptype = p.get("page_type") or "text"
        if ptype == "thin":
            thin_buf.append(p)
            wc = word_count("\n\n".join((x.get("text_with_tables") or "") for x in thin_buf))
            if wc >= 100:
                flush_thin()
            continue
        flush_thin()
        txt = (p.get("text_with_tables") or "").strip()
        pn = int(p["page_number"])
        segments.append(
            Segment(
                text=txt,
                page_start=pn,
                page_end=pn,
                dominant_type=ptype,
                has_table=bool((p.get("tables") or [])),
                source_pages=[pn],
            )
        )
    flush_thin()
    return segments, thin_merged_count


def _nearest_section_titles(text_before: str) -> tuple[str, str]:
    """From accumulated document prefix, get last major section + subsection titles."""
    section = ""
    subsection = ""
    for m in SECTION_HEADING_LINE.finditer(text_before):
        num = m.group("num")
        title = (m.group("title") or "").strip()[:200]
        label = f"{num} {title}".strip()
        parts = num.split(".")
        if len(parts) <= 2:
            section = label
            subsection = ""
        else:
            subsection = label
    return section, subsection


def _extract_procedure_id(text: str) -> str | None:
    m = PROCEDURE_ID_CAPTURE.search(text)
    if not m:
        return None
    for g in m.groups():
        if g:
            return g[:64]
    return None


def _is_procedure_heavy(text: str) -> bool:
    return len(STEP_PATTERN.findall(text)) >= 2


def _split_by_major_sections(text: str) -> list[str]:
    """Split on lines that look like numbered section headings (1.2, 2.3.4)."""
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    chunks_lines: list[list[str]] = []
    cur: list[str] = []
    heading_re = re.compile(r"^\s*\d+(?:\.\d+)+\s+\S")

    for ln in lines:
        if heading_re.match(ln) and cur and word_count("".join(cur)) > 40:
            chunks_lines.append(cur)
            cur = [ln]
        else:
            cur.append(ln)
    if cur:
        chunks_lines.append(cur)
    return ["".join(c).strip() for c in chunks_lines if "".join(c).strip()]


def _split_procedure_oversized(text: str, max_tokens: int) -> list[str]:
    """Split long procedure text on blank lines / step boundaries without breaking mid-step line."""
    if count_tokens(text) <= max_tokens:
        return [text]
    parts = re.split(r"\n\s*\n+", text)
    out: list[str] = []
    buf = ""
    for para in parts:
        trial = f"{buf}\n\n{para}".strip() if buf else para
        if count_tokens(trial) <= max_tokens:
            buf = trial
        else:
            if buf:
                out.append(buf)
            if count_tokens(para) > max_tokens:
                # line-based split inside huge paragraph
                start = 0
                approx = tokens_to_chars_approx(max_tokens - 20)
                while start < len(para):
                    end = min(len(para), start + approx)
                    end = sentence_boundary_rfind(para, end)
                    if end <= start:
                        end = min(len(para), start + approx)
                        end = line_boundary_rfind(para, end)
                    out.append(para[start:end].strip())
                    start = end
                buf = ""
            else:
                buf = para
    if buf:
        out.append(buf)
    return [x for x in out if x]


def _split_text_to_token_budget(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    min_words: int,
    allow_overlap: bool,
) -> list[str]:
    if not text.strip():
        return []
    if count_tokens(text) <= max_tokens:
        return [text.strip()]

    chunks: list[str] = []
    start = 0
    n = len(text)
    approx = tokens_to_chars_approx(max_tokens - 40)
    overlap_chars = tokens_to_chars_approx(overlap_tokens) if allow_overlap else 0

    while start < n:
        end = min(n, start + approx)
        if end < n:
            end = sentence_boundary_rfind(text, end)
            if end <= start + max(80, approx // 5):
                end = line_boundary_rfind(text, min(n, start + approx))
            if end <= start:
                end = min(n, start + approx)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        start = end - overlap_chars if allow_overlap and overlap_chars > 0 else end
        if start >= end:
            start = end
    return chunks


def segments_to_chunks(
    segments: list[Segment],
    *,
    source_file: str,
    min_words: int,
    max_tokens: int,
    overlap_tokens: int,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    prefix_for_sections = ""

    for seg in segments:
        if not seg.text.strip():
            continue

        def append_chunk(
            content: str,
            *,
            ctype: str,
            has_table: bool,
            ps: int = seg.page_start,
            pe: int = seg.page_end,
        ) -> None:
            nonlocal prefix_for_sections
            section, subsection = _nearest_section_titles(prefix_for_sections)
            pid = _extract_procedure_id(content)
            chunks.append(
                _make_chunk_dict(
                    content=content,
                    source_file=source_file,
                    page_start=ps,
                    page_end=pe,
                    section=section,
                    subsection=subsection,
                    ctype=ctype,
                    procedure_id=pid,
                    has_table=has_table,
                    chunk_index=len(chunks),
                )
            )

        blocks = _split_table_and_text_blocks(seg.text)
        for kind, block in blocks:
            prefix_for_sections += "\n" + block
            if kind == "table":
                append_chunk(block, ctype="table_heavy", has_table=True)
                continue

            subpieces = _split_by_major_sections(block) or [block]
            for piece in subpieces:
                proc_heavy = _is_procedure_heavy(piece) or seg.dominant_type == "procedure"
                if proc_heavy:
                    for ptext in _split_procedure_oversized(piece, max_tokens):
                        append_chunk(ptext, ctype="procedure", has_table=False)
                    continue

                parts = _split_text_to_token_budget(
                    piece,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    min_words=min_words,
                    allow_overlap=True,
                )
                for si, sf in enumerate(parts):
                    if (
                        si > 0
                        and overlap_tokens > 0
                        and chunks
                        and "[TABLE:" not in chunks[-1]["content"]
                        and "procedure" not in chunks[-1]["metadata"]["type"]
                    ):
                        prev = chunks[-1]["content"]
                        ov = _overlap_suffix(prev, overlap_tokens)
                        if ov:
                            sf = f"{ov}\n\n{sf}"
                    append_chunk(sf, ctype=seg.dominant_type, has_table="[TABLE:" in sf)

    # Cap merges: few "words" (e.g. tables of numbers) can still be huge in tokens.
    max_merge_tokens = max(max_tokens * 6, min_words * 25, 2500)
    merged = merge_undersized_chunks(chunks, min_words, max_merge_tokens=max_merge_tokens)
    for i, c in enumerate(merged):
        c["metadata"]["chunk_index"] = i
    return merged


def _overlap_suffix(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    approx = tokens_to_chars_approx(overlap_tokens)
    if len(text) <= approx:
        return text.strip()
    tail = text[-approx:]
    cut = tail.find("\n")
    if cut != -1 and cut < len(tail) // 3:
        tail = tail[cut + 1 :]
    return tail.strip()


def _make_chunk_dict(
    *,
    content: str,
    source_file: str,
    page_start: int,
    page_end: int,
    section: str,
    subsection: str,
    ctype: str,
    procedure_id: str | None,
    has_table: bool,
    chunk_index: int,
) -> dict[str, Any]:
    wc = word_count(content)
    return {
        "content": content,
        "metadata": {
            "source_file": source_file,
            "page_start": page_start,
            "page_end": page_end,
            "section": section or "",
            "subsection": subsection or "",
            "type": ctype,
            "procedure_id": procedure_id,
            "has_table": has_table,
            "word_count": wc,
            "chunk_index": chunk_index,
        },
    }


def merge_undersized_chunks(
    chunks: list[dict[str, Any]],
    min_words: int,
    *,
    max_merge_tokens: int = 6000,
) -> list[dict[str, Any]]:
    if not chunks:
        return []
    out: list[dict[str, Any]] = []
    buf: dict[str, Any] | None = None
    for c in chunks:
        if buf is None:
            buf = _copy_chunk(c)
            continue
        if buf["metadata"]["word_count"] < min_words:
            merged = _combine_chunks(buf, c)
            if count_tokens(merged["content"]) > max_merge_tokens:
                out.append(buf)
                buf = _copy_chunk(c)
            else:
                buf = merged
        else:
            out.append(buf)
            buf = _copy_chunk(c)
    if buf is not None:
        out.append(buf)
    for i, c in enumerate(out):
        c["metadata"]["chunk_index"] = i
    return out


def _copy_chunk(c: dict[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(c)


def _combine_chunks(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    a = _copy_chunk(a)
    a["content"] = f"{a['content']}\n\n{b['content']}"
    a["metadata"]["word_count"] = word_count(a["content"])
    a["metadata"]["page_end"] = max(a["metadata"]["page_end"], b["metadata"]["page_end"])
    a["metadata"]["has_table"] = a["metadata"]["has_table"] or b["metadata"]["has_table"]
    if b["metadata"].get("procedure_id"):
        a["metadata"]["procedure_id"] = b["metadata"]["procedure_id"]
    return a
