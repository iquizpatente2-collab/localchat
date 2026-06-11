"""
Bridge Localchat RAG index build to manual_pdf_pipeline (pdfplumber-based ingestion).
"""
from __future__ import annotations

from typing import Any


def pipeline_json_to_store_chunks(chunks_json: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map pipeline chunk records to VectorStore chunk dicts (text + page + optional fields)."""
    out: list[dict[str, Any]] = []
    for c in chunks_json:
        meta = c.get("metadata") or {}
        text = (c.get("content") or "").strip()
        if not text:
            continue
        ps = int(meta.get("page_start") or 1)
        pe = int(meta.get("page_end") or ps)
        out.append(
            {
                "text": text,
                "page": ps,
                "page_end": pe,
                "section": (meta.get("section") or "")[:500],
                "subsection": (meta.get("subsection") or "")[:500],
                "chunk_type": meta.get("type") or "",
                "procedure_id": meta.get("procedure_id"),
                "has_table": bool(meta.get("has_table")),
                "topic_id": "",
                "image_ids": [],
            }
        )
    return out


def enriched_pages_to_recipe_pages(enriched: list[dict[str, Any]]) -> list[tuple[int, str]]:
    """(page_number, text_with_tables) for recipe index — same shape as extract_pages_cleaned."""
    pages: list[tuple[int, str]] = []
    for p in enriched:
        if p.get("skipped"):
            continue
        txt = (p.get("text_with_tables") or "").strip()
        if not txt:
            continue
        pages.append((int(p["page_number"]), txt))
    return pages
