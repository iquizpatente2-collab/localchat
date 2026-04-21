"""
Serialize pdfplumber tables to readable plain text; detect titles from text above table.
"""
from __future__ import annotations

import re
from typing import Any

import pdfplumber
from loguru import logger
from tqdm import tqdm

from .utils import normalize_whitespace, table_cell_empty


def _forward_fill_merged(rows: list[list[Any]]) -> list[list[str]]:
    """Carry last non-empty cell down within each column (merged-cell heuristic)."""
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    filled: list[list[str]] = []
    last_val: list[str] = [""] * ncols
    for row in rows:
        out_row: list[str] = []
        for ci in range(ncols):
            cell = row[ci] if ci < len(row) else None
            if table_cell_empty(cell):
                out_row.append(last_val[ci])
            else:
                s = str(cell).strip().replace("\n", " ")
                last_val[ci] = s
                out_row.append(s)
        filled.append(out_row)
    return filled


def _table_empty_ratio(rows: list[list[str]]) -> float:
    if not rows:
        return 1.0
    total = 0
    empty = 0
    for row in rows:
        for c in row:
            total += 1
            if not str(c).strip():
                empty += 1
    return empty / max(total, 1)


def _all_cells_garbage(rows: list[list[Any]]) -> bool:
    for row in rows:
        for c in row:
            if c is None:
                continue
            s = str(c).strip()
            if s and s.lower() not in {"none", "null"}:
                return False
    return True


def _title_from_bbox_page(page: pdfplumber.page.Page, bbox: tuple[float, float, float, float]) -> str:
    """Text in region strictly above table bbox."""
    try:
        x0, top, x1, bottom = bbox
        h = max(0, top - 1)
        if h < 8:
            return ""
        crop = page.crop((0, 0, page.width, h))
        t = crop.extract_text(layout=True) or crop.extract_text() or ""
        lines = [normalize_whitespace(ln) for ln in t.splitlines() if normalize_whitespace(ln)]
        if not lines:
            return ""
        # Prefer line that looks like a table caption
        for ln in reversed(lines[-6:]):
            if re.match(r"(?i)^table\s+[\d.\-]+", ln) or ln.lower().startswith("table "):
                return ln[:200]
        return lines[-1][:200]
    except Exception as e:
        logger.debug("Title bbox crop failed: {}", e)
        return ""


def serialize_page_tables(
    page: pdfplumber.page.Page,
    page_num: int,
    tables: list[list[list[Any]]],
    bboxes: list[tuple[float, float, float, float]],
) -> tuple[str, int]:
    """
    Returns (serialized_block, tables_kept_count).
    """
    blocks: list[str] = []
    kept = 0
    for m, tbl in enumerate(tables):
        if not tbl or _all_cells_garbage(tbl):
            logger.warning("Skipping garbage table {} on page {}", m, page_num)
            continue
        filled = _forward_fill_merged(tbl)
        if _table_empty_ratio(filled) > 0.5:
            logger.warning("Skipping sparse table (>50% empty) {} page {}", m, page_num)
            continue
        title = ""
        if m < len(bboxes):
            title = _title_from_bbox_page(page, bboxes[m])
        if not title.strip():
            title = f"Table page_{page_num} table_{m + 1}"
        lines = ["[TABLE: {}]".format(title)]
        for row in filled:
            lines.append(" | ".join(str(c) if c else "" for c in row))
        blocks.append("\n".join(lines))
        kept += 1
    return "\n\n".join(blocks), kept


def append_tables_to_pages(
    pdf_path: str,
    classified_pages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Opens PDF once more to crop titles; mutates copies with key `text_with_tables`.
    """
    from pathlib import Path

    path = Path(pdf_path)
    out: list[dict[str, Any]] = []
    tables_serialized = 0
    with pdfplumber.open(path) as pdf:
        for d in tqdm(classified_pages, desc="Serialize tables", unit="pg"):
            dc = dict(d)
            if dc.get("skipped"):
                dc["text_with_tables"] = dc.get("raw_text") or ""
                out.append(dc)
                continue
            pnum = int(dc["page_number"])
            idx = pnum - 1
            if idx < 0 or idx >= len(pdf.pages):
                dc["text_with_tables"] = dc.get("raw_text") or ""
                out.append(dc)
                continue
            page = pdf.pages[idx]
            tbls = dc.get("tables") or []
            bbs = dc.get("table_bboxes") or []
            try:
                ser, k = serialize_page_tables(page, pnum, tbls, bbs)
                tables_serialized += k
            except Exception as e:
                logger.error("Table serialization failed page {}: {}", pnum, e)
                ser, k = "", 0
            base = (dc.get("raw_text") or "").rstrip()
            if ser:
                dc["text_with_tables"] = f"{base}\n\n{ser}" if base else ser
            else:
                dc["text_with_tables"] = base
            dc["_tables_serialized_count"] = k
            out.append(dc)
    return out
