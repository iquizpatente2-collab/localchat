"""
pdfplumber-based extraction: raw text (layout), tables with optional bbox, word counts.
Never crashes the whole run on a single bad page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pdfplumber
from loguru import logger
from tqdm import tqdm

from .utils import word_count


@dataclass
class ExtractedPage:
    page_number: int
    raw_text: str
    tables: list[list[list[str | None]]] = field(default_factory=list)
    table_bboxes: list[tuple[float, float, float, float]] = field(default_factory=list)
    word_count: int = 0
    skipped: bool = False
    error: str | None = None


def _safe_extract_tables(page: pdfplumber.page.Page) -> tuple[list[list[list[Any]]], list[tuple[float, float, float, float]]]:
    """Use find_tables for geometry; fall back to extract_tables()."""
    rows_out: list[list[list[Any]]] = []
    bboxes: list[tuple[float, float, float, float]] = []
    try:
        found = page.find_tables() or []
        for ft in found:
            try:
                data = ft.extract()
                if not data:
                    continue
                bbox = getattr(ft, "bbox", None)
                if bbox and len(bbox) >= 4:
                    bboxes.append((float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])))
                else:
                    bboxes.append((0.0, 0.0, float(page.width), float(page.height)))
                rows_out.append(data)
            except Exception as e:
                logger.warning("Table extract failed on page {}: {}", page.page_number, e)
    except Exception as e:
        logger.warning("find_tables failed on page {}: {}", page.page_number, e)

    if not rows_out:
        try:
            legacy = page.extract_tables() or []
            for tbl in legacy:
                if tbl and any(any(c is not None and str(c).strip() for c in row) for row in tbl):
                    rows_out.append(tbl)
                    bboxes.append((0.0, 0.0, float(page.width), float(page.height)))
        except Exception as e:
            logger.warning("extract_tables failed on page {}: {}", page.page_number, e)

    return rows_out, bboxes


def extract_pdf(
    path: Path,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[ExtractedPage]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    pages: list[ExtractedPage] = []
    try:
        with pdfplumber.open(path) as pdf:
            total = len(pdf.pages)
            first = 1 if page_start is None else max(1, page_start)
            last = total if page_end is None else min(total, page_end)
            if first > last:
                return []
            idx_range = range(first - 1, last)
            for i in tqdm(idx_range, desc="Extract pages", unit="pg"):
                pnum = i + 1
                page = pdf.pages[i]
                try:
                    raw = page.extract_text(layout=True)
                    if raw is None:
                        raw = ""
                    raw = raw.replace("\x00", "")
                    wc = word_count(raw)
                    tbls, bbs = _safe_extract_tables(page)
                    skipped = wc == 0 and not tbls
                    ep = ExtractedPage(
                        page_number=pnum,
                        raw_text=raw,
                        tables=tbls,
                        table_bboxes=bbs,
                        word_count=wc,
                        skipped=skipped,
                    )
                    if skipped:
                        logger.warning("Page {} empty after extraction — tagged skipped", pnum)
                    pages.append(ep)
                except Exception as e:
                    logger.error("Extraction error page {}: {}", pnum, e)
                    pages.append(
                        ExtractedPage(
                            page_number=pnum,
                            raw_text="",
                            tables=[],
                            table_bboxes=[],
                            word_count=0,
                            skipped=True,
                            error=str(e),
                        )
                    )
    except Exception as e:
        logger.exception("Failed to open PDF: {}", e)
        raise RuntimeError(f"Could not open PDF {path}: {e}") from e

    return pages


def pages_to_dicts(pages: list[ExtractedPage]) -> list[dict[str, Any]]:
    return [
        {
            "page_number": p.page_number,
            "raw_text": p.raw_text,
            "tables": p.tables,
            "table_bboxes": p.table_bboxes,
            "word_count": p.word_count,
            "skipped": p.skipped,
            "error": p.error,
        }
        for p in pages
    ]
