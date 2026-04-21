#!/usr/bin/env python3
"""
CLI orchestrator: extract → classify → serialize tables → chunk → enrich → JSON outputs.
Run from this directory:  python pipeline.py --input ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from .chunker import merge_thin_pages, segments_to_chunks
from .classifier import classify_all
from .enricher import build_page_snippet_map, enrich_chunks
from .extractor import extract_pdf, pages_to_dicts
from .table_serializer import append_tables_to_pages
from .utils import word_count


def _configure_logging(verbose: bool, *, quiet: bool = False) -> None:
    logger.remove()
    if quiet:
        level = "WARNING"
    elif verbose:
        level = "DEBUG"
    else:
        level = "INFO"
    logger.add(sys.stderr, level=level, format="<level>{level}</level> | {message}")


def extract_enriched_pages(
    pdf_path: Path,
    page_start: int | None = None,
    page_end: int | None = None,
) -> list[dict[str, Any]]:
    """
    pdfplumber extract → classify → serialize tables into each page's text_with_tables.
    """
    pdf_path = Path(pdf_path).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    raw_pages = extract_pdf(pdf_path, page_start=page_start, page_end=page_end)
    page_dicts = pages_to_dicts(raw_pages)
    if not page_dicts:
        raise RuntimeError("No pages in selected range.")
    classified = classify_all(page_dicts)
    return append_tables_to_pages(str(pdf_path), classified)


def apply_normalized_pages_to_enriched(
    enriched: list[dict[str, Any]],
    pages: list[tuple[int, str]],
) -> None:
    """After LLM recipe normalize, push text back so chunking sees cleaned pages."""
    by_num = {int(pn): t for pn, t in pages}
    for ep in enriched:
        if ep.get("skipped"):
            continue
        pn = int(ep["page_number"])
        if pn in by_num:
            ep["text_with_tables"] = by_num[pn]
            ep["word_count"] = word_count(by_num[pn])


def finalize_chunks_from_enriched(
    enriched_pages: list[dict[str, Any]],
    doc_name: str,
    *,
    min_words: int,
    max_tokens: int,
    overlap_tokens: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """merge thin pages → chunk → xref/header enrich; build stats."""
    tables_serialized = sum(int(p.get("_tables_serialized_count", 0) or 0) for p in enriched_pages)
    segments, thin_merged = merge_thin_pages(enriched_pages)
    chunks = segments_to_chunks(
        segments,
        source_file=doc_name,
        min_words=min_words,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
    )
    page_map = build_page_snippet_map(enriched_pages)
    chunks_final, xref_n = enrich_chunks(chunks, page_map, doc_name)

    pages_by_type: dict[str, int] = {}
    skipped_nums: list[int] = []
    for p in enriched_pages:
        pt = str(p.get("page_type", "unknown"))
        pages_by_type[pt] = pages_by_type.get(pt, 0) + 1
        if p.get("skipped"):
            skipped_nums.append(int(p["page_number"]))

    wc_list = [int(c["metadata"]["word_count"]) for c in chunks_final]
    avg_w = sum(wc_list) / max(len(wc_list), 1)

    stats: dict[str, Any] = {
        "total_pages": len(enriched_pages),
        "pages_by_type": pages_by_type,
        "total_chunks": len(chunks_final),
        "avg_chunk_words": round(avg_w, 2),
        "thin_pages_merged": thin_merged,
        "tables_serialized": tables_serialized,
        "xrefs_resolved": xref_n,
        "pages_skipped": len(skipped_nums),
        "skipped_page_numbers": sorted(skipped_nums),
    }
    return chunks_final, stats


def run_pipeline_core(
    input_pdf: Path,
    *,
    min_words: int = 80,
    max_tokens: int = 500,
    overlap_tokens: int = 50,
    page_start: int | None = None,
    page_end: int | None = None,
    verbose: bool = False,
    quiet: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    """
    In-memory pipeline for RAG integration.
    Returns (chunks with content+metadata, stats dict, enriched page dicts).
    """
    _configure_logging(verbose, quiet=quiet)
    input_pdf = Path(input_pdf).resolve()
    if not input_pdf.exists():
        raise FileNotFoundError(input_pdf)

    doc_name = input_pdf.name

    try:
        if not quiet:
            logger.info("Extracting PDF with pdfplumber: {}", input_pdf)
        enriched_pages = extract_enriched_pages(input_pdf, page_start=page_start, page_end=page_end)
        chunks_final, stats = finalize_chunks_from_enriched(
            enriched_pages,
            doc_name,
            min_words=min_words,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        return chunks_final, stats, enriched_pages
    except Exception as e:
        logger.exception("Pipeline failed: {}", e)
        raise RuntimeError(f"Pipeline failed: {e}") from e


def run_pipeline(
    input_pdf: Path,
    output_chunks: Path,
    output_stats: Path,
    *,
    min_words: int,
    max_tokens: int,
    overlap_tokens: int,
    page_start: int | None,
    page_end: int | None,
    verbose: bool,
) -> None:
    chunks_final, stats, _ = run_pipeline_core(
        input_pdf,
        min_words=min_words,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        page_start=page_start,
        page_end=page_end,
        verbose=verbose,
        quiet=False,
    )

    output_chunks = Path(output_chunks)
    output_stats = Path(output_stats)
    output_chunks.parent.mkdir(parents=True, exist_ok=True)
    output_stats.parent.mkdir(parents=True, exist_ok=True)

    with open(output_chunks, "w", encoding="utf-8") as f:
        json.dump(chunks_final, f, ensure_ascii=False, indent=2)
    with open(output_stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Wrote {} chunks → {}", len(chunks_final), output_chunks)
    logger.info("Wrote stats → {}", output_stats)


def main() -> None:
    p = argparse.ArgumentParser(description="Production PDF → RAG chunks pipeline (pdfplumber).")
    p.add_argument("--input", required=True, help="Path to input PDF")
    p.add_argument("--output", default="chunks.json", help="Output chunks JSON path")
    p.add_argument("--stats", default="stats.json", help="Output stats JSON path")
    p.add_argument("--min-words", type=int, default=80, help="Minimum chunk size (words); smaller chunks merged")
    p.add_argument("--max-tokens", type=int, default=500, help="Target max tokens per chunk")
    p.add_argument("--overlap", type=int, default=50, help="Token overlap between text chunks (not inside tables/procedures)")
    p.add_argument("--pages", nargs=2, type=int, metavar=("START", "END"), help="1-based inclusive page range (testing)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    ps, pe = None, None
    if args.pages:
        ps, pe = args.pages
        if ps < 1 or pe < ps:
            p.error("Invalid --pages range")

    try:
        run_pipeline(
            Path(args.input),
            Path(args.output),
            Path(args.stats),
            min_words=args.min_words,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap,
            page_start=ps,
            page_end=pe,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
