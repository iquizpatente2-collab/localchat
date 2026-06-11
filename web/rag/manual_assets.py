"""
Extract figures from PDFs and link them to RAG chunks by page + section (topic).

Phase A: PyMuPDF layout blocks → section heading above figure → topic_id on chunks.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    fitz = None

FIGURE_CAPTION = re.compile(
    r"(?i)^\s*(?:fig(?:ura)?\.?|figure|fig\.)\s*(\d+)\s*[:\-–.]?\s*(.*)$"
)
_SECTION_NUM = re.compile(r"^\s*(\d+(?:\.\d+){0,5})\s+(\S.+)$")


def _slug(text: str, *, max_len: int = 56) -> str:
    t = re.sub(r"[^\w\s\-]", "", (text or "").lower())
    t = re.sub(r"\s+", "-", t.strip())
    return (t[:max_len] or "topic").strip("-") or "topic"


def _is_section_line(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 160:
        return False
    if _SECTION_NUM.match(line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    upper = sum(1 for c in letters if c.isupper())
    return upper / len(letters) > 0.55 and len(line) < 100


def _heading_section_title(line: str) -> str:
    line = line.strip()
    m = _SECTION_NUM.match(line)
    if m:
        return f"{m.group(1)} {m.group(2).strip()}"
    return line


class ManualAssetCatalog:
    """Persisted figure index: catalog.json + PNG files under files/."""

    def __init__(self, dir_path: Path):
        self.dir_path = Path(dir_path)
        self.files_dir = self.dir_path / "files"
        self._catalog_path = self.dir_path / "catalog.json"
        self.assets: list[dict[str, Any]] = []
        self.source_file: str | None = None

    def load(self) -> bool:
        if not self._catalog_path.is_file():
            return False
        try:
            data = json.loads(self._catalog_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        self.assets = list(data.get("assets") or [])
        self.source_file = data.get("source_file")
        return bool(self.assets)

    def save(self) -> None:
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_file": self.source_file,
            "assets": self.assets,
        }
        self._catalog_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def clear(self) -> None:
        self.assets = []
        self.source_file = None
        if self._catalog_path.exists():
            try:
                self._catalog_path.unlink()
            except OSError:
                pass
        if self.files_dir.exists():
            for p in self.files_dir.glob("*.png"):
                try:
                    p.unlink()
                except OSError:
                    pass

    def get(self, asset_id: str) -> dict[str, Any] | None:
        aid = (asset_id or "").strip()
        for a in self.assets:
            if a.get("id") == aid:
                return a
        return None

    def file_path(self, asset_id: str) -> Path | None:
        rec = self.get(asset_id)
        if not rec:
            return None
        rel = rec.get("file")
        if not rel:
            return None
        p = self.files_dir / str(rel)
        return p if p.is_file() else None

    def resolve_for_chunks(self, chunks: list[dict]) -> list[dict[str, Any]]:
        """Unique figures linked to the given store chunks (for chat UI)."""
        if not self.assets:
            return []
        by_id = {a["id"]: a for a in self.assets if a.get("id")}
        ordered: list[str] = []
        for ch in chunks:
            for iid in ch.get("image_ids") or []:
                if iid in by_id and iid not in ordered:
                    ordered.append(iid)
        out: list[dict[str, Any]] = []
        for iid in ordered:
            a = dict(by_id[iid])
            a["url"] = f"/api/manual-asset/{iid}"
            out.append(a)
        return out


def _text_lines_from_block(block: dict[str, Any]) -> list[tuple[str, float]]:
    """(line_text, y0) from a text block."""
    lines: list[tuple[str, float]] = []
    if block.get("type") != 0:
        return lines
    for ln in block.get("lines") or []:
        spans = ln.get("spans") or []
        text = "".join(str(s.get("text") or "") for s in spans).strip()
        if not text:
            continue
        bbox = ln.get("bbox") or block.get("bbox") or (0, 0, 0, 0)
        lines.append((text, float(bbox[1])))
    return lines


def extract_figures_from_pdf(
    pdf_path: Path,
    assets_dir: Path,
    *,
    source_name: str,
    min_px: int = 48,
) -> list[dict[str, Any]]:
    """
    Extract embedded images per page; assign section topic from nearest heading above.
    """
    if fitz is None:
        logger.warning("PyMuPDF not installed — skip figure extraction (pip install pymupdf)")
        return []

    pdf_path = Path(pdf_path).resolve()
    assets_dir = Path(assets_dir)
    files_dir = assets_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    doc = fitz.open(str(pdf_path))

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            page_dict = page.get_text("dict")
            blocks = page_dict.get("blocks") or []

            headings: list[tuple[str, float]] = []
            text_lines: list[tuple[str, float, float]] = []

            for block in blocks:
                for line, y0 in _text_lines_from_block(block):
                    text_lines.append((line, y0, y0))
                    if _is_section_line(line):
                        headings.append((_heading_section_title(line), y0))

            headings.sort(key=lambda x: x[1])
            fig_on_page = 0

            for block in blocks:
                if block.get("type") != 1:
                    continue
                bbox = block.get("bbox") or (0, 0, 0, 0)
                x0, y0, x1, y1 = (float(b) for b in bbox[:4])
                w, h = max(0.0, x1 - x0), max(0.0, y1 - y0)
                if w < min_px or h < min_px:
                    continue

                section = ""
                for title, hy in reversed(headings):
                    if hy <= y0 + 2:
                        section = title
                        break

                caption = ""
                for line, ly, _ in text_lines:
                    if ly >= y1 - 2 and ly <= y1 + 80:
                        cm = FIGURE_CAPTION.match(line)
                        if cm:
                            caption = line.strip()
                            break

                fig_on_page += 1
                topic_slug = _slug(section) if section else f"page-{page_num}"
                topic_id = f"p{page_num:03d}-{topic_slug}-f{fig_on_page:02d}"
                digest = hashlib.sha1(
                    f"{source_name}|{page_num}|{fig_on_page}|{x0:.1f}|{y0:.1f}".encode()
                ).hexdigest()[:10]
                asset_id = f"{topic_id}-{digest}"

                try:
                    img_bytes = page.get_pixmap(clip=fitz.Rect(x0, y0, x1, y1), alpha=False).tobytes(
                        "png"
                    )
                except Exception as e:
                    logger.warning("Figure render failed page {}: {}", page_num, e)
                    continue

                fname = f"{asset_id}.png"
                (files_dir / fname).write_bytes(img_bytes)

                assets.append(
                    {
                        "id": asset_id,
                        "file": fname,
                        "page": page_num,
                        "topic_id": topic_id,
                        "section": section,
                        "caption": caption,
                        "width": int(w),
                        "height": int(h),
                        "bbox": [x0, y0, x1, y1],
                    }
                )
    finally:
        doc.close()

    logger.info("Extracted {} figure(s) from {}", len(assets), pdf_path.name)
    return assets


def _section_matches(chunk_section: str, asset_section: str) -> bool:
    cs = (chunk_section or "").strip().lower()
    ass = (asset_section or "").strip().lower()
    if not ass:
        return True
    if not cs:
        return True
    return cs in ass or ass in cs or cs.split()[0] == ass.split()[0]


def _chunk_gets_figure(ch: dict[str, Any], asset: dict[str, Any]) -> bool:
    ap = int(asset.get("page") or 0)
    page = int(ch.get("page") or 1)
    page_end = int(ch.get("page_end") or page)
    if ap < page or ap > page_end:
        return False
    ass = (asset.get("section") or "").strip()
    if not ass:
        return True
    section = (ch.get("section") or "").strip()
    return _section_matches(section, ass)


def attach_images_to_chunks(
    chunks: list[dict[str, Any]],
    assets: list[dict[str, Any]],
) -> None:
    """Set topic_id and image_ids on each chunk from matching page + section."""
    if not assets:
        return

    for ch in chunks:
        page = int(ch.get("page") or 1)
        subsection = (ch.get("subsection") or "").strip()
        section = (ch.get("section") or "").strip()
        topic_key = subsection or section or f"page-{page}"
        ch["topic_id"] = f"p{page:03d}-{_slug(topic_key)}"

        ids: list[str] = []
        for a in assets:
            if _chunk_gets_figure(ch, a):
                ids.append(str(a["id"]))
        ch["image_ids"] = list(dict.fromkeys(ids))


def extract_and_attach_figures(
    pdf_path: Path,
    assets_dir: Path,
    chunks: list[dict[str, Any]],
    *,
    source_name: str,
    catalog: ManualAssetCatalog | None = None,
) -> ManualAssetCatalog:
    """Extract figures, link to chunks, persist catalog."""
    cat = catalog or ManualAssetCatalog(assets_dir)
    cat.clear()
    cat.source_file = source_name
    assets = extract_figures_from_pdf(pdf_path, assets_dir, source_name=source_name)
    attach_images_to_chunks(chunks, assets)
    cat.assets = assets
    cat.save()
    return cat
