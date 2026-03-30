"""Parse normalized recipe pages into structured fields."""
from __future__ import annotations

import re


def _strip_bullet(s: str) -> str:
    return re.sub(r"^[-*•\u2022]+\s*", "", s.strip())


def _strip_step_num(s: str) -> str:
    return re.sub(r"^\d+[\).\]]\s*", "", s.strip())


def _looks_like_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if re.fullmatch(r"\d+[a-z]?", s.lower()):
        return False
    if len(re.findall(r"[A-Za-z]", s)) < 4:
        return False
    if len(s) > 70:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / len(letters)
    return upper_ratio >= 0.65


def _best_title_fallback(lines: list[str]) -> str:
    cleaned = [x.strip() for x in lines if x.strip()]
    for i, ln in enumerate(cleaned[:20]):
        if not _looks_like_heading(ln):
            continue
        title = ln
        if i + 1 < len(cleaned) and re.fullmatch(r"\([^)]+\)", cleaned[i + 1]):
            title = f"{title} {cleaned[i + 1]}"
        return title[:200]
    for ln in cleaned[:20]:
        if len(ln) >= 6 and len(re.findall(r"[A-Za-z]", ln)) >= 4:
            return ln[:200]
    return ""


def infer_title_from_text(text: str) -> str:
    """Best-effort title extraction from OCR/prose page text."""
    raw = (text or "").strip()
    if not raw:
        return "Untitled"
    title = _best_title_fallback(raw.splitlines())
    return title or "Untitled"


def parse_structured_recipe(text: str) -> dict[str, str | list[str]]:
    """
    Expect sections Recipe Name / Ingredients / Instructions (case-insensitive).
    Falls back to first line as title and full body as full_text only.
    """
    raw = text.strip()
    lines = raw.splitlines()
    section: str | None = None
    title_parts: list[str] = []
    ingredients: list[str] = []
    instructions: list[str] = []

    name_hdr = re.compile(r"(?i)^recipe\s*name:\s*(.*)$")
    ing_hdr = re.compile(r"(?i)^ingredients:\s*(.*)$")
    inst_hdr = re.compile(r"(?i)^instructions:\s*(.*)$")

    for line in lines:
        ls = line.strip()
        if not ls:
            continue

        m = name_hdr.match(ls)
        if m:
            section = "name"
            rest = m.group(1).strip()
            if rest:
                title_parts.append(rest)
            continue
        m = ing_hdr.match(ls)
        if m:
            section = "ing"
            rest = m.group(1).strip()
            if rest:
                ingredients.append(_strip_bullet(rest))
            continue
        m = inst_hdr.match(ls)
        if m:
            section = "inst"
            rest = m.group(1).strip()
            if rest:
                instructions.append(_strip_step_num(rest))
            continue

        if section == "name":
            title_parts.append(ls)
        elif section == "ing":
            ingredients.append(_strip_bullet(ls))
        elif section == "inst":
            instructions.append(_strip_step_num(ls))

    title = " ".join(title_parts).strip()
    if not title:
        title = infer_title_from_text(raw)
    if not title:
        title = "Untitled"

    return {
        "title": title,
        "ingredients": ingredients,
        "instructions": instructions,
        "full_text": raw,
    }


def build_keywords(recipe: dict) -> list[str]:
    """Lightweight keyword list for fuzzy / display (not for inventing content)."""
    seen: set[str] = set()
    out: list[str] = []

    def add(w: str) -> None:
        w = w.strip().lower()
        if len(w) < 2 or w in seen:
            return
        seen.add(w)
        out.append(w)

    for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", recipe.get("title", "")):
        add(w)
    for ing in recipe.get("ingredients") or []:
        if isinstance(ing, str):
            for w in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", ing)[:6]:
                add(w)
    return out[:40]
