"""Match user-reported cooking progress to stored recipe steps; report remaining steps."""
from __future__ import annotations

import re
from typing import Any

from rapidfuzz import fuzz


def fallback_steps_from_prose(full_text: str) -> list[str]:
    """If a recipe page is plain prose, split into sentence-like cooking steps."""
    if not full_text:
        return []
    txt = full_text.replace("\n", " ").strip()
    txt = re.sub(r"\s+", " ", txt)
    parts = re.split(r"(?<=[\.\!\?;])\s+", txt)
    out: list[str] = []
    for p in parts:
        p = p.strip(" -\t")
        if len(p) < 18:
            continue
        if re.fullmatch(r"(index|continued|page\s+\d+)", p.lower()):
            continue
        out.append(p)
        if len(out) >= 24:
            break
    return out


def steps_from_recipe(recipe: dict[str, Any]) -> list[str]:
    raw_instr = recipe.get("instructions") or []
    out: list[str] = []
    for x in raw_instr:
        s = str(x).strip()
        if s:
            out.append(s)
    if out:
        return out
    return fallback_steps_from_prose((recipe.get("full_text") or "").strip())


def split_user_completed_lines(text: str) -> list[str]:
    """Turn user's 'what I did' blob into separate phrases."""
    if not text or not text.strip():
        return []
    lines: list[str] = []
    for block in re.split(r"[\n\r]+", text):
        block = block.strip()
        if not block:
            continue
        for piece in re.split(r"[;•]+", block):
            piece = re.sub(r"^[-*]\s*", "", piece.strip())
            piece = re.sub(r"^\d+[\).\]]\s*", "", piece)
            if len(piece) >= 3:
                lines.append(piece)
    return lines


def extract_completed_from_natural_message(message: str) -> list[str]:
    """
    Extract done-actions from free-form follow-ups that may omit recipe name.
    Examples:
    - "i have added salt and pepper, now what?"
    - "after boiling tomato what next"
    """
    raw = (message or "").strip()
    if not raw:
        return []
    out: list[str] = []

    m_after = re.search(
        r"\bafter\s+(.+?)(?:\s*,?\s*(?:what(?:\s+to\s+do)?\s+next|what\s+next|now\s+what|then)\b|$)",
        raw,
        flags=re.I,
    )
    if m_after:
        done = m_after.group(1).strip(" ,.?")
        if len(done) >= 3:
            out.append(done)

    m_have = re.search(
        r"\bi\s+(?:have|['’]ve)\s+(.+?)(?:\s*,?\s*(?:what(?:\s+to\s+do)?\s+next|what\s+next|now\s+what|then)\b|$)",
        raw,
        flags=re.I,
    )
    if m_have:
        done = m_have.group(1).strip(" ,.?")
        if len(done) >= 3:
            out.append(done)

    if out:
        return split_user_completed_lines("; ".join(out))
    return []


def _strip_progress_trailing_question(s: str) -> str:
    t = s.strip()
    t = re.sub(r"\?+\s*$", "", t)
    t = re.sub(
        r",?\s*\b(what\s*(?:to\s*do\s*)?(?:next|now)\??|whats\s*next|what\s+should\s+i\s+do)\b.*$",
        "",
        t,
        flags=re.I,
    ).strip()
    return t.rstrip(",.")


def _dish_from_leading_phrase(dish_part: str) -> str:
    """Remove leading 'I am making' / 'making' so the query matches the book title."""
    d = dish_part.strip()
    d = re.sub(
        r"^(?:i\s*['']?m\s+making|i\s+am\s+making|making|i\s+want\s+to\s+make|cooking)\s+",
        "",
        d,
        flags=re.I,
    ).strip()
    return d.rstrip(",.")


def _parse_natural_progress_sentence(raw: str) -> tuple[str, str]:
    """
    Handle free-form asks like:
    - "what to add in salsa di pomodoro after boiling tomato"
    - "for risotto con piselli, after browning onion, what next?"
    """
    text = (raw or "").strip()
    if not text:
        return "", ""
    lo = text.lower()

    # Extract completed action from "after <action>".
    done = ""
    m_after = re.search(
        r"\bafter\s+(.+?)(?:\s*,?\s*(?:what(?:\s+to\s+do)?\s+next|now\s+what|what\s+next|then)\b|$)",
        text,
        flags=re.I,
    )
    if m_after:
        done = m_after.group(1).strip(" ,.?")

    # Try to extract recipe from "in/for <dish>".
    dish = ""
    m_dish = re.search(
        r"\b(?:in|for)\s+([a-z][a-z0-9\s'_-]{2,}?)(?:\s+\bafter\b|[,.!?]|$)",
        lo,
        flags=re.I,
    )
    if m_dish:
        dish = m_dish.group(1).strip(" ,.?")

    # Backup: "recipe <dish>" phrase.
    if not dish:
        m_recipe = re.search(r"\brecipe\s+(?:of\s+)?([a-z][a-z0-9\s'_-]{2,})", lo, flags=re.I)
        if m_recipe:
            dish = m_recipe.group(1).strip(" ,.?")

    # Backup: "making/cooking <dish>" without explicit "I have".
    if not dish:
        m_make = re.search(
            r"\b(?:making|cooking)\s+([a-z][a-z0-9\s'_-]{2,}?)(?:\s+\bafter\b|[,.!?]|$)",
            lo,
            flags=re.I,
        )
        if m_make:
            dish = m_make.group(1).strip(" ,.?")

    # If we only found "after ..." but no recipe, use left side as weak dish hint.
    if done and not dish:
        left = lo.split(" after ", 1)[0]
        left = re.sub(
            r"\b(?:what|which|how|should|can|could|would|do|i|add|make|to|next|now|then)\b",
            " ",
            left,
            flags=re.I,
        )
        left = re.sub(r"\s+", " ", left).strip(" ,.?")
        if len(left) >= 3:
            dish = left

    return dish, done


def infer_recipe_focus_query(message: str) -> str:
    """
    Extract likely recipe name from free-form asks, even when no progress steps are given.
    Examples:
    - "i am making risotto milanaise, what ingredients do i need"
    - "ingredients for salsa di pomodoro"
    """
    raw = (message or "").strip()
    if not raw:
        return ""
    lo = raw.lower().strip()

    # "i am making/cooking <dish> ..."
    m_make = re.search(
        r"\b(?:i\s*['']?m\s+making|i\s+am\s+making|making|cooking)\s+(.+?)(?:[,?.!]|$)",
        lo,
        flags=re.I,
    )
    if m_make:
        d = _dish_from_leading_phrase(m_make.group(1))
        d = re.sub(
            r"\b(what|which|how|ingredients?|steps?|method|need|should|do|next)\b.*$",
            "",
            d,
            flags=re.I,
        ).strip(" ,.?")
        if len(d) >= 3:
            return d

    # "ingredients for <dish>", "recipe for <dish>"
    m_for = re.search(
        r"\b(?:ingredients?|recipe|steps?|method)\s+(?:for|of)\s+(.+?)(?:[,?.!]|$)",
        lo,
        flags=re.I,
    )
    if m_for:
        d = m_for.group(1).strip(" ,.?")
        if len(d) >= 3:
            return d

    # Reuse progress parser fallback ("in/for <dish> after <step>")
    dish, _done = _parse_natural_progress_sentence(lo)
    if dish and len(dish) >= 3:
        return dish

    return raw


def split_recipe_progress_message(message: str) -> tuple[str, str]:
    """
    Names the dish + what you already did.

    Supported shapes:
    - ``Recipe: Name`` then body (newline) with completed steps
    - First line = dish name, following lines = completed steps
    - One line: ``… making risotto X, i have browned the onions, what next?``
    """
    raw = (message or "").strip()
    if not raw:
        return "", ""
    m = re.match(r"(?is)^recipe\s*:?\s*(.+?)(?:\n+|\Z)", raw)
    if m:
        recipe_q = m.group(1).strip()
        completed = raw[m.end() :].strip()
        return recipe_q, completed
    if "\n" in raw:
        parts = raw.split("\n", 1)
        return parts[0].strip(), parts[1].strip()

    # Single line: "… dish …, i have / i've …"
    parts = re.split(r",\s*i(?:'ve|\s+have)\s+", raw, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        dish = _dish_from_leading_phrase(parts[0])
        done = _strip_progress_trailing_question(parts[1])
        if dish and done:
            return dish, done

    # "making risotto … i have …" without comma before i have (rare)
    m2 = re.search(
        r"(?:i\s*['']?m\s+making|i\s+am\s+making|making)\s+(.+?)\s+i\s+(?:have|'ve)\s+(.+)$",
        raw,
        flags=re.I,
    )
    if m2:
        dish = _dish_from_leading_phrase(m2.group(1))
        done = _strip_progress_trailing_question(m2.group(2))
        if dish and done:
            return dish, done

    # Natural-language fallback: "what to add in X after Y"
    dish, done = _parse_natural_progress_sentence(raw)
    if dish and done:
        return dish, done

    parts = raw.split("\n", 1)
    recipe_q = parts[0].strip()
    completed = parts[1].strip() if len(parts) > 1 else ""
    return recipe_q, completed


def _compact_alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _line_step_score(user_line: str, step: str) -> float:
    u = user_line.lower().strip()
    s = step.lower().strip()
    if len(u) < 3 or len(s) < 5:
        return 0.0
    a = fuzz.token_set_ratio(u, s) / 100.0
    b = fuzz.partial_ratio(u, s) / 100.0
    c = fuzz.partial_ratio(u, s[: min(len(s), 400)]) / 100.0
    cu, cs = _compact_alnum(u), _compact_alnum(s)
    d = (fuzz.partial_ratio(cu, cs) / 100.0) if len(cu) >= 4 and len(cs) >= 6 else 0.0
    return float(max(a, b, c, d))


def match_completed_steps(
    user_lines: list[str],
    steps: list[str],
    *,
    match_threshold: float = 0.58,
) -> tuple[list[bool], list[tuple[int, float, str]]]:
    """
    Returns (done_mask per step, list of (step_index, best_score, best_user_line) for matched steps).
    A step is 'done' if any user line scores >= match_threshold against it.
    """
    n = len(steps)
    done = [False] * n
    best_for_step: list[tuple[float, str]] = [(0.0, "")] * n
    for i, step in enumerate(steps):
        best_sc = 0.0
        best_ul = ""
        for ul in user_lines:
            sc = _line_step_score(ul, step)
            if sc > best_sc:
                best_sc = sc
                best_ul = ul
        best_for_step[i] = (best_sc, best_ul)
        if best_sc >= match_threshold:
            done[i] = True
    details = [(i, best_for_step[i][0], best_for_step[i][1]) for i in range(n) if done[i]]
    return done, details


def format_progress_answer(
    recipe: dict[str, Any],
    steps: list[str],
    done: list[bool],
    matched_detail: list[tuple[int, float, str]],
) -> str:
    title = (recipe.get("title") or "Recipe").strip()
    page = recipe.get("page", "?")
    lines: list[str] = [
        f"Recipe: {title} (page {page})",
        "",
    ]
    if not steps:
        lines.append(
            "This page has no clear numbered steps in the index. "
            "Use the full recipe text from recipe search mode, or re-ingest with "
            "RAG_RECIPE_NORMALIZE=1 for cleaner structure."
        )
        ft = (recipe.get("full_text") or "").strip()
        if ft:
            lines.append("")
            lines.append("Source text (excerpt):")
            lines.append(ft[:2800] + ("…" if len(ft) > 2800 else ""))
        return "\n".join(lines)

    matched_by_idx = {i: (sc, ul) for i, sc, ul in matched_detail}
    lines.append("Steps matched to what you said you already did:")
    any_m = False
    for i, step in enumerate(steps):
        if not done[i]:
            continue
        any_m = True
        sc, ul = matched_by_idx.get(i, (0.0, ""))
        hint = f' (matched from: "{ul}", score {sc:.2f})' if ul else ""
        lines.append(f"  [done] {i + 1}. {step}{hint}")
    if not any_m:
        lines.append("  (none — we could not confidently line up your list with the book’s steps.)")
        lines.append(
            "  Try rephrasing using words from the recipe, or name smaller actions (e.g. “sautéed onion”)."
        )

    remaining = [s for i, s in enumerate(steps) if not done[i]]
    lines.append("")
    if remaining:
        lines.append("What to do next (remaining steps, in order):")
        first_rem = next((i for i, d in enumerate(done) if not d), None)
        if first_rem is not None:
            lines.append("")
            lines.append(f"Next step: {first_rem + 1}. {steps[first_rem]}")
            lines.append("")
        if len(remaining) > 1:
            lines.append("Full remaining list:")
            for i, s in enumerate(steps):
                if not done[i]:
                    lines.append(f"  {i + 1}. {s}")
    else:
        lines.append(
            "All indexed steps for this page were matched to your list. "
            "If the book has more on the next page, say so or open the PDF."
        )

    lines.append("")
    lines.append(
        "Note: matches are fuzzy (offline). If something looks wrong, compare with the PDF on the cited page."
    )
    return "\n".join(lines)
