"""
Optional LLM pass to turn messy PDF recipe text into a consistent structure before RAG chunking.

Controlled by RAG_RECIPE_NORMALIZE in web.app (see env docs there).
"""
from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from typing import Any

RECIPE_NORMALIZE_SYSTEM = """You are a strict information extractor.

Only use the information explicitly present in the text.

Task:
Convert the following old recipe text into structured format.

Rules:
- DO NOT add any ingredient not mentioned
- DO NOT guess quantities
- DO NOT assume missing steps
- If something is unclear, keep it as-is but rewrite clearly
- Preserve original meaning

Output:

Recipe Name:
Ingredients:
Instructions:"""


def build_recipe_normalize_user_message(raw_text: str) -> str:
    return f"Text:\n\n{raw_text}"


def _index_like_page(text: str) -> bool:
    u = text.upper()
    head = u[:2000]
    if "INDEX" in head and ("CONTINUED" in head or re.search(r"\bPAGE\s+\d", head)):
        return True
    if "INDEX,CONTINUED" in u[:1200]:
        return True
    return False


def page_should_normalize(text: str, mode: str) -> bool:
    """mode: 'all' | 'auto'."""
    t = text.strip()
    if len(t) < 60:
        return False
    if _index_like_page(text):
        return False
    if mode == "all":
        return True
    lo = t.lower()
    signals = (
        "ingredient",
        "tablespoon",
        "teaspoon",
        "tbsp",
        "tsp",
        " oz.",
        " oz ",
        " lb.",
        "gram",
        "simmer",
        "boil",
        "saucepan",
        "oven",
        "bake",
        "recipe",
        "salsa",
        "pasta",
        "broth",
        "flour",
        "butter",
        "salted water",
        "table spoon",
    )
    return any(s in lo for s in signals)


def truncate_for_model(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars // 2 - 40
    tail = max_chars - head - 80
    return (
        text[:head]
        + "\n\n[... middle of page omitted for length ...]\n\n"
        + text[-tail:]
    )


ChatFn = Callable[..., Awaitable[str]]


async def normalize_recipe_pages(
    session: Any,
    pages: list[tuple[int, str]],
    *,
    chat_fn: ChatFn,
    model: str,
    mode: str = "auto",
    max_chars: int = 12000,
    concurrency: int = 8,
    timeout_s: float = 240.0,
) -> list[tuple[int, str]]:
    """chat_fn must match web.rag.ollama_rag.ollama_chat signature."""
    sem = asyncio.Semaphore(max(1, concurrency))
    to_run = sum(1 for _, t in pages if page_should_normalize(t, mode))
    lock = asyncio.Lock()
    done_llm = 0

    async def one(page_no: int, raw: str) -> tuple[int, str]:
        nonlocal done_llm
        if not page_should_normalize(raw, mode):
            return (page_no, raw)
        payload = truncate_for_model(raw, max_chars)
        messages = [
            {"role": "system", "content": RECIPE_NORMALIZE_SYSTEM},
            {"role": "user", "content": build_recipe_normalize_user_message(payload)},
        ]
        options = {"num_predict": 4096, "temperature": 0.1}
        async with sem:
            try:
                out = await chat_fn(
                    session,
                    model,
                    messages,
                    stream=False,
                    options=options,
                    timeout_s=timeout_s,
                )
            except Exception as e:
                print(f"[RAG] Recipe normalize page {page_no} failed: {e!s}")
                return (page_no, raw)
        cleaned = (out or "").strip()
        if len(cleaned) < 20:
            return (page_no, raw)
        async with lock:
            done_llm += 1
            if done_llm == 1 or done_llm % 4 == 0 or done_llm >= to_run:
                print(
                    f"[RAG] Recipe normalize progress: {done_llm}/{to_run} "
                    f"(parallelism={concurrency}, last page {page_no})"
                )
        return (page_no, cleaned)

    tasks = [one(pn, txt) for pn, txt in pages]
    return list(await asyncio.gather(*tasks))
