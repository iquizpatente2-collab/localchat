"""
Local web UI for manual Q&A using Ollama (embeddings + chat).

Run from repository root:
  uvicorn web.app:app --host 0.0.0.0 --port 8080

Env:
  OLLAMA_HOST          default http://127.0.0.1:11434
  OLLAMA_EMBED_MODEL   default nomic-embed-text
  OLLAMA_CHAT_MODEL    default qwen3.5:9b
  RAG_TOP_K            default 5
  RAG_RECIPE_NORMALIZE       0|1 — if 1, normalize recipe-like pages via Ollama before chunking/embed
  RAG_RECIPE_NORMALIZE_MODE  auto|all — auto skips index pages and non-recipe text
  RAG_RECIPE_MODEL           optional; defaults to OLLAMA_CHAT_MODEL
  RAG_RECIPE_MAX_PAGE_CHARS  max chars per page sent to the normalizer (default 12000)
  RAG_RECIPE_CONCURRENCY     parallel Ollama /api/chat calls while normalizing (default 12; try 8–16 for GPU)
  RAG_EMBED_CONCURRENCY      parallel embedding requests (default 16)
  OLLAMA_NUM_PARALLEL        set on the Ollama server (e.g. 8) so the daemon accepts enough concurrent jobs
  RAG_RECIPE_TIMEOUT_S       per-page chat timeout (default 300)
  (Startup repair: if chunk cache exists but recipe_store is missing, re-ingest skips LLM
   normalize so the server is not blocked for hours — set RAG_REPAIR_FULL_NORMALIZE=1 to force it.)

  Recipe index (structured + fuzzy + semantic hybrid):
  RECIPE_W_EMBED / RECIPE_W_FUZZY   hybrid weights (default 0.6 / 0.4)
  RECIPE_TOP_K                      top recipes after hybrid rank (default 5)
  RECIPE_CHAT_MAX_TOKENS            LLM budget for /api/recipe-chat (default 600)
  RECIPE_QUERY_SPELLCHECK           1 to enable TextBlob correction (optional: pip install textblob)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import aiohttp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.rag.ingest import extract_pages_cleaned, pages_to_chunks
from web.rag.ollama_rag import ollama_chat, ollama_embed, embed_many
from web.rag.recipe_catalog import (
    FAISS_AVAILABLE,
    RecipeCatalog,
    build_recipe_embeddings_texts,
    expand_query_for_embedding,
    maybe_spell_correct,
)
from web.rag.recipe_normalize import normalize_recipe_pages, page_should_normalize
from web.rag.recipe_prompts import (
    PROMPT_DIRECT_RECIPE,
    PROMPT_EXPLAIN_MATCH,
    PROMPT_SHOW_MATCHING,
    PROMPT_VAGUE,
    format_recipes_for_prompt,
)
from web.rag.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MANUALS_DIR = DATA_DIR / "manuals"
STORE_DIR = DATA_DIR / "rag_store"
RECIPE_STORE_DIR = DATA_DIR / "recipe_store"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = ROOT / "docs"
STATE_PATH = STORE_DIR / "source_state.json"

MANUALS_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)
RECIPE_STORE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen3.5:9b")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
MAX_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "220"))
CHAT_TIMEOUT_S = float(os.environ.get("RAG_CHAT_TIMEOUT_S", "240"))
CHAT_FALLBACK_MODEL = os.environ.get("OLLAMA_CHAT_FALLBACK", "qwen2.5:7b-instruct")
RAG_DOCS_FILE = os.environ.get("RAG_DOCS_FILE", "").strip()
RAG_AUTO_DOCS = os.environ.get("RAG_AUTO_DOCS", "1").strip() not in {"0", "false", "False"}
LEXICAL_K = int(os.environ.get("RAG_LEXICAL_K", "18"))
VECTOR_K = int(os.environ.get("RAG_VECTOR_K", "12"))

RAG_RECIPE_NORMALIZE = os.environ.get("RAG_RECIPE_NORMALIZE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}
RAG_RECIPE_NORMALIZE_MODE = os.environ.get("RAG_RECIPE_NORMALIZE_MODE", "auto").strip().lower()
if RAG_RECIPE_NORMALIZE_MODE not in {"auto", "all"}:
    RAG_RECIPE_NORMALIZE_MODE = "auto"
RAG_RECIPE_MODEL = os.environ.get("RAG_RECIPE_MODEL", "").strip() or CHAT_MODEL
RAG_RECIPE_MAX_PAGE_CHARS = int(os.environ.get("RAG_RECIPE_MAX_PAGE_CHARS", "12000"))
RAG_RECIPE_CONCURRENCY = int(os.environ.get("RAG_RECIPE_CONCURRENCY", "12"))
RAG_RECIPE_TIMEOUT_S = float(os.environ.get("RAG_RECIPE_TIMEOUT_S", "300"))
RAG_REPAIR_FULL_NORMALIZE = os.environ.get("RAG_REPAIR_FULL_NORMALIZE", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

RECIPE_W_EMBED = float(os.environ.get("RECIPE_W_EMBED", "0.6"))
RECIPE_W_FUZZY = float(os.environ.get("RECIPE_W_FUZZY", "0.4"))
RECIPE_TOP_K = int(os.environ.get("RECIPE_TOP_K", "5"))
RECIPE_CHAT_MAX_TOKENS = int(os.environ.get("RECIPE_CHAT_MAX_TOKENS", "600"))
RECIPE_SYSTEM = (
    "You only use the recipes provided in the user message. "
    "Never invent dishes, ingredients, or steps that are not supported by those recipes."
)

RAG_SYSTEM = """You are a manual assistant. Use ONLY the provided manual excerpts to answer.

The source may have OCR or printing typos (e.g. "Pomidoro" vs "Pomodoro") and may give the same recipe in English and Italian (e.g. "TOMATO SAUCE" and "Salsa di …"). Treat those as the same dish when the meaning matches.

If a "Retrieval note" below explains that the manual spells a word differently or uses an English title, you MUST treat that recipe as answering the user's question. Do NOT say the recipe is "not mentioned" solely because one letter differs or the title is in English.

Summarize steps from the best-matching excerpt. If the excerpts are unrelated or insufficient, say so briefly. Stay concise unless the user asks for detail."""

store = VectorStore(STORE_DIR)
recipe_catalog = RecipeCatalog(RECIPE_STORE_DIR)
_store_lock = asyncio.Lock()


def _recipe_query_embedding_ok(qvec: np.ndarray) -> tuple[bool, str]:
    """Ensure live Ollama embeddings match the stored recipe matrix (common failure after model swap)."""
    if recipe_catalog.embeddings is None:
        return False, "recipe embeddings missing"
    idx_dim = int(recipe_catalog.embeddings.shape[1])
    q = np.asarray(qvec, dtype=np.float32).reshape(-1)
    if q.shape[0] != idx_dim:
        return (
            False,
            f"embedding dimension mismatch: query has {q.shape[0]} dims but recipe index has {idx_dim}. "
            f"Re-ingest the PDF (or delete data/recipe_store) so it matches OLLAMA_EMBED_MODEL={EMBED_MODEL!r}.",
        )
    return True, ""


def _file_signature(path: Path) -> dict:
    st = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }


def _load_state() -> dict:
    if not STATE_PATH.exists():
        return {}
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _infer_recipe_mode(q: str) -> str:
    ql = q.lower().strip()
    if re.search(r"\b(why|how come|explain why|reason (these|those|they))\b", ql):
        return "explain"
    if re.search(r"\b(list|show)\s+(all|every)|\ball recipes\b", ql):
        return "list"
    if re.search(r"\b(full|complete|entire)\s+recipe|whole recipe|give me the recipe\b", ql):
        return "direct"
    if re.search(r"\b(something|anything|similar(\s+to)?|recipe ideas?)\b", ql):
        return "vague"
    return "list"


def _recipe_user_prompt(mode: str, query: str, recipes_block: str) -> str:
    if mode == "explain":
        return PROMPT_EXPLAIN_MATCH.format(QUERY=query, RECIPES=recipes_block)
    if mode == "vague":
        return PROMPT_VAGUE.format(QUERY=query, RECIPES=recipes_block)
    if mode == "direct":
        return PROMPT_DIRECT_RECIPE.format(QUERY=query, RECIPES=recipes_block)
    return PROMPT_SHOW_MATCHING.format(QUERY=query, RECIPES=recipes_block)


def _grounded_recipe_answer(query: str, recipe: dict, parts: dict[str, float]) -> str:
    """Deterministic, citation-friendly answer that only uses extracted catalog fields."""
    title = (recipe.get("title") or "Untitled").strip()
    page = recipe.get("page", "?")
    ingredients = [str(x).strip() for x in (recipe.get("ingredients") or []) if str(x).strip()]
    instructions = [str(x).strip() for x in (recipe.get("instructions") or []) if str(x).strip()]
    full_text = (recipe.get("full_text") or "").strip()
    fallback_steps = _fallback_steps_from_prose(full_text)

    lines: list[str] = [
        f"Recipe Name: {title}",
        "",
        "Ingredients:",
    ]
    if ingredients:
        lines.extend(f"- {x}" for x in ingredients)
    else:
        lines.append("- Not clearly extracted from source text")

    lines.append("")
    lines.append("Instructions:")
    if instructions:
        lines.extend(f"{i}. {x}" for i, x in enumerate(instructions, 1))
    elif fallback_steps:
        lines.extend(f"{i}. {x}" for i, x in enumerate(fallback_steps, 1))
    else:
        lines.append("1. Not clearly extracted from source text")

    lines.append("")
    lines.append(
        "Source note: This output is grounded only in extracted PDF text "
        f"(page {page})."
    )
    lines.append(
        f"Match reason for query '{query}': hybrid="
        f"{parts.get('embed', 0.0) * RECIPE_W_EMBED + parts.get('fuzzy', 0.0) * RECIPE_W_FUZZY:.3f} "
        f"(embed={parts.get('embed', 0.0):.3f}, fuzzy={parts.get('fuzzy', 0.0):.3f})."
    )
    if full_text:
        lines.append("")
        lines.append("Extracted Source Text:")
        lines.append(full_text if len(full_text) <= 3200 else full_text[:3200] + "\n[... truncated ...]")
    return "\n".join(lines)


def _fallback_steps_from_prose(full_text: str) -> list[str]:
    """
    If a recipe page is plain prose, split into sentence-like cooking steps.
    Keeps source wording; only light cleanup.
    """
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
        # Skip obvious running headers/boilerplate fragments.
        if re.fullmatch(r"(index|continued|page\s+\d+)", p.lower()):
            continue
        out.append(p)
        if len(out) >= 24:
            break
    return out


def _pick_docs_pdf() -> Path | None:
    if RAG_DOCS_FILE:
        p = Path(RAG_DOCS_FILE)
        if not p.is_absolute():
            p = ROOT / p
        if p.exists() and p.suffix.lower() == ".pdf":
            return p
        return None

    if not DOCS_DIR.exists():
        return None
    pdfs = sorted(DOCS_DIR.glob("*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pdfs[0] if pdfs else None


async def _build_index_from_pdf(
    pdf_path: Path,
    source_name: str,
    *,
    apply_recipe_normalize: bool | None = None,
) -> int:
    """
    apply_recipe_normalize: None = use env RAG_RECIPE_NORMALIZE; False = skip LLM page cleanup.
    """
    pages = extract_pages_cleaned(pdf_path)
    if not pages:
        raise RuntimeError(f"No extractable text in {pdf_path.name}")

    do_normalize = RAG_RECIPE_NORMALIZE if apply_recipe_normalize is None else apply_recipe_normalize
    if do_normalize:
        n_pages = len(pages)
        to_norm = sum(
            1 for _, t in pages if page_should_normalize(t, RAG_RECIPE_NORMALIZE_MODE)
        )
        print(
            f"[RAG] Recipe normalize: mode={RAG_RECIPE_NORMALIZE_MODE!r} "
            f"model={RAG_RECIPE_MODEL!r} parallel={RAG_RECIPE_CONCURRENCY} "
            f"(~{to_norm}/{n_pages} pages) ..."
        )
        async with aiohttp.ClientSession() as session:
            pages = await normalize_recipe_pages(
                session,
                pages,
                chat_fn=ollama_chat,
                model=RAG_RECIPE_MODEL,
                mode=RAG_RECIPE_NORMALIZE_MODE,
                max_chars=RAG_RECIPE_MAX_PAGE_CHARS,
                concurrency=RAG_RECIPE_CONCURRENCY,
                timeout_s=RAG_RECIPE_TIMEOUT_S,
            )

    chunks = pages_to_chunks(pages)
    if not chunks:
        raise RuntimeError(f"No chunks after processing {pdf_path.name}")

    print(f"[RAG] Ingesting {len(chunks)} chunks; embedding with {EMBED_MODEL} ...")
    recipes, recipe_embed_texts = build_recipe_embeddings_texts(pages, source_name)
    async with aiohttp.ClientSession() as session:
        texts = [c["text"] for c in chunks]
        emb = await embed_many(session, texts, EMBED_MODEL)
        recipe_emb = await embed_many(session, recipe_embed_texts, EMBED_MODEL)

    store.set_data(chunks, emb, source_file=source_name)
    store.save()
    recipe_catalog.set_recipes_with_embeddings(recipes, recipe_emb, source_name)
    _save_state(_file_signature(pdf_path))
    print(
        f"[RAG] Index saved: {len(chunks)} vectors; recipe catalog: {len(recipes)} pages"
    )
    return len(chunks)


def _format_context(results: list[tuple[dict, float]]) -> str:
    parts: list[str] = []
    for i, (ch, score) in enumerate(results, 1):
        page = ch.get("page", "?")
        parts.append(
            f"--- Excerpt {i} (page {page}, score {score:.3f}) ---\n{ch['text']}"
        )
    return "\n\n".join(parts)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def _typo_variants(term: str) -> list[str]:
    """Single-character o/i (and a few common OCR) swaps so 'pomodoro' matches 'pomidoro' in text."""
    if len(term) < 4:
        return [term]
    out: set[str] = {term}
    for i, c in enumerate(term):
        if c == "o":
            out.add(term[:i] + "i" + term[i + 1 :])
        elif c == "i":
            out.add(term[:i] + "o" + term[i + 1 :])
    return list(out)[:10]


def _embedding_query_boost(q: str) -> str:
    """Extra phrases for embedding so vector search aligns English titles with Italian queries."""
    ql = q.lower()
    extras: list[str] = []
    if any(x in ql for x in ("pomodoro", "pomidoro", "pomidor")) or (
        "salsa" in ql and "di" in ql
    ):
        extras.extend(
            [
                "tomato sauce",
                "TOMATO SAUCE",
                "Italian tomato sauce",
                "salsa tomato",
            ]
        )
    if "salsa" in ql:
        extras.append("sauce recipe")
    if not extras:
        return q
    return f"{q} {' '.join(extras)}".strip()


# Common OCR / scan variants vs standard spelling (expand lexical search both ways)
_OCR_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "pomodoro": ("pomidoro",),
    "pomidoro": ("pomodoro",),
}


def _expand_weighted_ocr_aliases(
    weighted: list[tuple[str, float]],
) -> list[tuple[str, float]]:
    out = list(weighted)
    have = {t for t, _ in out}
    for t, w in weighted:
        for alt in _OCR_TERM_ALIASES.get(t, ()):
            if alt not in have:
                out.append((alt, w * 0.99))
                have.add(alt)
    return out


def _mirror_ocr_for_embed(q: str) -> str:
    """Second embedding query: match text as the book scanned it (e.g. Pomidoro)."""
    return re.sub(r"\bpomodoro\b", "pomidoro", q, flags=re.IGNORECASE)


def _merge_dual_vector_hits(
    a: list[tuple[dict, float]],
    b: list[tuple[dict, float]],
    limit: int,
) -> list[tuple[dict, float]]:
    """Keep the stronger of two vector scores per chunk (standard vs OCR-mirrored query)."""

    def key_for(ch: dict) -> str:
        return f"{ch.get('page', '?')}|{ch.get('text', '')[:120]}"

    best: dict[str, tuple[dict, float]] = {}
    for ch, sc in a + b:
        k = key_for(ch)
        prev = best.get(k)
        if prev is None or sc > prev[1]:
            best[k] = (ch, sc)
    out = sorted(best.values(), key=lambda x: x[1], reverse=True)
    return out[:limit]


def _llm_spelling_bridge(q: str, hits: list[tuple[dict, float]]) -> str:
    """Short hint so the LLM accepts OCR/title mismatches when retrieval already found the right pages."""
    ql = q.lower()
    blob = "\n".join(h[0].get("text", "") for h in hits[:6]).lower()
    notes: list[str] = []
    if "pomodoro" in ql and "pomidoro" in blob:
        notes.append(
            "The manual spells this dish 'Pomidoro' in Italian; that is the same recipe as standard 'pomodoro'."
        )
    if any(x in ql for x in ("pomodoro", "pomidoro", "salsa")) and "tomato sauce" in blob:
        notes.append(
            "An English heading such as 'TOMATO SAUCE' may name the same recipe the user asked for in Italian."
        )
    if not notes:
        return ""
    return "Retrieval note (trust this for matching titles):\n- " + "\n- ".join(notes) + "\n\n"


def _italian_english_dish_bonus(lo: str, q: str) -> float:
    """When the user names an Italian dish, up-rank chunks whose English title matches (e.g. Tomato Sauce)."""
    ql = q.lower()
    if not any(
        x in ql for x in ("pomodoro", "pomidoro", "salsa di", "salsa ")
    ):
        return 0.0
    b = 0.0
    if "tomato sauce" in lo:
        b += 18.0
    if "tomato" in lo and "sauce" in lo:
        b += 10.0
    return b


# Generic query words that match too many chunks (preface, index, etc.)
_STOP = frozenset(
    {
        "how",
        "what",
        "when",
        "where",
        "why",
        "who",
        "make",
        "give",
        "tell",
        "recipe",
        "from",
        "the",
        "for",
        "and",
        "with",
        "this",
        "that",
        "book",
        "manual",
        "page",
        "about",
        "into",
        "your",
        "some",
        "any",
    }
)


def _query_terms(q: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) >= 3]


def _query_terms_weighted(q: str) -> list[tuple[str, float]]:
    """Stopwords removed; rare/long tokens weighted higher (e.g. napolitaine)."""
    terms = [t for t in _query_terms(q) if t not in _STOP]
    if not terms:
        terms = [t for t in _query_terms(q)]
    out: list[tuple[str, float]] = []
    for t in terms:
        w = 1.0 + max(0, len(t) - 5) * 0.35
        if len(t) >= 8:
            w += 4.0
        out.append((t, w))
    return out


def _compound_phrase_bonus(compact: str, terms: list[str]) -> float:
    """
    OCR often glues titles: MACARONINAPOLITAINE. Reward joined query tokens in order.
    """
    if len(terms) < 2:
        return 0.0
    joined = "".join(_norm(t) for t in terms)
    if len(joined) < 8:
        return 0.0
    if joined in compact:
        return 120.0
    # Pair longest two tokens (dish names)
    by_len = sorted(terms, key=len, reverse=True)[:4]
    for i in range(len(by_len)):
        for j in range(i + 1, len(by_len)):
            pair = _norm(by_len[i]) + _norm(by_len[j])
            if len(pair) >= 10 and pair in compact:
                return 90.0
    return 0.0


def _recipe_step_bonus(text: str) -> float:
    u = text.upper()
    hits = sum(
        x in u
        for x in (
            "SAUCEPAN",
            "TABLESPOON",
            "SIMMER",
            "GRIND",
            "BROWN",
            "DRAIN",
            "BOILING",
            "SALTED WATER",
        )
    )
    return min(2.5, 0.18 * hits)


def _is_index_chunk(text: str) -> bool:
    u = text.upper()
    return "INDEX" in u[:900] or "INDEX,CONTINUED" in u[:1200]


def _catalog_penalty(text: str) -> float:
    """Downrank index / copyright / library boilerplate that steals vector matches."""
    head = text[:2500].upper()
    if "UCSBLIBRARY" in text.upper():
        return 0.06
    if "INDEX" in head and ("CONTINUED" in head or "PAGE" in head):
        return 0.08
    if "COPYRIGHT" in head[:500] and "BY" in head[:500]:
        return 0.12
    if "THE ITALIAN COOKBOOK" in head.replace("\n", " ")[:120]:
        return 0.25
    if "PREFACE" in head[:400]:
        return 0.35
    return 1.0


def _keyword_hits(chunks: list[dict], q: str, top_k: int) -> list[tuple[dict, float]]:
    weighted = _expand_weighted_ocr_aliases(_query_terms_weighted(q))
    raw_terms = [t for t, _ in weighted]
    if not raw_terms:
        return []
    out: list[tuple[dict, float]] = []
    for ch in chunks:
        txt = ch.get("text", "")
        if not txt:
            continue
        lo = txt.lower()
        compact = _norm(txt)
        cb = _compound_phrase_bonus(compact, raw_terms)
        if _is_index_chunk(txt):
            cb *= 0.12
        score = cb
        for t, w in weighted:
            variants = _typo_variants(t)
            term_score = 0.0
            for tv in variants:
                term_score += lo.count(tv) + 0.85 * compact.count(_norm(tv))
            score += w * term_score
        score += _italian_english_dish_bonus(lo, q)
        score += _recipe_step_bonus(txt)
        score *= _catalog_penalty(txt)
        if score > 0:
            out.append((ch, float(score)))
    out.sort(key=lambda x: x[1], reverse=True)
    return out[:top_k]


def _merge_hits(
    vector_hits: list[tuple[dict, float]],
    lexical_hits: list[tuple[dict, float]],
    top_k: int,
) -> list[tuple[dict, float]]:
    merged: dict[str, tuple[dict, float]] = {}

    def key_for(ch: dict) -> str:
        return f"{ch.get('page','?')}|{ch.get('text','')[:120]}"

    for rank, (ch, sc) in enumerate(vector_hits):
        cat = _catalog_penalty(ch.get("text", ""))
        step = _recipe_step_bonus(ch.get("text", ""))
        score = (sc * 2.0) * cat + step + max(0.0, 1.0 - rank * 0.08)
        k = key_for(ch)
        prev = merged.get(k)
        if prev is None or score > prev[1]:
            merged[k] = (ch, score)

    for rank, (ch, sc) in enumerate(lexical_hits):
        score = (sc * 2.8) + max(0.0, 0.9 - rank * 0.05)
        k = key_for(ch)
        prev = merged.get(k)
        if prev is None:
            merged[k] = (ch, score)
        else:
            merged[k] = (prev[0], prev[1] + score)

    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


class ChatBody(BaseModel):
    message: str


class RecipeChatBody(BaseModel):
    message: str
    mode: str = "grounded"


class RecipeRankBody(BaseModel):
    message: str
    top_k: int = 5


def create_app() -> FastAPI:
    app = FastAPI(title="Localchat Manual RAG")

    @app.on_event("startup")
    async def _startup():
        loaded = store.load()
        if loaded:
            print(f"[RAG] Loaded index: {len(store.chunks)} chunks from {store.source_file!r}")
        else:
            print("[RAG] No index yet — upload a PDF manual.")

        if recipe_catalog.load():
            print(
                f"[RAG] Loaded recipe catalog: {len(recipe_catalog.recipes)} recipes "
                f"from {recipe_catalog.source_file!r}"
            )
        else:
            print("[RAG] No recipe catalog on disk — ingest a PDF to build it.")

        if not RAG_AUTO_DOCS:
            return

        docs_pdf = _pick_docs_pdf()
        if docs_pdf is None:
            return

        async with _store_lock:
            sig = _file_signature(docs_pdf)
            prev = _load_state()
            cache_hit = loaded and prev == sig
            rc_ready = bool(recipe_catalog.recipes) and recipe_catalog.embeddings is not None
            if cache_hit and rc_ready:
                print(f"[RAG] Using cached index for docs PDF: {docs_pdf.name}")
                return
            if cache_hit and not rc_ready:
                print(
                    f"[RAG] Cached chunk index OK but recipe catalog missing — "
                    f"re-ingesting {docs_pdf.name} to build recipe_store."
                )
                if RAG_RECIPE_NORMALIZE and not RAG_REPAIR_FULL_NORMALIZE:
                    print(
                        "[RAG] Skipping LLM recipe normalize on this repair run (fast). "
                        "Set RAG_REPAIR_FULL_NORMALIZE=1 to normalize during repair, or upload PDF again."
                    )

            print(f"[RAG] Auto-indexing docs PDF: {docs_pdf}")
            try:
                repair_skip_norm = (
                    cache_hit
                    and not rc_ready
                    and not RAG_REPAIR_FULL_NORMALIZE
                )
                await _build_index_from_pdf(
                    docs_pdf,
                    source_name=docs_pdf.name,
                    apply_recipe_normalize=False if repair_skip_norm else None,
                )
            except Exception as e:
                print(f"[RAG] Auto-index failed: {e}")

    @app.get("/api/health")
    async def health():
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{os.environ.get('OLLAMA_HOST', 'http://127.0.0.1:11434').rstrip('/')}/api/tags"
                ) as resp:
                    ok = resp.status == 200
            return {"ok": ok, "ollama": "reachable" if ok else "error"}
        except Exception as e:
            return {"ok": False, "ollama": str(e)}

    @app.get("/api/status")
    async def status():
        async with _store_lock:
            loaded = bool(store.chunks and store.embeddings is not None)
            n = len(store.chunks) if loaded else 0
            rc_loaded = bool(recipe_catalog.recipes and recipe_catalog.embeddings is not None)
            rec_dim = (
                int(recipe_catalog.embeddings.shape[1])
                if rc_loaded and recipe_catalog.embeddings is not None
                else None
            )
            return {
                "loaded": loaded,
                "chunks": n,
                "source_file": store.source_file,
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
                "recipe_catalog_loaded": rc_loaded,
                "recipe_count": len(recipe_catalog.recipes) if rc_loaded else 0,
                "recipe_source": recipe_catalog.source_file,
                "recipe_embed_dim": rec_dim,
                "recipe_index_backend": "faiss" if FAISS_AVAILABLE else "numpy",
            }

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Please upload a .pdf file")

        dest = MANUALS_DIR / "current_manual.pdf"
        data = await file.read()
        if len(data) > 40 * 1024 * 1024:
            raise HTTPException(400, "File too large (max 40 MB)")

        async with _store_lock:
            dest.write_bytes(data)
            try:
                n = await _build_index_from_pdf(dest, source_name=file.filename)
            except Exception as e:
                raise HTTPException(400, f"Manual indexing failed: {e}") from e

        return {
            "ok": True,
            "chunks": n,
            "filename": file.filename,
        }

    @app.post("/api/chat")
    async def chat(body: ChatBody):
        q = (body.message or "").strip()
        if not q:
            raise HTTPException(400, "message is empty")

        async with _store_lock:
            if not store.chunks or store.embeddings is None:
                raise HTTPException(400, "No manual loaded. Upload a PDF first.")

            try:
                async with aiohttp.ClientSession() as session:
                    q_std = _embedding_query_boost(q)
                    q_ocr = _embedding_query_boost(_mirror_ocr_for_embed(q))
                    emb_std = await ollama_embed(session, q_std, EMBED_MODEL)
                    emb_ocr = await ollama_embed(session, q_ocr, EMBED_MODEL)
                    vk = max(VECTOR_K, TOP_K + 2)
                    v_std = store.search(emb_std, top_k=vk)
                    v_ocr = store.search(emb_ocr, top_k=vk)
                    vector_hits = _merge_dual_vector_hits(v_std, v_ocr, limit=vk)
                    lexical_hits = _keyword_hits(store.chunks, q, top_k=LEXICAL_K)
                    hits = _merge_hits(vector_hits, lexical_hits, top_k=max(TOP_K, 5))
            except Exception as e:
                raise HTTPException(502, f"Retrieval failed: {e}") from e

        if not hits:
            raise HTTPException(500, "Search returned no chunks")

        context = _format_context(hits)
        bridge = _llm_spelling_bridge(q, hits)
        user_content = (
            f"Manual excerpts:\n\n{context}\n\n---\n\n{bridge}User question: {q}"
        )
        messages = [
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user_content},
        ]

        chat_options = {
            "num_predict": MAX_TOKENS,
            "temperature": 0.2,
        }

        primary_err = None
        used_model = CHAT_MODEL
        try:
            async with aiohttp.ClientSession() as session:
                answer = await ollama_chat(
                    session,
                    CHAT_MODEL,
                    messages,
                    stream=False,
                    options=chat_options,
                    timeout_s=CHAT_TIMEOUT_S,
                )
        except Exception as e:
            primary_err = str(e)
            answer = ""

        if not answer and CHAT_FALLBACK_MODEL and CHAT_FALLBACK_MODEL != CHAT_MODEL:
            used_model = CHAT_FALLBACK_MODEL
            try:
                async with aiohttp.ClientSession() as session:
                    answer = await ollama_chat(
                        session,
                        CHAT_FALLBACK_MODEL,
                        messages,
                        stream=False,
                        options=chat_options,
                        timeout_s=CHAT_TIMEOUT_S,
                    )
            except Exception as e:
                raise HTTPException(
                    502,
                    f"Ollama chat failed. Primary ({CHAT_MODEL}): {primary_err}. "
                    f"Fallback ({CHAT_FALLBACK_MODEL}): {e}",
                ) from e

        if not answer:
            raise HTTPException(502, f"Ollama chat failed. Primary ({CHAT_MODEL}): {primary_err}")

        return {
            "answer": answer,
            "model_used": used_model,
            "sources": [
                {"page": h[0].get("page"), "score": round(h[1], 4)}
                for h in hits
            ],
        }

    @app.post("/api/recipes/rank")
    async def recipes_rank(body: RecipeRankBody):
        """Layer 1–2 only: hybrid fuzzy + embedding ranks (no LLM)."""
        q = (body.message or "").strip()
        if not q:
            raise HTTPException(400, "message is empty")
        top_k = max(1, min(20, body.top_k))

        async with _store_lock:
            if not recipe_catalog.recipes or recipe_catalog.embeddings is None:
                raise HTTPException(
                    400,
                    "No recipe catalog. Upload and ingest a PDF first.",
                )
            try:
                async with aiohttp.ClientSession() as session:
                    q_prep = maybe_spell_correct(q)
                    q_embed = expand_query_for_embedding(q_prep)
                    qvec = await ollama_embed(session, q_embed, EMBED_MODEL)
                    ok, why = _recipe_query_embedding_ok(qvec)
                    if not ok:
                        raise HTTPException(502, why)
                    ranked = recipe_catalog.combined_search(
                        q_prep,
                        qvec,
                        top_k=top_k,
                        w_embed=RECIPE_W_EMBED,
                        w_fuzzy=RECIPE_W_FUZZY,
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    502,
                    f"Recipe search failed ({type(e).__name__}): {e!s}. "
                    "Check that Ollama is running and the embed model is loaded.",
                ) from e

        return {
            "query": q,
            "query_used": q_prep,
            "results": [
                {
                    "title": r.get("title"),
                    "page": r.get("page"),
                    "score": round(comb, 5),
                    "embed": round(parts["embed"], 5),
                    "fuzzy": round(parts["fuzzy"], 5),
                    "coverage": round(parts.get("coverage", 0.0), 5),
                }
                for r, comb, parts in ranked
            ],
        }

    @app.post("/api/recipe-chat")
    async def recipe_chat(body: RecipeChatBody):
        """Fuzzy + semantic retrieval, then LLM formatting (layer 3)."""
        q = (body.message or "").strip()
        if not q:
            raise HTTPException(400, "message is empty")
        mode_in = (body.mode or "grounded").strip().lower()
        if mode_in not in {"auto", "grounded", "list", "vague", "explain", "direct"}:
            raise HTTPException(400, "mode must be auto|grounded|list|vague|explain|direct")

        async with _store_lock:
            if not recipe_catalog.recipes or recipe_catalog.embeddings is None:
                raise HTTPException(
                    400,
                    "No recipe catalog. Upload and ingest a PDF first.",
                )
            try:
                async with aiohttp.ClientSession() as session:
                    q_prep = maybe_spell_correct(q)
                    q_embed = expand_query_for_embedding(q_prep)
                    qvec = await ollama_embed(session, q_embed, EMBED_MODEL)
                    ok, why = _recipe_query_embedding_ok(qvec)
                    if not ok:
                        raise HTTPException(502, why)
                    ranked = recipe_catalog.combined_search(
                        q_prep,
                        qvec,
                        top_k=RECIPE_TOP_K,
                        w_embed=RECIPE_W_EMBED,
                        w_fuzzy=RECIPE_W_FUZZY,
                    )
            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(
                    502,
                    f"Recipe retrieval failed ({type(e).__name__}): {e!s}. "
                    "Check that Ollama is running and `ollama pull "
                    f"{EMBED_MODEL}` matches the model used to build data/recipe_store.",
                ) from e

        if not ranked:
            raise HTTPException(500, "No matching recipes")

        mode = "grounded" if mode_in == "auto" else mode_in
        if mode == "grounded":
            best_recipe, _best_score, best_parts = ranked[0]
            answer = _grounded_recipe_answer(q_prep, best_recipe, best_parts)
            return {
                "answer": answer,
                "model_used": "grounded-extractor",
                "mode": mode,
                "matches": [
                    {
                        "title": r.get("title"),
                        "page": r.get("page"),
                        "score": round(comb, 5),
                        "embed": round(parts["embed"], 5),
                        "fuzzy": round(parts["fuzzy"], 5),
                        "coverage": round(parts.get("coverage", 0.0), 5),
                    }
                    for r, comb, parts in ranked
                ],
            }

        recipes_only = [r for r, _, _ in ranked]
        recipes_block = format_recipes_for_prompt(recipes_only)
        user_prompt = _recipe_user_prompt(mode, q_prep, recipes_block)
        messages = [
            {"role": "system", "content": RECIPE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        chat_options = {
            "num_predict": RECIPE_CHAT_MAX_TOKENS,
            "temperature": 0.2,
        }
        primary_err = None
        used_model = CHAT_MODEL
        try:
            async with aiohttp.ClientSession() as session:
                answer = await ollama_chat(
                    session,
                    CHAT_MODEL,
                    messages,
                    stream=False,
                    options=chat_options,
                    timeout_s=CHAT_TIMEOUT_S,
                )
        except Exception as e:
            primary_err = str(e)
            answer = ""

        if not answer and CHAT_FALLBACK_MODEL and CHAT_FALLBACK_MODEL != CHAT_MODEL:
            used_model = CHAT_FALLBACK_MODEL
            try:
                async with aiohttp.ClientSession() as session:
                    answer = await ollama_chat(
                        session,
                        CHAT_FALLBACK_MODEL,
                        messages,
                        stream=False,
                        options=chat_options,
                        timeout_s=CHAT_TIMEOUT_S,
                    )
            except Exception as e:
                raise HTTPException(
                    502,
                    f"Ollama chat failed. Primary ({CHAT_MODEL}): {primary_err}. "
                    f"Fallback ({CHAT_FALLBACK_MODEL}): {e}",
                ) from e

        if not answer:
            raise HTTPException(502, f"Ollama chat failed. Primary ({CHAT_MODEL}): {primary_err}")

        return {
            "answer": answer,
            "model_used": used_model,
            "mode": mode,
            "matches": [
                {
                    "title": r.get("title"),
                    "page": r.get("page"),
                    "score": round(comb, 5),
                    "embed": round(parts["embed"], 5),
                    "fuzzy": round(parts["fuzzy"], 5),
                    "coverage": round(parts.get("coverage", 0.0), 5),
                }
                for r, comb, parts in ranked
            ],
        }

    @app.get("/")
    async def root_page():
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, "Missing static/index.html")
        return FileResponse(index)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
