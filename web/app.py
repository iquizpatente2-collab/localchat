"""
Local web UI for manual Q&A using Ollama (embeddings + chat).

Run from repository root:
  uvicorn web.app:app --host 0.0.0.0 --port 8080

Env:
  OLLAMA_HOST          default http://127.0.0.1:11434
  OLLAMA_EMBED_MODEL   default nomic-embed-text
  OLLAMA_CHAT_MODEL    default qwen3.5:9b
  RAG_TOP_K            default 5
"""
from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path

import aiohttp
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web.rag.ingest import ingest_pdf_chunks
from web.rag.ollama_rag import ollama_chat, ollama_embed, embed_many
from web.rag.store import VectorStore

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MANUALS_DIR = DATA_DIR / "manuals"
STORE_DIR = DATA_DIR / "rag_store"
STATIC_DIR = Path(__file__).resolve().parent / "static"
DOCS_DIR = ROOT / "docs"
STATE_PATH = STORE_DIR / "source_state.json"

MANUALS_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)

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

RAG_SYSTEM = """You are a manual assistant. Use ONLY the provided manual excerpts to answer.

The source may have OCR or printing typos (e.g. "Pomidoro" vs "Pomodoro") and may give the same recipe in English and Italian (e.g. "TOMATO SAUCE" and "Salsa di …"). Treat those as the same dish when the meaning matches.

If a "Retrieval note" below explains that the manual spells a word differently or uses an English title, you MUST treat that recipe as answering the user's question. Do NOT say the recipe is "not mentioned" solely because one letter differs or the title is in English.

Summarize steps from the best-matching excerpt. If the excerpts are unrelated or insufficient, say so briefly. Stay concise unless the user asks for detail."""

store = VectorStore(STORE_DIR)
_store_lock = asyncio.Lock()


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


async def _build_index_from_pdf(pdf_path: Path, source_name: str) -> int:
    chunks = ingest_pdf_chunks(pdf_path)
    if not chunks:
        raise RuntimeError(f"No extractable text in {pdf_path.name}")

    print(f"[RAG] Ingesting {len(chunks)} chunks; embedding with {EMBED_MODEL} ...")
    async with aiohttp.ClientSession() as session:
        texts = [c["text"] for c in chunks]
        emb = await embed_many(session, texts, EMBED_MODEL)

    store.set_data(chunks, emb, source_file=source_name)
    store.save()
    _save_state(_file_signature(pdf_path))
    print(f"[RAG] Index saved: {len(chunks)} vectors")
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


def create_app() -> FastAPI:
    app = FastAPI(title="Localchat Manual RAG")

    @app.on_event("startup")
    async def _startup():
        loaded = store.load()
        if loaded:
            print(f"[RAG] Loaded index: {len(store.chunks)} chunks from {store.source_file!r}")
        else:
            print("[RAG] No index yet — upload a PDF manual.")

        if not RAG_AUTO_DOCS:
            return

        docs_pdf = _pick_docs_pdf()
        if docs_pdf is None:
            return

        async with _store_lock:
            sig = _file_signature(docs_pdf)
            prev = _load_state()
            if loaded and prev == sig:
                print(f"[RAG] Using cached index for docs PDF: {docs_pdf.name}")
                return

            print(f"[RAG] Auto-indexing docs PDF: {docs_pdf}")
            try:
                await _build_index_from_pdf(docs_pdf, source_name=docs_pdf.name)
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
            return {
                "loaded": loaded,
                "chunks": n,
                "source_file": store.source_file,
                "embed_model": EMBED_MODEL,
                "chat_model": CHAT_MODEL,
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

    @app.get("/")
    async def root_page():
        index = STATIC_DIR / "index.html"
        if not index.exists():
            raise HTTPException(500, "Missing static/index.html")
        return FileResponse(index)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


app = create_app()
