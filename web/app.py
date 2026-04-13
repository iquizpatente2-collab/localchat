"""
Local web UI for manual Q&A using Ollama (embeddings + chat).

Run from repository root:
  uvicorn web.app:app --host 0.0.0.0 --port 8080

Env:
  OLLAMA_HOST          default http://127.0.0.1:11434
  OLLAMA_EMBED_MODEL   default nomic-embed-text
  OLLAMA_CHAT_MODEL    default qwen2.5:3b
  RAG_TOP_K            default 5
  RAG_MAX_TOKENS       manual chat output budget (default 480; two-part answers need more room)
  RAG_CHAT_HISTORY_MAX max prior user/assistant turns sent with /api/chat (default 12)
  COMMUNITY_ENABLED       1|0 — user tips in Chroma (default 1)
  COMMUNITY_CHROMA_PATH   persist dir (default data/community_chroma)
  COMMUNITY_QUERY_TOP_K   neighbors to scan (default 8)
  COMMUNITY_MAX_DISTANCE  cosine distance max; lower stricter (default 0.28)
  COMMUNITY_LEXICAL_FILTER 1|0 — require on-topic overlap with saved tip question (default 1)
  COMMUNITY_INJECT_MAX    max tips injected into prompt (default 2)
  WHISPER_STT             1|0 — POST /api/transcribe (default 1 if faster-whisper or openai-whisper installed)
  WHISPER_MODEL           tiny|base|small|medium|large-v3 … (default small)
  WHISPER_DEVICE          auto|cpu|cuda (faster-whisper / openai device pick)
  WHISPER_COMPUTE_TYPE    faster-whisper only, e.g. int8, float16 (default int8 on CPU, float16 on CUDA)
  WHISPER_MAX_UPLOAD_MB   max audio upload size (default 25)
  WHISPER_VAD_FILTER      1|0 — faster-whisper VAD (default 0; VAD often strips browser WebM)
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
  RECIPE_PROGRESS_MATCH             fuzzy threshold 0–1 for /api/recipe-progress (default 0.58)

  Manual /api/chat retrieval:
  RAG_TITLE_PAGE_BOOST     1|0 — if recipe title matches query, pull chunks from that page (default 1)
  RAG_TITLE_MATCH_MIN      fuzzy title score 0–1 to trigger page boost (default 0.78)
  RAG_TITLE_PAGE_MAX_CHUNKS max chunks per matched page to merge in (default 3)
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import aiohttp
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

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
from web.rag.recipe_parse import infer_title_from_text
from web.rag.recipe_progress import (
    extract_completed_from_natural_message,
    fallback_steps_from_prose,
    format_progress_answer,
    infer_recipe_focus_query,
    match_completed_steps,
    split_recipe_progress_message,
    split_user_completed_lines,
    steps_from_recipe,
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
CURRENT_MANUAL_PATH = MANUALS_DIR / "current_manual.pdf"

MANUALS_DIR.mkdir(parents=True, exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)
RECIPE_STORE_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
MAX_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "480"))
RAG_CHAT_HISTORY_MAX = max(0, int(os.environ.get("RAG_CHAT_HISTORY_MAX", "12")))
CHAT_TIMEOUT_S = float(os.environ.get("RAG_CHAT_TIMEOUT_S", "240"))
CHAT_FALLBACK_MODEL = os.environ.get("OLLAMA_CHAT_FALLBACK", "qwen2.5:7b-instruct")
RAG_DOCS_FILE = os.environ.get("RAG_DOCS_FILE", "").strip()
RAG_AUTO_DOCS = os.environ.get("RAG_AUTO_DOCS", "1").strip() not in {"0", "false", "False"}
LEXICAL_K = int(os.environ.get("RAG_LEXICAL_K", "18"))
VECTOR_K = int(os.environ.get("RAG_VECTOR_K", "12"))
RAG_TITLE_PAGE_BOOST = os.environ.get("RAG_TITLE_PAGE_BOOST", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
RAG_TITLE_MATCH_MIN = float(os.environ.get("RAG_TITLE_MATCH_MIN", "0.78"))
RAG_TITLE_PAGE_MAX_CHUNKS = max(1, min(8, int(os.environ.get("RAG_TITLE_PAGE_MAX_CHUNKS", "3"))))

COMMUNITY_ENABLED = os.environ.get("COMMUNITY_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
COMMUNITY_CHROMA_DIR = Path(
    os.environ.get("COMMUNITY_CHROMA_PATH", str(DATA_DIR / "community_chroma"))
).resolve()
COMMUNITY_QUERY_TOP_K = max(1, int(os.environ.get("COMMUNITY_QUERY_TOP_K", "8")))
COMMUNITY_MAX_DISTANCE = float(os.environ.get("COMMUNITY_MAX_DISTANCE", "0.28"))
COMMUNITY_LEXICAL_FILTER = os.environ.get("COMMUNITY_LEXICAL_FILTER", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
COMMUNITY_INJECT_MAX = max(0, int(os.environ.get("COMMUNITY_INJECT_MAX", "2")))
COMMUNITY_SAVE_QUESTION_MAX = 8000
COMMUNITY_SAVE_COMMENT_MAX = 4000
COMMUNITY_SAVE_AUTHOR_MAX = 120
COMMUNITY_SAVE_ANSWER_MAX = 4000

_COMMUNITY_WORD_RE = re.compile(r"[a-zA-ZÀ-ÖØ-öø-ÿ]{3,}", re.UNICODE)
_COMMUNITY_STOPWORDS = frozenset(
    """
    the and for are but not you all can her was one our out day get has him his how its may new now
    old see two way who boy did let put say she too use that this with have from they been than into
    your will just like some what when tell best long each also any both few more most other such than
    them then these those very want make need help tips easy easiest step steps recipe using use used
    just only also into about after again against before being below between both under over while
    una uno per con che non sono alla nel gli dei delle come fare questo questa molto più anche solo
    """.split()
)


def _community_significant_tokens(text: str) -> set[str]:
    raw = {m.group(0).lower() for m in _COMMUNITY_WORD_RE.finditer(text or "")}
    return {t for t in raw if t not in _COMMUNITY_STOPWORDS}


def _community_one_line(s: str, max_len: int) -> str:
    t = " ".join((s or "").split())
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def _community_passes_lexical_gate(user_q: str, match: dict) -> bool:
    """Legacy full-doc overlap (used only when the query has no topic tokens after generic strip)."""
    q_toks = _community_significant_tokens(user_q)
    if not q_toks:
        return True
    doc = f"{match.get('question') or ''} {match.get('answer_excerpt') or ''} {match.get('comment') or ''}"
    d_toks = _community_significant_tokens(doc)
    inter = q_toks & d_toks
    if not inter:
        return False
    longest = max(len(t) for t in inter)
    nq = len(q_toks)
    if nq <= 1:
        return True
    if nq == 2:
        return len(inter) >= 2 or longest >= 4
    return len(inter) >= 2 or longest >= 6


# Stripped before topic matching so unrelated tips (e.g. brown stock vs fried chicken) do not match on generic cooking words.
_COMMUNITY_TOPIC_GENERIC = frozenset(
    """
    water broth salt pepper heat time recipe dish next then what when have has had make made like some your
    this that with from into about after before onion carrot celery garlic butter oil pan pot saucepan simmer
    boil boiled boiling add cup cups tablespoon teaspoons teaspoon minute minutes hour hours step steps deglaze
    deglazing simmering meat veal beef pork lamb sauce stock slice slices piece
    """.split()
)


def _community_topic_tokens(text: str) -> set[str]:
    return _community_significant_tokens(text) - _COMMUNITY_TOPIC_GENERIC


def _community_match_on_topic(user_q: str, match: dict) -> bool:
    """
    Require overlap between the *current* question and the tip's saved question (or comment),
    after removing generic cooking words. Stops unrelated tips from appearing via embedding-only similarity.
    """
    qt = _community_topic_tokens(user_q)
    if not qt:
        return _community_passes_lexical_gate(user_q, match)
    qs = _community_topic_tokens(match.get("question") or "")
    qc = _community_topic_tokens(match.get("comment") or "")
    inter_s = qt & qs
    inter_c = qt & qc
    if inter_s or inter_c:
        strong = {t for t in (inter_s | inter_c) if len(t) >= 5}
        if strong:
            return True
        if len(inter_s) >= 2 or len(inter_c) >= 2:
            return True
        for t in (inter_s | inter_c):
            if len(t) >= 4:
                return True
    return False


def _community_filter_matches(user_q: str, matches: list[dict]) -> list[dict]:
    """Keep tips that are on-topic vs the current question (not just vector-near in 'cooking' space)."""
    if not matches:
        return []
    if not COMMUNITY_LEXICAL_FILTER:
        return matches
    return [m for m in matches if _community_match_on_topic(user_q, m)]

WHISPER_STT_ENABLED = os.environ.get("WHISPER_STT", "1").strip().lower() not in {"0", "false", "no"}
WHISPER_MAX_UPLOAD_BYTES = max(1, int(os.environ.get("WHISPER_MAX_UPLOAD_MB", "25"))) * 1024 * 1024
_WHISPER_AUDIO_SUFFIXES = frozenset(
    {".webm", ".wav", ".mp3", ".mpeg", ".mp4", ".m4a", ".ogg", ".opus", ".flac"}
)

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
RECIPE_PROGRESS_MATCH = float(os.environ.get("RECIPE_PROGRESS_MATCH", "0.58"))
RECIPE_FAST_TITLE_MIN_SCORE = float(os.environ.get("RECIPE_FAST_TITLE_MIN_SCORE", "0.86"))
RECIPE_EMBED_CACHE_SIZE = max(0, int(os.environ.get("RECIPE_EMBED_CACHE_SIZE", "128")))
RECIPE_SESSION_TTL_S = max(60, int(os.environ.get("RECIPE_SESSION_TTL_S", "7200")))
RECIPE_SESSION_MAX = max(16, int(os.environ.get("RECIPE_SESSION_MAX", "300")))
RECIPE_TITLE_MATCH_MIN_SCORE = float(os.environ.get("RECIPE_TITLE_MATCH_MIN_SCORE", "0.72"))
RECIPE_SYSTEM = (
    "You only use the recipes provided in the user message. "
    "Never invent dishes, ingredients, or steps that are not supported by those recipes."
)

RAG_SYSTEM = """You are a manual assistant for a cookbook-style PDF. You will receive prior conversation (if any), then manual excerpts, then the user's latest question.

Answer in **exactly two** labeled parts (do not add a third part):

1) **From the manual** — Only what is supported by the excerpts below. Quote or paraphrase faithfully (ingredients, steps, times, titles). If the excerpts do not contain a recipe or direct answer, say briefly that the manual does not cover it in these passages. Do NOT invent steps or ingredients here. Never cite or imply "the book says" anything that is not in the excerpts.

2) **In my view (not from the manual)** — Short optional suggestions: modern substitutions, sides, safety tips, or variations using general cooking knowledge. This section MUST be clearly separate from part 1. You MUST NOT present this part as coming from the manual. If part 1 already fully answers a narrow factual question with no room for useful extras, you may use one line such as "No strong extras beyond the manual for this question."

**Do not** write any "Community", "other users", or similar section. **Do not** invent tips attributed to people. The application adds verified community tips separately when they exist.

Rules:
- OCR/typos: treat near-matches as the same dish when meaning matches (e.g. English vs Italian titles). The scan may use **souce** instead of **sauce** — treat as the same word.
- If a "Retrieval note" explains spelling/title variants, treat that recipe as matching the user's question.
- **Page numbers:** Only cite pages exactly as shown in the excerpt lines (e.g. `Excerpt 2 (page 29, ...)`). Never invent or guess a page (e.g. do not say "page 34" unless that page appears in an excerpt header you were given).
- **Proteins:** If the user asks about **chicken** or **fried chicken** / **Pollo fritto**, describe steps only from excerpts that actually mention **chicken** (or that Italian title in the same block). Do **not** substitute **lamb**, **veal**, or **mutton** from another excerpt unless the user asked for that meat.
- Use conversation history only for understanding references ("it", "that dish"); facts still come from excerpts in part 1.
- If the user names a specific dish (e.g. Pavese soup), base part 1 on excerpts that name or clearly describe that dish; do not substitute a different recipe (e.g. another bread soup) unless the excerpts provided do not contain the named dish at all — then say the manual excerpts do not show it and answer briefly from what is shown.
- Many recipes have **both** an English and an Italian title (e.g. BROWN STOCK and Sugo di Carne). Treat them as the same dish when the procedure matches; do not say the manual has no match if an excerpt clearly describes that stock/sauce.
- Stay concise unless the user asks for detail."""

store = VectorStore(STORE_DIR)
recipe_catalog = RecipeCatalog(RECIPE_STORE_DIR)
_store_lock = asyncio.Lock()
_recipe_embed_cache: OrderedDict[str, np.ndarray] = OrderedDict()
_recipe_session_ctx: OrderedDict[str, dict] = OrderedDict()
community_store = None


def _init_community_store() -> None:
    global community_store
    community_store = None
    if not COMMUNITY_ENABLED:
        print("[community] disabled (COMMUNITY_ENABLED=0)")
        return
    try:
        from web.rag.community_chroma import CommunitySolutionsStore

        community_store = CommunitySolutionsStore(COMMUNITY_CHROMA_DIR)
        n = community_store.count()
        print(f"[community] Chroma ready at {COMMUNITY_CHROMA_DIR} ({n} saved tip(s))")
    except Exception as e:
        print(f"[community] unavailable: {e}")


def _relative_saved_phrase(ts: int) -> str:
    if ts <= 0:
        return "previously"
    now = int(time.time())
    d = max(0, (now - ts) // 86400)
    if d <= 0:
        return "earlier today"
    if d == 1:
        return "1 day ago"
    if d < 14:
        return f"{d} days ago"
    if d < 60:
        return f"{d // 7} weeks ago"
    if d < 365:
        return f"{max(1, d // 30)} months ago"
    return f"{d // 365} year(s) ago"


def _strip_model_community_section(text: str) -> str:
    """Remove any model-written part 3; community is appended only from the database."""
    t = text or ""
    patterns = (
        r"(?is)\n+\s*3\)\s*\*\*Community\b.*",
        r"(?is)\n+\s*\*\*3\)\s*Community\b.*",
        r"(?is)\n+\s*###\s*3\.\s*Community\b.*",
    )
    for pat in patterns:
        t2 = re.sub(pat, "", t, count=1)
        if t2 != t:
            return t2.rstrip()
    return t.rstrip()


def _format_community_answer_append(matches: list[dict]) -> str:
    """Deterministic part 3 from Chroma only (no LLM)."""
    if not matches:
        return ""
    lines = [
        "",
        "3) **Community (other users, not verified)**",
        "",
    ]
    for m in matches:
        age = _relative_saved_phrase(int(m.get("saved_ts") or 0))
        auth = str(m.get("author") or "").strip() or "Someone"
        comm = (m.get("comment") or "").strip()
        qprev = _community_one_line(m.get("question") or "", 280)
        ansnap = _community_one_line(m.get("answer_excerpt") or "", 520)
        lines.append(f"- **{auth}** ({age})")
        if qprev:
            lines.append(f"  - They had asked: {qprev}")
        if ansnap:
            lines.append(f"  - Assistant reply (snapshot when saved): {ansnap}")
        lines.append(f"  - Tip: {comm if comm else '(no comment text)'}")
    return "\n".join(lines)


def _community_matches_api(matches: list[dict]) -> list[dict]:
    out: list[dict] = []
    for m in matches:
        ts = int(m.get("saved_ts") or 0)
        qpv = (m.get("question") or "")[:280]
        apv = (m.get("answer_excerpt") or "")[:360]
        tip = _community_one_line(m.get("comment") or "", 200)
        tip_line = f"Tip: {tip}" if tip else ""
        title = f"Q: {qpv}" + (f"\n\nA: {apv}" if apv else "") + (f"\n\n{tip_line}" if tip_line else "")
        out.append(
            {
                "author": m.get("author"),
                "age_phrase": _relative_saved_phrase(ts),
                "saved_ts": ts,
                "question_preview": (m.get("question") or "")[:240],
                "answer_preview": (m.get("answer_excerpt") or "")[:320],
                "tip_preview": (m.get("comment") or "")[:200],
                "chip_title": title[:900],
                "distance": round(float(m.get("distance", 0.0)), 4),
            }
        )
    return out


async def _community_lookup_matches(emb_std: np.ndarray) -> list[dict]:
    if community_store is None or COMMUNITY_INJECT_MAX <= 0:
        return []
    try:
        return await asyncio.to_thread(
            community_store.query_similar,
            emb_std,
            **{
                "n_results": COMMUNITY_QUERY_TOP_K,
                "max_distance": COMMUNITY_MAX_DISTANCE,
                "inject_max": COMMUNITY_INJECT_MAX,
            },
        )
    except Exception as e:
        print(f"[community] query failed: {e}")
        return []


def _whisper_lib_available() -> bool:
    try:
        from web.rag import whisper_transcribe as wt

        return wt.detect_backend() is not None
    except Exception:
        return False


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


async def _embed_recipe_query_cached(session: aiohttp.ClientSession, text: str) -> np.ndarray:
    """
    In-memory LRU cache for recipe query embeddings.
    Avoids repeated Ollama embed calls for similar follow-up requests.
    """
    key = f"{EMBED_MODEL}|{(text or '').strip().lower()}"
    if RECIPE_EMBED_CACHE_SIZE > 0 and key in _recipe_embed_cache:
        _recipe_embed_cache.move_to_end(key)
        return _recipe_embed_cache[key]
    qvec = await ollama_embed(session, text, EMBED_MODEL)
    arr = np.asarray(qvec, dtype=np.float32).reshape(-1)
    if RECIPE_EMBED_CACHE_SIZE > 0:
        _recipe_embed_cache[key] = arr
        _recipe_embed_cache.move_to_end(key)
        while len(_recipe_embed_cache) > RECIPE_EMBED_CACHE_SIZE:
            _recipe_embed_cache.popitem(last=False)
    return arr


def _prune_recipe_session_ctx(now_s: float | None = None) -> None:
    now = time.time() if now_s is None else now_s
    stale_keys = [
        sid
        for sid, ctx in _recipe_session_ctx.items()
        if float(ctx.get("updated_at", 0.0)) < (now - RECIPE_SESSION_TTL_S)
    ]
    for sid in stale_keys:
        _recipe_session_ctx.pop(sid, None)
    while len(_recipe_session_ctx) > RECIPE_SESSION_MAX:
        _recipe_session_ctx.popitem(last=False)


def _session_get_recipe_ctx(session_id: str) -> dict | None:
    if not session_id:
        return None
    _prune_recipe_session_ctx()
    ctx = _recipe_session_ctx.get(session_id)
    if ctx is not None:
        _recipe_session_ctx.move_to_end(session_id)
    return ctx


def _session_set_recipe_ctx(
    session_id: str,
    *,
    recipe_query: str,
    recipe_title: str | None,
    page: int | None,
) -> None:
    if not session_id:
        return
    now = time.time()
    _recipe_session_ctx[session_id] = {
        "recipe_query": (recipe_query or "").strip(),
        "recipe_title": (recipe_title or "").strip(),
        "page": page,
        "updated_at": now,
    }
    _recipe_session_ctx.move_to_end(session_id)
    _prune_recipe_session_ctx(now)


def _recipe_from_session_ctx(ctx: dict | None) -> dict | None:
    """Resolve previously matched recipe record from in-memory session context."""
    if not ctx:
        return None
    page = ctx.get("page")
    title = str(ctx.get("recipe_title") or "").strip().lower()
    if page is None and not title:
        return None
    for r in recipe_catalog.recipes:
        rp = r.get("page")
        rt = str(r.get("title") or "").strip().lower()
        if page is not None and rp == page:
            return r
        if title and rt and rt == title:
            return r
    return None


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


def _clear_runtime_indexes() -> None:
    """Remove persisted RAG/recipe index artifacts and reset in-memory stores."""
    for p in (
        STORE_DIR / "meta.json",
        STORE_DIR / "embeddings.npy",
        STATE_PATH,
        RECIPE_STORE_DIR / "recipes.json",
        RECIPE_STORE_DIR / "recipe_embeddings.npy",
    ):
        try:
            if p.exists():
                p.unlink()
        except Exception:
            pass
    store.chunks = []
    store.embeddings = None
    store.source_file = None
    recipe_catalog.recipes = []
    recipe_catalog.embeddings = None
    recipe_catalog.source_file = None
    recipe_catalog._faiss_index = None
    _recipe_embed_cache.clear()


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
    title = (recipe.get("title") or "").strip()
    page = recipe.get("page", "?")
    ingredients = [str(x).strip() for x in (recipe.get("ingredients") or []) if str(x).strip()]
    instructions = [str(x).strip() for x in (recipe.get("instructions") or []) if str(x).strip()]
    full_text = (recipe.get("full_text") or "").strip()
    if len(title) < 4 or len(re.findall(r"[A-Za-z]", title)) < 3:
        title = infer_title_from_text(full_text)
    fallback_steps = fallback_steps_from_prose(full_text)

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
    cov = parts.get("coverage")
    bg = parts.get("bigram")
    tail = f"(embed={parts.get('embed', 0.0):.3f}, fuzzy={parts.get('fuzzy', 0.0):.3f}"
    if cov is not None:
        tail += f", coverage={float(cov):.3f}"
    if bg is not None:
        tail += f", phrase={float(bg):.3f}"
    tail += ")."
    lines.append(
        f"Match reason for query '{query}': hybrid="
        f"{parts.get('embed', 0.0) * RECIPE_W_EMBED + parts.get('fuzzy', 0.0) * RECIPE_W_FUZZY:.3f} "
        + tail
    )
    if full_text:
        lines.append("")
        lines.append("Extracted Source Text:")
        lines.append(full_text if len(full_text) <= 3200 else full_text[:3200] + "\n[... truncated ...]")
    return "\n".join(lines)


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
    if "pavese" in ql:
        extras.extend(
            [
                "Zuppa alla Pavese",
                "PAVESE SOUP",
                "pavese soup",
                "toasted bread butter poached eggs broth parmesan",
            ]
        )
    if ("brown" in ql and "stock" in ql) or "brownstock" in ql.replace(" ", ""):
        extras.extend(
            [
                "BROWN STOCK",
                "Sugo di Carne",
                "beef stock bones onion carrot celery simmer strain",
            ]
        )
    if "sugo" in ql and "carne" in ql:
        extras.extend(
            [
                "BROWN STOCK",
                "Sugo di Carne",
                "brown stock meat broth",
            ]
        )
    if ("fried" in ql and "chicken" in ql) or ("pollo" in ql and "fritto" in ql):
        extras.extend(
            [
                "FRIED CHICKEN",
                "Pollo fritto",
                "POLLO FRITTO",
                "spring chicken flour egg bread crumbs oil",
                "chicken fried in oil boil then fry",
            ]
        )
    if "polenta" in ql:
        extras.extend(
            [
                "POLENTA PIE",
                "Polenta Pasticciata",
                "polenta pasticciata cornmeal mush milk carrot onion celery bacon",
            ]
        )
    if not extras:
        return q
    return f"{q} {' '.join(extras)}".strip()


# Common OCR / scan variants vs standard spelling (expand lexical search both ways)
_OCR_TERM_ALIASES: dict[str, tuple[str, ...]] = {
    "pomodoro": ("pomidoro",),
    "pomidoro": ("pomodoro",),
    "sauce": ("souce",),
    "souce": ("sauce",),
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
    """Second embedding query: match text as the book scanned it (e.g. Pomidoro, Polio)."""
    q = re.sub(r"\bpomodoro\b", "pomidoro", q, flags=re.IGNORECASE)
    q = re.sub(r"\bpollo\b", "polio", q, flags=re.IGNORECASE)
    return q


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
    if "pavese" in ql and "pavese" in blob:
        notes.append(
            "The user asked about Pavese soup; excerpts that name 'Pavese' or 'Zuppa alla Pavese' are the primary source for that dish."
        )
    stock_ask = ("brown" in ql and "stock" in ql) or ("sugo" in ql and "carne" in ql)
    stock_blob = ("brown" in blob and "stock" in blob) or (
        "sugo" in blob and "carne" in blob
    )
    if stock_ask and stock_blob:
        notes.append(
            "In this manual, **BROWN STOCK** and **Sugo di Carne** are the same recipe (English and Italian titles). "
            "If an excerpt describes browning meat with onion/carrot/celery, adding water, and long simmering/straining, "
            "that is the user's recipe even when their words differ slightly from the heading."
        )
    if ("fried" in ql and "chicken" in ql) or ("pollo" in ql and "fritto" in ql):
        if ("fried" in blob and "chicken" in blob) or ("pollo" in blob and "fritto" in blob) or (
            "polio" in blob and "fritto" in blob
        ):
            notes.append(
                "The manual may title this **FRIED CHICKEN** or **Pollo fritto** (OCR sometimes reads 'Polio fritto'); "
                "those name the same dish as the user's question. Use only the excerpt that describes **chicken** — "
                "do not mix in lamb, veal, or mutton from a different fried-meat recipe."
            )
    if "polenta" in ql and "polenta" in blob:
        notes.append(
            "The book often titles this **POLENTA PIE** with **(Polenta Pasticciata)**; vegetables and fillings "
            "may appear in the same excerpt (e.g. carrot, onion, celery, bacon, Bolognese). "
            "The scan may spell **souce** instead of **sauce**."
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


def _named_dish_token_bonus(lo: str, ql: str) -> float:
    """Large bonus when a distinctive dish token in the query appears in the chunk (disambiguates generic words like 'bread')."""
    bonus = 0.0
    compact = _norm(lo)
    if "pavese" in ql and "pavese" in lo:
        bonus += 58.0
    # Italian Cook Book: same recipe, English + Italian titles (often split across lines in OCR).
    brown_stock_q = ("brown" in ql and "stock" in ql) or "brownstock" in ql.replace(" ", "")
    sugo_carne_q = ("sugo" in ql and "carne" in ql) or "sugo di carne" in ql
    if brown_stock_q:
        if "brownstock" in compact or ("brown" in lo and "stock" in lo):
            bonus += 64.0
        if "sugo" in lo and "carne" in lo:
            bonus += 52.0
    if sugo_carne_q and not brown_stock_q:
        if "sugo di carne" in lo or ("sugo" in lo and "carne" in lo):
            bonus += 56.0
        if "brown" in lo and "stock" in lo:
            bonus += 50.0
    if ("fried" in ql and "chicken" in ql) or ("pollo" in ql and "fritto" in ql):
        if ("fried" in lo and "chicken" in lo) or "fried chicken" in lo:
            bonus += 62.0
        if "pollo" in lo and "fritto" in lo:
            bonus += 62.0
        if "polio" in lo and "fritto" in lo:
            bonus += 58.0
    if "polenta" in ql:
        if "polenta" in lo:
            bonus += 56.0
        if "pasticciata" in ql or "pie" in ql:
            if "pasticciata" in lo or "polenta pie" in lo or ("polenta" in lo and "pie" in lo):
                bonus += 48.0
    return bonus


def _fried_chicken_chunk_score_adj(lo: str, ql: str) -> float:
    """
    Prefer excerpts that are actually chicken/pollo fritto; demote other 'fritto' meat recipes (e.g. lamb)
    when the user asked for fried chicken.
    """
    ql = ql.lower()
    if not (("fried" in ql and "chicken" in ql) or ("pollo" in ql and "fritto" in ql)):
        return 0.0
    lo = lo.lower()
    adj = 0.0
    if (
        "chicken" in lo
        or "spring chicken" in lo
        or ("fried" in lo and "chicken" in lo)
        or ("pollo" in lo and "fritto" in lo)
        or ("polio" in lo and "fritto" in lo)
    ):
        adj += 32.0
    if "lamb" in lo and "chicken" not in lo and "pollo" not in lo and "spring" not in lo:
        adj -= 52.0
    if "veal" in lo and "chicken" not in lo and "pollo" not in lo:
        adj -= 44.0
    if "mutton" in lo and "chicken" not in lo and "pollo" not in lo:
        adj -= 40.0
    return adj


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
        score += _named_dish_token_bonus(lo, q.lower())
        score += _fried_chicken_chunk_score_adj(lo, q.lower())
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
    *,
    query: str = "",
) -> list[tuple[dict, float]]:
    merged: dict[str, tuple[dict, float]] = {}
    ql = (query or "").lower()

    def key_for(ch: dict) -> str:
        return f"{ch.get('page','?')}|{ch.get('text','')[:120]}"

    for rank, (ch, sc) in enumerate(vector_hits):
        cat = _catalog_penalty(ch.get("text", ""))
        step = _recipe_step_bonus(ch.get("text", ""))
        lo = (ch.get("text") or "").lower()
        dish = (
            _named_dish_token_bonus(lo, ql)
            + _italian_english_dish_bonus(lo, query)
            + _fried_chicken_chunk_score_adj(lo, ql)
        )
        score = (sc * 2.0) * cat + step + dish + max(0.0, 1.0 - rank * 0.08)
        k = key_for(ch)
        prev = merged.get(k)
        if prev is None or score > prev[1]:
            merged[k] = (ch, score)

    for rank, (ch, sc) in enumerate(lexical_hits):
        lo_lex = (ch.get("text") or "").lower()
        score = (sc * 2.8) + _fried_chicken_chunk_score_adj(lo_lex, ql) + max(0.0, 0.9 - rank * 0.05)
        k = key_for(ch)
        prev = merged.get(k)
        if prev is None:
            merged[k] = (ch, score)
        else:
            merged[k] = (prev[0], prev[1] + score)

    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _recipe_title_page_boost(
    q: str,
    hits: list[tuple[dict, float]],
    top_k: int,
) -> list[tuple[dict, float]]:
    """
    When the structured recipe index finds a strong title match, merge in chunks from that page.
    Stops vector search from returning only a generic 'bread' recipe when the user named a specific dish.
    """
    if not RAG_TITLE_PAGE_BOOST or not recipe_catalog.recipes or not store.chunks:
        return hits
    qstr = (q or "").strip()
    if len(qstr) < 5:
        return hits
    try:
        ranked = recipe_catalog.fast_title_search(qstr, top_k=2, min_score=RAG_TITLE_MATCH_MIN)
    except Exception:
        return hits
    if not ranked:
        return hits
    r, title_sc, _ = ranked[0]
    page = r.get("page")
    if page is None:
        return hits
    ql = qstr.lower()
    rare = [w for w in re.findall(r"[a-zà-öø-ÿ]{4,}", ql) if w not in _STOP]
    if "pavese" in ql and "pavese" not in rare:
        rare.append("pavese")

    def score_ch(ch: dict) -> float:
        t = (ch.get("text") or "").lower()
        return float(sum(t.count(w) for w in rare))

    same_page = [ch for ch in store.chunks if ch.get("page") == page]
    if not same_page:
        return hits
    same_page.sort(key=score_ch, reverse=True)
    inject = same_page[:RAG_TITLE_PAGE_MAX_CHUNKS]

    def kf(ch: dict) -> str:
        return f"{ch.get('page', '?')}|{ch.get('text', '')[:140]}"

    merged: dict[str, tuple[dict, float]] = {kf(h[0]): h for h in hits}
    inj = 14.0 + float(title_sc) * 30.0
    for ch in inject:
        kk = kf(ch)
        if kk not in merged:
            merged[kk] = (ch, inj)
        else:
            c0, s0 = merged[kk]
            merged[kk] = (c0, s0 + inj * 0.4)
    out = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return out[:top_k]


class ChatHistoryTurn(BaseModel):
    role: str
    content: str


class ChatBody(BaseModel):
    message: str
    session_id: str = ""
    history: list[ChatHistoryTurn] = Field(default_factory=list)


class CommunitySaveBody(BaseModel):
    question: str
    comment: str
    author: str = ""
    answer_snapshot: str = ""


class RecipeChatBody(BaseModel):
    message: str
    mode: str = "grounded"
    session_id: str = ""


class RecipeRankBody(BaseModel):
    message: str
    top_k: int = 5
    session_id: str = ""


class RecipeProgressBody(BaseModel):
    """First line or 'Recipe: …' = dish name; following lines = what you already did."""

    message: str
    session_id: str = ""


def _sanitize_manual_history(turns: list[ChatHistoryTurn]) -> list[dict[str, str]]:
    """Keep last N user/assistant turns for chat context (bounded size)."""
    max_n = RAG_CHAT_HISTORY_MAX if RAG_CHAT_HISTORY_MAX > 0 else 12
    per_msg = 4000
    out: list[dict[str, str]] = []
    for t in turns:
        role = str(t.role or "").strip().lower()
        content = str(t.content or "").strip()
        if role not in ("user", "assistant") or not content:
            continue
        out.append({"role": role, "content": content[:per_msg]})
    return out[-max_n:]


def _retrieval_query_from_history(history: list[dict[str, str]], q: str) -> str:
    """Blend recent turns into the embedding query so vague follow-ups stay on-topic."""
    if not history:
        return q
    tail = history[-6:]
    parts = [h["content"][:700] for h in tail]
    parts.append(q)
    return " \n".join(parts)[:3000]


def _lexical_query_from_history(history: list[dict[str, str]], q: str) -> str:
    """Keyword search: current question plus last user line (reduces noise vs full assistant text)."""
    if not history:
        return q
    last_user = ""
    for h in reversed(history):
        if h["role"] == "user":
            last_user = (h.get("content") or "").strip()[:500]
            break
    if not last_user:
        return q
    return f"{last_user}\n{q}"[:2000]


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

        _init_community_store()

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
                "community_enabled": bool(community_store is not None and COMMUNITY_ENABLED),
                "community_tips": int(community_store.count()) if community_store else 0,
                "whisper_stt_available": bool(WHISPER_STT_ENABLED and _whisper_lib_available()),
            }

    @app.post("/api/transcribe")
    async def transcribe_audio(
        file: UploadFile = File(...),
        language: str = Form(""),
    ):
        """Upload short audio; transcribe with local Whisper (faster-whisper or openai-whisper)."""
        if not WHISPER_STT_ENABLED:
            raise HTTPException(404, "Whisper STT is disabled (WHISPER_STT=0).")
        from web.rag import whisper_transcribe as wt

        if wt.detect_backend() is None:
            raise HTTPException(
                503,
                "Whisper not installed. Run: pip install faster-whisper  (or: pip install openai-whisper)",
            )
        raw = await file.read()
        if len(raw) > WHISPER_MAX_UPLOAD_BYTES:
            mb = WHISPER_MAX_UPLOAD_BYTES // (1024 * 1024)
            raise HTTPException(400, f"Audio too large (max {mb} MB).")
        if len(raw) < 200:
            raise HTTPException(400, "Audio file too small.")
        suffix = Path(file.filename or "rec.webm").suffix.lower()
        if suffix not in _WHISPER_AUDIO_SUFFIXES:
            suffix = ".webm"
        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(raw)
                tmp_path = Path(tmp.name)
            lang_in = (language or "").strip().lower()
            if lang_in in ("en-us", "english"):
                lang = "en"
            elif lang_in in ("it-it", "italian", "italiano"):
                lang = "it"
            elif not lang_in:
                lang = None
            else:
                lang = lang_in[:2]
            text, backend = await asyncio.to_thread(wt.transcribe_file, tmp_path, lang)
        except Exception as e:
            raise HTTPException(502, f"Transcription failed: {e}") from e
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
        return {"text": text, "backend": backend}

    @app.post("/api/upload")
    async def upload(file: UploadFile = File(...)):
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise HTTPException(400, "Please upload a .pdf file")

        dest = CURRENT_MANUAL_PATH
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

    @app.delete("/api/upload")
    async def remove_uploaded_manual():
        """Remove manually uploaded PDF from manuals/current_manual.pdf (does not change active index)."""
        try:
            existed = CURRENT_MANUAL_PATH.exists()
            if existed:
                CURRENT_MANUAL_PATH.unlink()
            return {
                "ok": True,
                "removed": existed,
                "path": str(CURRENT_MANUAL_PATH.relative_to(ROOT)),
            }
        except Exception as e:
            raise HTTPException(500, f"Failed to remove uploaded manual: {e}") from e

    @app.post("/api/use-docs")
    async def use_docs_pdf():
        """Rebuild active indexes from docs PDF and make docs the source of truth."""
        docs_pdf = _pick_docs_pdf()
        if docs_pdf is None:
            raise HTTPException(400, "No docs PDF found. Put a .pdf file in docs/ or set RAG_DOCS_FILE.")

        async with _store_lock:
            try:
                _clear_runtime_indexes()
                n = await _build_index_from_pdf(docs_pdf, source_name=docs_pdf.name)
            except Exception as e:
                raise HTTPException(500, f"Docs reindex failed: {e}") from e
        return {"ok": True, "chunks": n, "source": docs_pdf.name}

    @app.post("/api/chat")
    async def chat(body: ChatBody):
        q = (body.message or "").strip()
        if not q:
            raise HTTPException(400, "message is empty")

        hist = _sanitize_manual_history(body.history)
        q_embed = _retrieval_query_from_history(hist, q)
        q_lex = _lexical_query_from_history(hist, q)
        comm_matches: list[dict] = []

        async with _store_lock:
            if not store.chunks or store.embeddings is None:
                raise HTTPException(400, "No manual loaded. Upload a PDF first.")

            try:
                async with aiohttp.ClientSession() as session:
                    q_std = _embedding_query_boost(q_embed)
                    q_ocr = _embedding_query_boost(_mirror_ocr_for_embed(q_embed))
                    emb_std = await ollama_embed(session, q_std, EMBED_MODEL)
                    emb_ocr_coro = ollama_embed(session, q_ocr, EMBED_MODEL)
                    comm_coro = _community_lookup_matches(emb_std)
                    emb_ocr, comm_matches = await asyncio.gather(emb_ocr_coro, comm_coro)
                    comm_matches = _community_filter_matches(q, comm_matches)
                    vk = max(VECTOR_K, TOP_K + 2)
                    v_std = store.search(emb_std, top_k=vk)
                    v_ocr = store.search(emb_ocr, top_k=vk)
                    vector_hits = _merge_dual_vector_hits(v_std, v_ocr, limit=vk)
                    lexical_hits = _keyword_hits(store.chunks, q_lex, top_k=LEXICAL_K)
                    hits = _merge_hits(
                        vector_hits,
                        lexical_hits,
                        top_k=max(TOP_K, 5),
                        query=q,
                    )
                    hits = _recipe_title_page_boost(q, hits, top_k=max(TOP_K, 5))
            except Exception as e:
                raise HTTPException(502, f"Retrieval failed: {e}") from e

        if not hits:
            raise HTTPException(500, "Search returned no chunks")

        context = _format_context(hits)
        bridge = _llm_spelling_bridge(q, hits)
        # Community tips are not shown to the LLM — appended after generation from DB only.
        parts = [f"Manual excerpts:\n\n{context}"]
        parts.append(f"---\n\n{bridge}User question: {q}")
        user_content = "\n\n".join(parts)
        messages: list[dict[str, str]] = [{"role": "system", "content": RAG_SYSTEM}]
        for h in hist:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})

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

        answer = _strip_model_community_section(answer)
        if comm_matches:
            answer = answer.rstrip() + _format_community_answer_append(comm_matches)

        return {
            "answer": answer,
            "model_used": used_model,
            "sources": [
                {"page": h[0].get("page"), "score": round(h[1], 4)}
                for h in hits
            ],
            "community_matches": _community_matches_api(comm_matches),
        }

    @app.post("/api/community-save")
    async def community_save(body: CommunitySaveBody):
        if community_store is None:
            raise HTTPException(
                503,
                "Community tips are disabled or Chroma is unavailable. "
                "Install chromadb (pip install chromadb) and set COMMUNITY_ENABLED=1.",
            )
        q = (body.question or "").strip()
        c = (body.comment or "").strip()
        if not c:
            raise HTTPException(400, "comment is empty")
        if len(q) > COMMUNITY_SAVE_QUESTION_MAX:
            raise HTTPException(400, f"question too long (max {COMMUNITY_SAVE_QUESTION_MAX})")
        if len(c) > COMMUNITY_SAVE_COMMENT_MAX:
            raise HTTPException(400, f"comment too long (max {COMMUNITY_SAVE_COMMENT_MAX})")
        author = (body.author or "").strip()[:COMMUNITY_SAVE_AUTHOR_MAX]
        ans = (body.answer_snapshot or "").strip()[:COMMUNITY_SAVE_ANSWER_MAX]
        try:
            async with aiohttp.ClientSession() as session:
                q_boost = _embedding_query_boost(q[:COMMUNITY_SAVE_QUESTION_MAX])
                emb = await ollama_embed(session, q_boost, EMBED_MODEL)
            rid = await asyncio.to_thread(
                lambda: community_store.add_tip(
                    question=q[:COMMUNITY_SAVE_QUESTION_MAX],
                    embedding=emb,
                    comment=c[:COMMUNITY_SAVE_COMMENT_MAX],
                    author=author,
                    answer_excerpt=ans,
                )
            )
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(502, f"Failed to save community tip: {e}") from e
        return {"ok": True, "id": rid}

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
                    qvec = await _embed_recipe_query_cached(session, q_embed)
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

    @app.post("/api/recipe-progress")
    async def recipe_progress(body: RecipeProgressBody):
        """
        Offline 'what's next': find recipe by hybrid search, fuzzy-match user lines to indexed steps.
        Message format: first line = recipe name (or 'Recipe: Name'); following lines = completed work.
        """
        message = (body.message or "").strip()
        session_id = (body.session_id or "").strip()
        if not message:
            raise HTTPException(400, "message is empty")
        recipe_q, completed_raw = split_recipe_progress_message(message)
        user_lines = split_user_completed_lines(completed_raw)
        prev_ctx = _session_get_recipe_ctx(session_id)
        prev_recipe_q = (prev_ctx or {}).get("recipe_query", "").strip() if prev_ctx else ""
        prev_recipe_obj = _recipe_from_session_ctx(prev_ctx)
        if not user_lines:
            # Natural follow-up like "I have added salt... now what?".
            user_lines = extract_completed_from_natural_message(message)
        # If no explicit "completed steps" are present, fall back to normal grounded recipe QA.
        progress_mode = bool(user_lines)
        if progress_mode and recipe_q:
            search_query = recipe_q.strip()
        elif progress_mode and prev_recipe_q:
            search_query = prev_recipe_q
        else:
            search_query = infer_recipe_focus_query(message).strip()
            # If focus extraction is weak, keep conversational continuity.
            if prev_recipe_q and search_query.lower() == message.lower():
                search_query = prev_recipe_q
        if not search_query:
            raise HTTPException(400, "message is empty")

        # For follow-up progress asks, prefer pinned session recipe to avoid drifting to another dish.
        q_prep = maybe_spell_correct(search_query)
        if progress_mode and prev_recipe_obj is not None:
            ranked = [
                (
                    prev_recipe_obj,
                    1.0,
                    {
                        "embed": 1.0,
                        "fuzzy": 1.0,
                        "coverage": 1.0,
                    },
                )
            ]
        else:
            async with _store_lock:
                if not recipe_catalog.recipes or recipe_catalog.embeddings is None:
                    raise HTTPException(
                        400,
                        "No recipe catalog. Upload and ingest a PDF first.",
                    )
                try:
                    async with aiohttp.ClientSession() as session:
                        fast_ranked = recipe_catalog.fast_title_search(
                            q_prep,
                            top_k=min(RECIPE_TOP_K, 3),
                            min_score=RECIPE_FAST_TITLE_MIN_SCORE,
                        )
                        if fast_ranked:
                            ranked = fast_ranked
                        else:
                            q_embed = expand_query_for_embedding(q_prep)
                            qvec = await _embed_recipe_query_cached(session, q_embed)
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
                        "Check that Ollama is running and the embed model matches the recipe index.",
                    ) from e

        if not ranked:
            raise HTTPException(500, "No matching recipes")

        best_recipe, _best_comb, best_parts = ranked[0]
        if not progress_mode:
            best_title = str(best_recipe.get("title") or "")
            title_score = fuzz.token_set_ratio(q_prep.lower(), best_title.lower()) / 100.0
            if title_score < RECIPE_TITLE_MATCH_MIN_SCORE:
                raise HTTPException(
                    422,
                    "I could not confidently match your dish name to a recipe title in the indexed PDF. "
                    "Try the exact recipe title from the book, or ask with 'recipe: <title>'.",
                )
        if progress_mode:
            steps = steps_from_recipe(best_recipe)
            done, matched_detail = match_completed_steps(
                user_lines,
                steps,
                match_threshold=RECIPE_PROGRESS_MATCH,
            )
            answer = format_progress_answer(best_recipe, steps, done, matched_detail)
            steps_total = len(steps)
            steps_matched = sum(1 for d in done if d)
            model_used = "offline-step-match"
        else:
            answer = _grounded_recipe_answer(q_prep, best_recipe, best_parts)
            steps_total = 0
            steps_matched = 0
            model_used = "grounded-fallback"
        _session_set_recipe_ctx(
            session_id,
            recipe_query=search_query,
            recipe_title=str(best_recipe.get("title") or ""),
            page=best_recipe.get("page"),
        )

        return {
            "answer": answer,
            "model_used": model_used,
            "recipe_query": search_query,
            "recipe_title": best_recipe.get("title"),
            "page": best_recipe.get("page"),
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
            "steps_total": steps_total,
            "steps_matched": steps_matched,
        }

    @app.post("/api/recipe-chat")
    async def recipe_chat(body: RecipeChatBody):
        """Fuzzy + semantic retrieval, then LLM formatting (layer 3)."""
        q = (body.message or "").strip()
        session_id = (body.session_id or "").strip()
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
            _session_set_recipe_ctx(
                session_id,
                recipe_query=q_prep,
                recipe_title=str(best_recipe.get("title") or ""),
                page=best_recipe.get("page"),
            )
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
        if ranked:
            best_recipe, _best_score, _best_parts = ranked[0]
            _session_set_recipe_ctx(
                session_id,
                recipe_query=q_prep,
                recipe_title=str(best_recipe.get("title") or ""),
                page=best_recipe.get("page"),
            )

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
