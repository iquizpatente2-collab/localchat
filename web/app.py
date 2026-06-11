"""
Local web UI for manual Q&A using Ollama (embeddings + chat).

Run from repository root:
  uvicorn web.app:app --host 0.0.0.0 --port 8080

Env:
  OLLAMA_HOST          default http://127.0.0.1:11434
  OLLAMA_EMBED_MODEL   default nomic-embed-text
  RAG_EMBED_INPUT_MAX_TOKENS if set, overrides per-model defaults (ollama_rag picks ~7000 for nomic, 512 for unknown embed models).
  RAG_EMBED_INPUT_MAX_CHARS if >0, extra hard character cap after token trim (optional).
  RAG_PDF_PIPELINE         1|0 — use pdfplumber manual_pdf_pipeline for ingest (default 1). Set 0 for legacy pypdf chunking.
  RAG_PIPELINE_MIN_WORDS   pipeline min chunk words (default 80)
  RAG_PIPELINE_MAX_TOKENS  pipeline target max tokens per chunk (default 500)
  RAG_PIPELINE_OVERLAP     pipeline text overlap in tokens (default 50)
  RAG_PIPELINE_PAGE_START / RAG_PIPELINE_PAGE_END optional 1-based inclusive page window for testing
  OLLAMA_CHAT_MODEL    default qwen2.5:14b-instruct
  OLLAMA_CHAT_FALLBACK default qwen2.5:7b-instruct (used if primary chat fails)
  RAG_TOP_K            default 5
  RAG_MAX_TOKENS       manual chat output budget (default 480; two-part answers need more room)
  RAG_CHAT_HISTORY_MAX max prior user/assistant turns sent with /api/chat (default 12)
  COMMUNITY_ENABLED       1|0 — user tips in Chroma (default 1)
  COMMUNITY_CHROMA_PATH   persist dir (default data/community_chroma)
  COMMUNITY_QUERY_TOP_K   neighbors to scan (default 8)
  COMMUNITY_MAX_DISTANCE  cosine distance max; lower stricter (default 0.28)
  COMMUNITY_LEXICAL_FILTER 1|0 — require on-topic overlap with saved tip question (default 1)
  COMMUNITY_INJECT_MAX    max tips from Chroma considered after distance filter (default 2); lexical filter still applies.
  COMMUNITY_DISPLAY_MAX_DISTANCE  stricter Chroma distance cap to append/show a tip in the answer (default 0.20).
  COMMUNITY_DISPLAY_MIN_FUZZ      rapidfuzz token_set_ratio min vs saved Q+comment for display (default 58).
  COMMUNITY_DISPLAY_ULTRA_DISTANCE / COMMUNITY_DISPLAY_ULTRA_MIN_FUZZ  relax fuzz slightly when distance is very low (defaults 0.14 / 48).
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
  RAG_RETRIEVAL_BLEND       always|never|smart — how to mix chat history into vector/lexical query (default smart).
                            smart: if the new question is unlike the last exchange (embedding cosine), search as a fresh query.
  RAG_TOPIC_CONTINUATION_SIM cosine threshold for “same thread” when smart (default 0.48; higher = stricter about reusing history)
  RAG_TOPIC_CONTEXT_MAX_CHARS cap on prior user+assistant text used for that similarity check (default 2400)
  RAG_FOLLOWUP_MAX_CHARS     max length for “short follow-up” detection; longer messages always use the similarity check (default 48)
  RAG_MANUAL_ORGANIZE        1|0 — extra LLM pass to Markdown-layout retrieved excerpts in the reply (default 1)
  RAG_MANUAL_ORGANIZE_MAX_CHARS / RAG_MANUAL_ORGANIZE_MAX_TOKENS / RAG_MANUAL_ORGANIZE_TIMEOUT_S
  RAG_MANUAL_ORGANIZE_MODEL  optional; defaults to OLLAMA_CHAT_MODEL (e.g. set qwen2.5:3b for a faster layout pass)
  RAG_MANUAL_MAX_PASSAGES     max passages sent to chat + shown under “Estratti” after focus ranking (default 3)
  RAG_MANUAL_FOCUS_SCORE_GAP  if best passage beats 2nd by this focus score, show at most 2; if > ~2.2× gap, show 1
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import aiohttp
import numpy as np
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

from web.rag.ingest import extract_pages_cleaned, pages_to_chunks
from web.rag.pipeline_bridge import enriched_pages_to_recipe_pages, pipeline_json_to_store_chunks
from web.rag.ollama_rag import ollama_chat, ollama_chat_stream, ollama_embed, embed_many
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
CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:14b-instruct")
TOP_K = int(os.environ.get("RAG_TOP_K", "4"))
MAX_TOKENS = int(os.environ.get("RAG_MAX_TOKENS", "480"))
RAG_CHAT_HISTORY_MAX = max(0, int(os.environ.get("RAG_CHAT_HISTORY_MAX", "12")))
CHAT_TIMEOUT_S = float(os.environ.get("RAG_CHAT_TIMEOUT_S", "240"))
CHAT_FALLBACK_MODEL = os.environ.get("OLLAMA_CHAT_FALLBACK", "qwen2.5:7b-instruct")
_raw_chat_model_options = os.environ.get("OLLAMA_CHAT_MODELS", "").strip()
RAG_DOCS_FILE = os.environ.get("RAG_DOCS_FILE", "").strip()
RAG_AUTO_DOCS = os.environ.get("RAG_AUTO_DOCS", "1").strip() not in {"0", "false", "False"}
LEXICAL_K = int(os.environ.get("RAG_LEXICAL_K", "20"))
VECTOR_K = int(os.environ.get("RAG_VECTOR_K", "14"))
RAG_EXCERPT_MAX_CHARS = max(300, int(os.environ.get("RAG_EXCERPT_MAX_CHARS", "1200")))
RAG_MANUAL_ORGANIZE = os.environ.get("RAG_MANUAL_ORGANIZE", "1").strip().lower() not in {"0", "false", "no"}
RAG_MANUAL_ORGANIZE_MAX_CHARS = max(2000, int(os.environ.get("RAG_MANUAL_ORGANIZE_MAX_CHARS", "16000")))
RAG_MANUAL_ORGANIZE_MAX_TOKENS = max(120, int(os.environ.get("RAG_MANUAL_ORGANIZE_MAX_TOKENS", "520")))
RAG_MANUAL_ORGANIZE_TIMEOUT_S = float(os.environ.get("RAG_MANUAL_ORGANIZE_TIMEOUT_S", "90"))
_manual_front_mode_raw = os.environ.get("RAG_MANUAL_FRONT_MODE", "auto").strip().lower()
RAG_MANUAL_FRONT_MODE = _manual_front_mode_raw if _manual_front_mode_raw in {"auto", "always", "never"} else "auto"
# After reranking: keep only the best-matching passages in chat + UI (reduces irrelevant OCR bulk).
RAG_MANUAL_MAX_PASSAGES = max(1, min(12, int(os.environ.get("RAG_MANUAL_MAX_PASSAGES", "3"))))
RAG_MANUAL_FOCUS_SCORE_GAP = float(os.environ.get("RAG_MANUAL_FOCUS_SCORE_GAP", "26"))
RAG_TITLE_PAGE_BOOST = os.environ.get("RAG_TITLE_PAGE_BOOST", "1").strip().lower() not in {
    "0",
    "false",
    "no",
}
RAG_TITLE_MATCH_MIN = float(os.environ.get("RAG_TITLE_MATCH_MIN", "0.78"))
RAG_TITLE_PAGE_MAX_CHUNKS = max(1, min(8, int(os.environ.get("RAG_TITLE_PAGE_MAX_CHUNKS", "3"))))

_rag_blend_raw = os.environ.get("RAG_RETRIEVAL_BLEND", "smart").strip().lower()
RAG_RETRIEVAL_BLEND = _rag_blend_raw if _rag_blend_raw in {"always", "never", "smart"} else "smart"
RAG_TOPIC_CONTINUATION_SIM = float(os.environ.get("RAG_TOPIC_CONTINUATION_SIM", "0.48"))
RAG_TOPIC_CONTEXT_MAX_CHARS = max(400, int(os.environ.get("RAG_TOPIC_CONTEXT_MAX_CHARS", "2400")))
RAG_FOLLOWUP_MAX_CHARS = max(12, int(os.environ.get("RAG_FOLLOWUP_MAX_CHARS", "48")))

RAG_PDF_PIPELINE = os.environ.get("RAG_PDF_PIPELINE", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "legacy",
    "pypdf",
}
RAG_PIPELINE_MIN_WORDS = max(20, int(os.environ.get("RAG_PIPELINE_MIN_WORDS", "80")))
RAG_PIPELINE_MAX_TOKENS = max(100, int(os.environ.get("RAG_PIPELINE_MAX_TOKENS", "500")))
RAG_PIPELINE_OVERLAP = max(0, int(os.environ.get("RAG_PIPELINE_OVERLAP", "50")))
_rps = os.environ.get("RAG_PIPELINE_PAGE_START", "").strip()
_rpe = os.environ.get("RAG_PIPELINE_PAGE_END", "").strip()
RAG_PIPELINE_PAGE_START = int(_rps) if _rps.isdigit() else None
RAG_PIPELINE_PAGE_END = int(_rpe) if _rpe.isdigit() else None

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
# Stricter than COMMUNITY_MAX_DISTANCE: only tips meeting this + fuzzy gate appear in the final Community section.
COMMUNITY_DISPLAY_MAX_DISTANCE = float(os.environ.get("COMMUNITY_DISPLAY_MAX_DISTANCE", "0.20"))
COMMUNITY_DISPLAY_MIN_FUZZ = max(30, min(100, int(os.environ.get("COMMUNITY_DISPLAY_MIN_FUZZ", "58"))))
# If embedding is very strong, allow slightly lower fuzzy overlap (still on-topic via prior filter).
COMMUNITY_DISPLAY_ULTRA_DISTANCE = float(os.environ.get("COMMUNITY_DISPLAY_ULTRA_DISTANCE", "0.14"))
COMMUNITY_DISPLAY_ULTRA_MIN_FUZZ = max(30, min(100, int(os.environ.get("COMMUNITY_DISPLAY_ULTRA_MIN_FUZZ", "48"))))
COMMUNITY_SAVE_QUESTION_MAX = 8000
COMMUNITY_SAVE_COMMENT_MAX = 4000
COMMUNITY_SAVE_AUTHOR_MAX = 120
COMMUNITY_SAVE_ANSWER_MAX = 4000

# Admin gate for PDF ingest and community note management (override via env in production).
ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin").strip() or "admin"
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin")
ADMIN_SESSION_SECRET = os.environ.get(
    "ADMIN_SESSION_SECRET",
    "localchat-admin-session-secret-change-me",
)
ADMIN_SESSION_TTL_S = max(300, int(os.environ.get("ADMIN_SESSION_TTL_S", "43200")))


def _admin_issue_token() -> str:
    exp = int(time.time()) + ADMIN_SESSION_TTL_S
    payload = str(exp)
    sig = hmac.new(
        ADMIN_SESSION_SECRET.encode(),
        payload.encode(),
        hashlib.sha256,
    ).hexdigest()
    return f"{payload}.{sig}"


def _admin_verify_token(token: str | None) -> bool:
    if not token:
        return False
    try:
        exp_s, sig = token.rsplit(".", 1)
        exp = int(exp_s)
        if time.time() > exp:
            return False
        expected = hmac.new(
            ADMIN_SESSION_SECRET.encode(),
            exp_s.encode(),
            hashlib.sha256,
        ).hexdigest()
        return hmac.compare_digest(sig, expected)
    except (ValueError, TypeError):
        return False


def _admin_token_from_headers(
    authorization: str | None,
    x_admin_token: str | None,
) -> str | None:
    if x_admin_token and x_admin_token.strip():
        return x_admin_token.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return None


async def require_admin(
    authorization: str | None = Header(None),
    x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
) -> None:
    tok = _admin_token_from_headers(authorization, x_admin_token)
    if not _admin_verify_token(tok):
        raise HTTPException(status_code=401, detail="Admin login required")


def _configured_chat_models() -> list[str]:
    """Ordered chat model choices exposed to frontend selector."""
    out: list[str] = []

    def add(model: str) -> None:
        m = (model or "").strip()
        if m and m not in out:
            out.append(m)

    add(CHAT_MODEL)
    add(CHAT_FALLBACK_MODEL)
    if _raw_chat_model_options:
        for item in _raw_chat_model_options.replace(";", ",").split(","):
            add(item)
    return out


def _safe_selected_model(raw: str | None) -> str | None:
    """Accept a user-selected model only when it looks valid."""
    if raw is None:
        return None
    m = str(raw).strip()
    if not m:
        return None
    # Simple guard against accidental garbage/whitespace-only payloads.
    if len(m) > 120 or any(ch in m for ch in "\r\n\t"):
        return None
    return m


# Pipeline prepends this block to each chunk; strip for cleaner UI/LLM context (see manual_pdf_pipeline/enricher.py).
_CHUNK_CTX_HEADER_RE = re.compile(
    r"(?is)^Document:\s*[^\n]+\n"
    r"Section:\s*[^\n]+\n"
    r"Subsection:\s*[^\n]+\n"
    r"Page:\s*[^\n]+\n"
    r"Type:\s*[^\n]+\n"
    r"Procedure ID:\s*[^\n]+\n"
    r"Frequency:\s*[^\n]+\n"
    r"-{2,}\s*\n+",
)

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


def _community_confident_for_display(user_q: str, match: dict) -> bool:
    """
    Final UI / appendix: only tips that are both vector-close and lexically aligned with this question.
    Chroma distance is cosine distance (lower = more similar).
    """
    dist = float(match.get("distance", 999.0))
    if dist > COMMUNITY_DISPLAY_MAX_DISTANCE:
        return False
    q = (user_q or "").strip()
    if len(q) < 4:
        return False
    blob = f"{match.get('question') or ''} {match.get('comment') or ''}".strip()
    if not blob:
        return False
    score = fuzz.token_set_ratio(q.lower(), blob.lower())
    if score >= COMMUNITY_DISPLAY_MIN_FUZZ:
        return True
    if dist <= COMMUNITY_DISPLAY_ULTRA_DISTANCE and score >= COMMUNITY_DISPLAY_ULTRA_MIN_FUZZ:
        return True
    return False


def _community_context_for_llm(matches: list[dict]) -> str:
    """Inject into the chat prompt so the model can use tips as non-authoritative context."""
    if not matches:
        return ""
    lines = [
        "---",
        "Community field notes (NOT from the manual; user-contributed; may be wrong).",
        "You may use them only to inform practical suggestions in your Assistant notes.",
        "Never present them as OEM manual facts; never copy them as if they were excerpt text.",
        "",
    ]
    for i, m in enumerate(matches, 1):
        qv = _community_one_line(m.get("question") or "", 360)
        cv = _community_one_line(m.get("comment") or "", 520)
        av = _community_one_line(m.get("answer_excerpt") or "", 480)
        lines.append(f"Note {i} (similarity distance {float(m.get('distance', 0.0)):.3f}):")
        if qv:
            lines.append(f"  Saved question: {qv}")
        if av:
            lines.append(f"  Saved assistant snapshot: {av}")
        if cv:
            lines.append(f"  User comment: {cv}")
        lines.append("")
    return "\n".join(lines).rstrip()

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
    "You only use the manual records provided in the user message. "
    "Never invent fault codes, part numbers, procedures, values, or steps not supported by those records. "
    "Always answer in the same language as the user's latest message (Italian or English)."
)

RAG_SYSTEM = """You are an offline assistant for industrial machine manuals. You will receive prior conversation (if any), manual excerpts, optional community field notes, then the user's latest question.

The application **shows the user a focused subset of retrieved passages** (best match first as Excerpt 1 / Passage 1 — same text as in your message). Your reply must be **only** the section below — do **not** re-paste large blocks of excerpt text. When you mention a technical fact (fault code, torque, pressure, temperature, part name, step order, warning, or a counted list of symbol types), **tie it explicitly** to the supporting excerpt (e.g. "Excerpt 1 …") or say clearly that **the retrieved passages do not contain** that detail. If Excerpt 1 lists enumerated items that answer a "how many types" question, you may count only those items that are explicitly named in that excerpt. Never state numbers or items not supported by the excerpts.

If a "Community field notes" block is present, it is **user-contributed, not OEM manual text**. You may use it only to inform **general** practical suggestions; never cite it as authoritative specifications.

Begin your reply with this exact heading line:
### Assistant notes (interpretation — general practice; manual facts only when tied to an excerpt above)

Then:
- Answer strictly from the manual excerpts for **factual** claims; for anything not in the excerpts, say it is not in the retrieved passages (do not guess).
- Add brief practical guidance (diagnostics, isolation, safety) as your own view, clearly separate from manual facts.
- If community notes align with the question, you may mention that "saved field notes suggest …" in one short phrase — still not as manual truth.

**Do not** write a "Community" appendix yourself. **Do not** invent attributed tips. The application appends a verified community section only when a tip strongly matches the question.

Rules:
- OCR/typos: treat near-matches as the same machine term when meaning matches (e.g. alarm code variants, English vs Italian labels).
- If a "Retrieval note" explains spelling/title variants, treat that machine item as matching the user's question.
- **Page numbers:** Only cite pages exactly as shown in the excerpt lines (e.g. `Excerpt 2 (page 29, ...)` or "Passage 1"). Never invent or guess a page.
- When the user asks about a specific machine/subsystem/fault code, prefer excerpts that mention that exact target; do not substitute another subsystem unless no matching excerpt exists.
- Use conversation history only for understanding references ("it", "that unit"); facts still come from the excerpts.
- Language: same as the user's latest question (Italian or English).
- Stay concise unless the user asks for detail."""

MANUAL_ORGANIZE_SYSTEM = """You reformat manual excerpt text for readability only (layout task).

Hard rules:
- Output Markdown only. Do not start with `#` / `##` / `###` headings; begin with bullets or **short labels** so the host UI can wrap your output in its own section title.
- Use short bullet lists for distinct hazard/symbol/category labels that appear as separate phrases or lines in the source. Remove duplicate lines and collapse excessive blank lines.
- Do NOT add facts, counts, ISO norms, or wording not clearly supported by the input. If the source lists names like "Pericolo generico" / "Pericolo di schiacciamento", reproduce them as bullets without inventing extra types.
- If the user question asks how many types/categories appear and the source text enumerates distinct names on separate lines, include every such name as a bullet; then add exactly one closing line: `Riassunto: nel testo compaiono N voci distinte.` where N is the number of bullets you listed (only if N is at least 2 and every bullet came from the source).
- If the source does not state a total count, do not state a number; list only what appears.
- Keep [TABLE: ...] material: one bullet per table caption row when it encodes AVVISO/PERICOLO/AVVERTIMENTO blocks.
- If you see "[... excerpt truncated ...]", do not fill in missing text.
- Match the dominant language of the excerpts (Italian vs English). No chitchat, no preface about yourself."""

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
        r"(?is)\n+\s*###\s*Community\b.*",
    )
    for pat in patterns:
        t2 = re.sub(pat, "", t, count=1)
        if t2 != t:
            return t2.rstrip()
    return t.rstrip()


def _format_community_answer_append(matches: list[dict]) -> str:
    """Deterministic community block from Chroma only (no LLM), after manual + assistant."""
    if not matches:
        return ""
    lines = [
        "",
        "---",
        "",
        "### Community (other users, not verified)",
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
    lang = "Italian" if _detect_answer_language(query) == "it" else "English"
    if mode == "explain":
        base = PROMPT_EXPLAIN_MATCH.format(QUERY=query, RECIPES=recipes_block)
        return f"{base}\n\nRespond in {lang}."
    if mode == "vague":
        base = PROMPT_VAGUE.format(QUERY=query, RECIPES=recipes_block)
        return f"{base}\n\nRespond in {lang}."
    if mode == "direct":
        base = PROMPT_DIRECT_RECIPE.format(QUERY=query, RECIPES=recipes_block)
        return f"{base}\n\nRespond in {lang}."
    base = PROMPT_SHOW_MATCHING.format(QUERY=query, RECIPES=recipes_block)
    return f"{base}\n\nRespond in {lang}."


_ITALIAN_MARKERS = {
    "il",
    "lo",
    "la",
    "gli",
    "le",
    "dei",
    "delle",
    "della",
    "dello",
    "nel",
    "nella",
    "con",
    "per",
    "come",
    "dopo",
    "prima",
    "ricetta",
    "manuale",
    "ingrediente",
    "ingredienti",
    "procedura",
    "passo",
    "passi",
    "macchina",
    "manutenzione",
}
_ENGLISH_MARKERS = {
    "the",
    "and",
    "with",
    "for",
    "from",
    "after",
    "before",
    "recipe",
    "manual",
    "ingredients",
    "procedure",
    "step",
    "steps",
    "machine",
    "maintenance",
    "how",
    "what",
}


def _detect_answer_language(text: str) -> str:
    """
    Lightweight language guess between Italian and English.
    Defaults to English on ties to keep behavior stable.
    """
    toks = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ']+", (text or "").lower())
    if not toks:
        return "en"
    it_score = sum(1 for t in toks if t in _ITALIAN_MARKERS)
    en_score = sum(1 for t in toks if t in _ENGLISH_MARKERS)
    # Italian accented chars are a strong hint.
    if re.search(r"[àèéìòù]", text.lower()):
        it_score += 2
    return "it" if it_score > en_score else "en"


def _answer_language_directive(user_q: str) -> str:
    lang = _detect_answer_language(user_q)
    if lang == "it":
        return (
            "Language directive: Rispondi in italiano. "
            "Usa termini tecnici chiari e naturali in italiano."
        )
    return "Language directive: Respond in English. Use clear, concise technical English."


def _grounded_recipe_answer(query: str, recipe: dict, parts: dict[str, float]) -> str:
    """Deterministic, citation-friendly answer using extracted manual record fields only."""
    lang = _detect_answer_language(query)
    title = (recipe.get("title") or "").strip()
    page = recipe.get("page", "?")
    ingredients = [str(x).strip() for x in (recipe.get("ingredients") or []) if str(x).strip()]
    instructions = [str(x).strip() for x in (recipe.get("instructions") or []) if str(x).strip()]
    full_text = (recipe.get("full_text") or "").strip()
    if len(title) < 4 or len(re.findall(r"[A-Za-z]", title)) < 3:
        title = infer_title_from_text(full_text)
    fallback_steps = fallback_steps_from_prose(full_text)

    if lang == "it":
        recipe_name_label = "Titolo Procedura"
        ingredients_label = "Componenti / Parametri"
        instructions_label = "Passi Operativi"
        unclear = "Non chiaramente estratto dal testo sorgente"
        source_note = (
            "Nota fonte: Questa risposta e basata solo sul testo estratto dal PDF "
            f"(pagina {page})."
        )
        match_reason = "Motivo della corrispondenza alla richiesta"
        extracted_text_label = "Testo Sorgente Estratto"
    else:
        recipe_name_label = "Procedure Title"
        ingredients_label = "Components / Parameters"
        instructions_label = "Operational Steps"
        unclear = "Not clearly extracted from source text"
        source_note = (
            "Source note: This output is grounded only in extracted PDF text "
            f"(page {page})."
        )
        match_reason = "Match reason for query"
        extracted_text_label = "Extracted Source Text"

    lines: list[str] = [f"{recipe_name_label}: {title}", "", f"{ingredients_label}:"]
    if ingredients:
        lines.extend(f"- {x}" for x in ingredients)
    else:
        lines.append(f"- {unclear}")

    lines.append("")
    lines.append(f"{instructions_label}:")
    if instructions:
        lines.extend(f"{i}. {x}" for i, x in enumerate(instructions, 1))
    elif fallback_steps:
        lines.extend(f"{i}. {x}" for i, x in enumerate(fallback_steps, 1))
    else:
        lines.append(f"1. {unclear}")

    lines.append("")
    lines.append(source_note)
    cov = parts.get("coverage")
    bg = parts.get("bigram")
    tail = f"(embed={parts.get('embed', 0.0):.3f}, fuzzy={parts.get('fuzzy', 0.0):.3f}"
    if cov is not None:
        tail += f", coverage={float(cov):.3f}"
    if bg is not None:
        tail += f", phrase={float(bg):.3f}"
    tail += ")."
    lines.append(
        f"{match_reason} '{query}': hybrid="
        f"{parts.get('embed', 0.0) * RECIPE_W_EMBED + parts.get('fuzzy', 0.0) * RECIPE_W_FUZZY:.3f} "
        + tail
    )
    if full_text:
        lines.append("")
        lines.append(f"{extracted_text_label}:")
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


def _resolve_active_manual_pdf_path() -> Path | None:
    """Filesystem path for the PDF that built the active vector index (for /api/manual)."""
    name = (store.source_file or "").strip()
    if name:
        base = Path(name).name
        if base:
            for d in (MANUALS_DIR, DOCS_DIR):
                cand = d / base
                if cand.is_file() and cand.suffix.lower() == ".pdf":
                    return cand
    if CURRENT_MANUAL_PATH.is_file():
        return CURRENT_MANUAL_PATH
    doc = _pick_docs_pdf()
    if doc is not None and doc.is_file():
        return doc
    return None


async def _build_index_from_pdf(
    pdf_path: Path,
    source_name: str,
    *,
    apply_recipe_normalize: bool | None = None,
) -> int:
    """
    apply_recipe_normalize: None = use env RAG_RECIPE_NORMALIZE; False = skip LLM page cleanup.
    """
    use_pdf_pipeline = RAG_PDF_PIPELINE
    if use_pdf_pipeline:
        try:
            from manual_pdf_pipeline.pipeline import (
                apply_normalized_pages_to_enriched,
                extract_enriched_pages,
                finalize_chunks_from_enriched,
            )
        except ImportError as e:
            print(f"[RAG] manual_pdf_pipeline not available ({e}); falling back to pypdf ingest.")
            use_pdf_pipeline = False

    pages: list[tuple[int, str]]
    chunks: list[dict]

    if use_pdf_pipeline:
        print(f"[RAG] Ingesting with pdfplumber pipeline: {pdf_path.name}")
        enriched = extract_enriched_pages(
            pdf_path,
            page_start=RAG_PIPELINE_PAGE_START,
            page_end=RAG_PIPELINE_PAGE_END,
        )
        pages = enriched_pages_to_recipe_pages(enriched)
        if not pages:
            raise RuntimeError(f"No extractable text in {pdf_path.name} (pdf pipeline)")
    else:
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

    if use_pdf_pipeline:
        apply_normalized_pages_to_enriched(enriched, pages)
        chunks_final, pstats = finalize_chunks_from_enriched(
            enriched,
            source_name or pdf_path.name,
            min_words=RAG_PIPELINE_MIN_WORDS,
            max_tokens=RAG_PIPELINE_MAX_TOKENS,
            overlap_tokens=RAG_PIPELINE_OVERLAP,
        )
        chunks = pipeline_json_to_store_chunks(chunks_final)
        print(
            f"[RAG] Pipeline: {pstats.get('total_chunks')} chunks; "
            f"tables_serialized={pstats.get('tables_serialized')} "
            f"thin_merged={pstats.get('thin_pages_merged')}"
        )
    else:
        chunks = pages_to_chunks(pages)

    if not chunks:
        raise RuntimeError(f"No chunks after processing {pdf_path.name}")

    print(
        f"[RAG] Ingesting {len(chunks)} chunks; embedding with {EMBED_MODEL} "
        f"(concurrency={os.environ.get('RAG_EMBED_CONCURRENCY', '4')}) ..."
    )
    recipes, recipe_embed_texts = build_recipe_embeddings_texts(pages, source_name)
    async with aiohttp.ClientSession() as session:
        texts = [c["text"] for c in chunks]
        emb = await embed_many(session, texts, EMBED_MODEL)
        print(
            f"[RAG] Chunk embeddings done ({len(texts)}). "
            f"Recipe page embeddings ({len(recipe_embed_texts)}) ..."
        )
        recipe_emb = await embed_many(session, recipe_embed_texts, EMBED_MODEL)

    store.set_data(chunks, emb, source_file=source_name)
    store.save()
    recipe_catalog.set_recipes_with_embeddings(recipes, recipe_emb, source_name)
    _save_state(_file_signature(pdf_path))
    print(
        f"[RAG] Index saved: {len(chunks)} vectors; recipe catalog: {len(recipes)} pages"
    )
    return len(chunks)


def _strip_chunk_context_header(text: str) -> str:
    """Remove pdf_pipeline context header (Document/Section/.../---) from chunk body."""
    return _CHUNK_CTX_HEADER_RE.sub("", (text or "").lstrip())


def _chunk_text_for_display(text: str) -> str:
    """Strip internal metadata header and mild whitespace cleanup before excerpting."""
    t = _strip_chunk_context_header((text or "").strip())
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _organize_chunk_text_lines(text: str) -> str:
    """Deterministic cleanup: strip trailing spaces per line, drop consecutive duplicate lines."""
    lines = [ln.rstrip() for ln in (text or "").splitlines()]
    out: list[str] = []
    prev_nonempty: str | None = None
    for ln in lines:
        if not ln.strip():
            if out and out[-1] != "":
                out.append("")
            continue
        core = re.sub(r" {2,}", " ", ln.strip())
        if core == prev_nonempty:
            continue
        prev_nonempty = core
        out.append(core)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _strip_md_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\s*\n?", "", t)
        t = re.sub(r"\n?```\s*$", "", t)
    return t.strip()


def _hits_blob_for_organize(hits: list[tuple[dict, float]], max_chars: int) -> str:
    parts: list[str] = []
    n = 0
    for ch, _sc in hits:
        ps = ch.get("page", "?")
        pe = ch.get("page_end")
        disp = f"{ps}–{pe}" if pe is not None and pe != ps else str(ps)
        body = _organize_chunk_text_lines(_chunk_text_for_display(ch.get("text", "") or ""))
        room = max_chars - n - 24
        if room < 200:
            break
        piece = f"--- Pages {disp} ---\n{body[:room]}"
        parts.append(piece)
        n += len(piece) + 2
    return "\n\n".join(parts)[:max_chars]


async def _organize_manual_excerpts_llm(
    session: aiohttp.ClientSession,
    query: str,
    hits: list[tuple[dict, float]],
) -> str:
    """Second-pass layout: structured Markdown from retrieved text only (strict)."""
    blob = _hits_blob_for_organize(hits, RAG_MANUAL_ORGANIZE_MAX_CHARS)
    if not blob.strip():
        return ""
    user_msg = (
        "User question (context only — do not invent facts beyond the manual text):\n"
        f"{query}\n\n"
        "Prioritize passages that directly answer this question (e.g. Simbologia / hazard categories). "
        "De-emphasize boilerplate from other sections if it is not in the excerpt text below.\n\n"
        "---\n\nManual excerpt text to reorganize:\n"
        f"{blob}"
    )
    opts = {"num_predict": RAG_MANUAL_ORGANIZE_MAX_TOKENS, "temperature": 0.0}
    org_model = os.environ.get("RAG_MANUAL_ORGANIZE_MODEL", "").strip() or CHAT_MODEL
    raw = await ollama_chat(
        session,
        org_model,
        [
            {"role": "system", "content": MANUAL_ORGANIZE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        stream=False,
        options=opts,
        timeout_s=RAG_MANUAL_ORGANIZE_TIMEOUT_S,
    )
    out = _strip_md_fences(raw)
    out = re.sub(r"^(?:#{1,6}\s.*\n)+", "", out.strip(), flags=re.MULTILINE)
    return out.strip()


def _format_context(results: list[tuple[dict, float]]) -> str:
    parts: list[str] = []
    for i, (ch, score) in enumerate(results, 1):
        ps = ch.get("page", "?")
        pe = ch.get("page_end")
        if pe is not None and pe != ps:
            page_disp = f"{ps}-{pe}"
        else:
            page_disp = str(ps)
        txt = _chunk_text_for_display(ch.get("text", "") or "")
        if len(txt) > RAG_EXCERPT_MAX_CHARS:
            txt = txt[:RAG_EXCERPT_MAX_CHARS].rstrip() + "\n[... excerpt truncated ...]"
        parts.append(f"--- Excerpt {i} (page {page_disp}, score {score:.3f}) ---\n{txt}")
    return "\n\n".join(parts)


def _format_manual_answer_preamble(
    results: list[tuple[dict, float]],
    organized_markdown: str | None = None,
) -> str:
    """
    Manual block: optional LLM-organized view (strict), then per-page passages with light layout cleanup.
    """
    lines: list[str] = [
        "### From the manual (retrieved passages)",
        "",
        "Sotto trovi i passaggi **più pertinenti** a questa domanda (RAG: riordinati per rilevanza; Passaggio 1 = miglior corrispondenza). "
        "La sintesi strutturata riordina solo quel testo, senza aggiungere fatti.",
        "",
    ]
    org = (organized_markdown or "").strip()
    if org:
        lines.append("### Sintesi strutturata (solo dal testo degli estratti)")
        lines.append("")
        lines.append(org)
        lines.append("")
        lines.append("### Estratti per pagina (testo indicizzato)")
        lines.append("")
    for i, (ch, _score) in enumerate(results, 1):
        ps = ch.get("page", "?")
        pe = ch.get("page_end")
        if pe is not None and pe != ps:
            page_disp = f"pages {ps}–{pe}"
        else:
            page_disp = f"page {ps}"
        raw = _chunk_text_for_display(ch.get("text", "") or "")
        txt = _organize_chunk_text_lines(raw)
        if len(txt) > RAG_EXCERPT_MAX_CHARS:
            txt = txt[:RAG_EXCERPT_MAX_CHARS].rstrip() + "\n[... excerpt truncated ...]"
        lines.append(f"**Passage {i}** ({page_disp})")
        lines.append("")
        lines.append(txt)
        lines.append("")
    return "\n".join(lines).rstrip()


def _should_include_manual_front(query: str) -> bool:
    if RAG_MANUAL_FRONT_MODE == "always":
        return True
    if RAG_MANUAL_FRONT_MODE == "never":
        return False
    q = (query or "").strip().lower()
    if not q:
        return False
    # Show full excerpt blocks only when the user explicitly asks for source text/citations.
    hints = (
        "show excerpt",
        "show passage",
        "quote",
        "verbatim",
        "exact text",
        "source text",
        "cite source",
        "show source",
        "mostra estratto",
        "mostra passaggio",
        "riporta testuale",
        "testo esatto",
        "cita fonte",
        "mostra fonte",
    )
    return any(h in q for h in hints)


_LEGACY_LLM_PART2 = re.compile(
    r"(?is)\n*\s*2\)\s*\*\*In my view(?:\s*\([^)]*\))?\*\*\s*",
)


def _strip_legacy_manual_part_from_llm(text: str) -> str:
    """If the model still emits the old two-part answer, drop part 1 (manual) — the server prepends passages."""
    t = (text or "").strip()
    m = _LEGACY_LLM_PART2.search(t)
    if m:
        return t[m.end() :].strip()
    return t


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
    # Industrial manual bridge: maintenance frequencies, periodic plans, and service sections.
    if any(x in ql for x in ("manutenzione", "maintenance", "service", "ispezione", "inspection")):
        extras.extend(
            [
                "maintenance plan",
                "preventive maintenance",
                "periodic maintenance",
                "scheduled service",
                "maintenance tasks and checklist",
            ]
        )
    if any(x in ql for x in ("annuale", "annually", "yearly", "annually")):
        extras.extend(["manutenzione annuale", "annual maintenance", "yearly maintenance"])
    if any(x in ql for x in ("mensile", "monthly")):
        extras.extend(["manutenzione mensile", "monthly maintenance"])
    if any(x in ql for x in ("settimanale", "weekly")):
        extras.extend(["manutenzione settimanale", "weekly maintenance"])
    if any(x in ql for x in ("giornaliera", "daily", "quotidiana")):
        extras.extend(["manutenzione giornaliera", "daily maintenance"])
    if any(x in ql for x in ("trimestrale", "quarterly")):
        extras.extend(["manutenzione trimestrale", "quarterly maintenance"])
    if any(x in ql for x in ("quadrimestrale", "every 4 months", "4 months")):
        extras.extend(["manutenzione quadrimestrale", "every four months maintenance"])
    if "2000 ore" in ql or "2000 hours" in ql:
        extras.extend(["maintenance every 2000 hours", "manutenzione ogni 2000 ore"])
    # Safety pictograms / hazard labels (Italian manuals).
    if any(x in ql for x in ("simbol", "simbolog", "pittogram", "pericol", "avvert", "avvis")) or (
        "sicurezza" in ql and "simbol" in ql
    ):
        extras.extend(
            [
                "Simbologia simboli di sicurezza messaggi PERICOLO AVVERTIMENTO AVVISO",
                "safety symbols hazard warning notice pictograms",
                "sezione simbologia sicurezza",
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


_FOOD_HINT_TOKENS = frozenset(
    {
        "recipe",
        "recipes",
        "dish",
        "soup",
        "sauce",
        "stock",
        "broth",
        "pomodoro",
        "pomidoro",
        "pollo",
        "chicken",
        "veal",
        "lamb",
        "mutton",
        "polenta",
        "pasta",
        "macaroni",
        "spaghetti",
        "ingredient",
        "ingredients",
        "bake",
        "boil",
        "fry",
    }
)


def _is_food_query(q: str) -> bool:
    ql = (q or "").lower()
    toks = set(re.findall(r"[a-zà-öø-ÿ]{3,}", ql))
    return len(toks & _FOOD_HINT_TOKENS) >= 1


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


def _query_coverage_ratio(text: str, q: str) -> float:
    """
    Fraction of distinctive query tokens found in a chunk.
    Helps keep retrieval anchored to the requested machine/procedure terms.
    """
    q_terms = [t for t in _query_terms(q) if t not in _STOP]
    if not q_terms:
        q_terms = _query_terms(q)
    if not q_terms:
        return 0.0
    lo = (text or "").lower()
    compact = _norm(text or "")
    uniq = list(dict.fromkeys(q_terms))
    hit = 0
    for t in uniq:
        nt = _norm(t)
        if (t in lo) or (nt and nt in compact):
            hit += 1
    return hit / max(1, len(uniq))


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


def _maintenance_chunk_bonus(text: str, q: str) -> float:
    """Industrial maintenance lexical booster: frequencies, section labels, and code IDs."""
    lo = (text or "").lower()
    ql = (q or "").lower()
    bonus = 0.0
    if not lo or not ql:
        return 0.0

    # Period/frequency intent
    if any(x in ql for x in ("manutenzione", "maintenance", "service", "ispezione", "inspection")):
        if any(x in lo for x in ("manutenzione", "maintenance", "service", "ispezione", "inspection")):
            bonus += 14.0
    pairs = (
        (("annuale", "yearly", "annual"), ("annuale", "annual", "yearly")),
        (("mensile", "monthly"), ("mensile", "monthly")),
        (("settimanale", "weekly"), ("settimanale", "weekly")),
        (("giornaliera", "daily", "quotidiana"), ("giornaliera", "daily", "quotidiana")),
        (("trimestrale", "quarterly"), ("trimestrale", "quarterly")),
        (("quadrimestrale", "4 months", "every 4 months"), ("quadrimestrale", "four months", "4 months")),
    )
    for ask_terms, hit_terms in pairs:
        if any(t in ql for t in ask_terms) and any(t in lo for t in hit_terms):
            bonus += 22.0

    if ("2000 ore" in ql or "2000 hours" in ql) and ("2000 ore" in lo or "2000 hours" in lo):
        bonus += 24.0

    # Procedure/code references: e.g. [MQ1], IM1, MY13
    q_codes = [c.upper() for c in re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", q.upper())]
    if q_codes:
        up = (text or "").upper()
        for c in q_codes:
            if f"[{c}]" in up or c in up:
                bonus += 34.0
    return bonus


def _extract_page_refs_from_query(q: str) -> set[int]:
    ql = (q or "").lower()
    refs: set[int] = set()
    for m in re.finditer(r"\b(?:pagina|pag|page|p)\s*[:.]?\s*(\d{1,4})\b", ql):
        try:
            n = int(m.group(1))
        except Exception:
            continue
        if 1 <= n <= 5000:
            refs.add(n)
    return refs


def _chunk_covers_page(ch: dict, page_ref: int) -> bool:
    p = ch.get("page")
    pe = ch.get("page_end")
    if not isinstance(p, int):
        return False
    if isinstance(pe, int):
        lo, hi = (p, pe) if p <= pe else (pe, p)
        return lo <= page_ref <= hi
    return p == page_ref


def _page_reference_bonus(q: str, ch: dict) -> float:
    refs = _extract_page_refs_from_query(q)
    if not refs:
        return 0.0
    return 120.0 if any(_chunk_covers_page(ch, r) for r in refs) else 0.0


def _mechanical_component_bonus(text: str, q: str) -> float:
    lo = (text or "").lower()
    ql = (q or "").lower()
    if not lo or not ql:
        return 0.0
    roots = (
        "cingh",
        "caten",
        "belt",
        "chain",
        "tendicinghia",
        "tendicatena",
        "pulegg",
        "sprocket",
        "trasmission",
    )
    if not any(r in ql for r in roots):
        return 0.0
    bonus = 0.0
    part_hits = sum(1 for r in roots if r in lo)
    bonus += min(56.0, part_hits * 18.0)
    action_terms = ("sostitu", "replace", "cambia", "change", "regola", "adjust", "tension", "allent")
    procedure_terms = ("procedura", "procedure", "passo", "step", "rimuov", "remove", "install", "installa", "sostituz")
    if any(t in ql for t in action_terms) and any(t in lo for t in procedure_terms):
        bonus += 24.0
    return bonus


def _manual_safety_symbols_query_bonus(q: str, text: str) -> float:
    """Up-rank chunks about safety pictograms / hazard messages when the user asks in that area."""
    ql = (q or "").lower()
    lo = (text or "").lower()
    if not ql or not lo:
        return 0.0
    wants = any(
        x in ql
        for x in (
            "simbol",
            "simbolog",
            "pittogram",
            "pericol",
            "avvert",
            "avvis",
            "messaggi di sicurezza",
            "warning symbol",
            "hazard symbol",
        )
    )
    if not wants:
        return 0.0
    bonus = 0.0
    if "simbologia" in lo:
        bonus += 52.0
    if "simboli" in lo and "sicurezza" in lo:
        bonus += 38.0
    if "messaggi di sicurezza" in lo:
        bonus += 36.0
    if "simboli specifici" in lo or ("categ" in lo and "simbol" in lo):
        bonus += 28.0
    if "pericolo" in lo and ("avvertimento" in lo or "avviso" in lo):
        bonus += 22.0
    for needle in ("pericolo", "avvertimento", "avviso", "grave infortunio", "infortunio o morte"):
        if needle in lo:
            bonus += 5.0
    return min(bonus, 130.0)


def _hard_lexical_recall(chunks: list[dict], q: str, top_k: int = 6) -> list[tuple[dict, float]]:
    """
    Force recall of chunks that contain exact multi-word phrases/codes from the query.
    Useful for maintenance labels where vector search can drift.
    """
    ql = (q or "").lower()
    extra: list[str] = []
    if any(x in ql for x in ("simbol", "simbolog", "pittogram", "pericol", "avvert", "avvis")):
        extra.extend(
            [
                "simbologia",
                "simboli",
                "messaggi di sicurezza",
                "simboli specifici",
                "pericolo avvertimento",
            ]
        )
    toks = [t for t in _query_terms(q) if t not in _STOP]
    if len(toks) < 2 and not extra:
        return []
    phrases: list[str] = list(extra)
    for n in (3, 2):
        for i in range(0, len(toks) - n + 1):
            ph = " ".join(toks[i : i + n])
            if len(ph) >= 8:
                phrases.append(ph)
    phrases = list(dict.fromkeys(phrases))[:20]
    if not phrases:
        return []
    q_codes = [c.upper() for c in re.findall(r"\b[A-Z]{1,3}\d{1,4}\b", q.upper())]

    scored: list[tuple[dict, float]] = []
    for ch in chunks:
        txt = ch.get("text", "")
        if not txt:
            continue
        lo = txt.lower()
        compact = _norm(txt)
        score = 0.0
        for ph in phrases:
            score += 22.0 * lo.count(ph)
            score += 16.0 * compact.count(_norm(ph))
        if q_codes:
            up = txt.upper()
            for c in q_codes:
                if f"[{c}]" in up:
                    score += 42.0
                elif c in up:
                    score += 22.0
        if score > 0:
            score *= _catalog_penalty(txt)
            scored.append((ch, float(score)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


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
    food_q = _is_food_query(q)
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
        if food_q:
            score += _italian_english_dish_bonus(lo, q)
            score += _named_dish_token_bonus(lo, q.lower())
            score += _fried_chicken_chunk_score_adj(lo, q.lower())
            score += _recipe_step_bonus(txt)
        score += _maintenance_chunk_bonus(txt, q)
        score += _mechanical_component_bonus(txt, q)
        score += _manual_safety_symbols_query_bonus(q, txt)
        score += _page_reference_bonus(q, ch)
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
    food_q = _is_food_query(query)

    def key_for(ch: dict) -> str:
        return f"{ch.get('page','?')}|{ch.get('text','')[:120]}"

    for rank, (ch, sc) in enumerate(vector_hits):
        cat = _catalog_penalty(ch.get("text", ""))
        if food_q:
            step = _recipe_step_bonus(ch.get("text", ""))
            lo = (ch.get("text") or "").lower()
            dish = (
                _named_dish_token_bonus(lo, ql)
                + _italian_english_dish_bonus(lo, query)
                + _fried_chicken_chunk_score_adj(lo, ql)
            )
        else:
            step = 0.0
            dish = 0.0
        cov = _query_coverage_ratio(ch.get("text", ""), query)
        score = (sc * 2.0) * cat + step + dish + max(0.0, 1.0 - rank * 0.08) + (cov * 42.0)
        score += _maintenance_chunk_bonus(ch.get("text", ""), query)
        score += _mechanical_component_bonus(ch.get("text", ""), query)
        score += _manual_safety_symbols_query_bonus(query, ch.get("text", ""))
        score += _page_reference_bonus(query, ch)
        k = key_for(ch)
        prev = merged.get(k)
        if prev is None or score > prev[1]:
            merged[k] = (ch, score)

    for rank, (ch, sc) in enumerate(lexical_hits):
        lo_lex = (ch.get("text") or "").lower()
        score = (sc * 2.8) + max(0.0, 0.9 - rank * 0.05)
        if food_q:
            score += _fried_chicken_chunk_score_adj(lo_lex, ql)
        score += _query_coverage_ratio(ch.get("text", ""), query) * 58.0
        score += _maintenance_chunk_bonus(ch.get("text", ""), query)
        score += _mechanical_component_bonus(ch.get("text", ""), query)
        score += _manual_safety_symbols_query_bonus(query, ch.get("text", ""))
        score += _page_reference_bonus(query, ch)
        k = key_for(ch)
        prev = merged.get(k)
        if prev is None:
            merged[k] = (ch, score)
        else:
            merged[k] = (prev[0], prev[1] + score)

    ranked = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]


def _rerank_hits_for_query(q: str, hits: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """Reorder merged hits so chunks that clearly match the question (e.g. simbologia) surface first."""
    if len(hits) < 2:
        return hits

    def sort_key(item: tuple[dict, float]) -> tuple[float, float]:
        ch, sc = item
        txt = ch.get("text", "") or ""
        cov = _query_coverage_ratio(txt, q)
        sym = _manual_safety_symbols_query_bonus(q, txt)
        mech = _mechanical_component_bonus(txt, q)
        page_boost = _page_reference_bonus(q, ch)
        return (sym + mech + page_boost + cov * 14.0, sc)

    return sorted(hits, key=sort_key, reverse=True)


def _focus_hits_for_qa(q: str, hits: list[tuple[dict, float]]) -> list[tuple[dict, float]]:
    """
    Narrow merged hits to the passages most relevant to this question (Passage 1 = best match).
    Reduces long, low-relevance OCR dumps in chat and in the user-visible manual block.
    """
    if not hits:
        return hits
    max_p = RAG_MANUAL_MAX_PASSAGES
    gap_thr = RAG_MANUAL_FOCUS_SCORE_GAP
    rows: list[tuple[float, dict, float]] = []
    for ch, sc in hits:
        txt = ch.get("text", "") or ""
        fs = (
            _manual_safety_symbols_query_bonus(q, txt)
            + _mechanical_component_bonus(txt, q)
            + _page_reference_bonus(q, ch)
            + _query_coverage_ratio(txt, q) * 22.0
            + float(sc) * 0.018
        )
        rows.append((fs, ch, sc))
    rows.sort(key=lambda x: x[0], reverse=True)
    gap = rows[0][0] - rows[1][0] if len(rows) >= 2 else 0.0
    if len(rows) >= 2 and gap >= gap_thr * 2.2:
        take = 1
    elif len(rows) >= 2 and gap >= gap_thr:
        take = max(1, min(max_p, 2))
    else:
        take = max_p
    take = min(take, len(rows), max_p)
    return [(ch, sc) for _, ch, sc in rows[:take]]


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
    if not _is_food_query(q):
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
    model: str | None = None


class CommunitySaveBody(BaseModel):
    question: str
    comment: str
    author: str = ""
    answer_snapshot: str = ""


class CommunityUpdateBody(BaseModel):
    """Replace fields on an existing tip; same length limits as save."""

    question: str
    comment: str
    author: str = ""
    answer_snapshot: str = ""


class AdminLoginBody(BaseModel):
    username: str
    password: str


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


_RAG_SHORT_FOLLOWUP_RE = re.compile(
    r"\b("
    r"what\s+('?s|is)\s+next|what\s+do\s+i\s+do\s+next|what\s+next|"
    r"next\s+step|then\s+what|and\s+then|after\s+that|now\s+what|"
    r"how\s+long|how\s+much|how\s+many|"
    r"temperature|degrees|minutes?|hours?|"
    r"oven|bake|baking\s+time|"
    r"serve|serving|"
    r"substitut|instead\s+of"
    r")\b",
    re.I,
)


def _is_short_followup_question(q: str) -> bool:
    """Very short continuations stay on the previous recipe; longer messages run the topic-similarity check."""
    s = (q or "").strip()
    if len(s) > RAG_FOLLOWUP_MAX_CHARS:
        return False
    return bool(_RAG_SHORT_FOLLOWUP_RE.search(s))


def _recent_context_blob(history: list[dict[str, str]]) -> str:
    """Last few user/assistant turns (for “is this still the same topic?”)."""
    if not history:
        return ""
    tail = history[-4:]
    lines: list[str] = []
    for h in tail:
        role = h.get("role") or ""
        content = (h.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{role}: {content[:900]}")
    blob = "\n".join(lines)
    return blob[:RAG_TOPIC_CONTEXT_MAX_CHARS]


def _cosine_sim_vec(a: np.ndarray, b: np.ndarray) -> float:
    a64 = a.astype(np.float64, copy=False)
    b64 = b.astype(np.float64, copy=False)
    na = float(np.linalg.norm(a64))
    nb = float(np.linalg.norm(b64))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a64, b64) / (na * nb))


async def _retrieval_should_blend_history(
    session: aiohttp.ClientSession,
    history: list[dict[str, str]],
    q: str,
    embed_model: str,
) -> bool:
    """
    True → blend recent chat into embedding + lexical queries (original behavior).
    False → treat this turn as a new topic: search with the current message only.
    """
    if RAG_RETRIEVAL_BLEND == "always":
        return True
    if RAG_RETRIEVAL_BLEND == "never":
        return False
    if not history:
        return False
    if _is_short_followup_question(q):
        return True
    ctx = _recent_context_blob(history)
    if not ctx.strip():
        return True
    try:
        vecs = await embed_many(session, [ctx, q.strip()], embed_model, concurrency=2)
        sim = _cosine_sim_vec(vecs[0], vecs[1])
        return sim >= RAG_TOPIC_CONTINUATION_SIM
    except Exception:
        return True


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

    @app.post("/api/admin/login")
    async def admin_login(body: AdminLoginBody):
        user = (body.username or "").strip()
        if user != ADMIN_USERNAME or body.password != ADMIN_PASSWORD:
            raise HTTPException(status_code=401, detail="Invalid username or password")
        return {"ok": True, "token": _admin_issue_token()}

    @app.get("/api/admin/session")
    async def admin_session(
        authorization: str | None = Header(None),
        x_admin_token: str | None = Header(None, alias="X-Admin-Token"),
    ):
        tok = _admin_token_from_headers(authorization, x_admin_token)
        return {"ok": True, "logged_in": _admin_verify_token(tok)}

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
                "chat_model_options": _configured_chat_models(),
                "recipe_catalog_loaded": rc_loaded,
                "recipe_count": len(recipe_catalog.recipes) if rc_loaded else 0,
                "recipe_source": recipe_catalog.source_file,
                "recipe_embed_dim": rec_dim,
                "recipe_index_backend": "faiss" if FAISS_AVAILABLE else "numpy",
                "community_enabled": bool(community_store is not None and COMMUNITY_ENABLED),
                "community_tips": int(community_store.count()) if community_store else 0,
                "whisper_stt_available": bool(WHISPER_STT_ENABLED and _whisper_lib_available()),
                "manual_pdf_available": bool(_resolve_active_manual_pdf_path()),
            }

    @app.get("/api/models")
    async def models():
        configured = _configured_chat_models()
        tags_models: list[str] = []
        ollama_reachable = False
        ollama_error = ""
        ollama_host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{ollama_host}/api/tags", timeout=aiohttp.ClientTimeout(total=12)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        tags_models = [
                            str(m.get("name") or "").strip()
                            for m in (data.get("models") or [])
                            if str(m.get("name") or "").strip()
                        ]
                        ollama_reachable = True
                    else:
                        ollama_error = f"Ollama /api/tags HTTP {resp.status}"
        except Exception as e:
            ollama_error = str(e)

        merged: list[str] = []
        for model in configured + tags_models:
            if model and model not in merged:
                merged.append(model)
        if not merged and configured:
            merged = list(configured)

        return {
            "default": CHAT_MODEL,
            "fallback": CHAT_FALLBACK_MODEL,
            "configured": configured,
            "available": merged,
            "ollama_host": ollama_host,
            "ollama_reachable": ollama_reachable,
            "ollama_error": ollama_error,
        }

    @app.get("/api/manual")
    async def serve_active_manual():
        """Serve the PDF that matches the loaded index (inline) for viewer + #page=N deep links."""
        async with _store_lock:
            path = _resolve_active_manual_pdf_path()
        if path is None or not path.is_file():
            raise HTTPException(
                404,
                "No manual PDF found for the active index. Upload a manual or place a PDF in docs/.",
            )
        return FileResponse(
            path,
            media_type="application/pdf",
            filename=path.name,
            content_disposition_type="inline",
        )

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
    async def upload(
        file: UploadFile = File(...),
        _: None = Depends(require_admin),
    ):
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
                import traceback

                print(f"[RAG] Manual indexing failed: {e}")
                traceback.print_exc()
                raise HTTPException(400, f"Manual indexing failed: {e}") from e

        return {
            "ok": True,
            "chunks": n,
            "filename": file.filename,
        }

    @app.delete("/api/upload")
    async def remove_uploaded_manual(_: None = Depends(require_admin)):
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
    async def use_docs_pdf(_: None = Depends(require_admin)):
        """Rebuild active indexes from docs PDF and make docs the source of truth."""
        docs_pdf = _pick_docs_pdf()
        if docs_pdf is None:
            raise HTTPException(400, "No docs PDF found. Put a .pdf file in docs/ or set RAG_DOCS_FILE.")

        async with _store_lock:
            try:
                _clear_runtime_indexes()
                n = await _build_index_from_pdf(docs_pdf, source_name=docs_pdf.name)
            except Exception as e:
                import traceback

                print(f"[RAG] Docs reindex failed: {e}")
                traceback.print_exc()
                raise HTTPException(500, f"Docs reindex failed: {e}") from e
        return {"ok": True, "chunks": n, "source": docs_pdf.name}

    async def _prepare_manual_chat_context(q: str, hist: list[dict[str, str]]):
        comm_matches: list[dict] = []
        async with _store_lock:
            if not store.chunks or store.embeddings is None:
                raise HTTPException(400, "No manual loaded. Upload a PDF first.")
            try:
                async with aiohttp.ClientSession() as session:
                    blend_hist = await _retrieval_should_blend_history(session, hist, q, EMBED_MODEL)
                    if blend_hist:
                        q_embed = _retrieval_query_from_history(hist, q)
                        q_lex = _lexical_query_from_history(hist, q)
                    else:
                        q_embed = q
                        q_lex = q
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
                    hard_hits = _hard_lexical_recall(store.chunks, q, top_k=6)
                    hits = _merge_hits(
                        vector_hits,
                        lexical_hits + hard_hits,
                        top_k=max(TOP_K, 6),
                        query=q,
                    )
                    hits = _recipe_title_page_boost(q, hits, top_k=max(TOP_K, 6))
                    hits = _rerank_hits_for_query(q, hits)
            except Exception as e:
                raise HTTPException(502, f"Retrieval failed: {e}") from e
        if not hits:
            raise HTTPException(500, "Search returned no chunks")
        hits_focused = _focus_hits_for_qa(q, hits)
        organized_md = ""
        if RAG_MANUAL_ORGANIZE:
            try:
                async with aiohttp.ClientSession() as org_session:
                    organized_md = await _organize_manual_excerpts_llm(org_session, q, hits_focused)
            except Exception as e:
                print(f"[RAG] manual organize skipped: {e!s}")
                organized_md = ""
        context = _format_context(hits_focused)
        bridge = _llm_spelling_bridge(q, hits_focused)
        comm_for_llm = list(comm_matches)
        comm_for_display = [m for m in comm_for_llm if _community_confident_for_display(q, m)]
        lang_directive = _answer_language_directive(q)
        parts = [
            "Retrieval note: The excerpts below are **focused** on the best-matching passages for this "
            "question (Excerpt 1 = strongest match). Prefer Excerpt 1 when it directly answers the question.\n\n"
            f"Manual excerpts:\n\n{context}",
        ]
        if organized_md.strip():
            parts.append(
                "---\nStructured excerpt view (same indexed text as above; no new facts):\n"
                + organized_md.strip()
            )
        if comm_for_llm:
            parts.append(_community_context_for_llm(comm_for_llm))
        parts.append(f"---\n\n{bridge}User question: {q}")
        parts.append(f"---\n\n{lang_directive}")
        user_content = "\n\n".join(parts)
        messages: list[dict[str, str]] = [{"role": "system", "content": RAG_SYSTEM}]
        for h in hist:
            messages.append({"role": h["role"], "content": h["content"]})
        messages.append({"role": "user", "content": user_content})
        return hits_focused, organized_md, comm_for_display, messages

    def _finalize_manual_answer(
        q: str,
        answer: str,
        hits_focused: list[tuple[dict, float]],
        organized_md: str,
        comm_for_display: list[dict],
    ) -> str:
        out = _strip_model_community_section(answer or "")
        out = _strip_legacy_manual_part_from_llm(out)
        if _should_include_manual_front(q):
            manual_front = _format_manual_answer_preamble(hits_focused, organized_md or None)
            out = f"{manual_front}\n\n---\n\n{out.strip()}".strip()
        else:
            out = out.strip()
        if comm_for_display:
            out = out.rstrip() + _format_community_answer_append(comm_for_display)
        return out

    def _manual_chat_payload(
        answer: str,
        used_model: str,
        hits_focused: list[tuple[dict, float]],
        comm_for_display: list[dict],
    ) -> dict:
        return {
            "answer": answer,
            "model_used": used_model,
            "manual_pdf_available": bool(_resolve_active_manual_pdf_path()),
            "sources": [
                {
                    "page": h[0].get("page"),
                    "page_end": h[0].get("page_end"),
                    "score": round(h[1], 4),
                }
                for h in hits_focused
            ],
            "community_matches": _community_matches_api(comm_for_display),
        }

    @app.post("/api/chat")
    async def chat(body: ChatBody):
        q = (body.message or "").strip()
        if not q:
            raise HTTPException(400, "message is empty")
        selected_model = _safe_selected_model(body.model)
        primary_model = selected_model or CHAT_MODEL
        fallback_model = None if selected_model else CHAT_FALLBACK_MODEL
        hist = _sanitize_manual_history(body.history)
        hits_focused, organized_md, comm_for_display, messages = await _prepare_manual_chat_context(q, hist)
        chat_options = {"num_predict": MAX_TOKENS, "temperature": 0.05}
        primary_err = None
        used_model = primary_model
        try:
            async with aiohttp.ClientSession() as session:
                answer = await ollama_chat(
                    session,
                    primary_model,
                    messages,
                    stream=False,
                    options=chat_options,
                    timeout_s=CHAT_TIMEOUT_S,
                )
        except Exception as e:
            primary_err = str(e)
            answer = ""
        if not answer and fallback_model and fallback_model != primary_model:
            used_model = fallback_model
            try:
                async with aiohttp.ClientSession() as session:
                    answer = await ollama_chat(
                        session,
                        fallback_model,
                        messages,
                        stream=False,
                        options=chat_options,
                        timeout_s=CHAT_TIMEOUT_S,
                    )
            except Exception as e:
                raise HTTPException(
                    502,
                    f"Ollama chat failed. Primary ({primary_model}): {primary_err}. "
                    f"Fallback ({fallback_model}): {e}",
                ) from e
        if not answer:
            raise HTTPException(502, f"Ollama chat failed. Primary ({primary_model}): {primary_err}")
        final_answer = _finalize_manual_answer(q, answer, hits_focused, organized_md, comm_for_display)
        return _manual_chat_payload(final_answer, used_model, hits_focused, comm_for_display)

    @app.post("/api/chat-stream")
    async def chat_stream(body: ChatBody):
        q = (body.message or "").strip()
        if not q:
            raise HTTPException(400, "message is empty")
        selected_model = _safe_selected_model(body.model)
        primary_model = selected_model or CHAT_MODEL
        fallback_model = None if selected_model else CHAT_FALLBACK_MODEL
        hist = _sanitize_manual_history(body.history)
        hits_focused, organized_md, comm_for_display, messages = await _prepare_manual_chat_context(q, hist)
        chat_options = {"num_predict": MAX_TOKENS, "temperature": 0.05}

        async def event_gen():
            used_model = primary_model
            primary_err = None
            pieces: list[str] = []
            yielded_any = False
            try:
                async with aiohttp.ClientSession() as session:
                    async for delta in ollama_chat_stream(
                        session,
                        primary_model,
                        messages,
                        options=chat_options,
                        timeout_s=CHAT_TIMEOUT_S,
                    ):
                        yielded_any = True
                        pieces.append(delta)
                        yield json.dumps({"type": "token", "delta": delta}, ensure_ascii=False) + "\n"
            except Exception as e:
                primary_err = str(e)

            if not yielded_any and fallback_model and fallback_model != primary_model:
                used_model = fallback_model
                try:
                    async with aiohttp.ClientSession() as session:
                        async for delta in ollama_chat_stream(
                            session,
                            fallback_model,
                            messages,
                            options=chat_options,
                            timeout_s=CHAT_TIMEOUT_S,
                        ):
                            yielded_any = True
                            pieces.append(delta)
                            yield json.dumps({"type": "token", "delta": delta}, ensure_ascii=False) + "\n"
                except Exception as e:
                    detail = (
                        f"Ollama chat failed. Primary ({primary_model}): {primary_err}. "
                        f"Fallback ({fallback_model}): {e}"
                    )
                    yield json.dumps({"type": "error", "detail": detail}, ensure_ascii=False) + "\n"
                    return

            raw_answer = "".join(pieces).strip()
            if not raw_answer:
                detail = f"Ollama chat failed. Primary ({primary_model}): {primary_err}"
                yield json.dumps({"type": "error", "detail": detail}, ensure_ascii=False) + "\n"
                return
            final_answer = _finalize_manual_answer(q, raw_answer, hits_focused, organized_md, comm_for_display)
            payload = _manual_chat_payload(final_answer, used_model, hits_focused, comm_for_display)
            payload["type"] = "done"
            yield json.dumps(payload, ensure_ascii=False) + "\n"

        return StreamingResponse(event_gen(), media_type="application/x-ndjson")

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

    @app.get("/api/community")
    async def community_list(limit: int = 100, _: None = Depends(require_admin)):
        if community_store is None:
            raise HTTPException(
                503,
                "Community tips are disabled or Chroma is unavailable. "
                "Install chromadb (pip install chromadb) and set COMMUNITY_ENABLED=1.",
            )
        try:
            rows = await asyncio.to_thread(lambda: community_store.list_tips(limit=limit))
        except Exception as e:
            raise HTTPException(502, f"Failed to list community tips: {e}") from e
        return {"ok": True, "count": len(rows), "items": rows}

    @app.delete("/api/community/{tip_id}")
    async def community_delete(tip_id: str, _: None = Depends(require_admin)):
        if community_store is None:
            raise HTTPException(
                503,
                "Community tips are disabled or Chroma is unavailable. "
                "Install chromadb (pip install chromadb) and set COMMUNITY_ENABLED=1.",
            )
        tid = (tip_id or "").strip()
        if not tid:
            raise HTTPException(400, "tip id is empty")
        try:
            deleted = await asyncio.to_thread(lambda: community_store.delete_tip(tid))
        except Exception as e:
            raise HTTPException(502, f"Failed to delete community tip: {e}") from e
        if not deleted:
            raise HTTPException(404, "Community tip not found")
        return {"ok": True, "deleted": True, "id": tid}

    @app.put("/api/community/{tip_id}")
    async def community_update(
        tip_id: str,
        body: CommunityUpdateBody,
        _: None = Depends(require_admin),
    ):
        if community_store is None:
            raise HTTPException(
                503,
                "Community tips are disabled or Chroma is unavailable. "
                "Install chromadb (pip install chromadb) and set COMMUNITY_ENABLED=1.",
            )
        tid = (tip_id or "").strip()
        if not tid:
            raise HTTPException(400, "tip id is empty")
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
            existing = await asyncio.to_thread(lambda: community_store.get_tip(tid))
        except Exception as e:
            raise HTTPException(502, f"Failed to load community tip: {e}") from e
        if not existing:
            raise HTTPException(404, "Community tip not found")
        try:
            async with aiohttp.ClientSession() as session:
                q_boost = _embedding_query_boost(q[:COMMUNITY_SAVE_QUESTION_MAX])
                emb = await ollama_embed(session, q_boost, EMBED_MODEL)
            ok = await asyncio.to_thread(
                lambda: community_store.update_tip(
                    tid,
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
            raise HTTPException(502, f"Failed to update community tip: {e}") from e
        if not ok:
            raise HTTPException(404, "Community tip not found")
        return {"ok": True, "id": tid}

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
