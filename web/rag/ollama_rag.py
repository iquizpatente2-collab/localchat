import asyncio
import json
import os
from typing import Any, AsyncIterator

import aiohttp
import numpy as np


def _ollama_base() -> str:
    return os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def _embed_cap_tokens(model: str) -> int:
    """
    Ollama enforces the *model's* tokenizer context (often 512); cl100k length is only an estimate.
    If RAG_EMBED_INPUT_MAX_TOKENS is set, it wins. Otherwise pick a safe default from the model id.
    """
    raw = os.environ.get("RAG_EMBED_INPUT_MAX_TOKENS", "").strip()
    if raw.isdigit():
        return max(32, int(raw))
    ml = (model or "").lower()
    # Stay below published limits; server tokenizer can count higher than cl100k.
    if "nomic-embed" in ml or ml.startswith("nomic") or "nomic_embed" in ml:
        # Ollama's tokenizer often stricter than cl100k; 2048 is safe on all builds.
        return 2048
    if "qwen3-embedding" in ml or "qwen3_embedding" in ml:
        return 2048
    if "snowflake" in ml or "arctic-embed" in ml:
        return 2048
    # minilm, e5-small, many Ollama embedding ports — very small context
    return 512


def _default_embed_char_cap(model: str) -> int:
    raw = os.environ.get("RAG_EMBED_INPUT_MAX_CHARS", "").strip()
    if raw.isdigit():
        return max(256, int(raw))
    ml = (model or "").lower()
    if "nomic" in ml or "qwen3-embed" in ml:
        return 6000
    return 2048


def _truncate_for_embedding(
    text: str,
    model: str,
    *,
    max_tok_override: int | None = None,
) -> str:
    text = text or ""
    if not text:
        return text
    max_tok = max_tok_override if max_tok_override is not None else _embed_cap_tokens(model)
    # Avoid huge encode() cost on pathological strings
    pre_cap = min(len(text), max(12_000, max_tok * 12))
    text = text[:pre_cap]
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        ids = enc.encode(text)
        if len(ids) > max_tok:
            text = enc.decode(ids[:max_tok])
    except Exception:
        # ~4 chars/token heuristic when tiktoken missing or encode fails
        text = text[: max(max_tok * 4, 512)]
    # Hard character ceiling — Ollama may count tokens higher than cl100k for technical PDF text.
    char_ceiling = _default_embed_char_cap(model)
    if max_tok_override is not None:
        char_ceiling = min(char_ceiling, max(max_tok_override * 3, 512))
    text = text[:char_ceiling]
    return text


def _parse_embedding_vector(data: dict[str, Any]) -> np.ndarray | None:
    embs = data.get("embeddings")
    if isinstance(embs, list) and len(embs) > 0:
        return np.array(embs[0], dtype=np.float32)
    emb = data.get("embedding")
    if emb is not None:
        return np.array(emb, dtype=np.float32)
    return None


_embed_inflight: asyncio.Semaphore | None = None


def _embed_inflight_sem() -> asyncio.Semaphore:
    global _embed_inflight
    if _embed_inflight is None:
        cap = max(1, int(os.environ.get("OLLAMA_EMBED_MAX_INFLIGHT", "1")))
        _embed_inflight = asyncio.Semaphore(cap)
    return _embed_inflight


def _embed_retry_limit() -> int:
    return max(1, int(os.environ.get("RAG_EMBED_RETRY_MAX", "24")))


def _embed_retryable_status(status: int) -> bool:
    return status in (429, 503, 502, 504)


def _embed_context_too_long(status: int, body: str) -> bool:
    if status not in (400, 500):
        return False
    bl = (body or "").lower()
    return "context length" in bl or "too long" in bl or "maximum context" in bl


async def ollama_embed(session: aiohttp.ClientSession, text: str, model: str) -> np.ndarray:
    """Try modern /api/embed first, then legacy /api/embeddings."""
    async with _embed_inflight_sem():
        base = _ollama_base()
        raw = text or ""
        tok_limit = _embed_cap_tokens(model)
        timeout_s = float(os.environ.get("RAG_EMBED_TIMEOUT_S", "300"))
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        last_err = ""
        max_tries = _embed_retry_limit()

        for shrink in range(5):
            limit = max(64, tok_limit // (2**shrink))
            chunk = _truncate_for_embedding(raw, model, max_tok_override=limit)
            endpoints: list[tuple[str, dict[str, Any]]] = [
                (f"{base}/api/embed", {"model": model, "input": chunk}),
                (f"{base}/api/embeddings", {"model": model, "prompt": chunk}),
            ]
            too_long = False
            for url, payload in endpoints:
                for attempt in range(max_tries):
                    async with session.post(url, json=payload, timeout=timeout) as resp:
                        body = await resp.text()
                        if resp.status != 200:
                            last_err = f"{url} -> HTTP {resp.status}: {body[:500]}"
                            if _embed_context_too_long(resp.status, body):
                                too_long = True
                                break
                            if _embed_retryable_status(resp.status) and attempt + 1 < max_tries:
                                await asyncio.sleep(min(90.0, 2.0 * (2**attempt)))
                                continue
                            break
                        try:
                            data = await resp.json()
                        except Exception:
                            last_err = f"{url} -> invalid JSON"
                            break
                    vec = _parse_embedding_vector(data)
                    if vec is not None:
                        return vec
                    last_err = f"{url} -> unexpected JSON keys: {list(data.keys())}"
                    break
                if too_long:
                    break
            if too_long:
                continue

        raise RuntimeError(f"Embeddings failed. Last error: {last_err}")


async def ollama_chat(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, str]],
    stream: bool = False,
    options: dict[str, Any] | None = None,
    timeout_s: float = 180.0,
) -> str:
    url = f"{_ollama_base()}/api/chat"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    # Qwen reasoning models can return huge "thinking" traces and sometimes empty content.
    # Default to think=false unless explicitly enabled.
    think_raw = os.environ.get("OLLAMA_THINK", "0").strip().lower()
    if think_raw in ("0", "false", "no", "off"):
        payload["think"] = False
    elif think_raw in ("1", "true", "yes", "on"):
        payload["think"] = True
    if options:
        payload["options"] = options
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Chat HTTP {resp.status}: {body}")
        data = await resp.json()
    return (data.get("message") or {}).get("content", "").strip()


async def ollama_chat_stream(
    session: aiohttp.ClientSession,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
    timeout_s: float = 180.0,
) -> AsyncIterator[str]:
    """Yield incremental chat content chunks from Ollama /api/chat stream."""
    url = f"{_ollama_base()}/api/chat"
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": True}
    think_raw = os.environ.get("OLLAMA_THINK", "0").strip().lower()
    if think_raw in ("0", "false", "no", "off"):
        payload["think"] = False
    elif think_raw in ("1", "true", "yes", "on"):
        payload["think"] = True
    if options:
        payload["options"] = options
    timeout = aiohttp.ClientTimeout(total=timeout_s)
    async with session.post(url, json=payload, timeout=timeout) as resp:
        if resp.status != 200:
            body = await resp.text()
            raise RuntimeError(f"Chat HTTP {resp.status}: {body}")
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = data.get("message") or {}
            delta = msg.get("content")
            if isinstance(delta, str) and delta:
                yield delta


def _embed_concurrency() -> int:
    return max(1, int(os.environ.get("RAG_EMBED_CONCURRENCY", "1")))


def _embed_pause_s() -> float:
    raw = os.environ.get("RAG_EMBED_PAUSE_S", "0.75").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 0.75


async def embed_many(
    session: aiohttp.ClientSession,
    texts: list[str],
    model: str,
    batch_pause: float = 0.0,
    *,
    concurrency: int | None = None,
) -> np.ndarray:
    """
    Embeddings in parallel (bounded) so Ollama can keep the GPU busier than strict serial calls.
    Set RAG_EMBED_CONCURRENCY (default 16) or pass concurrency=.
    (batch_pause is ignored when concurrency > 1.)
    """
    if not texts:
        raise ValueError("embed_many: empty texts")
    conc = max(1, concurrency if concurrency is not None else _embed_concurrency())
    pause = batch_pause if batch_pause > 0 else _embed_pause_s()
    if conc == 1:
        vecs: list[np.ndarray] = []
        n = len(texts)
        log_every = max(1, int(os.environ.get("RAG_EMBED_PROGRESS_EVERY", "25")))
        for i, t in enumerate(texts):
            vecs.append(await ollama_embed(session, t, model))
            if (i + 1) == n or (i + 1) % log_every == 0:
                print(f"[RAG] Embeddings progress: {i + 1}/{n}")
            if pause and i + 1 < n:
                await asyncio.sleep(pause)
        return np.stack(vecs, axis=0)

    sem = asyncio.Semaphore(conc)
    n = len(texts)
    out: list[np.ndarray | None] = [None] * n
    done = 0
    progress_lock = asyncio.Lock()
    log_every = max(1, int(os.environ.get("RAG_EMBED_PROGRESS_EVERY", "25")))

    async def one(i: int, t: str) -> None:
        nonlocal done
        async with sem:
            out[i] = await ollama_embed(session, t, model)
        async with progress_lock:
            done += 1
            if done == n or done % log_every == 0:
                print(f"[RAG] Embeddings progress: {done}/{n}")

    await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
    return np.stack([out[i] for i in range(n)], axis=0)
