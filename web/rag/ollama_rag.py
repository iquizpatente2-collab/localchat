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
        return 7000
    if "qwen3-embedding" in ml or "qwen3_embedding" in ml:
        return 7000
    if "snowflake" in ml or "arctic-embed" in ml:
        return 7000
    # minilm, e5-small, many Ollama embedding ports — very small context
    return 512


def _truncate_for_embedding(text: str, model: str) -> str:
    text = text or ""
    if not text:
        return text
    max_tok = _embed_cap_tokens(model)
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
    # Final hard ceiling: some servers still stricter than cl100k estimate
    char_ceiling = int(os.environ.get("RAG_EMBED_INPUT_MAX_CHARS", "0"))
    if char_ceiling > 0:
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


async def ollama_embed(session: aiohttp.ClientSession, text: str, model: str) -> np.ndarray:
    """Try modern /api/embed first, then legacy /api/embeddings."""
    text = _truncate_for_embedding(text or "", model)
    base = _ollama_base()
    attempts: list[tuple[str, dict[str, Any]]] = [
        (f"{base}/api/embed", {"model": model, "input": text}),
        (f"{base}/api/embeddings", {"model": model, "prompt": text}),
    ]
    last_err = ""
    for url, payload in attempts:
        async with session.post(url, json=payload) as resp:
            body = await resp.text()
            if resp.status != 200:
                last_err = f"{url} -> HTTP {resp.status}: {body[:500]}"
                continue
            try:
                data = await resp.json()
            except Exception:
                last_err = f"{url} -> invalid JSON"
                continue
        vec = _parse_embedding_vector(data)
        if vec is not None:
            return vec
        last_err = f"{url} -> unexpected JSON keys: {list(data.keys())}"
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
    return max(1, int(os.environ.get("RAG_EMBED_CONCURRENCY", "16")))


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
    if conc == 1:
        vecs: list[np.ndarray] = []
        for i, t in enumerate(texts):
            vecs.append(await ollama_embed(session, t, model))
            if batch_pause and i + 1 < len(texts):
                await asyncio.sleep(batch_pause)
        return np.stack(vecs, axis=0)

    sem = asyncio.Semaphore(conc)
    n = len(texts)
    out: list[np.ndarray | None] = [None] * n

    async def one(i: int, t: str) -> None:
        async with sem:
            out[i] = await ollama_embed(session, t, model)

    await asyncio.gather(*(one(i, t) for i, t in enumerate(texts)))
    return np.stack([out[i] for i in range(n)], axis=0)
