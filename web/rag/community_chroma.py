"""
Local ChromaDB store for user-contributed "community" tips linked to questions.

Embeddings are supplied by the caller (same Ollama model as RAG) so vectors stay
compatible with manual retrieval space.
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

try:
    import chromadb
except ImportError:
    chromadb = None


class CommunitySolutionsStore:
    """Persistent Chroma collection: one row per saved tip."""

    def __init__(self, persist_dir: Path) -> None:
        if chromadb is None:
            raise RuntimeError("chromadb is not installed (pip install chromadb)")
        persist_dir = Path(persist_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._path = persist_dir
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._col = self._client.get_or_create_collection(
            name="community_solutions",
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def path(self) -> Path:
        return self._path

    def count(self) -> int:
        return int(self._col.count())

    def add_tip(
        self,
        *,
        question: str,
        embedding: np.ndarray,
        comment: str,
        author: str,
        answer_excerpt: str,
    ) -> str:
        rid = str(uuid.uuid4())
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        ts = int(time.time())
        self._col.add(
            ids=[rid],
            embeddings=[vec.tolist()],
            documents=[question.strip()[:8000]],
            metadatas=[
                {
                    "comment": comment.strip()[:4000],
                    "author": (author or "Anonymous").strip()[:120] or "Anonymous",
                    "answer_excerpt": (answer_excerpt or "").strip()[:4000],
                    "saved_ts": str(ts),
                }
            ],
        )
        return rid

    def delete_tip(self, tip_id: str) -> bool:
        tid = (tip_id or "").strip()
        if not tid:
            return False
        try:
            got = self._col.get(ids=[tid], include=[])
            if not (got.get("ids") or []):
                return False
            self._col.delete(ids=[tid])
            return True
        except Exception:
            return False

    def get_tip(self, tip_id: str) -> dict[str, Any] | None:
        tid = (tip_id or "").strip()
        if not tid:
            return None
        try:
            got = self._col.get(ids=[tid], include=["documents", "metadatas"])
            ids = got.get("ids") or []
            if not ids:
                return None
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            meta = metas[0] if metas else {}
            doc = docs[0] if docs else ""
            ts_raw = meta.get("saved_ts", "0")
            try:
                saved_ts = int(float(str(ts_raw)))
            except (TypeError, ValueError):
                saved_ts = 0
            return {
                "id": str(ids[0]),
                "question": (doc or "")[:2000],
                "comment": str(meta.get("comment") or "")[:4000],
                "author": str(meta.get("author") or "Anonymous")[:120],
                "answer_excerpt": str(meta.get("answer_excerpt") or "")[:4000],
                "saved_ts": saved_ts,
            }
        except Exception:
            return None

    def update_tip(
        self,
        tip_id: str,
        *,
        question: str,
        embedding: np.ndarray,
        comment: str,
        author: str,
        answer_excerpt: str,
    ) -> bool:
        tid = (tip_id or "").strip()
        if not tid:
            return False
        try:
            got = self._col.get(ids=[tid], include=[])
            if not (got.get("ids") or []):
                return False
        except Exception:
            return False
        vec = np.asarray(embedding, dtype=np.float32).reshape(-1)
        ts = int(time.time())
        try:
            self._col.update(
                ids=[tid],
                embeddings=[vec.tolist()],
                documents=[question.strip()[:8000]],
                metadatas=[
                    {
                        "comment": comment.strip()[:4000],
                        "author": (author or "Anonymous").strip()[:120] or "Anonymous",
                        "answer_excerpt": (answer_excerpt or "").strip()[:4000],
                        "saved_ts": str(ts),
                    }
                ],
            )
            return True
        except Exception:
            return False

    def list_tips(self, *, limit: int = 100) -> list[dict[str, Any]]:
        if self._col.count() == 0:
            return []
        k = max(1, min(int(limit), self._col.count()))
        got = self._col.get(include=["documents", "metadatas"], limit=k)
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        out: list[dict[str, Any]] = []
        for i, rid in enumerate(ids):
            meta = metas[i] if i < len(metas) and metas[i] else {}
            doc = docs[i] if i < len(docs) else ""
            ts_raw = meta.get("saved_ts", "0")
            try:
                saved_ts = int(float(str(ts_raw)))
            except (TypeError, ValueError):
                saved_ts = 0
            out.append(
                {
                    "id": str(rid or ""),
                    "question": (doc or "")[:2000],
                    "comment": str(meta.get("comment") or "")[:4000],
                    "author": str(meta.get("author") or "Anonymous")[:120],
                    "answer_excerpt": str(meta.get("answer_excerpt") or "")[:4000],
                    "saved_ts": saved_ts,
                }
            )
        out.sort(key=lambda x: x.get("saved_ts", 0), reverse=True)
        return out

    def query_similar(
        self,
        query_embedding: np.ndarray,
        *,
        n_results: int,
        max_distance: float,
        inject_max: int,
    ) -> list[dict[str, Any]]:
        """
        Cosine space: Chroma distance ≈ 1 - cosine_similarity. Lower is more similar.
        """
        if inject_max <= 0 or self._col.count() == 0:
            return []
        vec = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        k = min(max(1, n_results), self._col.count())
        res = self._col.query(
            query_embeddings=[vec.tolist()],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids_list = res.get("ids") or []
        dist_list = res.get("distances") or []
        docs_list = res.get("documents") or []
        meta_list = res.get("metadatas") or []
        if not ids_list or not dist_list:
            return []
        ids0 = ids_list[0]
        dist0 = dist_list[0]
        docs0 = docs_list[0] if docs_list else []
        meta0 = meta_list[0] if meta_list else []

        out: list[dict[str, Any]] = []
        for i, dist in enumerate(dist0):
            if dist is None or float(dist) > max_distance:
                continue
            meta = meta0[i] if i < len(meta0) and meta0[i] else {}
            doc = docs0[i] if i < len(docs0) else ""
            ts_raw = meta.get("saved_ts", "0")
            try:
                saved_ts = int(float(str(ts_raw)))
            except (TypeError, ValueError):
                saved_ts = 0
            out.append(
                {
                    "id": ids0[i] if i < len(ids0) else "",
                    "distance": float(dist),
                    "question": (doc or "")[:2000],
                    "comment": str(meta.get("comment") or "")[:4000],
                    "author": str(meta.get("author") or "Anonymous")[:120],
                    "answer_excerpt": str(meta.get("answer_excerpt") or "")[:4000],
                    "saved_ts": saved_ts,
                }
            )
            if len(out) >= inject_max:
                break
        return out
