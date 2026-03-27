import json
from pathlib import Path

import numpy as np


class VectorStore:
    """Simple numpy cosine store persisted as JSON + .npy."""

    def __init__(self, dir_path: Path):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir_path / "meta.json"
        self._emb_path = self.dir_path / "embeddings.npy"
        self.chunks: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self.source_file: str | None = None

    def load(self) -> bool:
        if not self._meta_path.exists() or not self._emb_path.exists():
            return False
        with open(self._meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.chunks = meta.get("chunks", [])
        self.source_file = meta.get("source_file")
        self.embeddings = np.load(self._emb_path)
        return len(self.chunks) > 0 and self.embeddings is not None

    def save(self) -> None:
        meta = {
            "chunks": self.chunks,
            "source_file": self.source_file,
        }
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if self.embeddings is not None:
            np.save(self._emb_path, self.embeddings.astype(np.float32))

    def set_data(
        self,
        chunks: list[dict],
        embeddings: np.ndarray,
        source_file: str | None = None,
    ) -> None:
        if len(chunks) != embeddings.shape[0]:
            raise ValueError("chunks and embeddings length mismatch")
        self.chunks = chunks
        self.embeddings = embeddings
        if source_file is not None:
            self.source_file = source_file

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[tuple[dict, float]]:
        if not self.chunks or self.embeddings is None:
            return []
        q = query_embedding.astype(np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-10)
        M = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-10
        M = M / norms
        scores = M @ q
        k = min(top_k, len(scores))
        idx = np.argsort(-scores)[:k]
        return [(self.chunks[int(i)], float(scores[i])) for i in idx]
