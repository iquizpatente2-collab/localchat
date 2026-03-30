"""
Structured recipe index: JSON records + embeddings, fuzzy + semantic hybrid search.

Uses FAISS IndexFlatIP when `faiss` is installed; otherwise numpy cosine (Pi-friendly).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
from rapidfuzz import fuzz

try:
    import faiss

    _HAS_FAISS = True
except ImportError:
    faiss = None  # type: ignore
    _HAS_FAISS = False

# Public flag for status APIs
FAISS_AVAILABLE = _HAS_FAISS

from web.rag.recipe_parse import build_keywords, parse_structured_recipe

# Optional query correction (heavy dependency — off unless enabled)
def maybe_spell_correct(query: str) -> str:
    if os.environ.get("RECIPE_QUERY_SPELLCHECK", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
    }:
        return query
    try:
        from textblob import TextBlob

        return str(TextBlob(query).correct())
    except Exception:
        return query


_MEAT_TOKENS = frozenset({"meat", "meats"})
_LIGHT_STOP = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "and",
        "or",
        "to",
        "for",
        "with",
        "di",
        "de",
        "la",
        "is",
        "what",
        "how",
        "make",
        "prepare",
        "recipe",
    }
)


def expand_query_for_embedding(query: str) -> str:
    """Append related tokens for embedding only (does not change user-facing query)."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    extra: list[str] = []
    if words & _MEAT_TOKENS:
        extra.extend(["mutton", "beef", "veal", "lamb", "pork"])
    if words & {"cutlet", "cutlets", "mutton", "veal", "filet", "fillet"}:
        extra.extend(
            [
                "mutton cutlets",
                "mutton cutlet",
                "filet of veal",
                "veal",
                "braciuole",
                "cutlets",
            ]
        )
    if any(x in words for x in ("salsa", "sauce", "pomodoro", "pomidoro")):
        extra.extend(
            [
                "tomato sauce",
                "salsa di pomodoro",
                "salsa di pomidoro",
                "italian tomato sauce",
            ]
        )
    if {"pollo", "fritto"} <= words or {"fried", "chicken"} <= words:
        extra.extend(
            [
                "fried chicken",
                "pollo fritto",
                "chicken fried in oil",
            ]
        )
    if not extra:
        return query
    return f"{query} {' '.join(extra)}"


def query_ngrams(query: str, max_n: int = 3) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) > 1]
    out: list[str] = []
    for n in range(1, min(max_n, len(toks)) + 1):
        for i in range(len(toks) - n + 1):
            out.append(" ".join(toks[i : i + n]))
    return out


def _compact_alnum(s: str) -> str:
    """Strip non-alphanumeric so fused PDF lines still substring-match (e.g. mutton+cutlet)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def embed_text_for_recipe(recipe: dict) -> str:
    title = recipe.get("title") or ""
    body_full = recipe.get("full_text") or ""
    first_line = body_full.split("\n", 1)[0].strip() if body_full else ""
    compact = _compact_alnum(f"{title} {first_line} {body_full[:1800]}")
    kws = " ".join(recipe.get("keywords") or [])
    body = body_full[:2500]
    return f"{title}\n{first_line}\n{compact}\n{kws}\n{body}"


def fuzzy_recipe_score(query: str, recipe: dict) -> float:
    """0..1 from rapidfuzz (partial + token-aware + compact match for spaceless PDF text)."""
    q = query.lower().strip()
    if not q:
        return 0.0
    title = (recipe.get("title") or "").lower()
    kws = " ".join(recipe.get("keywords") or []).lower()
    blob = f"{title} {kws} {(recipe.get('full_text') or '')[:2000]}".lower()
    s1 = fuzz.partial_ratio(q, title)
    s2 = fuzz.partial_ratio(q, blob)
    s3 = fuzz.token_set_ratio(q, title)
    s4 = max((fuzz.partial_ratio(ng, title) for ng in query_ngrams(query)), default=0)
    cq = _compact_alnum(q)
    if len(cq) >= 4:
        ct = _compact_alnum(title)
        cb = _compact_alnum(blob)
        s5 = fuzz.partial_ratio(cq, ct)
        s6 = fuzz.partial_ratio(cq, cb)
    else:
        s5 = s6 = 0
    base = max(s1, s2, s3, s4, s5, s6) / 100.0
    return _query_intent_adjustment(q, title, blob, base)


def _query_intent_adjustment(query: str, title: str, blob: str, score: float) -> float:
    """
    Improve disambiguation for title collisions in old cookbooks.
    Example: "salsa di pomidoro" should prefer TOMATO SAUCE over UOVA AL POMIDORO.
    """
    q_words = {w for w in re.findall(r"[a-z0-9]+", query) if w not in _LIGHT_STOP}
    t_words = set(re.findall(r"[a-z0-9]+", title))
    b_words = set(re.findall(r"[a-z0-9]+", blob))
    words = t_words | b_words

    # If query asks for sauce/salsa, heavily prefer sauce-like candidates.
    asks_sauce = ("salsa" in q_words) or ("sauce" in q_words)
    has_sauce = ("salsa" in words) or ("sauce" in words)
    if asks_sauce and not has_sauce:
        score *= 0.62

    # Tomato-sauce intent bridge across language/OCR variants.
    asks_tomato = any(x in q_words for x in ("pomodoro", "pomidoro", "tomato"))
    title_blob = f"{title} {blob}"
    if asks_sauce and asks_tomato:
        if ("tomato sauce" in title_blob) or ("salsa di pomidoro" in title_blob) or ("salsa di pomodoro" in title_blob):
            score += 0.23
        if ("uova al pomidoro" in title_blob) or ("eggs with tomatoes" in title_blob):
            score -= 0.12

    # Fried chicken bilingual bridge ("fried chicken" <-> "pollo fritto").
    asks_fried_chicken = (
        {"fried", "chicken"} <= q_words
        or {"pollo", "fritto"} <= q_words
    )
    has_fried_chicken = (
        ("fried chicken" in title_blob)
        or ("pollo fritto" in title_blob)
        or ("polio fritto" in title_blob)  # OCR variant observed in this book
    )
    if asks_fried_chicken:
        if has_fried_chicken:
            score += 0.26
        else:
            score *= 0.70

    return float(max(0.0, min(1.0, score)))


def _query_focus_tokens(query: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", query.lower()) if len(t) >= 3]
    return [t for t in toks if t not in _LIGHT_STOP]


def _token_coverage_score(query: str, title: str, blob: str) -> float:
    """
    0..1 ratio of query focus tokens present in candidate text (compact + normal forms).
    Helps keep fuzzy search from latching onto generic matches.
    """
    qtok = _query_focus_tokens(query)
    if not qtok:
        return 1.0
    t_lo = f"{title} {blob}".lower()
    t_compact = _compact_alnum(t_lo)
    hit = 0
    for t in qtok:
        if t in t_lo or _compact_alnum(t) in t_compact:
            hit += 1
    return hit / max(1, len(qtok))


def _query_content_bigrams(query: str) -> list[tuple[str, str]]:
    """Consecutive non-stop tokens (length >= 3) for phrase-level checks."""
    toks = [
        t
        for t in re.findall(r"[a-z0-9]+", query.lower())
        if t not in _LIGHT_STOP and len(t) >= 3
    ]
    if len(toks) < 2:
        return []
    return [(toks[i], toks[i + 1]) for i in range(len(toks) - 1)]


def _bigram_match_score(a: str, b: str, title: str, blob: str) -> float:
    """
    How well a query bigram matches the candidate (1.0 = exact phrase or tight proximity).
    Downweights pages that mention both words far apart (e.g. soup with 'chicken' and
    unrelated 'fried in deep fat' for croutons) when the user asked for a phrase like
    'fried chicken'.
    """
    text = f"{title} {blob}".lower()
    if not text.strip():
        return 1.0
    phrase = f"{a} {b}"
    if phrase in text:
        return 1.0
    ctext = _compact_alnum(text)
    if _compact_alnum(a + b) in ctext:
        return 1.0
    if a not in text or b not in text:
        return 0.45
    # Proximity window: both tokens must appear near each other to count as a real phrase hit.
    span = max(len(a), len(b)) + 96
    i = 0
    while True:
        pa = text.find(a, i)
        if pa < 0:
            break
        win = text[pa : pa + span]
        if b in win:
            return 0.9
        i = pa + 1
    i = 0
    while True:
        pb = text.find(b, i)
        if pb < 0:
            break
        win = text[pb : pb + span]
        if a in win:
            return 0.9
        i = pb + 1
    # Both tokens exist but never in the same local window — weak match.
    return 0.48


def _multi_bigram_score(query: str, title: str, blob: str) -> float:
    pairs = _query_content_bigrams(query)
    if not pairs:
        return 1.0
    scores = [_bigram_match_score(a, b, title, blob) for a, b in pairs]
    return float(min(scores))


def _cosine_matrix(q: np.ndarray, M: np.ndarray) -> np.ndarray:
    q = q.astype(np.float32).reshape(-1)
    q = q / (np.linalg.norm(q) + 1e-10)
    M = M.astype(np.float32)
    norms = np.linalg.norm(M, axis=1, keepdims=True) + 1e-10
    M = M / norms
    return M @ q


def _emb_scores_to_unit_interval(scores: np.ndarray) -> np.ndarray:
    """Cosine in [-1,1] -> [0,1]."""
    return np.clip((scores.astype(np.float64) + 1.0) / 2.0, 0.0, 1.0)


class RecipeCatalog:
    def __init__(self, dir_path: Path):
        self.dir_path = Path(dir_path)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        self._meta_path = self.dir_path / "recipes.json"
        self._emb_path = self.dir_path / "recipe_embeddings.npy"
        self.recipes: list[dict[str, Any]] = []
        self.embeddings: np.ndarray | None = None
        self.source_file: str | None = None
        self._faiss_index: Any = None

    def _rebuild_faiss(self) -> None:
        self._faiss_index = None
        if not _HAS_FAISS or self.embeddings is None or len(self.embeddings) == 0:
            return
        d = int(self.embeddings.shape[1])
        idx = faiss.IndexFlatIP(d)
        mat = self.embeddings.astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-10
        mat = mat / norms
        idx.add(mat)
        self._faiss_index = idx

    def load(self) -> bool:
        if not self._meta_path.exists() or not self._emb_path.exists():
            return False
        try:
            meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
        except Exception:
            return False
        self.recipes = meta.get("recipes", [])
        self.source_file = meta.get("source_file")
        self.embeddings = np.load(self._emb_path)
        if len(self.recipes) != (self.embeddings.shape[0] if self.embeddings is not None else 0):
            return False
        self._rebuild_faiss()
        return len(self.recipes) > 0

    def save(self) -> None:
        meta: dict[str, Any] = {
            "recipes": self.recipes,
            "source_file": self.source_file,
            "faiss_backend": FAISS_AVAILABLE,
        }
        if self.embeddings is not None:
            meta["embed_dim"] = int(self.embeddings.shape[1])
        self._meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.embeddings is not None:
            np.save(self._emb_path, self.embeddings.astype(np.float32))

    def set_recipes_with_embeddings(
        self,
        recipes: list[dict[str, Any]],
        embeddings: np.ndarray,
        source_name: str,
    ) -> None:
        if len(recipes) != embeddings.shape[0]:
            raise ValueError("recipes count must match embedding rows")
        self.recipes = recipes
        self.embeddings = embeddings.astype(np.float32)
        self.source_file = source_name
        self._rebuild_faiss()
        self.save()

    def clear(self) -> None:
        self.recipes = []
        self.embeddings = None
        self.source_file = None
        self._faiss_index = None
        for p in (self._meta_path, self._emb_path):
            if p.exists():
                p.unlink()

    def embedding_search_raw(self, query_vec: np.ndarray, top_k: int) -> list[tuple[int, float]]:
        """Return (index, cosine_similarity) for top_k."""
        if not self.recipes or self.embeddings is None:
            return []
        k = min(top_k, len(self.recipes))
        q = query_vec.astype(np.float32).reshape(-1)

        if self._faiss_index is not None:
            qn = q / (np.linalg.norm(q) + 1e-10)
            qn = qn.reshape(1, -1).astype(np.float32)
            sims, idxs = self._faiss_index.search(qn, k)
            row_sims = sims[0]
            row_idx = idxs[0]
            out: list[tuple[int, float]] = []
            for i in range(k):
                ii = int(row_idx[i])
                if ii < 0:
                    continue
                out.append((ii, float(row_sims[i])))
            return out

        scores = _cosine_matrix(q, self.embeddings)
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i])) for i in idx]

    def combined_search(
        self,
        query: str,
        query_vec: np.ndarray,
        *,
        top_k: int = 5,
        fuzzy_threshold: float = 0.55,
        w_embed: float = 0.6,
        w_fuzzy: float = 0.4,
        embed_pool: int = 24,
    ) -> list[tuple[dict[str, Any], float, dict[str, float]]]:
        """
        Hybrid rank: combined = w_embed * embed_norm + w_fuzzy * fuzzy_norm.
        embed_norm is cosine similarity mapped from [-1,1] to [0,1].
        """
        if not self.recipes or self.embeddings is None:
            return []

        emb_hits = self.embedding_search_raw(query_vec, min(embed_pool, len(self.recipes)))
        candidates: set[int] = {i for i, _ in emb_hits}

        fuzzy_scores: list[float] = [fuzzy_recipe_score(query, r) for r in self.recipes]
        for i, fz in enumerate(fuzzy_scores):
            if fz >= fuzzy_threshold:
                candidates.add(i)

        q = query_vec.astype(np.float32).reshape(-1)
        q = q / (np.linalg.norm(q) + 1e-10)

        scored: list[tuple[int, float, dict[str, float]]] = []
        for i in candidates:
            row = self.embeddings[i].astype(np.float32)
            row = row / (np.linalg.norm(row) + 1e-10)
            cos = float(np.dot(row, q))
            emb_n = (cos + 1.0) / 2.0
            fz = fuzzy_scores[i]
            combined = w_embed * emb_n + w_fuzzy * fz
            title = str(self.recipes[i].get("title") or "")
            blob = str(self.recipes[i].get("full_text") or "")[:2200]
            cov = _token_coverage_score(query, title, blob)
            # Strongly prefer candidates that actually cover the query focus tokens.
            combined = (combined * 0.82) + (cov * 0.18)
            bgram = _multi_bigram_score(query, title, blob)
            combined *= bgram
            scored.append(
                (
                    i,
                    combined,
                    {
                        "embed": emb_n,
                        "fuzzy": fz,
                        "coverage": cov,
                        "bigram": bgram,
                        "cosine_raw": cos,
                    },
                )
            )

        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            (self.recipes[i], comb, parts) for i, comb, parts in scored[:top_k]
        ]


def build_recipe_embeddings_texts(pages: list[tuple[int, str]], source_name: str) -> tuple[list[dict], list[str]]:
    """Same order as pages: one recipe record + one embed string per page."""
    recipes: list[dict] = []
    texts: list[str] = []
    for page_no, text in pages:
        parsed = parse_structured_recipe(text)
        parsed["id"] = f"{source_name}#p{page_no}"
        parsed["page"] = page_no
        parsed["title_lower"] = (parsed.get("title") or "").lower()
        parsed["keywords"] = build_keywords(parsed)
        recipes.append(parsed)
        texts.append(embed_text_for_recipe(parsed))
    return recipes, texts
