"""Embeddings with a graceful fallback.

Real path: BGE via sentence-transformers. Fallback path: a deterministic hashed
bag-of-words vectorizer in pure numpy, so the demo runs with no model download.
Cosine over the fallback approximates lexical overlap, which is enough to show
dense + sparse fusion working.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

from gridfin.config import get_settings

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Embedder:
    def __init__(self, dim: int = 384):
        self.settings = get_settings()
        self.dim = dim
        self._model = None
        self.backend = "hash"
        if self.settings.has_sentence_transformers:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer("BAAI/bge-small-en-v1.5")
                self.dim = self._model.get_sentence_embedding_dimension()
                self.backend = "bge"
            except Exception:  # pragma: no cover - model download/runtime issues
                self._model = None

    def encode(self, texts: list[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True)
            return np.asarray(vecs, dtype=np.float32)
        return self._hash_encode(texts)

    def _hash_encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in _tokens(text):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                idx = h % self.dim
                sign = 1.0 if (h >> 7) & 1 else -1.0
                out[i, idx] += sign
            norm = np.linalg.norm(out[i])
            if norm > 0:
                out[i] /= norm
        return out
