"""Local sentence embeddings for the RAG index.

Uses a small sentence-transformers model rather than a hosted embeddings
API: it runs offline/free and keeps the only paid API dependency (Anthropic)
limited to the two steps that actually need a strong LLM - structured
extraction and answer generation.
"""

from __future__ import annotations

import os
import threading

DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_lock = threading.Lock()
_model_cache: dict[str, object] = {}


class Embedder:
    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL):
        self.model_name = model_name
        self._model = self._load(model_name)

    @staticmethod
    def _load(model_name: str):
        with _lock:
            if model_name not in _model_cache:
                from sentence_transformers import SentenceTransformer

                _model_cache[model_name] = SentenceTransformer(model_name)
            return _model_cache[model_name]

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]
