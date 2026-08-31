"""Embedding generation and cosine similarity for ALLI v0.5."""

from __future__ import annotations

import os
import math
from typing import Iterable, Optional, Sequence

import ollama

from . import config


class EmbeddingError(RuntimeError):
    """Raised when the embedding model is missing or unavailable."""


class EmbeddingStore:
    """Generate and cache text embeddings via Ollama."""

    def __init__(self, model: Optional[str] = None, client=None):
        self.model = model or config.EMBEDDING_MODEL
        # Use explicit client if provided, otherwise use the configured Ollama host
        if client is not None:
            self._client = client
        else:
            # Read OLLAMA_HOST from environment, default to 127.0.0.1:11434
            self._client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self._cache: dict[str, list[float]] = {}
        self._verified = False

    def verify_model(self) -> None:
        """Ensure the configured embedding model is installed and usable."""
        if self._verified:
            return

        try:
            listed = self._client.list()
        except Exception as error:
            raise EmbeddingError(
                "Memory embeddings are unavailable because Ollama is not "
                f"reachable: {error}"
            ) from error

        names = set()
        models = getattr(listed, "models", None)
        if models is None and isinstance(listed, dict):
            models = listed.get("models", [])

        for item in models or []:
            name = getattr(item, "model", None) or getattr(item, "name", None)
            if name is None and isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if name:
                names.add(str(name))
                # Ollama may report "all-minilm:latest"
                names.add(str(name).split(":")[0])

        short = self.model.split(":")[0]
        if self.model not in names and short not in names:
            raise EmbeddingError(
                f"Embedding model '{self.model}' is not installed.\n"
                f"Install it with: ollama pull {self.model}"
            )

        # Probe that embeddings actually work for this model.
        try:
            probe = self._client.embed(model=self.model, input="alli probe")
            vectors = _extract_embeddings(probe)
            if not vectors or not vectors[0]:
                raise EmbeddingError(
                    f"Embedding model '{self.model}' did not return vectors."
                )
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(
                "Memory embeddings are unavailable because the embedding "
                f"model is not installed/running: {error}"
            ) from error

        self._verified = True

    def generate_embedding(self, text: str) -> list[float]:
        """Embed a single string, using an in-process cache."""
        key = text.strip()
        if not key:
            return []

        cached = self._cache.get(key)
        if cached is not None:
            return list(cached)

        self.verify_model()
        try:
            response = self._client.embed(model=self.model, input=key)
            vectors = _extract_embeddings(response)
        except EmbeddingError:
            raise
        except Exception as error:
            raise EmbeddingError(
                "Memory embeddings are unavailable because the embedding "
                f"model is not installed/running: {error}"
            ) from error

        if not vectors or not vectors[0]:
            raise EmbeddingError(
                f"Embedding model '{self.model}' returned an empty embedding."
            )

        vector = [float(x) for x in vectors[0]]
        self._cache[key] = vector
        return list(vector)

    def generate_embeddings(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed multiple strings, preferring a single batch call."""
        if not texts:
            return []

        results: list[Optional[list[float]]] = [None] * len(texts)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        for index, text in enumerate(texts):
            key = text.strip()
            if not key:
                results[index] = []
                continue
            cached = self._cache.get(key)
            if cached is not None:
                results[index] = list(cached)
            else:
                missing_indices.append(index)
                missing_texts.append(key)

        if missing_texts:
            self.verify_model()
            try:
                response = self._client.embed(
                    model=self.model,
                    input=missing_texts,
                )
                vectors = _extract_embeddings(response)
            except EmbeddingError:
                raise
            except Exception as error:
                raise EmbeddingError(
                    "Memory embeddings are unavailable because the embedding "
                    f"model is not installed/running: {error}"
                ) from error

            if len(vectors) != len(missing_texts):
                # Fall back to one-by-one if batch shape is unexpected.
                for index, text in zip(missing_indices, missing_texts):
                    results[index] = self.generate_embedding(text)
            else:
                for index, text, vector in zip(
                    missing_indices, missing_texts, vectors
                ):
                    clean = [float(x) for x in vector]
                    self._cache[text] = clean
                    results[index] = list(clean)

        return [item if item is not None else [] for item in results]

    def clear_cache(self) -> None:
        self._cache.clear()


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine similarity in [-1, 1], or 0.0 for empty/mismatched vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b):
        dot += x * y
        norm_a += x * x
        norm_b += y * y

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0

    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _extract_embeddings(response) -> list[list[float]]:
    embeddings = getattr(response, "embeddings", None)
    if embeddings is None and isinstance(response, dict):
        embeddings = response.get("embeddings")
    if embeddings is None:
        # Older single-embedding shape
        embedding = getattr(response, "embedding", None)
        if embedding is None and isinstance(response, dict):
            embedding = response.get("embedding")
        if embedding is not None:
            return [list(embedding)]
        return []
    return [list(item) for item in embeddings]
