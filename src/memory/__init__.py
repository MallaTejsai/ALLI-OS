"""ALLI v0.5 semantic memory package."""

from .config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    LONG_TERM_MEMORY_FILE,
    MEMORY_SIMILARITY_THRESHOLD,
)
from .embeddings import EmbeddingError, EmbeddingStore, cosine_similarity
from .extractor import MemoryExtractor
from .manager import SafeMemoryManager
from .query_intent import QueryIntentDetector
from .retriever import MemoryRetriever
from .store import MemoryStore, load_json_list, save_json

__all__ = [
    "CHAT_MODEL",
    "EMBEDDING_MODEL",
    "LONG_TERM_MEMORY_FILE",
    "MEMORY_SIMILARITY_THRESHOLD",
    "EmbeddingError",
    "EmbeddingStore",
    "cosine_similarity",
    "MemoryExtractor",
    "SafeMemoryManager",
    "QueryIntentDetector",
    "MemoryRetriever",
    "MemoryStore",
    "load_json_list",
    "save_json",
]
