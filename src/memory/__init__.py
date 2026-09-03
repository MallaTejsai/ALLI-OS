"""ALLI v0.6 semantic memory package."""

from .config import (
    CHAT_MODEL,
    EMBEDDING_MODEL,
    LONG_TERM_MEMORY_FILE,
    MEMORY_SIMILARITY_THRESHOLD,
)
from .embeddings import EmbeddingError, EmbeddingStore, cosine_similarity
from .entity_resolver import extract_entity, EntityExtraction, Entity
from .extractor import MemoryExtractor
from .manager import SafeMemoryManager
from .query_entity import QueryEntity, resolve_query_entity
from .query_intent import QueryIntentDetector
from .retriever import MemoryRetriever, RetrievalResult
from .store import MemoryStore, compute_expires_at, is_expired, load_json_list, save_json
from .structured_extractor import StructuredExtractor, create_structured_extractor
from .validation import (
    validate_memory_input,
    validate_value,
    validate_subject,
    validate_content,
    is_garbage,
    is_vague,
    is_temporal_only,
    ValidationResult,
)

__all__ = [
    "CHAT_MODEL",
    "EMBEDDING_MODEL",
    "LONG_TERM_MEMORY_FILE",
    "MEMORY_SIMILARITY_THRESHOLD",
    "EmbeddingError",
    "EmbeddingStore",
    "cosine_similarity",
    "Entity",
    "EntityExtraction",
    "extract_entity",
    "MemoryExtractor",
    "SafeMemoryManager",
    "QueryEntity",
    "resolve_query_entity",
    "QueryIntentDetector",
    "MemoryRetriever",
    "RetrievalResult",
    "MemoryStore",
    "compute_expires_at",
    "is_expired",
    "load_json_list",
    "save_json",
    "StructuredExtractor",
    "create_structured_extractor",
    "validate_memory_input",
    "validate_value",
    "validate_subject",
    "validate_content",
    "is_garbage",
    "is_vague",
    "is_temporal_only",
    "ValidationResult",
]
