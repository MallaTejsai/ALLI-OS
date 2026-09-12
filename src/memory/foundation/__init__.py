"""ALLI v0.6 Cognitive Memory Engine — foundation package.

Provides canonical data models, storage abstraction, validation,
identifier generation, and the write pipeline for the memory engine.
"""

from .models import (
    Entity,
    EntityRef,
    EntityType,
    Episode,
    Fact,
    ImportanceLevel,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    MemoryType,
    EventType,
    Provenance,
    ProvenanceSource,
    Relationship,
    ScoredMemory,
    TemporalScope,
)
from .storage import InMemoryStore, MemoryStore
from .validation import ValidationResult, validate_memory, validate_entity_ref
from .ids import (
    generate_entity_id,
    generate_memory_id,
    generate_event_id,
    generate_episode_id,
)
from .understanding import MessageUnderstanding, MessageType, understand_message
from .entity_resolution import ResolvedEntity, resolve_entity
from .extraction import MemoryOperation, ExtractionResult, PatternExtractor
from .operations import OperationResult, DecisionReceipt, apply_operation
from .pipeline import WritePipeline, WriteResult

__all__ = [
    # Models
    "Entity",
    "EntityRef",
    "EntityType",
    "Episode",
    "Fact",
    "ImportanceLevel",
    "Lifecycle",
    "LifecycleState",
    "Memory",
    "MemoryEvent",
    "MemoryType",
    "EventType",
    "Provenance",
    "ProvenanceSource",
    "Relationship",
    "ScoredMemory",
    "TemporalScope",
    # Storage
    "InMemoryStore",
    "MemoryStore",
    # Validation
    "ValidationResult",
    "validate_memory",
    "validate_entity_ref",
    # IDs
    "generate_entity_id",
    "generate_memory_id",
    "generate_event_id",
    "generate_episode_id",
    # Understanding
    "MessageUnderstanding",
    "MessageType",
    "understand_message",
    # Entity Resolution
    "ResolvedEntity",
    "resolve_entity",
    # Extraction
    "MemoryOperation",
    "ExtractionResult",
    "PatternExtractor",
    # Operations
    "OperationResult",
    "DecisionReceipt",
    "apply_operation",
    # Pipeline
    "WritePipeline",
    "WriteResult",
]
