"""Canonical data models for ALLI v0.6 Cognitive Memory Engine.

Every memory in ALLI is represented using these models. They are
deliberately flat enough for JSON storage now, but map naturally
to relational + vector storage later.

Design rule: categories are derived tags on facts, not the
primary organizing structure. The same entity→fact→lifecycle
model works for all domains.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class EntityType(str, Enum):
    """Types of entities the system can represent."""

    SELF = "self"
    PERSON = "person"
    ORGANIZATION = "organization"
    CONCEPT = "concept"
    DEVICE = "device"
    UNKNOWN = "unknown"


class MemoryType(str, Enum):
    """Classification of memory content.

    - EPISODIC: Specific interaction events (episodes)
    - SEMANTIC: General knowledge and facts
    - PROCEDURAL: Skills, procedures, how-to knowledge
    - TEMPORARY: Time-bound context, temporary state
    """

    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    TEMPORARY = "temporary"


class ImportanceLevel(str, Enum):
    """Importance classification for memories.

    Per architecture §6.1: critical | high | medium | low
    Category-driven policy (not ML model).
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class LifecycleState(str, Enum):
    """Lifecycle states a memory can occupy.

    State machine:
        proposed → active → inactive → expired
    """

    PROPOSED = "proposed"
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"


class EventType(str, Enum):
    """Types of memory events."""

    CREATED = "created"
    UPDATED = "updated"
    SUPERSEDED = "superseded"
    DEACTIVATED = "deactivated"
    EXPIRED = "expired"
    RESTORED = "restored"


class ProvenanceSource(str, Enum):
    """Origin of a memory."""

    USER_STATEMENT = "user_statement"
    MODEL_EXTRACTION = "model_extraction"
    INFERRED = "inferred"
    SYSTEM_EVENT = "system_event"
    MIGRATION = "migration"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    """Return current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class EntityRef:
    """Lightweight reference to an entity within a memory record.

    Every fact is scoped to an entity via this reference. This prevents
    cross-entity contamination: a query about MOM never returns USER's data.
    """

    name: str
    entity_type: str = EntityType.SELF.value
    relation: Optional[str] = None
    entity_id: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("EntityRef.name must be non-empty")

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "relation": self.relation,
            "entity_id": self.entity_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityRef":
        """Deserialize from dictionary."""
        return cls(
            name=data["name"],
            entity_type=data.get("entity_type", EntityType.SELF.value),
            relation=data.get("relation"),
            entity_id=data.get("entity_id"),
        )


@dataclass(frozen=True)
class Fact:
    """A discrete piece of knowledge: subject + predicate + value.

    The triple structure enables generic matching:
    - Two facts about the same subject with the same predicate
      are candidates for contradiction or update.
    - The content field preserves the human-readable form.
    """

    fact_type: str
    subject: str
    predicate: str
    value: str
    content: str
    polarity: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.fact_type:
            raise ValueError("Fact.fact_type must be non-empty")
        if not self.content:
            raise ValueError("Fact.content must be non-empty")

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "fact_type": self.fact_type,
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "content": self.content,
            "polarity": self.polarity,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Fact":
        """Deserialize from dictionary."""
        return cls(
            fact_type=data["fact_type"],
            subject=data["subject"],
            predicate=data["predicate"],
            value=data["value"],
            content=data["content"],
            polarity=data.get("polarity"),
        )


@dataclass(frozen=True)
class TemporalScope:
    """Time window during which a fact is considered current.

    Supports:
    - Persistent facts (valid_until=None, is_persistent=True)
    - Temporary state (valid_until computed from hint)
    - Events/tasks (valid_until from natural language hint)
    - Historical facts (valid_until from supersession)
    """

    valid_from: str
    valid_until: Optional[str] = None
    is_persistent: bool = True
    temporal_hint: Optional[str] = None

    def is_current_at(self, timestamp: str | None = None) -> bool:
        """Check if this temporal scope is valid at the given time."""
        check = timestamp or _utc_now_iso()
        if check < self.valid_from:
            return False
        if self.valid_until is not None and check > self.valid_until:
            return False
        return True

    def is_expired_at(self, timestamp: str | None = None) -> bool:
        """Check if this temporal scope has expired."""
        return not self.is_current_at(timestamp)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "is_persistent": self.is_persistent,
            "temporal_hint": self.temporal_hint,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TemporalScope":
        """Deserialize from dictionary."""
        return cls(
            valid_from=data["valid_from"],
            valid_until=data.get("valid_until"),
            is_persistent=data.get("is_persistent", True),
            temporal_hint=data.get("temporal_hint"),
        )


@dataclass(frozen=True)
class Provenance:
    """Origin and history of a memory.

    Preserves enough information to answer: "Where did this memory come from?"
    Distinguishes explicit user statements from model inferences.
    """

    source: str = ProvenanceSource.USER_STATEMENT.value
    extraction_method: Optional[str] = None
    message_id: Optional[str] = None
    episode_id: Optional[str] = None
    reason: Optional[str] = None

    def is_user_stated(self) -> bool:
        return self.source == ProvenanceSource.USER_STATEMENT.value

    def is_inferred(self) -> bool:
        return self.source == ProvenanceSource.INFERRED.value

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "source": self.source,
            "extraction_method": self.extraction_method,
            "message_id": self.message_id,
            "episode_id": self.episode_id,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Provenance":
        """Deserialize from dictionary."""
        return cls(
            source=data.get("source", ProvenanceSource.USER_STATEMENT.value),
            extraction_method=data.get("extraction_method"),
            message_id=data.get("message_id"),
            episode_id=data.get("episode_id"),
            reason=data.get("reason"),
        )


@dataclass
class Lifecycle:
    """State machine for a memory: proposed → active → inactive → expired.

    Preserves history via supersession links rather than deletion.
    """

    state: str = LifecycleState.PROPOSED.value
    superseded_by: Optional[str] = None
    superseded_at: Optional[str] = None
    deactivated_at: Optional[str] = None
    expired_at: Optional[str] = None

    @property
    def is_active(self) -> bool:
        return self.state == LifecycleState.ACTIVE.value

    @property
    def is_inactive(self) -> bool:
        return self.state == LifecycleState.INACTIVE.value

    @property
    def is_expired(self) -> bool:
        return self.state == LifecycleState.EXPIRED.value

    def activate(self) -> None:
        """Transition to active state."""
        self.state = LifecycleState.ACTIVE.value

    def deactivate(self) -> None:
        """Transition to inactive state."""
        self.state = LifecycleState.INACTIVE.value
        self.deactivated_at = _utc_now_iso()

    def expire(self) -> None:
        """Transition to expired state."""
        self.state = LifecycleState.EXPIRED.value
        self.expired_at = _utc_now_iso()

    def supersede(self, winner_id: str) -> None:
        """Mark this memory as superseded by another."""
        self.state = LifecycleState.INACTIVE.value
        self.superseded_by = winner_id
        self.superseded_at = _utc_now_iso()

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "state": self.state,
            "superseded_by": self.superseded_by,
            "superseded_at": self.superseded_at,
            "deactivated_at": self.deactivated_at,
            "expired_at": self.expired_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Lifecycle":
        """Deserialize from dictionary."""
        return cls(
            state=data.get("state", LifecycleState.PROPOSED.value),
            superseded_by=data.get("superseded_by"),
            superseded_at=data.get("superseded_at"),
            deactivated_at=data.get("deactivated_at"),
            expired_at=data.get("expired_at"),
        )


# ---------------------------------------------------------------------------
# Primary models
# ---------------------------------------------------------------------------


@dataclass
class Entity:
    """A thing that facts can be about: a person, organization, or concept.

    Entity identity is independent from natural-language wording.
    Names are normalized for consistent matching.
    """

    id: str
    name: str
    entity_type: str = EntityType.SELF.value
    aliases: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_ref(self) -> EntityRef:
        """Create a lightweight EntityRef from this entity."""
        return EntityRef(
            name=self.name,
            entity_type=self.entity_type,
            entity_id=self.id,
        )

    def matches_name(self, other_name: str) -> bool:
        """Check if a name matches this entity (case-insensitive)."""
        normalized = other_name.strip().lower()
        if normalized == self.name.lower():
            return True
        return normalized in [a.lower() for a in self.aliases]

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Entity":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data.get("entity_type", EntityType.SELF.value),
            aliases=data.get("aliases", []),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", _utc_now_iso()),
            updated_at=data.get("updated_at", _utc_now_iso()),
        )


@dataclass
class Relationship:
    """A connection between two entities.

    Relationships are first-class and separate from facts about entities.
    They enable entity-scoped retrieval across the relationship graph.
    """

    id: str
    entity_a: EntityRef
    entity_b: EntityRef
    relation_type: str
    confidence: float = 1.0
    provenance: Provenance = field(default_factory=Provenance)
    temporal: TemporalScope = field(default_factory=lambda: TemporalScope(
        valid_from=_utc_now_iso()
    ))
    lifecycle: Lifecycle = field(default_factory=lambda: Lifecycle(
        state=LifecycleState.ACTIVE.value
    ))
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "entity_a": self.entity_a.to_dict(),
            "entity_b": self.entity_b.to_dict(),
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "provenance": self.provenance.to_dict(),
            "temporal": self.temporal.to_dict(),
            "lifecycle": self.lifecycle.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Relationship":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            entity_a=EntityRef.from_dict(data["entity_a"]),
            entity_b=EntityRef.from_dict(data["entity_b"]),
            relation_type=data["relation_type"],
            confidence=data.get("confidence", 1.0),
            provenance=Provenance.from_dict(data.get("provenance", {})),
            temporal=TemporalScope.from_dict(data.get("temporal", {"valid_from": _utc_now_iso()})),
            lifecycle=Lifecycle.from_dict(data.get("lifecycle", {})),
            created_at=data.get("created_at", _utc_now_iso()),
            updated_at=data.get("updated_at", _utc_now_iso()),
        )


@dataclass
class Memory:
    """The canonical memory record.

    Combines a fact + its lifecycle state + provenance + temporal
    validity + confidence. This is the unit of storage.
    """

    id: str
    entity: EntityRef
    fact: Fact
    domain: str
    lifecycle: Lifecycle
    provenance: Provenance
    temporal: TemporalScope
    confidence: float = 1.0
    embedding: Optional[list[float]] = None
    embedding_version: int = 0
    schema_version: int = 1
    importance: str = ImportanceLevel.MEDIUM.value
    memory_type: str = MemoryType.SEMANTIC.value
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)

    @property
    def is_active(self) -> bool:
        return self.lifecycle.is_active

    @property
    def is_expired(self) -> bool:
        return self.temporal.is_expired_at()

    @property
    def entity_name(self) -> str:
        return self.entity.name

    @property
    def fact_type(self) -> str:
        return self.fact.fact_type

    @property
    def subject(self) -> str:
        return self.fact.subject

    @property
    def content(self) -> str:
        return self.fact.content

    def matches_entity_and_subject(
        self,
        entity_name: str,
        fact_type: str,
        subject: str,
    ) -> bool:
        """Check if this memory represents the same logical fact."""
        return (
            self.entity.name == entity_name
            and self.fact.fact_type == fact_type
            and self.fact.subject == subject
        )

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "entity": self.entity.to_dict(),
            "fact": self.fact.to_dict(),
            "domain": self.domain,
            "lifecycle": self.lifecycle.to_dict(),
            "provenance": self.provenance.to_dict(),
            "temporal": self.temporal.to_dict(),
            "confidence": self.confidence,
            "embedding": self.embedding,
            "embedding_version": self.embedding_version,
            "schema_version": self.schema_version,
            "importance": self.importance,
            "memory_type": self.memory_type,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            entity=EntityRef.from_dict(data["entity"]),
            fact=Fact.from_dict(data["fact"]),
            domain=data["domain"],
            lifecycle=Lifecycle.from_dict(data.get("lifecycle", {})),
            provenance=Provenance.from_dict(data.get("provenance", {})),
            temporal=TemporalScope.from_dict(data.get("temporal", {"valid_from": _utc_now_iso()})),
            confidence=data.get("confidence", 1.0),
            embedding=data.get("embedding"),
            embedding_version=data.get("embedding_version", 0),
            schema_version=data.get("schema_version", 1),
            importance=data.get("importance", "medium"),
            memory_type=data.get("memory_type", MemoryType.SEMANTIC.value),
            created_at=data.get("created_at", _utc_now_iso()),
            updated_at=data.get("updated_at", _utc_now_iso()),
        )


@dataclass
class ScoredMemory:
    """A memory with a retrieval score."""

    memory: Memory
    score: float
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    domain_match: bool = True
    entity_match: bool = True


@dataclass
class MemoryEvent:
    """A recorded operation on memory.

    Events enable debugging, audit trails, and undo capabilities.
    Every write operation produces exactly one event.
    """

    id: str
    event_type: str
    memory_id: str
    timestamp: str = field(default_factory=_utc_now_iso)
    trigger: str = "user_statement"
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "event_type": self.event_type,
            "memory_id": self.memory_id,
            "timestamp": self.timestamp,
            "trigger": self.trigger,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEvent":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            event_type=data["event_type"],
            memory_id=data["memory_id"],
            timestamp=data.get("timestamp", _utc_now_iso()),
            trigger=data.get("trigger", "user_statement"),
            details=data.get("details", {}),
        )


@dataclass
class Episode:
    """A recorded interaction event.

    Preserves the source interaction, timestamp, participating entities,
    extracted facts, and provenance. Foundation for episodic memory.
    """

    id: str
    user_message: str
    timestamp: str = field(default_factory=_utc_now_iso)
    entities_involved: list[EntityRef] = field(default_factory=list)
    extracted_facts: list[str] = field(default_factory=list)
    extracted_relationships: list[str] = field(default_factory=list)
    memory_events: list[str] = field(default_factory=list)
    provenance: Provenance = field(default_factory=lambda: Provenance(
        source=ProvenanceSource.USER_STATEMENT.value
    ))
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "user_message": self.user_message,
            "timestamp": self.timestamp,
            "entities_involved": [e.to_dict() for e in self.entities_involved],
            "extracted_facts": self.extracted_facts,
            "extracted_relationships": self.extracted_relationships,
            "memory_events": self.memory_events,
            "provenance": self.provenance.to_dict(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Episode":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            user_message=data["user_message"],
            timestamp=data.get("timestamp", _utc_now_iso()),
            entities_involved=[EntityRef.from_dict(e) for e in data.get("entities_involved", [])],
            extracted_facts=data.get("extracted_facts", []),
            extracted_relationships=data.get("extracted_relationships", []),
            memory_events=data.get("memory_events", []),
            provenance=Provenance.from_dict(data.get("provenance", {})),
            metadata=data.get("metadata", {}),
        )
