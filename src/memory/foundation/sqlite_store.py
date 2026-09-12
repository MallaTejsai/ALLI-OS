"""SQLite-backed persistent storage for ALLI v0.6 Cognitive Memory Engine.

Implements the MemoryStore protocol using SQLite for structured persistence.
Preserves historical information rather than destroying it.

Architecture:
  WritePipeline → SQLiteStore (MemoryStore protocol) → SQLite database

Design:
  - All foundation models map to SQLite tables
  - JSON serialization for complex fields (aliases, metadata, details)
  - BLOB for embeddings
  - Transactions for atomicity
  - History preserved via lifecycle state and memory events
  - No data destruction - everything is kept
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from typing import Optional

from .models import (
    Entity,
    EntityRef,
    Episode,
    Fact,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    Provenance,
    Relationship,
    ScoredMemory,
    TemporalScope,
)
from .storage import MemoryStore, _cosine_similarity


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 1

_SCHEMA_SQL = """
-- Entity table: stores unique entities (people, devices, concepts, etc.)
CREATE TABLE IF NOT EXISTS entities (
    entity_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    aliases TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Memory table: stores all facts/knowledge with full lifecycle
CREATE TABLE IF NOT EXISTS memories (
    memory_id TEXT PRIMARY KEY,
    entity_id TEXT NOT NULL,

    -- Fact fields
    fact_type TEXT NOT NULL,
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL,
    value TEXT NOT NULL,
    content TEXT NOT NULL,
    polarity TEXT,

    -- Domain (derived tag)
    domain TEXT NOT NULL,

    -- Lifecycle state
    lifecycle_state TEXT NOT NULL,
    superseded_by TEXT,
    superseded_at TEXT,
    deactivated_at TEXT,
    expired_at TEXT,

    -- Provenance
    provenance_source TEXT NOT NULL,
    extraction_method TEXT,
    message_id TEXT,
    episode_id TEXT,
    provenance_reason TEXT,

    -- Temporal scope
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    is_persistent INTEGER NOT NULL DEFAULT 1,
    temporal_hint TEXT,

    -- Metadata
    confidence REAL NOT NULL DEFAULT 1.0,
    embedding BLOB,
    embedding_version INTEGER NOT NULL DEFAULT 0,
    schema_version INTEGER NOT NULL DEFAULT 1,
    importance TEXT NOT NULL DEFAULT 'medium',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (entity_id) REFERENCES entities(entity_id)
);

-- Relationship table: stores connections between entities
CREATE TABLE IF NOT EXISTS relationships (
    relationship_id TEXT PRIMARY KEY,
    entity_a_id TEXT NOT NULL,
    entity_b_id TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 1.0,

    -- Provenance
    provenance_source TEXT NOT NULL,
    extraction_method TEXT,
    message_id TEXT,
    episode_id TEXT,
    provenance_reason TEXT,

    -- Temporal scope
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    is_persistent INTEGER NOT NULL DEFAULT 1,
    temporal_hint TEXT,

    -- Lifecycle
    lifecycle_state TEXT NOT NULL,
    superseded_by TEXT,
    superseded_at TEXT,
    deactivated_at TEXT,
    expired_at TEXT,

    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,

    FOREIGN KEY (entity_a_id) REFERENCES entities(entity_id),
    FOREIGN KEY (entity_b_id) REFERENCES entities(entity_id)
);

-- MemoryEvent table: audit trail for all memory operations
CREATE TABLE IF NOT EXISTS memory_events (
    event_id TEXT PRIMARY KEY,
    memory_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    trigger TEXT NOT NULL DEFAULT 'user_statement',
    details TEXT DEFAULT '{}',

    FOREIGN KEY (memory_id) REFERENCES memories(memory_id)
);

-- Episode table: records user interactions
CREATE TABLE IF NOT EXISTS episodes (
    episode_id TEXT PRIMARY KEY,
    user_message TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    entities_involved TEXT DEFAULT '[]',
    extracted_facts TEXT DEFAULT '[]',
    extracted_relationships TEXT DEFAULT '[]',
    memory_events TEXT DEFAULT '[]',
    provenance_source TEXT NOT NULL,
    provenance_reason TEXT,
    metadata TEXT DEFAULT '{}'
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_memories_entity ON memories(entity_id);
CREATE INDEX IF NOT EXISTS idx_memories_domain ON memories(domain);
CREATE INDEX IF NOT EXISTS idx_memories_lifecycle ON memories(lifecycle_state);
CREATE INDEX IF NOT EXISTS idx_memories_fact_type ON memories(fact_type);
CREATE INDEX IF NOT EXISTS idx_memory_events_memory ON memory_events(memory_id);
CREATE INDEX IF NOT EXISTS idx_relationships_entity_a ON relationships(entity_a_id);
CREATE INDEX IF NOT EXISTS idx_relationships_entity_b ON relationships(entity_b_id);

-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_json(value: object) -> str:
    """Serialize a value to JSON string."""
    return json.dumps(value, ensure_ascii=False)


def _deserialize_json(text: str, default: object = None) -> object:
    """Deserialize a JSON string to a Python object."""
    if not text:
        return default if default is not None else {}
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default if default is not None else {}


def _serialize_embedding(embedding: Optional[list[float]]) -> Optional[bytes]:
    """Serialize an embedding vector to bytes for BLOB storage."""
    if embedding is None:
        return None
    # Store as JSON string of floats
    return json.dumps(embedding, ensure_ascii=False).encode('utf-8')


def _deserialize_embedding(data: Optional[bytes]) -> Optional[list[float]]:
    """Deserialize bytes from BLOB to an embedding vector."""
    if data is None:
        return None
    try:
        return json.loads(data.decode('utf-8'))
    except (json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Row converters
# ---------------------------------------------------------------------------

def _row_to_memory(row: sqlite3.Row) -> Memory:
    """Convert a SQLite row to a foundation Memory object."""
    return Memory(
        id=row["memory_id"],
        entity=EntityRef(
            name=row["entity_name"],
            entity_type=row["entity_type"],
            entity_id=row["entity_id"],
        ),
        fact=Fact(
            fact_type=row["fact_type"],
            subject=row["subject"],
            predicate=row["predicate"],
            value=row["value"],
            content=row["content"],
            polarity=row["polarity"],
        ),
        domain=row["domain"],
        lifecycle=Lifecycle(
            state=row["lifecycle_state"],
            superseded_by=row["superseded_by"],
            superseded_at=row["superseded_at"],
            deactivated_at=row["deactivated_at"],
            expired_at=row["expired_at"],
        ),
        provenance=Provenance(
            source=row["provenance_source"],
            extraction_method=row["extraction_method"],
            message_id=row["message_id"],
            episode_id=row["episode_id"],
            reason=row["provenance_reason"],
        ),
        temporal=TemporalScope(
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            is_persistent=bool(row["is_persistent"]),
            temporal_hint=row["temporal_hint"],
        ),
        confidence=row["confidence"],
        embedding=_deserialize_embedding(row["embedding"]),
        embedding_version=row["embedding_version"],
        schema_version=row["schema_version"],
        importance=row["importance"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _memory_to_params(memory: Memory) -> dict:
    """Convert a foundation Memory object to SQLite parameters."""
    return {
        "memory_id": memory.id,
        "entity_id": memory.entity.entity_id or memory.entity.name,
        "fact_type": memory.fact.fact_type,
        "subject": memory.fact.subject,
        "predicate": memory.fact.predicate,
        "value": memory.fact.value,
        "content": memory.fact.content,
        "polarity": memory.fact.polarity,
        "domain": memory.domain,
        "lifecycle_state": memory.lifecycle.state,
        "superseded_by": memory.lifecycle.superseded_by,
        "superseded_at": memory.lifecycle.superseded_at,
        "deactivated_at": memory.lifecycle.deactivated_at,
        "expired_at": memory.lifecycle.expired_at,
        "provenance_source": memory.provenance.source,
        "extraction_method": memory.provenance.extraction_method,
        "message_id": memory.provenance.message_id,
        "episode_id": memory.provenance.episode_id,
        "provenance_reason": memory.provenance.reason,
        "valid_from": memory.temporal.valid_from,
        "valid_until": memory.temporal.valid_until,
        "is_persistent": int(memory.temporal.is_persistent),
        "temporal_hint": memory.temporal.temporal_hint,
        "confidence": memory.confidence,
        "embedding": _serialize_embedding(memory.embedding),
        "embedding_version": memory.embedding_version,
        "schema_version": memory.schema_version,
        "importance": memory.importance,
        "created_at": memory.created_at,
        "updated_at": memory.updated_at,
    }


def _row_to_event(row: sqlite3.Row) -> MemoryEvent:
    """Convert a SQLite row to a MemoryEvent object."""
    return MemoryEvent(
        id=row["event_id"],
        event_type=row["event_type"],
        memory_id=row["memory_id"],
        timestamp=row["timestamp"],
        trigger=row["trigger"],
        details=_deserialize_json(row["details"], {}),
    )


def _row_to_entity(row: sqlite3.Row) -> Entity:
    """Convert a SQLite row to an Entity object."""
    return Entity(
        id=row["entity_id"],
        name=row["name"],
        entity_type=row["entity_type"],
        aliases=_deserialize_json(row["aliases"], []),
        metadata=_deserialize_json(row["metadata"], {}),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def _row_to_relationship(row: sqlite3.Row) -> Relationship:
    """Convert a SQLite row to a Relationship object."""
    return Relationship(
        id=row["relationship_id"],
        entity_a=EntityRef(name=row["entity_a_name"], entity_type=row["entity_a_type"], entity_id=row["entity_a_id"]),
        entity_b=EntityRef(name=row["entity_b_name"], entity_type=row["entity_b_type"], entity_id=row["entity_b_id"]),
        relation_type=row["relation_type"],
        confidence=row["confidence"],
        provenance=Provenance(
            source=row["provenance_source"],
            extraction_method=row["extraction_method"],
            message_id=row["message_id"],
            episode_id=row["episode_id"],
            reason=row["provenance_reason"],
        ),
        temporal=TemporalScope(
            valid_from=row["valid_from"],
            valid_until=row["valid_until"],
            is_persistent=bool(row["is_persistent"]),
            temporal_hint=row["temporal_hint"],
        ),
        lifecycle=Lifecycle(
            state=row["lifecycle_state"],
            superseded_by=row["superseded_by"],
            superseded_at=row["superseded_at"],
            deactivated_at=row["deactivated_at"],
            expired_at=row["expired_at"],
        ),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


# ---------------------------------------------------------------------------
# SQLiteStore implementation
# ---------------------------------------------------------------------------

class SQLiteStore:
    """SQLite-backed persistent storage for the v0.6 memory engine.

    Implements the MemoryStore protocol using SQLite for structured persistence.
    Preserves historical information and provides atomic transactions.
    """

    def __init__(self, db_path: str) -> None:
        """Initialize the SQLite store.

        Args:
            db_path: Path to the SQLite database file.
        """
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()
        # Initialize ID counters from existing database to prevent collisions
        from .ids import init_counters_from_db
        init_counters_from_db(self._get_conn())

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(self._db_path)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        conn.commit()

    @contextmanager
    def _transaction(self):
        """Context manager for database transactions."""
        conn = self._get_conn()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    # -----------------------------------------------------------------------
    # Entity operations
    # -----------------------------------------------------------------------

    def get_or_create_entity(
        self,
        name: str,
        entity_type: str = "self",
        aliases: list[str] = None,
        metadata: dict = None,
    ) -> Entity:
        """Get an existing entity or create a new one."""
        conn = self._get_conn()
        now = __import__('datetime').datetime.now(__import__('datetime').timezone.utc).replace(microsecond=0).isoformat()

        # Try to find existing
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?",
            (name.lower(),)
        ).fetchone()

        if row:
            return _row_to_entity(row)

        # Create new entity
        entity_id = f"ent_{name.lower().replace(' ', '_')}_{now[:10]}"
        with self._transaction() as conn:
            conn.execute(
                """INSERT INTO entities (entity_id, name, entity_type, aliases, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (entity_id, name.lower(), entity_type,
                 _serialize_json(aliases or []),
                 _serialize_json(metadata or {}),
                 now, now)
            )

        return Entity(
            id=entity_id,
            name=name.lower(),
            entity_type=entity_type,
            aliases=aliases or [],
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Retrieve an entity by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE entity_id = ?",
            (entity_id,)
        ).fetchone()
        return _row_to_entity(row) if row else None

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Retrieve an entity by name."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM entities WHERE name = ?",
            (name.lower(),)
        ).fetchone()
        return _row_to_entity(row) if row else None

    # -----------------------------------------------------------------------
    # MemoryStore protocol implementation
    # -----------------------------------------------------------------------

    def load(self) -> list[Memory]:
        """Load all memories from the database."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT m.*, e.name as entity_name, e.entity_type as entity_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.entity_id
            ORDER BY m.created_at
        """).fetchall()
        return [_row_to_memory(row) for row in rows]

    def save(self, memories: list[Memory]) -> None:
        """Save all memories to the database (replace mode)."""
        with self._transaction() as conn:
            # Delete all existing memories
            conn.execute("DELETE FROM memories")
            # Insert all memories
            for memory in memories:
                # Ensure entity exists
                self.get_or_create_entity(
                    name=memory.entity.name,
                    entity_type=memory.entity.entity_type,
                )
                params = _memory_to_params(memory)
                conn.execute(
                    """INSERT OR REPLACE INTO memories
                       (memory_id, entity_id, fact_type, subject, predicate, value, content,
                        polarity, domain, lifecycle_state, superseded_by, superseded_at,
                        deactivated_at, expired_at, provenance_source, extraction_method,
                        message_id, episode_id, provenance_reason, valid_from, valid_until,
                        is_persistent, temporal_hint, confidence, embedding, embedding_version,
                        schema_version, importance, created_at, updated_at)
                       VALUES
                       (:memory_id, :entity_id, :fact_type, :subject, :predicate, :value, :content,
                        :polarity, :domain, :lifecycle_state, :superseded_by, :superseded_at,
                        :deactivated_at, :expired_at, :provenance_source, :extraction_method,
                        :message_id, :episode_id, :provenance_reason, :valid_from, :valid_until,
                        :is_persistent, :temporal_hint, :confidence, :embedding, :embedding_version,
                        :schema_version, :importance, :created_at, :updated_at)""",
                    params
                )

    def add(self, memory: Memory) -> Memory:
        """Add a new memory record."""
        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).replace(microsecond=0).isoformat()

        # Ensure entity exists (create if needed)
        entity_name = memory.entity.name.lower()
        entity_type = memory.entity.entity_type
        entity_id = memory.entity.entity_id or f"ent_{entity_name.replace(' ', '_')}_{now[:10]}"

        params = _memory_to_params(memory)
        params["entity_id"] = entity_id

        with self._transaction() as conn:
            # Create entity if it doesn't exist
            existing_entity = conn.execute(
                "SELECT entity_id FROM entities WHERE name = ?",
                (entity_name,)
            ).fetchone()

            if not existing_entity:
                conn.execute(
                    """INSERT INTO entities (entity_id, name, entity_type, aliases, metadata, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (entity_id, entity_name, entity_type,
                     _serialize_json(memory.entity.aliases if hasattr(memory.entity, 'aliases') else []),
                     _serialize_json({}),
                     now, now)
                )
            else:
                entity_id = existing_entity["entity_id"]
                params["entity_id"] = entity_id

            conn.execute(
                """INSERT OR REPLACE INTO memories
                   (memory_id, entity_id, fact_type, subject, predicate, value, content,
                    polarity, domain, lifecycle_state, superseded_by, superseded_at,
                    deactivated_at, expired_at, provenance_source, extraction_method,
                    message_id, episode_id, provenance_reason, valid_from, valid_until,
                    is_persistent, temporal_hint, confidence, embedding, embedding_version,
                    schema_version, importance, created_at, updated_at)
                   VALUES
                   (:memory_id, :entity_id, :fact_type, :subject, :predicate, :value, :content,
                    :polarity, :domain, :lifecycle_state, :superseded_by, :superseded_at,
                    :deactivated_at, :expired_at, :provenance_source, :extraction_method,
                    :message_id, :episode_id, :provenance_reason, :valid_from, :valid_until,
                    :is_persistent, :temporal_hint, :confidence, :embedding, :embedding_version,
                    :schema_version, :importance, :created_at, :updated_at)""",
                params
            )
        return memory

    def update(self, memory_id: str, changes: dict) -> Optional[Memory]:
        """Update an existing memory by ID."""
        existing = self.get_by_id(memory_id)
        if existing is None:
            return None

        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).replace(microsecond=0).isoformat()

        # Apply changes to the memory object
        for key, value in changes.items():
            if key == "content":
                # content is a property on Fact; update the underlying field
                existing.fact = Fact(
                    fact_type=existing.fact.fact_type,
                    subject=existing.fact.subject,
                    predicate=existing.fact.predicate,
                    value=existing.fact.value,
                    content=value,
                    polarity=existing.fact.polarity,
                )
            elif hasattr(existing, key) and not isinstance(getattr(type(existing), key, None), property):
                setattr(existing, key, value)

        existing.updated_at = now

        # Save back to database
        params = _memory_to_params(existing)
        with self._transaction() as conn:
            conn.execute(
                """UPDATE memories SET
                   fact_type = :fact_type, subject = :subject, predicate = :predicate,
                   value = :value, content = :content, polarity = :polarity,
                   domain = :domain, lifecycle_state = :lifecycle_state,
                   superseded_by = :superseded_by, superseded_at = :superseded_at,
                   deactivated_at = :deactivated_at, expired_at = :expired_at,
                   provenance_source = :provenance_source, extraction_method = :extraction_method,
                   message_id = :message_id, episode_id = :episode_id,
                   provenance_reason = :provenance_reason, valid_from = :valid_from,
                   valid_until = :valid_until, is_persistent = :is_persistent,
                   temporal_hint = :temporal_hint, confidence = :confidence,
                   embedding = :embedding, embedding_version = :embedding_version,
                   schema_version = :schema_version, importance = :importance,
                   updated_at = :updated_at
                   WHERE memory_id = :memory_id""",
                params
            )
        return existing

    def deactivate(self, memory_id: str) -> bool:
        """Mark a memory as inactive."""
        memory = self.get_by_id(memory_id)
        if memory is None:
            return False

        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).replace(microsecond=0).isoformat()

        memory.lifecycle.deactivate()

        with self._transaction() as conn:
            conn.execute(
                """UPDATE memories SET
                   lifecycle_state = ?, deactivated_at = ?, updated_at = ?
                   WHERE memory_id = ?""",
                (memory.lifecycle.state, memory.lifecycle.deactivated_at,
                 now, memory_id)
            )
        return True

    def supersede(self, winner_id: str, loser_id: str) -> bool:
        """Mark loser as superseded by winner."""
        loser = self.get_by_id(loser_id)
        if loser is None:
            return False
        if not loser.lifecycle.is_active:
            return True

        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).replace(microsecond=0).isoformat()

        loser.lifecycle.supersede(winner_id)

        with self._transaction() as conn:
            conn.execute(
                """UPDATE memories SET
                   lifecycle_state = ?, superseded_by = ?, superseded_at = ?,
                   updated_at = ?
                   WHERE memory_id = ?""",
                (loser.lifecycle.state, loser.lifecycle.superseded_by,
                 loser.lifecycle.superseded_at, now, loser_id)
            )
        return True

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        conn = self._get_conn()
        row = conn.execute("""
            SELECT m.*, e.name as entity_name, e.entity_type as entity_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.entity_id
            WHERE m.memory_id = ?
        """, (memory_id,)).fetchone()
        return _row_to_memory(row) if row else None

    def get_active(self) -> list[Memory]:
        """Retrieve all active, non-expired memories."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT m.*, e.name as entity_name, e.entity_type as entity_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.entity_id
            WHERE m.lifecycle_state = 'active'
        """).fetchall()
        return [_row_to_memory(row) for row in rows]

    def get_by_entity(self, entity_name: str) -> list[Memory]:
        """Retrieve all memories scoped to a specific entity."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT m.*, e.name as entity_name, e.entity_type as entity_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.entity_id
            WHERE e.name = ?
        """, (entity_name.lower(),)).fetchall()
        return [_row_to_memory(row) for row in rows]

    def get_by_domain(self, domain: str) -> list[Memory]:
        """Retrieve all memories in a specific domain."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT m.*, e.name as entity_name, e.entity_type as entity_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.entity_id
            WHERE m.domain = ?
        """, (domain,)).fetchall()
        return [_row_to_memory(row) for row in rows]

    def search_semantic(
        self, query_vector: list[float], top_k: int = 8
    ) -> list[ScoredMemory]:
        """Search memories by semantic similarity to a query vector."""
        if not query_vector:
            return []

        # Load all memories with embeddings
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT m.*, e.name as entity_name, e.entity_type as entity_type
            FROM memories m
            JOIN entities e ON m.entity_id = e.entity_id
            WHERE m.embedding IS NOT NULL
        """).fetchall()

        scored: list[ScoredMemory] = []
        for row in rows:
            memory = _row_to_memory(row)
            if memory.embedding is not None:
                sim = _cosine_similarity(query_vector, memory.embedding)
                scored.append(ScoredMemory(memory=memory, score=sim, semantic_score=sim))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return total number of memories in storage."""
        conn = self._get_conn()
        row = conn.execute("SELECT COUNT(*) as cnt FROM memories").fetchone()
        return row["cnt"] if row else 0

    # -----------------------------------------------------------------------
    # Event operations
    # -----------------------------------------------------------------------

    def add_event(self, event: MemoryEvent) -> MemoryEvent:
        """Record a memory event."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO memory_events
                   (event_id, memory_id, event_type, timestamp, trigger, details)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (event.id, event.memory_id, event.event_type,
                 event.timestamp, event.trigger,
                 _serialize_json(event.details))
            )
        return event

    def get_events_for_memory(self, memory_id: str) -> list[MemoryEvent]:
        """Retrieve all events for a specific memory."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memory_events WHERE memory_id = ? ORDER BY timestamp",
            (memory_id,)
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    def get_all_events(self) -> list[MemoryEvent]:
        """Retrieve all memory events."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM memory_events ORDER BY timestamp"
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    # -----------------------------------------------------------------------
    # Relationship operations
    # -----------------------------------------------------------------------

    def add_relationship(self, relationship: Relationship) -> Relationship:
        """Store a relationship between entities."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO relationships
                   (relationship_id, entity_a_id, entity_b_id, relation_type,
                    confidence, provenance_source, extraction_method, message_id,
                    episode_id, provenance_reason, valid_from, valid_until,
                    is_persistent, temporal_hint, lifecycle_state, superseded_by,
                    superseded_at, deactivated_at, expired_at, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (relationship.id, relationship.entity_a.entity_id,
                 relationship.entity_b.entity_id, relationship.relation_type,
                 relationship.confidence, relationship.provenance.source,
                 relationship.provenance.extraction_method,
                 relationship.provenance.message_id,
                 relationship.provenance.episode_id,
                 relationship.provenance.reason,
                 relationship.temporal.valid_from,
                 relationship.temporal.valid_until,
                 int(relationship.temporal.is_persistent),
                 relationship.temporal.temporal_hint,
                 relationship.lifecycle.state,
                 relationship.lifecycle.superseded_by,
                 relationship.lifecycle.superseded_at,
                 relationship.lifecycle.deactivated_at,
                 relationship.lifecycle.expired_at,
                 relationship.created_at,
                 relationship.updated_at)
            )
        return relationship

    def get_relationships_for_entity(self, entity_id: str) -> list[Relationship]:
        """Retrieve all relationships involving an entity."""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT r.*,
                   ea.name as entity_a_name, ea.entity_type as entity_a_type,
                   eb.name as entity_b_name, eb.entity_type as entity_b_type
            FROM relationships r
            JOIN entities ea ON r.entity_a_id = ea.entity_id
            JOIN entities eb ON r.entity_b_id = eb.entity_id
            WHERE r.entity_a_id = ? OR r.entity_b_id = ?
        """, (entity_id, entity_id)).fetchall()
        return [_row_to_relationship(row) for row in rows]

    # -----------------------------------------------------------------------
    # Episode operations
    # -----------------------------------------------------------------------

    def add_episode(self, episode: Episode) -> Episode:
        """Store an episode."""
        with self._transaction() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO episodes
                   (episode_id, user_message, timestamp, entities_involved,
                    extracted_facts, extracted_relationships, memory_events,
                    provenance_source, provenance_reason, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (episode.id, episode.user_message, episode.timestamp,
                 _serialize_json([e.name for e in episode.entities_involved]),
                 _serialize_json(episode.extracted_facts),
                 _serialize_json(episode.extracted_relationships),
                 _serialize_json(episode.memory_events),
                 episode.provenance.source,
                 episode.provenance.reason,
                 _serialize_json(episode.metadata))
            )
        return episode

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve an episode by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM episodes WHERE episode_id = ?",
            (episode_id,)
        ).fetchone()
        if not row:
            return None
        return Episode(
            id=row["episode_id"],
            user_message=row["user_message"],
            timestamp=row["timestamp"],
            entities_involved=[
                EntityRef(name=n) for n in _deserialize_json(row["entities_involved"], [])
            ],
            extracted_facts=_deserialize_json(row["extracted_facts"], []),
            extracted_relationships=_deserialize_json(row["extracted_relationships"], []),
            memory_events=_deserialize_json(row["memory_events"], []),
            provenance=Provenance(
                source=row["provenance_source"],
                reason=row["provenance_reason"],
            ),
            metadata=_deserialize_json(row["metadata"], {}),
        )

    # -----------------------------------------------------------------------
    # Lifecycle operations (atomic)
    # -----------------------------------------------------------------------

    def supersede_atomic(self, loser_id: str, winner: Memory) -> tuple[bool, MemoryEvent]:
        """Atomically supersede a memory and create the new one.

        Returns (success, event).
        """
        loser = self.get_by_id(loser_id)
        if loser is None or not loser.lifecycle.is_active:
            return False, MemoryEvent(
                id="", event_type="failed", memory_id=loser_id,
                details={"reason": "not_found_or_inactive"}
            )

        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).replace(microsecond=0).isoformat()

        # Mark loser as superseded
        loser.lifecycle.supersede(winner.id)

        # Create event
        event = MemoryEvent(
            id=f"evt_sup_{loser_id}_{now[:10]}",
            event_type="superseded",
            memory_id=loser_id,
            timestamp=now,
            trigger="user_statement",
            details={
                "winner_id": winner.id,
                "old_content": loser.content,
                "new_content": winner.content,
            }
        )

        with self._transaction() as conn:
            # Update loser
            conn.execute(
                """UPDATE memories SET
                   lifecycle_state = ?, superseded_by = ?, superseded_at = ?,
                   updated_at = ?
                   WHERE memory_id = ?""",
                (loser.lifecycle.state, loser.lifecycle.superseded_by,
                 loser.lifecycle.superseded_at, now, loser_id)
            )
            # Add winner
            self.add(winner)
            # Record event
            self.add_event(event)

        return True, event

    def deactivate_atomic(self, memory_id: str, reason: str = "user_deletion") -> tuple[bool, MemoryEvent]:
        """Atomically deactivate a memory.

        Returns (success, event).
        """
        memory = self.get_by_id(memory_id)
        if memory is None:
            return False, MemoryEvent(
                id="", event_type="failed", memory_id=memory_id,
                details={"reason": "not_found"}
            )

        now = __import__('datetime').datetime.now(
            __import__('datetime').timezone.utc
        ).replace(microsecond=0).isoformat()

        memory.lifecycle.deactivate()

        event = MemoryEvent(
            id=f"evt_deact_{memory_id}_{now[:10]}",
            event_type="deactivated",
            memory_id=memory_id,
            timestamp=now,
            trigger="user_statement",
            details={"reason": reason, "content": memory.content}
        )

        with self._transaction() as conn:
            conn.execute(
                """UPDATE memories SET
                   lifecycle_state = ?, deactivated_at = ?, updated_at = ?
                   WHERE memory_id = ?""",
                (memory.lifecycle.state, memory.lifecycle.deactivated_at,
                 now, memory_id)
            )
            self.add_event(event)

        return True, event

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
