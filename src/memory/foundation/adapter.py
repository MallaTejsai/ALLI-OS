"""Persistent storage adapter — bridges foundation Memory objects to old JSON format.

This adapter allows the new WritePipeline to persist memories through
the old MemoryStore's JSON persistence while using the new foundation
Memory model internally.

Architecture:
  WritePipeline → PersistentStore (foundation Protocol) → old MemoryStore (JSON)

No old code is modified. The adapter converts between formats bidirectionally.
"""

from __future__ import annotations

from typing import Optional

from .models import (
    EntityRef,
    Fact,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    Provenance,
    ProvenanceSource,
    ScoredMemory,
    TemporalScope,
)
from .storage import MemoryStore as FoundationMemoryStore, _cosine_similarity


# Old status → new lifecycle state
_OLD_STATUS_MAP = {
    "active": LifecycleState.ACTIVE.value,
    "inactive": LifecycleState.INACTIVE.value,
    "expired": LifecycleState.EXPIRED.value,
    "superseded": LifecycleState.INACTIVE.value,
    "negated": LifecycleState.NEGATED.value,
}

# New lifecycle state → old status
_NEW_STATE_MAP = {v: k for k, v in _OLD_STATUS_MAP.items()}
_NEW_STATE_MAP[LifecycleState.PROPOSED.value] = "active"
# Override: use "inactive" for inactive state (old store only recognizes
# active/inactive/expired, not "superseded")
_NEW_STATE_MAP[LifecycleState.INACTIVE.value] = "inactive"
# NEGATED is preserved as its own status: it is current state for reads
# (evidence grounding + retrieval handle "negated"), but not "active" so
# behavioral checks and active_memories() do not count it as a stored fact.
_NEW_STATE_MAP[LifecycleState.NEGATED.value] = "negated"


def _old_dict_to_memory(d: dict) -> Memory:
    """Convert an old-format memory dict to a foundation Memory object."""
    entity_obj = d.get("entity", {})
    if isinstance(entity_obj, dict):
        entity_name = entity_obj.get("name", "user")
        entity_type = entity_obj.get("type", "self")
    else:
        entity_name = "user"
        entity_type = "self"

    temporal_obj = d.get("temporal", {})
    if isinstance(temporal_obj, dict):
        valid_from = temporal_obj.get("valid_from") or d.get("created_at", "")
        valid_until = temporal_obj.get("valid_until") or d.get("expires_at")
        is_persistent = temporal_obj.get("is_persistent", valid_until is None)
    else:
        valid_from = d.get("created_at", "")
        valid_until = d.get("expires_at")
        is_persistent = valid_until is None

    provenance_obj = d.get("provenance", {})
    if isinstance(provenance_obj, dict):
        source = provenance_obj.get("reason", d.get("source", "user_statement"))
    else:
        source = d.get("source", "user_statement")

    old_status = d.get("status", "active")
    lifecycle_state = _OLD_STATUS_MAP.get(old_status, LifecycleState.ACTIVE.value)

    return Memory(
        id=d.get("id", ""),
        entity=EntityRef(
            name=entity_name,
            entity_type=entity_type,
        ),
        fact=Fact(
            fact_type=d.get("fact_type", "general"),
            subject=d.get("subject") or d.get("value") or "",
            predicate=f"has_{d.get('fact_type', 'general')}",
            value=d.get("value") or "",
            content=d.get("content", ""),
        ),
        domain=d.get("category", "other"),
        lifecycle=Lifecycle(
            state=lifecycle_state,
            superseded_by=d.get("superseded_by"),
            superseded_at=d.get("superseded_at"),
            deactivated_at=d.get("deactivated_at"),
            expired_at=d.get("expired_at"),
        ),
        provenance=Provenance(
            source=source or ProvenanceSource.USER_STATEMENT.value,
            reason=source,
        ),
        temporal=TemporalScope(
            valid_from=valid_from or "",
            valid_until=valid_until,
            is_persistent=is_persistent,
        ),
        confidence=float(d.get("confidence", 1.0)),
        embedding=d.get("embedding"),
        embedding_version=d.get("embedding_version", 0),
        schema_version=d.get("schema_version", 1),
        importance=d.get("importance", "medium"),
        created_at=d.get("created_at", ""),
        updated_at=d.get("updated_at", d.get("created_at", "")),
    )


def _memory_to_old_dict(m: Memory) -> dict:
    """Convert a foundation Memory object back to old-format dict."""
    old_status = _NEW_STATE_MAP.get(m.lifecycle.state, "active")

    result = {
        "id": m.id,
        "category": m.domain,
        "fact_type": m.fact.fact_type,
        "subject": m.fact.subject,
        "value": m.fact.value,
        "content": m.fact.content,
        "importance": m.importance,
        "status": old_status,
        "embedding": m.embedding,
        "embedding_version": m.embedding_version,
        "created_at": m.created_at,
        "updated_at": m.updated_at,
        "source": m.provenance.source,
        "confidence": m.confidence,
        "entity": {
            "name": m.entity.name,
            "type": m.entity.entity_type,
            "id": m.entity.entity_id,
        },
        "relationships": [],
        "provenance": {
            "created_by": "user",
            "message_id": None,
            "reason": m.provenance.reason or m.provenance.source,
        },
        "temporal": {
            "valid_from": m.temporal.valid_from,
            "valid_until": m.temporal.valid_until,
            "recurring": False,
            "is_persistent": m.temporal.is_persistent,
        },
        "schema_version": m.schema_version,
    }

    # Preserve old expires_at for backward compatibility
    if m.temporal.valid_until:
        result["expires_at"] = m.temporal.valid_until

    # Preserve supersession info
    if m.lifecycle.superseded_by:
        result["superseded_by"] = m.lifecycle.superseded_by
    if m.lifecycle.superseded_at:
        result["superseded_at"] = m.lifecycle.superseded_at

    return result


class PersistentStore:
    """Foundation MemoryStore that persists through the old MemoryStore JSON format.

    Implements the foundation MemoryStore Protocol so WritePipeline can use it
    directly. Converts between Memory objects and old dict format for persistence.
    """

    def __init__(self, old_store: object) -> None:
        """Initialize with an existing old-format MemoryStore.

        Args:
            old_store: A memory.store.MemoryStore instance with .memories,
                       .add_memory(), .save(), .active_memories(), etc.
        """
        self._old_store = old_store
        self._memories: list[Memory] = []

    def load(self) -> list[Memory]:
        """Load memories from old store and convert to Memory objects."""
        # Ensure old store has loaded its data
        if not self._old_store.memories:
            self._old_store.load(ensure_embeddings=False)

        self._memories = []
        for old_dict in self._old_store.memories:
            if isinstance(old_dict, dict):
                try:
                    mem = _old_dict_to_memory(old_dict)
                    self._memories.append(mem)
                except Exception:
                    continue
        return list(self._memories)

    def save(self, memories: list[Memory]) -> None:
        """Save Memory objects by converting back to old format and persisting.

        Regenerates old_store.memories from foundation Memories to ensure
        lifecycle state changes (superseded, deactivated) are preserved.
        """
        old_dicts = []
        for mem in memories:
            old_dicts.append(_memory_to_old_dict(mem))

        self._old_store.memories = old_dicts
        self._old_store.save()
        self._memories = list(memories)

    def persist(self) -> None:
        """Persist all current foundation memories to disk.

        Regenerates old_store.memories from foundation Memories to ensure
        lifecycle state changes are preserved.
        """
        self.save(self._memories)

    def add(self, memory: Memory) -> Memory:
        """Add a new memory to the foundation store.

        The old store's memory list is regenerated from foundation Memories
        when save() is called, ensuring lifecycle state changes are preserved.
        """
        self._memories.append(memory)
        return memory

    def update(self, memory_id: str, changes: dict) -> Optional[Memory]:
        """Update an existing memory."""
        for mem in self._memories:
            if mem.id == memory_id:
                for key, value in changes.items():
                    if hasattr(mem, key):
                        setattr(mem, key, value)
                mem.updated_at = _utc_now_iso()
                return mem
        return None

    def deactivate(self, memory_id: str) -> bool:
        """Mark a memory as inactive."""
        for mem in self._memories:
            if mem.id == memory_id:
                mem.lifecycle.deactivate()
                # Also update old store
                for old_mem in self._old_store.memories:
                    if isinstance(old_mem, dict) and old_mem.get("id") == memory_id:
                        old_mem["status"] = "inactive"
                        break
                return True
        return False

    def supersede(self, winner_id: str, loser_id: str) -> bool:
        """Mark loser as superseded by winner."""
        loser = self.get_by_id(loser_id)
        if loser is None:
            return False
        if not loser.lifecycle.is_active:
            return True
        loser.lifecycle.supersede(winner_id)
        # Also update old store
        for old_mem in self._old_store.memories:
            if isinstance(old_mem, dict) and old_mem.get("id") == loser_id:
                old_mem["status"] = "superseded"
                old_mem["superseded_by"] = winner_id
                old_mem["superseded_at"] = loser.lifecycle.superseded_at
                break
        return True

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        for mem in self._memories:
            if mem.id == memory_id:
                return mem
        return None

    def get_active(self) -> list[Memory]:
        """Retrieve all active, non-expired memories."""
        return [
            m for m in self._memories
            if m.lifecycle.is_active and not m.temporal.is_expired_at()
        ]

    def get_by_entity(self, entity_name: str) -> list[Memory]:
        """Retrieve all memories for a specific entity."""
        return [
            m for m in self._memories
            if m.entity.name == entity_name
        ]

    def get_by_domain(self, domain: str) -> list[Memory]:
        """Retrieve all memories in a specific domain."""
        return [
            m for m in self._memories
            if m.domain == domain
        ]

    def search_semantic(
        self, query_vector: list[float], top_k: int = 8
    ) -> list[ScoredMemory]:
        """Search by semantic similarity."""
        if not query_vector:
            return []

        scored: list[ScoredMemory] = []
        for memory in self._memories:
            if memory.embedding is None:
                continue
            sim = _cosine_similarity(query_vector, memory.embedding)
            scored.append(ScoredMemory(memory=memory, score=sim, semantic_score=sim))

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        """Return total number of memories."""
        return len(self._memories)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SQLiteAdapter:
    """Foundation MemoryStore that persists through SQLite.

    Implements the foundation MemoryStore Protocol so WritePipeline can use it
    directly. Uses SQLiteStore for structured persistence with history.
    """

    def __init__(self, sqlite_store: object) -> None:
        """Initialize with a SQLiteStore instance.

        Args:
            sqlite_store: A SQLiteStore instance with .load(), .add(), etc.
        """
        self._sqlite_store = sqlite_store
        self._memories: list[Memory] = []

    def load(self) -> list[Memory]:
        """Load memories from SQLite store."""
        self._memories = self._sqlite_store.load()
        return list(self._memories)

    def save(self, memories: list[Memory]) -> None:
        """Save all memories to SQLite store."""
        self._sqlite_store.save(memories)
        self._memories = list(memories)

    def persist(self) -> None:
        """Persist all current foundation memories to SQLite."""
        self.save(self._memories)

    def add(self, memory: Memory) -> Memory:
        """Add a new memory to the SQLite store."""
        self._sqlite_store.add(memory)
        self._memories.append(memory)
        return memory

    def update(self, memory_id: str, changes: dict) -> Optional[Memory]:
        """Update an existing memory."""
        result = self._sqlite_store.update(memory_id, changes)
        if result:
            # Update in-memory list
            for i, mem in enumerate(self._memories):
                if mem.id == memory_id:
                    self._memories[i] = result
                    break
        return result

    def deactivate(self, memory_id: str) -> bool:
        """Mark a memory as inactive."""
        result = self._sqlite_store.deactivate(memory_id)
        if result:
            # Update in-memory list
            for mem in self._memories:
                if mem.id == memory_id:
                    mem.lifecycle.deactivate()
                    break
        return result

    def supersede(self, winner_id: str, loser_id: str) -> bool:
        """Mark loser as superseded by winner."""
        result = self._sqlite_store.supersede(winner_id, loser_id)
        if result:
            # Update in-memory list
            for mem in self._memories:
                if mem.id == loser_id:
                    mem.lifecycle.supersede(winner_id)
                    break
        return result

    def add_event(self, event: MemoryEvent) -> MemoryEvent:
        """Persist a memory event to SQLite."""
        return self._sqlite_store.add_event(event)

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        return self._sqlite_store.get_by_id(memory_id)

    def get_active(self) -> list[Memory]:
        """Retrieve all active, non-expired memories."""
        return self._sqlite_store.get_active()

    def get_by_entity(self, entity_name: str) -> list[Memory]:
        """Retrieve all memories for a specific entity."""
        return self._sqlite_store.get_by_entity(entity_name)

    def get_by_domain(self, domain: str) -> list[Memory]:
        """Retrieve all memories in a specific domain."""
        return self._sqlite_store.get_by_domain(domain)

    def search_semantic(
        self, query_vector: list[float], top_k: int = 8
    ) -> list[ScoredMemory]:
        """Search by semantic similarity."""
        return self._sqlite_store.search_semantic(query_vector, top_k)

    def count(self) -> int:
        """Return total number of memories."""
        return self._sqlite_store.count()

    def close(self) -> None:
        """Close the SQLite connection."""
        self._sqlite_store.close()


def sync_sqlite_to_old_store(sqlite_adapter: SQLiteAdapter, old_store: object) -> None:
    """Sync SQLite memories into the old JSON store's in-memory list.

    This enables the existing read path (MemoryRetriever) to find memories
    that were written to SQLite. Old JSON memories are preserved; SQLite
    memories are added (or merged) into the old store's memory list.

    Called once at startup before the retriever is initialized.

    IMPORTANT: Does NOT save to old JSON file — only modifies in-memory list.
    """
    sqlite_memories = sqlite_adapter.load()

    # Build a set of existing old-store IDs to avoid duplicates
    existing_ids = {m.get("id") for m in getattr(old_store, "memories", []) if isinstance(m, dict)}

    # Convert SQLite memories to old dict format and inject
    for mem in sqlite_memories:
        if mem.id in existing_ids:
            # Update existing entry with latest state from SQLite
            for i, old_mem in enumerate(old_store.memories):
                if isinstance(old_mem, dict) and old_mem.get("id") == mem.id:
                    old_store.memories[i] = _memory_to_old_dict(mem)
                    break
        else:
            # New memory from SQLite — add to old store
            old_dict = _memory_to_old_dict(mem)
            old_store.memories.append(old_dict)

    # Generate embeddings for any synced memories that lack them
    if hasattr(old_store, 'ensure_embeddings'):
        try:
            old_store.ensure_embeddings()
        except Exception:
            pass  # embeddings unavailable — sync still succeeds
