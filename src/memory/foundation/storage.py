"""Storage abstraction for ALLI v0.6 Cognitive Memory Engine.

The memory engine depends on this abstract interface, not on JSON
or SQLite directly. This ensures:
- Switching backends requires only implementing the interface
- No engine code changes when storage changes
- Testing can use in-memory storage (no file I/O)
- Future backends can be added without engine changes
"""

from __future__ import annotations

import math
from typing import Protocol, Optional, runtime_checkable

from .models import Memory, ScoredMemory


@runtime_checkable
class MemoryStore(Protocol):
    """Abstract storage interface for memory records.

    Every backend (JSON, SQLite, future vector store) implements
    this interface. The memory engine never touches files directly.
    """

    def load(self) -> list[Memory]:
        """Load all memories from storage."""
        ...

    def save(self, memories: list[Memory]) -> None:
        """Persist all memories to storage."""
        ...

    def add(self, memory: Memory) -> Memory:
        """Add a new memory record. Returns the stored memory."""
        ...

    def update(self, memory_id: str, changes: dict) -> Optional[Memory]:
        """Update an existing memory by ID. Returns updated memory or None."""
        ...

    def deactivate(self, memory_id: str) -> bool:
        """Mark a memory as inactive. Returns True if found and deactivated."""
        ...

    def supersede(self, winner_id: str, loser_id: str) -> bool:
        """Mark loser as superseded by winner. Returns True on success."""
        ...

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        """Retrieve a single memory by ID."""
        ...

    def get_active(self) -> list[Memory]:
        """Retrieve all active, non-expired memories."""
        ...

    def get_by_entity(self, entity_name: str) -> list[Memory]:
        """Retrieve all memories scoped to a specific entity."""
        ...

    def get_by_domain(self, domain: str) -> list[Memory]:
        """Retrieve all memories in a specific domain."""
        ...

    def search_semantic(
        self, query_vector: list[float], top_k: int = 8
    ) -> list[ScoredMemory]:
        """Search memories by semantic similarity to a query vector."""
        ...

    def count(self) -> int:
        """Return total number of memories in storage."""
        ...


class InMemoryStore:
    """In-memory storage backend for testing.

    Implements the MemoryStore interface without any file I/O.
    Data lives only in Python memory and is lost when the
    process exits.
    """

    def __init__(self) -> None:
        self._memories: list[Memory] = []

    def load(self) -> list[Memory]:
        return list(self._memories)

    def save(self, memories: list[Memory]) -> None:
        self._memories = list(memories)

    def add(self, memory: Memory) -> Memory:
        self._memories.append(memory)
        return memory

    def update(self, memory_id: str, changes: dict) -> Optional[Memory]:
        for memory in self._memories:
            if memory.id == memory_id:
                for key, value in changes.items():
                    if hasattr(memory, key):
                        setattr(memory, key, value)
                memory.updated_at = _utc_now_iso()
                return memory
        return None

    def deactivate(self, memory_id: str) -> bool:
        for memory in self._memories:
            if memory.id == memory_id:
                memory.lifecycle.deactivate()
                return True
        return False

    def supersede(self, winner_id: str, loser_id: str) -> bool:
        loser = self.get_by_id(loser_id)
        if loser is None:
            return False
        if not loser.lifecycle.is_active:
            return True
        loser.lifecycle.supersede(winner_id)
        return True

    def get_by_id(self, memory_id: str) -> Optional[Memory]:
        for memory in self._memories:
            if memory.id == memory_id:
                return memory
        return None

    def get_active(self) -> list[Memory]:
        return [
            m for m in self._memories
            if m.lifecycle.is_active and not m.temporal.is_expired_at()
        ]

    def get_by_entity(self, entity_name: str) -> list[Memory]:
        return [
            m for m in self._memories
            if m.entity.name == entity_name
        ]

    def get_by_domain(self, domain: str) -> list[Memory]:
        return [
            m for m in self._memories
            if m.domain == domain
        ]

    def search_semantic(
        self, query_vector: list[float], top_k: int = 8
    ) -> list[ScoredMemory]:
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
        return len(self._memories)


def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
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
