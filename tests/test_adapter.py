"""Integration tests for the PersistentStore adapter.

Tests that the adapter correctly bridges foundation Memory objects
to/from the old JSON persistence format.
"""

from __future__ import annotations

import os
import sys
import tempfile
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.foundation.adapter import PersistentStore, _old_dict_to_memory, _memory_to_old_dict
from memory.foundation.models import Memory, EntityRef, Fact, Lifecycle, LifecycleState
from memory.foundation.models import Provenance, TemporalScope


# ---------------------------------------------------------------------------
# Mock old MemoryStore for testing
# ---------------------------------------------------------------------------

class MockOldMemoryStore:
    """Minimal mock of the old MemoryStore for testing the adapter."""

    def __init__(self):
        self.memories = []
        self._save_count = 0
        self._added = []

    def load(self, ensure_embeddings=True):
        return self.memories

    def save(self):
        self._save_count += 1

    def add_memory(self, *, category, content, fact_type=None, importance=None,
                   value=None, subject=None, entity_name="user", entity_type="self",
                   confidence=1.0, **kwargs):
        memory = {
            "id": f"memory_{len(self.memories) + 1:03d}",
            "category": category,
            "fact_type": fact_type or "general",
            "subject": subject or value or "",
            "value": value or "",
            "content": content,
            "importance": importance or "medium",
            "status": "active",
            "embedding": None,
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
            "source": "user_statement",
            "confidence": confidence,
            "entity": {"name": entity_name, "type": entity_type, "id": None},
            "relationships": [],
            "provenance": {"created_by": "user", "message_id": None, "reason": "user_statement"},
            "temporal": {"valid_from": "2026-01-01T00:00:00", "valid_until": None, "recurring": False},
            "schema_version": 1,
        }
        self.memories.append(memory)
        self._added.append(memory)
        return memory


# ---------------------------------------------------------------------------
# Conversion tests
# ---------------------------------------------------------------------------

class TestOldDictToMemory:
    def test_basic_conversion(self) -> None:
        old = {
            "id": "memory_001",
            "category": "interests",
            "fact_type": "preference",
            "subject": "coffee",
            "value": "coffee",
            "content": "User likes coffee",
            "importance": "medium",
            "status": "active",
            "embedding": None,
            "created_at": "2026-01-01T00:00:00",
            "confidence": 0.9,
            "entity": {"name": "self", "type": "self", "id": None},
        }
        mem = _old_dict_to_memory(old)
        assert mem.id == "memory_001"
        assert mem.domain == "interests"
        assert mem.entity.name == "self"
        assert mem.fact.content == "User likes coffee"
        assert mem.lifecycle.is_active

    def test_inactive_status(self) -> None:
        old = {"id": "m1", "status": "inactive", "category": "other",
               "content": "test", "entity": {"name": "self", "type": "self"}}
        mem = _old_dict_to_memory(old)
        assert mem.lifecycle.state == LifecycleState.INACTIVE.value

    def test_entity_mapping(self) -> None:
        old = {"id": "m1", "status": "active", "category": "other",
               "content": "test", "entity": {"name": "mom", "type": "person"}}
        mem = _old_dict_to_memory(old)
        assert mem.entity.name == "mom"
        assert mem.entity.entity_type == "person"


class TestMemoryToOldDict:
    def test_basic_conversion(self) -> None:
        mem = Memory(
            id="mem_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="has_preference",
                      value="coffee", content="User likes coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(source="user_statement"),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00"),
        )
        old = _memory_to_old_dict(mem)
        assert old["id"] == "mem_001"
        assert old["category"] == "interests"
        assert old["status"] == "active"
        assert old["entity"]["name"] == "self"
        assert old["content"] == "User likes coffee"

    def test_inactive_state(self) -> None:
        mem = Memory(
            id="mem_002",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="general", subject="x", predicate="has_general",
                      value="x", content="test"),
            domain="other",
            lifecycle=Lifecycle(state=LifecycleState.INACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from=""),
        )
        old = _memory_to_old_dict(mem)
        assert old["status"] in ("inactive", "superseded")

    def test_superseded_preserved(self) -> None:
        mem = Memory(
            id="mem_003",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="general", subject="x", predicate="has_general",
                      value="x", content="test"),
            domain="other",
            lifecycle=Lifecycle(
                state=LifecycleState.INACTIVE.value,
                superseded_by="mem_004",
                superseded_at="2026-01-02T00:00:00",
            ),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from=""),
        )
        old = _memory_to_old_dict(mem)
        assert old["superseded_by"] == "mem_004"
        assert old["superseded_at"] == "2026-01-02T00:00:00"


# ---------------------------------------------------------------------------
# PersistentStore adapter tests
# ---------------------------------------------------------------------------

class TestPersistentStore:
    def test_load_converts_old_memories(self) -> None:
        old_store = MockOldMemoryStore()
        old_store.memories = [
            {"id": "m1", "category": "interests", "content": "likes coffee",
             "status": "active", "entity": {"name": "self", "type": "self"}},
            {"id": "m2", "category": "learning", "content": "learning Python",
             "status": "active", "entity": {"name": "self", "type": "self"}},
        ]
        store = PersistentStore(old_store)
        memories = store.load()
        assert len(memories) == 2
        assert memories[0].id == "m1"
        assert memories[0].domain == "interests"
        assert memories[1].id == "m2"

    def test_add_adds_to_foundation_store(self) -> None:
        old_store = MockOldMemoryStore()
        store = PersistentStore(old_store)
        store.load()

        mem = Memory(
            id="new_1",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="tea", predicate="has_preference",
                      value="tea", content="likes tea"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from=""),
        )
        result = store.add(mem)
        assert result is not None
        assert len(store._memories) == 1
        assert store._memories[0].fact.subject == "tea"

    def test_get_active_filters_inactive(self) -> None:
        old_store = MockOldMemoryStore()
        old_store.memories = [
            {"id": "m1", "category": "a", "content": "a", "status": "active",
             "entity": {"name": "self", "type": "self"}},
            {"id": "m2", "category": "b", "content": "b", "status": "inactive",
             "entity": {"name": "self", "type": "self"}},
        ]
        store = PersistentStore(old_store)
        store.load()
        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == "m1"

    def test_deactivate(self) -> None:
        old_store = MockOldMemoryStore()
        old_store.memories = [
            {"id": "m1", "category": "a", "content": "a", "status": "active",
             "entity": {"name": "self", "type": "self"}},
        ]
        store = PersistentStore(old_store)
        store.load()
        assert store.deactivate("m1")
        assert old_store.memories[0]["status"] == "inactive"

    def test_get_by_entity(self) -> None:
        old_store = MockOldMemoryStore()
        old_store.memories = [
            {"id": "m1", "category": "a", "content": "a", "status": "active",
             "entity": {"name": "self", "type": "self"}},
            {"id": "m2", "category": "b", "content": "b", "status": "active",
             "entity": {"name": "mom", "type": "person"}},
        ]
        store = PersistentStore(old_store)
        store.load()
        self_mems = store.get_by_entity("self")
        assert len(self_mems) == 1
        mom_mems = store.get_by_entity("mom")
        assert len(mom_mems) == 1

    def test_roundtrip_preserves_data(self) -> None:
        old_store = MockOldMemoryStore()
        old_store.memories = [
            {"id": "m1", "category": "interests", "fact_type": "preference",
             "subject": "coffee", "value": "coffee", "content": "User likes coffee",
             "status": "active", "importance": "medium", "confidence": 0.9,
             "entity": {"name": "self", "type": "self"},
             "temporal": {"valid_from": "2026-01-01", "valid_until": None}},
        ]
        store = PersistentStore(old_store)
        store.load()

        # Convert to old format and verify
        old_dict = _memory_to_old_dict(store.get_by_id("m1"))
        assert old_dict["category"] == "interests"
        assert old_dict["content"] == "User likes coffee"
        assert old_dict["entity"]["name"] == "self"
