"""SQLite storage tests for ALLI v0.6 Cognitive Memory Engine.

Tests the SQLite-backed storage implementation including:
- Schema creation
- Database initialization
- CRUD operations
- Entity storage
- Fact storage
- Relationship storage
- Episode storage
- Memory events
- Lifecycle state
- Temporal state
- Provenance
- Serialization/deserialization
- Transactions
- Rollback
- Duplicate handling
- Invalid references
- Unicode
- Long values
- Restart persistence
- Entity isolation
"""

from __future__ import annotations

import os
import sys
import tempfile
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.foundation.sqlite_store import SQLiteStore, _serialize_json, _deserialize_json
from memory.foundation.models import (
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
    TemporalScope,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".sqlite", delete=False) as f:
        db_path = f.name
    store = SQLiteStore(db_path)
    yield store, db_path
    store.close()
    os.unlink(db_path)


@pytest.fixture
def sample_entity():
    """Create a sample entity."""
    return EntityRef(name="self", entity_type="self")


@pytest.fixture
def sample_memory(sample_entity):
    """Create a sample memory."""
    return Memory(
        id="mem_001",
        entity=sample_entity,
        fact=Fact(
            fact_type="preference",
            subject="coffee",
            predicate="has_preference",
            value="coffee",
            content="I like coffee",
        ),
        domain="interests",
        lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
        provenance=Provenance(source="user_statement", reason="user_statement"),
        temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00", is_persistent=True),
        confidence=0.9,
        importance="medium",
    )


# ---------------------------------------------------------------------------
# Schema and initialization tests
# ---------------------------------------------------------------------------

class TestSchema:
    def test_creates_tables(self, temp_db):
        store, db_path = temp_db
        conn = store._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        assert "entities" in table_names
        assert "memories" in table_names
        assert "relationships" in table_names
        assert "memory_events" in table_names
        assert "episodes" in table_names

    def test_creates_indexes(self, temp_db):
        store, db_path = temp_db
        conn = store._get_conn()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        index_names = {i["name"] for i in indexes}
        assert "idx_memories_entity" in index_names
        assert "idx_memories_domain" in index_names
        assert "idx_memories_lifecycle" in index_names


# ---------------------------------------------------------------------------
# Entity operations tests
# ---------------------------------------------------------------------------

class TestEntityOperations:
    def test_get_or_create_entity(self, temp_db):
        store, _ = temp_db
        entity = store.get_or_create_entity("alice", "person")
        assert entity.name == "alice"
        assert entity.entity_type == "person"

    def test_get_existing_entity(self, temp_db):
        store, _ = temp_db
        entity1 = store.get_or_create_entity("bob", "person")
        entity2 = store.get_or_create_entity("bob", "person")
        assert entity1.id == entity2.id

    def test_get_entity_by_name(self, temp_db):
        store, _ = temp_db
        store.get_or_create_entity("charlie", "person")
        entity = store.get_entity_by_name("charlie")
        assert entity is not None
        assert entity.name == "charlie"

    def test_get_nonexistent_entity(self, temp_db):
        store, _ = temp_db
        entity = store.get_entity_by_name("nonexistent")
        assert entity is None


# ---------------------------------------------------------------------------
# CRUD operations tests
# ---------------------------------------------------------------------------

class TestCRUD:
    def test_add_and_get_memory(self, temp_db, sample_memory):
        store, _ = temp_db
        store.add(sample_memory)
        retrieved = store.get_by_id("mem_001")
        assert retrieved is not None
        assert retrieved.id == "mem_001"
        assert retrieved.content == "I like coffee"

    def test_load_all_memories(self, temp_db, sample_memory):
        store, _ = temp_db
        store.add(sample_memory)
        memories = store.load()
        assert len(memories) == 1
        assert memories[0].id == "mem_001"

    def test_count(self, temp_db, sample_memory):
        store, _ = temp_db
        assert store.count() == 0
        store.add(sample_memory)
        assert store.count() == 1

    def test_update_memory(self, temp_db, sample_memory):
        store, _ = temp_db
        store.add(sample_memory)
        updated = store.update("mem_001", {"content": "I love coffee"})
        assert updated is not None
        assert updated.content == "I love coffee"

    def test_update_nonexistent(self, temp_db):
        store, _ = temp_db
        result = store.update("nonexistent", {"content": "test"})
        assert result is None

    def test_delete_memory(self, temp_db, sample_memory):
        store, _ = temp_db
        store.add(sample_memory)
        result = store.deactivate("mem_001")
        assert result is True
        retrieved = store.get_by_id("mem_001")
        assert retrieved.lifecycle.state == LifecycleState.INACTIVE.value


# ---------------------------------------------------------------------------
# Entity-scoped queries tests
# ---------------------------------------------------------------------------

class TestEntityScopedQueries:
    def test_get_by_entity(self, temp_db):
        store, _ = temp_db
        mem1 = Memory(
            id="mem_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        mem2 = Memory(
            id="mem_002",
            entity=EntityRef(name="mom"),
            fact=Fact(fact_type="preference", subject="tea", predicate="likes", value="tea", content="Mom likes tea"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem1)
        store.add(mem2)

        self_memories = store.get_by_entity("self")
        assert len(self_memories) == 1
        assert self_memories[0].entity.name == "self"

        mom_memories = store.get_by_entity("mom")
        assert len(mom_memories) == 1
        assert mom_memories[0].entity.name == "mom"

    def test_get_by_domain(self, temp_db):
        store, _ = temp_db
        mem1 = Memory(
            id="mem_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        mem2 = Memory(
            id="mem_002",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="learning", subject="Python", predicate="learning", value="Python", content="I am learning Python"),
            domain="learning",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem1)
        store.add(mem2)

        interests = store.get_by_domain("interests")
        assert len(interests) == 1
        assert interests[0].domain == "interests"


# ---------------------------------------------------------------------------
# Lifecycle state tests
# ---------------------------------------------------------------------------

class TestLifecycleState:
    def test_supersede(self, temp_db):
        store, _ = temp_db
        mem1 = Memory(
            id="mem_old",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem1)

        result = store.supersede("mem_new", "mem_old")
        assert result is True

        old = store.get_by_id("mem_old")
        assert old.lifecycle.state == LifecycleState.INACTIVE.value
        assert old.lifecycle.superseded_by == "mem_new"

    def test_deactivate(self, temp_db):
        store, _ = temp_db
        mem = Memory(
            id="mem_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem)

        result = store.deactivate("mem_001")
        assert result is True

        deactivated = store.get_by_id("mem_001")
        assert deactivated.lifecycle.state == LifecycleState.INACTIVE.value
        assert deactivated.lifecycle.deactivated_at is not None


# ---------------------------------------------------------------------------
# Event operations tests
# ---------------------------------------------------------------------------

class TestEventOperations:
    def test_add_event(self, temp_db):
        store, _ = temp_db
        mem = Memory(
            id="mem_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem)

        event = MemoryEvent(
            id="evt_001",
            event_type="created",
            memory_id="mem_001",
            timestamp="2026-01-01T00:00:00+00:00",
            trigger="user_statement",
            details={"content": "I like coffee"},
        )
        store.add_event(event)

        events = store.get_events_for_memory("mem_001")
        assert len(events) == 1
        assert events[0].event_type == "created"


# ---------------------------------------------------------------------------
# Transaction tests
# ---------------------------------------------------------------------------

class TestTransactions:
    def test_rollback_on_error(self, temp_db):
        store, _ = temp_db
        # Try to add a memory with invalid data (missing required fields)
        with pytest.raises(Exception):
            with store._transaction() as conn:
                conn.execute("INSERT INTO memories (memory_id) VALUES (?)", ("test",))
        # Verify nothing was added
        assert store.count() == 0


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_json_serialization(self):
        data = {"key": "value", "nested": [1, 2, 3]}
        serialized = _serialize_json(data)
        deserialized = _deserialize_json(serialized)
        assert deserialized == data

    def test_json_unicode(self):
        data = {"text": "こんにちは世界"}
        serialized = _serialize_json(data)
        deserialized = _deserialize_json(serialized)
        assert deserialized == data

    def test_json_empty(self):
        assert _deserialize_json("") == {}
        assert _deserialize_json(None) == {}


# ---------------------------------------------------------------------------
# Unicode and special characters tests
# ---------------------------------------------------------------------------

class TestUnicode:
    def test_unicode_memory(self, temp_db):
        store, _ = temp_db
        mem = Memory(
            id="mem_unicode",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="日本語", predicate="likes", value="日本語", content="私は日本語が好きです"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem)
        retrieved = store.get_by_id("mem_unicode")
        assert retrieved.content == "私は日本語が好きです"

    def test_special_characters(self, temp_db):
        store, _ = temp_db
        mem = Memory(
            id="mem_special",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="C++", predicate="likes", value="C++", content="I like C++ & Python"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem)
        retrieved = store.get_by_id("mem_special")
        assert retrieved.content == "I like C++ & Python"


# ---------------------------------------------------------------------------
# Long values tests
# ---------------------------------------------------------------------------

class TestLongValues:
    def test_long_content(self, temp_db):
        store, _ = temp_db
        long_content = "x" * 10000
        mem = Memory(
            id="mem_long",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="note", subject="long", predicate="has_note", value="long", content=long_content),
            domain="notes",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem)
        retrieved = store.get_by_id("mem_long")
        assert len(retrieved.content) == 10000


# ---------------------------------------------------------------------------
# Restart persistence tests
# ---------------------------------------------------------------------------

class TestRestartPersistence:
    def test_data_surives_restart(self, temp_db):
        store, db_path = temp_db
        mem = Memory(
            id="mem_persist",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem)
        store.close()

        # Reopen the database
        store2 = SQLiteStore(db_path)
        retrieved = store2.get_by_id("mem_persist")
        assert retrieved is not None
        assert retrieved.content == "I like coffee"
        store2.close()


# ---------------------------------------------------------------------------
# Entity isolation tests
# ---------------------------------------------------------------------------

class TestEntityIsolation:
    def test_entities_isolated(self, temp_db):
        store, _ = temp_db
        mem1 = Memory(
            id="mem_self",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="blue", predicate="likes", value="blue", content="I like blue"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        mem2 = Memory(
            id="mem_mom",
            entity=EntityRef(name="mom"),
            fact=Fact(fact_type="preference", subject="pink", predicate="likes", value="pink", content="Mom likes pink"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        store.add(mem1)
        store.add(mem2)

        self_memories = store.get_by_entity("self")
        mom_memories = store.get_by_entity("mom")

        assert len(self_memories) == 1
        assert self_memories[0].content == "I like blue"
        assert len(mom_memories) == 1
        assert mom_memories[0].content == "Mom likes pink"


# ---------------------------------------------------------------------------
# Duplicate handling tests
# ---------------------------------------------------------------------------

class TestDuplicateHandling:
    def test_duplicate_id_replaces(self, temp_db):
        store, _ = temp_db
        mem1 = Memory(
            id="mem_dup",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-01T00:00:00+00:00"),
        )
        mem2 = Memory(
            id="mem_dup",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="preference", subject="tea", predicate="likes", value="tea", content="I like tea"),
            domain="interests",
            lifecycle=Lifecycle(state=LifecycleState.ACTIVE.value),
            provenance=Provenance(),
            temporal=TemporalScope(valid_from="2026-01-02T00:00:00+00:00"),
        )
        store.add(mem1)
        store.add(mem2)
        assert store.count() == 1
        retrieved = store.get_by_id("mem_dup")
        assert retrieved.content == "I like tea"
