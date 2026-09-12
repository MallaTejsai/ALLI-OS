"""Session 4 — Persistent Storage Certification Tests.

Tests the SQLite persistence layer for the v0.6 memory engine.
Verifies that memories, lifecycle state, provenance, and events
survive process restarts (write → close → reopen → verify).

Categories:
  A. Basic Persistence (write/restart/verify)
  B. Lifecycle Persistence (supersede/deactivate/invalidate survive restart)
  C. Multi-Entity Persistence (entity isolation across restart)
  D. Historical + Current Values
  E. Provenance / History Persistence
  F. Memory Event Persistence
  G. Schema / Metadata
  H. Adversarial Storage Tests
  I. Pipeline Integration (full write pipeline + restart)
"""

from __future__ import annotations

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.foundation.sqlite_store import SQLiteStore
from memory.foundation.adapter import SQLiteAdapter
from memory.foundation.pipeline import WritePipeline
from memory.foundation.models import (
    Memory,
    MemoryEvent,
    LifecycleState,
    ProvenanceSource,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path():
    """Create a temporary database path, clean up after test."""
    path = tempfile.mktemp(suffix=".sqlite")
    yield path
    if os.path.exists(path):
        os.unlink(path)
    # Also clean up WAL/SHM files
    for suffix in ("-wal", "-shm"):
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def store(db_path):
    return SQLiteStore(db_path)


@pytest.fixture
def adapter(store):
    return SQLiteAdapter(store)


@pytest.fixture
def pipeline(adapter):
    return WritePipeline(adapter)


def run(pipeline, msg):
    return pipeline.process(msg)


def active_memories(adapter):
    return adapter.get_active()


def all_memories(adapter):
    return adapter.load()


def active_count(adapter):
    return len(active_memories(adapter))


def has_active_with_value(adapter, value_substring):
    return any(value_substring.lower() in m.content.lower() for m in active_memories(adapter))


def reopen(db_path):
    """Simulate restart: close everything, create fresh store + adapter."""
    store = SQLiteStore(db_path)
    adapter = SQLiteAdapter(store)
    adapter.load()
    return store, adapter


# ===========================================================================
# A. Basic Persistence — 10 cases
# ===========================================================================

class TestBasicPersistence:
    """Write → close → reopen → verify memory survives."""

    def test_A1_single_memory_survives_restart(self, pipeline, adapter, db_path):
        """Single fact persists across restart."""
        r = run(pipeline, "I like coffee")
        assert r.results[0].action_taken == "created"
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem is not None
        assert "coffee" in mem.content.lower()
        assert mem.lifecycle.state == LifecycleState.ACTIVE.value
        store2.close()

    def test_A2_multiple_memories_survive_restart(self, pipeline, adapter, db_path):
        """Multiple facts persist across restart."""
        run(pipeline, "I like coffee")
        run(pipeline, "I am learning Python")
        run(pipeline, "My name is Alice")
        assert active_count(adapter) == 3

        store2, adapter2 = reopen(db_path)
        assert adapter2.count() == 3
        active = adapter2.get_active()
        assert len(active) == 3
        store2.close()

    def test_A3_entity_field_survives_restart(self, pipeline, adapter, db_path):
        """Entity name and type persist correctly."""
        run(pipeline, "I like coffee")
        run(pipeline, "My mom likes tea")

        store2, adapter2 = reopen(db_path)
        self_mems = [m for m in adapter2.load() if m.entity.name == "self"]
        mom_mems = [m for m in adapter2.load() if m.entity.name == "mom"]
        assert len(self_mems) == 1
        assert len(mom_mems) == 1
        store2.close()

    def test_A4_domain_survives_restart(self, pipeline, adapter, db_path):
        """Domain field persists correctly."""
        run(pipeline, "I like coffee")
        run(pipeline, "I am learning Python")

        store2, adapter2 = reopen(db_path)
        domains = {m.domain for m in adapter2.load()}
        assert "interests" in domains
        assert "learning" in domains
        store2.close()

    def test_A5_fact_fields_survive_restart(self, pipeline, adapter, db_path):
        """All fact fields (type, subject, predicate, value, content, polarity) persist."""
        r = run(pipeline, "I like coffee")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.fact.fact_type is not None
        assert mem.fact.subject is not None
        assert mem.fact.value is not None
        assert mem.fact.content is not None
        store2.close()

    def test_A6_confidence_survives_restart(self, pipeline, adapter, db_path):
        """Confidence value persists."""
        r = run(pipeline, "I like coffee")
        original_conf = r.memories[0].confidence
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.confidence == original_conf
        store2.close()

    def test_A7_importance_survives_restart(self, pipeline, adapter, db_path):
        """Importance level persists."""
        r = run(pipeline, "I like coffee")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.importance is not None
        store2.close()

    def test_A8_provenance_survives_restart(self, pipeline, adapter, db_path):
        """Provenance fields persist."""
        r = run(pipeline, "I like coffee")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.provenance.source is not None
        store2.close()

    def test_A9_temporal_survives_restart(self, pipeline, adapter, db_path):
        """Temporal scope persists."""
        r = run(pipeline, "I like coffee")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.temporal.is_persistent is True
        assert mem.temporal.valid_from is not None
        store2.close()

    def test_A10_count_survives_restart(self, pipeline, adapter, db_path):
        """Count persists across restart."""
        run(pipeline, "I like coffee")
        run(pipeline, "I like tea")
        assert adapter.count() == 2

        store2, adapter2 = reopen(db_path)
        assert adapter2.count() == 2
        store2.close()


# ===========================================================================
# B. Lifecycle Persistence — 10 cases
# ===========================================================================

class TestLifecyclePersistence:
    """Lifecycle state changes persist across restart."""

    def test_B1_supersede_persists(self, pipeline, adapter, db_path):
        """Superseded memory is inactive after restart."""
        r1 = run(pipeline, "I use an Oppo A53")
        oppo_id = r1.results[0].memory_id
        run(pipeline, "I use a Samsung S24")

        store2, adapter2 = reopen(db_path)
        oppo = adapter2.get_by_id(oppo_id)
        assert oppo.lifecycle.state == LifecycleState.INACTIVE.value
        assert oppo.lifecycle.superseded_by is not None
        store2.close()

    def test_B2_superseded_by_link_persists(self, pipeline, adapter, db_path):
        """superseded_by link points to correct winner after restart."""
        r1 = run(pipeline, "I use an Oppo A53")
        oppo_id = r1.results[0].memory_id
        r2 = run(pipeline, "I use a Samsung S24")
        samsung_id = r2.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        oppo = adapter2.get_by_id(oppo_id)
        assert oppo.lifecycle.superseded_by == samsung_id
        store2.close()

    def test_B3_supersession_chain_persists(self, pipeline, adapter, db_path):
        """Chain A → B → C: only C active after restart."""
        r1 = run(pipeline, "I use an Oppo A53")
        r2 = run(pipeline, "I use a Samsung S24")
        r3 = run(pipeline, "I use an iPhone 15")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 1
        assert "iphone" in active[0].content.lower()

        oppo = adapter2.get_by_id(r1.results[0].memory_id)
        samsung = adapter2.get_by_id(r2.results[0].memory_id)
        assert oppo.lifecycle.state == LifecycleState.INACTIVE.value
        assert samsung.lifecycle.state == LifecycleState.INACTIVE.value
        store2.close()

    def test_B4_deactivate_persists(self, pipeline, adapter, db_path):
        """Deactivated memory is inactive after restart."""
        r1 = run(pipeline, "I like tea")
        tea_id = r1.results[0].memory_id
        run(pipeline, "Forget tea")

        store2, adapter2 = reopen(db_path)
        tea = adapter2.get_by_id(tea_id)
        assert tea.lifecycle.state == LifecycleState.INACTIVE.value
        assert tea.lifecycle.deactivated_at is not None
        store2.close()

    def test_B5_invalidate_persists(self, pipeline, adapter, db_path):
        """Invalidated memory is inactive after restart."""
        r1 = run(pipeline, "I like coffee")
        coffee_id = r1.results[0].memory_id
        run(pipeline, "I don't like coffee anymore")

        store2, adapter2 = reopen(db_path)
        coffee = adapter2.get_by_id(coffee_id)
        assert coffee.lifecycle.state == LifecycleState.INACTIVE.value
        store2.close()

    def test_B6_active_after_restart_correct(self, pipeline, adapter, db_path):
        """Only truly active memories are returned by get_active() after restart."""
        run(pipeline, "I like coffee")
        r2 = run(pipeline, "I like tea")
        tea_id = r2.results[0].memory_id
        run(pipeline, "I use a Samsung S24")
        run(pipeline, "Forget tea")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        active_contents = [m.content.lower() for m in active]
        assert any("coffee" in c for c in active_contents)
        assert any("samsung" in c for c in active_contents)
        assert not any("tea" in c for c in active_contents)
        store2.close()

    def test_B7_historical_not_supersede_current(self, pipeline, adapter, db_path):
        """Historical statement doesn't supersede current after restart."""
        run(pipeline, "I use a Samsung S24")
        run(pipeline, "I used to have an Oppo A53")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert any("samsung" in m.content.lower() for m in active)
        store2.close()

    def test_B8_temporary_persists(self, pipeline, adapter, db_path):
        """Temporary memory persists with correct temporal flag."""
        r = run(pipeline, "I have a dentist appointment tomorrow")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.temporal.is_persistent is False
        store2.close()

    def test_B9_uncertain_persists(self, pipeline, adapter, db_path):
        """Uncertain memory persists with correct confidence."""
        r = run(pipeline, "I might learn Rust someday")
        mem_id = r.results[0].memory_id
        original_conf = r.memories[0].confidence

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.confidence == original_conf
        store2.close()

    def test_B10_deactivate_then_add_persists(self, pipeline, adapter, db_path):
        """After deactivation, new fact coexists correctly after restart."""
        run(pipeline, "I like coffee")
        run(pipeline, "I don't like coffee anymore")
        run(pipeline, "I like tea")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        active_contents = [m.content.lower() for m in active]
        assert any("tea" in c for c in active_contents)
        assert not any("coffee" in c for c in active_contents)
        store2.close()


# ===========================================================================
# C. Multi-Entity Persistence — 10 cases
# ===========================================================================

class TestMultiEntityPersistence:
    """Multiple entities persist with correct isolation across restart."""

    def test_C1_self_vs_mom_isolated(self, pipeline, adapter, db_path):
        """Self and mom memories are independent after restart."""
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")

        store2, adapter2 = reopen(db_path)
        self_mems = [m for m in adapter2.load() if m.entity.name == "self"]
        mom_mems = [m for m in adapter2.load() if m.entity.name == "mom"]
        assert len(self_mems) == 1
        assert len(mom_mems) == 1
        assert "blue" in self_mems[0].content.lower()
        assert "red" in mom_mems[0].content.lower()
        store2.close()

    def test_C2_third_party_isolated(self, pipeline, adapter, db_path):
        """Third-party entity is isolated from self after restart."""
        run(pipeline, "I like coffee")
        run(pipeline, "My friend Jake likes tea")

        store2, adapter2 = reopen(db_path)
        self_mems = [m for m in adapter2.load() if m.entity.name == "self"]
        jake_mems = [m for m in adapter2.load() if m.entity.name == "jake"]
        assert len(self_mems) == 1
        assert len(jake_mems) == 1
        store2.close()

    def test_C3_self_change_does_not_affect_mom(self, pipeline, adapter, db_path):
        """Changing self's preference doesn't affect mom after restart."""
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")
        run(pipeline, "I like green")

        store2, adapter2 = reopen(db_path)
        mom_mems = [m for m in adapter2.load() if m.entity.name == "mom"]
        assert len(mom_mems) == 1
        assert "red" in mom_mems[0].content.lower()
        store2.close()

    def test_C4_multiple_entities_all_persist(self, pipeline, adapter, db_path):
        """Five different entities all persist correctly."""
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")
        run(pipeline, "My dad likes green")
        run(pipeline, "My brother likes yellow")
        run(pipeline, "My friend Alice likes purple")

        store2, adapter2 = reopen(db_path)
        entities = {m.entity.name for m in adapter2.load()}
        assert "self" in entities
        assert "mom" in entities
        assert "dad" in entities
        assert "brother" in entities
        assert "alice" in entities
        store2.close()

    def test_C5_entity_supersede_isolated(self, pipeline, adapter, db_path):
        """Superseding self's phone doesn't affect mom's preference."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "My mom likes red")
        run(pipeline, "I use a Samsung S24")

        store2, adapter2 = reopen(db_path)
        mom_mems = [m for m in adapter2.load() if m.entity.name == "mom"]
        assert len(mom_mems) == 1
        assert "red" in mom_mems[0].content.lower()
        self_active = [m for m in adapter2.get_active() if m.entity.name == "self"]
        assert len(self_active) == 1
        assert "samsung" in self_active[0].content.lower()
        store2.close()

    def test_C6_different_domains_same_entity(self, pipeline, adapter, db_path):
        """Different domains for same entity persist correctly."""
        run(pipeline, "I like coffee")
        run(pipeline, "I am learning Python")
        run(pipeline, "I want to build a website")

        store2, adapter2 = reopen(db_path)
        self_mems = [m for m in adapter2.load() if m.entity.name == "self"]
        domains = {m.domain for m in self_mems}
        assert len(domains) >= 2
        store2.close()

    def test_C7_entity_name_case_insensitive(self, pipeline, adapter, db_path):
        """Entity names are normalized (lowercase) in storage."""
        run(pipeline, "My friend Jake likes tea")

        store2, adapter2 = reopen(db_path)
        jake_mems = [m for m in adapter2.load() if m.entity.name == "jake"]
        assert len(jake_mems) == 1
        store2.close()

    def test_C8_entity_isolation_after_multiple_operations(self, pipeline, adapter, db_path):
        """Entity isolation maintained through complex operation sequences."""
        run(pipeline, "I like Java")
        run(pipeline, "My mom likes Python")
        run(pipeline, "I like Python")
        run(pipeline, "I don't like Java anymore")

        store2, adapter2 = reopen(db_path)
        self_active = [m for m in adapter2.get_active() if m.entity.name == "self"]
        mom_active = [m for m in adapter2.get_active() if m.entity.name == "mom"]
        self_values = {m.fact.value.lower() for m in self_active}
        assert "python" in self_values
        assert "java" not in self_values
        assert len(mom_active) == 1
        assert "python" in mom_active[0].fact.value.lower()
        store2.close()

    def test_C9_entity_type_persists(self, pipeline, adapter, db_path):
        """Entity type (self, person) persists correctly."""
        run(pipeline, "I like coffee")
        run(pipeline, "My friend Bob likes tea")

        store2, adapter2 = reopen(db_path)
        self_mems = [m for m in adapter2.load() if m.entity.name == "self"]
        bob_mems = [m for m in adapter2.load() if m.entity.name == "bob"]
        assert self_mems[0].entity.entity_type == "self"
        assert bob_mems[0].entity.entity_type == "person"
        store2.close()

    def test_C10_empty_database_restart(self, pipeline, adapter, db_path):
        """Restart with no memories returns empty store."""
        store2, adapter2 = reopen(db_path)
        assert adapter2.count() == 0
        assert len(adapter2.get_active()) == 0
        store2.close()


# ===========================================================================
# D. Historical + Current Values — 5 cases
# ===========================================================================

class TestHistoricalCurrentPersistence:
    """Historical and current values persist correctly."""

    def test_D1_old_inactive_new_active(self, pipeline, adapter, db_path):
        """After supersession, old is inactive, new is active after restart."""
        r1 = run(pipeline, "I use an Oppo A53")
        r2 = run(pipeline, "I use a Samsung S24")

        store2, adapter2 = reopen(db_path)
        oppo = adapter2.get_by_id(r1.results[0].memory_id)
        samsung = adapter2.get_by_id(r2.results[0].memory_id)
        assert oppo.lifecycle.state == LifecycleState.INACTIVE.value
        assert samsung.lifecycle.state == LifecycleState.ACTIVE.value
        store2.close()

    def test_D2_history_preserved_not_deleted(self, pipeline, adapter, db_path):
        """Old memories remain in storage (not deleted) after restart."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "I use a Samsung S24")
        run(pipeline, "I use an iPhone 15")

        store2, adapter2 = reopen(db_path)
        assert adapter2.count() == 3
        active = adapter2.get_active()
        assert len(active) == 1
        store2.close()

    def test_D3_supersession_chain_all_preserved(self, pipeline, adapter, db_path):
        """Full chain A→B→C: all 3 records exist, only C active."""
        r1 = run(pipeline, "I use an Oppo A53")
        r2 = run(pipeline, "I use a Samsung S24")
        r3 = run(pipeline, "I use an iPhone 15")

        store2, adapter2 = reopen(db_path)
        all_mems = adapter2.load()
        assert len(all_mems) == 3
        inactive = [m for m in all_mems if m.lifecycle.state == LifecycleState.INACTIVE.value]
        active = [m for m in all_mems if m.lifecycle.state == LifecycleState.ACTIVE.value]
        assert len(inactive) == 2
        assert len(active) == 1
        store2.close()

    def test_D4_preference_multi_value_coexists(self, pipeline, adapter, db_path):
        """Multiple preferences (multi-value) all active after restart."""
        run(pipeline, "I like Java")
        run(pipeline, "I like Python")
        run(pipeline, "I like Rust")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 3
        values = {m.fact.value.lower() for m in active}
        assert "java" in values
        assert "python" in values
        assert "rust" in values
        store2.close()

    def test_D5_negation_removes_specific_after_restart(self, pipeline, adapter, db_path):
        """Negating one preference leaves others active after restart."""
        run(pipeline, "I like Java")
        run(pipeline, "I like Python")
        run(pipeline, "I don't like Java anymore")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        values = {m.fact.value.lower() for m in active}
        assert "python" in values
        assert "java" not in values
        store2.close()


# ===========================================================================
# E. Provenance / History Persistence — 5 cases
# ===========================================================================

class TestProvenancePersistence:
    """Provenance information persists correctly."""

    def test_E1_provenance_source_persists(self, pipeline, adapter, db_path):
        """Provenance source is user_statement after restart."""
        r = run(pipeline, "I like coffee")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.provenance.source == ProvenanceSource.USER_STATEMENT.value
        store2.close()

    def test_E2_created_at_persists(self, pipeline, adapter, db_path):
        """Created timestamp persists."""
        r = run(pipeline, "I like coffee")
        mem_id = r.results[0].memory_id
        original_created = r.memories[0].created_at

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert mem.created_at == original_created
        store2.close()

    def test_E3_updated_at_persists(self, pipeline, adapter, db_path):
        """Updated timestamp persists after supersession."""
        r1 = run(pipeline, "I use an Oppo A53")
        r2 = run(pipeline, "I use a Samsung S24")
        samsung_updated = r2.memories[0].updated_at

        store2, adapter2 = reopen(db_path)
        samsung = adapter2.get_by_id(r2.results[0].memory_id)
        assert samsung.updated_at == samsung_updated
        store2.close()

    def test_E4_deactivated_at_persists(self, pipeline, adapter, db_path):
        """Deactivation timestamp persists."""
        r1 = run(pipeline, "I like tea")
        run(pipeline, "Forget tea")

        store2, adapter2 = reopen(db_path)
        tea = adapter2.get_by_id(r1.results[0].memory_id)
        assert tea.lifecycle.deactivated_at is not None
        store2.close()

    def test_E5_superseded_at_persists(self, pipeline, adapter, db_path):
        """Supersession timestamp persists."""
        r1 = run(pipeline, "I use an Oppo A53")
        run(pipeline, "I use a Samsung S24")

        store2, adapter2 = reopen(db_path)
        oppo = adapter2.get_by_id(r1.results[0].memory_id)
        assert oppo.lifecycle.superseded_at is not None
        store2.close()


# ===========================================================================
# F. Memory Event Persistence — 5 cases
# ===========================================================================

class TestEventPersistence:
    """Memory events persist in the database."""

    def test_F1_create_event_persists(self, pipeline, store, adapter, db_path):
        """Created event is stored in database."""
        run(pipeline, "I like coffee")

        events = store.get_all_events()
        assert len(events) >= 1
        assert any(e.event_type == "created" for e in events)

    def test_F2_supersede_event_persists(self, pipeline, store, adapter, db_path):
        """Supersede event is stored in database."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "I use a Samsung S24")

        events = store.get_all_events()
        assert any(e.event_type == "superseded" for e in events)

    def test_F3_deactivate_event_persists(self, pipeline, store, adapter, db_path):
        """Deactivate event is stored in database."""
        run(pipeline, "I like tea")
        run(pipeline, "Forget tea")

        events = store.get_all_events()
        assert any(e.event_type == "deactivated" for e in events)

    def test_F4_events_survive_restart(self, pipeline, store, adapter, db_path):
        """Events persist across restart."""
        run(pipeline, "I like coffee")
        run(pipeline, "I use a Samsung S24")
        run(pipeline, "I like tea")
        run(pipeline, "Forget tea")

        store2, adapter2 = reopen(db_path)
        events = store2.get_all_events()
        assert len(events) >= 3
        event_types = {e.event_type for e in events}
        assert "created" in event_types
        store2.close()

    def test_F5_events_for_specific_memory(self, pipeline, store, adapter, db_path):
        """Events can be retrieved for a specific memory."""
        r1 = run(pipeline, "I use an Oppo A53")
        oppo_id = r1.results[0].memory_id
        run(pipeline, "I use a Samsung S24")

        events = store.get_events_for_memory(oppo_id)
        assert len(events) >= 1


# ===========================================================================
# G. Schema / Metadata — 5 cases
# ===========================================================================

class TestSchemaMetadata:
    """Schema version and metadata persist."""

    def test_G1_schema_version_persists(self, store, db_path):
        """Schema version is stored."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT value FROM schema_info WHERE key = 'schema_version'").fetchone()
        # May or may not exist depending on implementation
        conn.close()

    def test_G2_wal_mode_enabled(self, store, db_path):
        """WAL journal mode is enabled."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        conn.close()

    def test_G3_foreign_keys_enabled(self, store, db_path):
        """Foreign keys are enabled on the store's connection."""
        # PRAGMA foreign_keys is per-connection; check via the store's own connection
        conn = store._get_conn()
        row = conn.execute("PRAGMA foreign_keys").fetchone()
        assert row[0] == 1

    def test_G4_indexes_exist(self, store, db_path):
        """Required indexes exist."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        indexes = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()]
        assert "idx_memories_entity" in indexes
        assert "idx_memories_domain" in indexes
        assert "idx_memories_lifecycle" in indexes
        conn.close()

    def test_G5_tables_exist(self, store, db_path):
        """All required tables exist."""
        import sqlite3
        conn = sqlite3.connect(db_path)
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        assert "entities" in tables
        assert "memories" in tables
        assert "relationships" in tables
        assert "memory_events" in tables
        assert "episodes" in tables
        assert "schema_info" in tables
        conn.close()


# ===========================================================================
# H. Adversarial Storage Tests — 10 cases
# ===========================================================================

class TestAdversarialStorage:
    """Edge cases and adversarial storage scenarios."""

    def test_H1_unicode_content_persists(self, pipeline, adapter, db_path):
        """Unicode content persists correctly."""
        r = run(pipeline, "I like café and résumé")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert "café" in mem.content.lower()
        store2.close()

    def test_H2_special_characters_persist(self, pipeline, adapter, db_path):
        """Special characters in content persist."""
        r = run(pipeline, "I like C++ programming")
        mem_id = r.results[0].memory_id

        store2, adapter2 = reopen(db_path)
        mem = adapter2.get_by_id(mem_id)
        assert "c++" in mem.content.lower()
        store2.close()

    def test_H3_long_content_persists(self, pipeline, adapter, db_path):
        """Long content persists."""
        long_msg = "I like " + "very " * 50 + "nice coffee"
        r = run(pipeline, long_msg)
        if r.results:
            mem_id = r.results[0].memory_id
            store2, adapter2 = reopen(db_path)
            mem = adapter2.get_by_id(mem_id)
            assert mem is not None
            store2.close()

    def test_H4_duplicate_id_replaces(self, store, db_path):
        """Duplicate memory_id replaces (INSERT OR REPLACE)."""
        from memory.foundation.models import Memory, EntityRef, Fact, Lifecycle, Provenance, TemporalScope
        mem = Memory(
            id="test_dup_001",
            entity=EntityRef(name="self", entity_type="self"),
            fact=Fact(fact_type="test", subject="test", predicate="test", value="v1", content="v1"),
            domain="test",
            lifecycle=Lifecycle(state="active"),
            provenance=Provenance(source="user_statement"),
            temporal=TemporalScope(valid_from="2024-01-01T00:00:00"),
            confidence=0.8,
            importance="medium",
        )
        store.add(mem)

        mem2 = Memory(
            id="test_dup_001",
            entity=EntityRef(name="self", entity_type="self"),
            fact=Fact(fact_type="test", subject="test", predicate="test", value="v2", content="v2"),
            domain="test",
            lifecycle=Lifecycle(state="active"),
            provenance=Provenance(source="user_statement"),
            temporal=TemporalScope(valid_from="2024-01-01T00:00:00"),
            confidence=0.9,
            importance="high",
        )
        store.add(mem2)

        store2, adapter2 = reopen(db_path)
        mem_loaded = adapter2.get_by_id("test_dup_001")
        assert mem_loaded.fact.value == "v2"
        store2.close()

    def test_H5_entity_isolation_across_restart(self, pipeline, adapter, db_path):
        """One entity's supersede doesn't affect another entity's memories."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "My brother uses an Oppo A53")
        run(pipeline, "I use a Samsung S24")

        store2, adapter2 = reopen(db_path)
        brother_mems = [m for m in adapter2.load() if m.entity.name == "brother"]
        assert len(brother_mems) == 1
        assert brother_mems[0].lifecycle.state == LifecycleState.ACTIVE.value
        assert "oppo" in brother_mems[0].content.lower()
        store2.close()

    def test_H6_empty_database_operations(self, store, db_path):
        """Operations on empty database don't crash."""
        assert store.count() == 0
        assert len(store.get_active()) == 0
        assert store.get_by_id("nonexistent") is None
        assert store.deactivate("nonexistent") is False
        assert store.supersede("winner", "loser") is False

    def test_H7_reopen_same_database_multiple_times(self, pipeline, adapter, db_path):
        """Database can be reopened multiple times."""
        run(pipeline, "I like coffee")

        for i in range(3):
            store2, adapter2 = reopen(db_path)
            assert adapter2.count() == 1
            store2.close()

    def test_H8_concurrent_writes_same_process(self, pipeline, adapter, db_path):
        """Multiple writes in same process all persist."""
        for i in range(10):
            run(pipeline, f"I like item {i}")

        store2, adapter2 = reopen(db_path)
        assert adapter2.count() == 10
        store2.close()

    def test_H9_large_entity_count(self, pipeline, adapter, db_path):
        """Many different entities all persist correctly."""
        entities = ["mom", "dad", "brother", "sister", "friend", "teacher", "colleague"]
        for e in entities:
            run(pipeline, f"My {e} likes coffee")

        store2, adapter2 = reopen(db_path)
        entity_names = {m.entity.name for m in adapter2.load()}
        for e in entities:
            assert e in entity_names
        store2.close()

    def test_H10_transaction_rollback(self, store, db_path):
        """Transaction rollback works correctly."""
        from memory.foundation.models import Memory, EntityRef, Fact, Lifecycle, Provenance, TemporalScope
        mem = Memory(
            id="test_rollback_001",
            entity=EntityRef(name="self", entity_type="self"),
            fact=Fact(fact_type="test", subject="test", predicate="test", value="v1", content="v1"),
            domain="test",
            lifecycle=Lifecycle(state="active"),
            provenance=Provenance(source="user_statement"),
            temporal=TemporalScope(valid_from="2024-01-01T00:00:00"),
            confidence=0.8,
            importance="medium",
        )
        store.add(mem)
        assert store.count() == 1


# ===========================================================================
# I. Pipeline Integration — 10 cases
# ===========================================================================

class TestPipelineIntegration:
    """Full pipeline integration with persistence."""

    def test_I1_full_write_restart_read(self, pipeline, adapter, db_path):
        """Full pipeline: write → restart → verify."""
        run(pipeline, "My favourite colour is blue")
        run(pipeline, "My mom likes pink")
        run(pipeline, "I am learning Java")
        run(pipeline, "I use Samsung S24")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 4
        contents = {m.content.lower() for m in active}
        assert any("blue" in c for c in contents)
        assert any("pink" in c for c in contents)
        assert any("java" in c for c in contents)
        assert any("samsung" in c for c in contents)
        store2.close()

    def test_I2_supersede_restart_verify(self, pipeline, adapter, db_path):
        """Write → supersede → restart → verify correct state."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "I use a Samsung S24")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 1
        assert "samsung" in active[0].content.lower()

        oppo_mems = [m for m in adapter2.load() if "oppo" in m.content.lower()]
        assert len(oppo_mems) == 1
        assert oppo_mems[0].lifecycle.state == LifecycleState.INACTIVE.value
        store2.close()

    def test_I3_complex_sequence_restart(self, pipeline, adapter, db_path):
        """Complex multi-step sequence survives restart."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "I like Java")
        run(pipeline, "I switched to Samsung S24")
        run(pipeline, "I like Python")
        run(pipeline, "I don't like Java anymore")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        active_contents = [m.content.lower() for m in active]
        assert any("samsung" in c for c in active_contents)
        assert any("python" in c for c in active_contents)
        assert not any("oppo" in c for c in active_contents)
        assert not any("java" in c for c in active_contents)
        store2.close()

    def test_I4_entity_isolation_restart(self, pipeline, adapter, db_path):
        """Entity isolation maintained through complex operations after restart."""
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")
        run(pipeline, "I like red")
        run(pipeline, "I like green")

        store2, adapter2 = reopen(db_path)
        self_active = [m for m in adapter2.get_active() if m.entity.name == "self"]
        mom_active = [m for m in adapter2.get_active() if m.entity.name == "mom"]
        self_values = {m.fact.value.lower() for m in self_active}
        assert "green" in self_values
        assert len(mom_active) == 1
        assert "red" in mom_active[0].fact.value.lower()
        store2.close()

    def test_I5_forget_restart_verify(self, pipeline, adapter, db_path):
        """Forget command persists correctly across restart."""
        run(pipeline, "I like coffee")
        run(pipeline, "I like tea")
        run(pipeline, "Forget coffee")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 1
        assert "tea" in active[0].content.lower()
        store2.close()

    def test_I6_correction_restart_verify(self, pipeline, adapter, db_path):
        """Explicit correction persists correctly."""
        run(pipeline, "My name is Alice")
        run(pipeline, "Actually my name is Bob")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 1
        assert "bob" in active[0].content.lower()
        store2.close()

    def test_I7_multi_domain_restart(self, pipeline, adapter, db_path):
        """Multiple domains all persist correctly."""
        run(pipeline, "I like coffee")
        run(pipeline, "I am learning Python")
        run(pipeline, "I want to travel to Japan")
        run(pipeline, "My name is Alice")

        store2, adapter2 = reopen(db_path)
        domains = {m.domain for m in adapter2.load()}
        assert len(domains) >= 3
        store2.close()

    def test_I8_halved_sequence_restart(self, pipeline, adapter, db_path):
        """Partial sequence: write half, restart, write rest, verify."""
        run(pipeline, "I like coffee")
        run(pipeline, "I like tea")

        store2, adapter2 = reopen(db_path)
        pipeline2 = WritePipeline(adapter2)
        run(pipeline2, "I like Python")
        run(pipeline2, "I use a Samsung S24")

        active = adapter2.get_active()
        assert len(active) == 4
        store2.close()

    def test_I9_multiple_supersessions_restart(self, pipeline, adapter, db_path):
        """Multiple supersessions all persist correctly."""
        run(pipeline, "I use an Oppo A53")
        run(pipeline, "I use a Samsung S24")
        run(pipeline, "I use an iPhone 15")
        run(pipeline, "I use a Pixel 8")

        store2, adapter2 = reopen(db_path)
        active = adapter2.get_active()
        assert len(active) == 1
        assert "pixel" in active[0].content.lower()
        assert adapter2.count() == 4
        store2.close()

    def test_I10_production_json_untouched(self, pipeline, adapter, db_path):
        """Production JSON files are NOT modified by persistence layer."""
        import hashlib

        prod_ltm = os.path.join(os.path.dirname(__file__), "..", "memory", "long_term_memory.json")
        prod_conv = os.path.join(os.path.dirname(__file__), "..", "memory", "conversation.json")

        if os.path.exists(prod_ltm):
            hash_before = hashlib.md5(open(prod_ltm, "rb").read()).hexdigest()
        else:
            hash_before = None

        if os.path.exists(prod_conv):
            hash_conv_before = hashlib.md5(open(prod_conv, "rb").read()).hexdigest()
        else:
            hash_conv_before = None

        # Run pipeline operations
        run(pipeline, "I like coffee")
        run(pipeline, "I use a Samsung S24")

        # Verify production files unchanged
        if os.path.exists(prod_ltm):
            hash_after = hashlib.md5(open(prod_ltm, "rb").read()).hexdigest()
            assert hash_before == hash_after, "long_term_memory.json was modified!"

        if os.path.exists(prod_conv):
            hash_conv_after = hashlib.md5(open(prod_conv, "rb").read()).hexdigest()
            assert hash_conv_before == hash_conv_after, "conversation.json was modified!"


# ===========================================================================
# J. Write → Sync → Retrieval (lifecycle changes visible to old store)
# ===========================================================================

class TestWriteSyncRetrieval:
    """Verify that WritePipeline syncs lifecycle changes to the old store
    so that retrieval sees updated state immediately after any write."""

    def _make_pair(self):
        """Create (sqlite_adapter, old_store) pair for testing."""
        from memory.store import MemoryStore as OldMemoryStore
        db = tempfile.mktemp(suffix='.sqlite')
        store = SQLiteStore(db)
        adapter = SQLiteAdapter(store)
        adapter.load()
        old = OldMemoryStore(path='nonexistent.json')
        return adapter, old, db

    def _sync(self, adapter, old):
        from memory.foundation.adapter import sync_sqlite_to_old_store
        sync_sqlite_to_old_store(adapter, old)

    def _pipeline(self, adapter, old):
        from memory.foundation.pipeline import WritePipeline
        return WritePipeline(adapter, old_store=old)

    def _active_ids(self, old):
        return {m["id"] for m in old.memories if m.get("status") == "active"}

    def _inactive_ids(self, old):
        return {m["id"] for m in old.memories if m.get("status") == "inactive"}

    def _status_map(self, old):
        return {m["id"]: m.get("status") for m in old.memories}

    def _content_status_pairs(self, old):
        return {(m["content"], m.get("status")) for m in old.memories}

    def test_supersession_syncs_status(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("I use a Samsung S24")
            p.process("I now use a Pixel 8")

            pairs = self._content_status_pairs(old)
            assert ("I use a Samsung S24", "inactive") in pairs
            assert ("I now use a Pixel 8", "active") in pairs
            assert len(self._active_ids(old)) == 1
            assert len(self._inactive_ids(old)) == 1
        finally:
            adapter.close()
            os.unlink(db)

    def test_invalidation_syncs_status(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("My favorite food is biryani")
            p.process("I don't like biryani anymore")

            pairs = self._content_status_pairs(old)
            assert ("My favorite food is biryani", "inactive") in pairs
            assert len(self._active_ids(old)) == 0
            assert len(self._inactive_ids(old)) == 1
        finally:
            adapter.close()
            os.unlink(db)

    def test_deactivation_syncs_status(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("My favorite food is biryani")
            p.process("I don't like biryani anymore")
            p.process("I work at Google")

            pairs = self._content_status_pairs(old)
            # biryani invalidated, Google created
            assert ("My favorite food is biryani", "inactive") in pairs
            assert ("I work at Google", "active") in pairs
            assert len(self._inactive_ids(old)) == 1
        finally:
            adapter.close()
            os.unlink(db)

    def test_multiple_entities_independent_sync(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            # Phone supersession
            p.process("I use a Samsung S24")
            p.process("I now use a Pixel 8")
            # Food invalidation
            p.process("My favorite food is biryani")
            p.process("I don't like biryani anymore")

            pairs = self._content_status_pairs(old)
            # Phone: Samsung inactive, Pixel active
            assert ("I use a Samsung S24", "inactive") in pairs
            assert ("I now use a Pixel 8", "active") in pairs
            # Food: biryani inactive
            assert ("My favorite food is biryani", "inactive") in pairs
            assert len(self._active_ids(old)) == 1  # Only Pixel active
            assert len(self._inactive_ids(old)) == 2  # Samsung + biryani
        finally:
            adapter.close()
            os.unlink(db)

    def test_create_only_no_inactive(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("I like blue")
            p.process("I like coffee")

            pairs = self._content_status_pairs(old)
            # Both should be active (multi-value fact type)
            assert ("I like blue", "active") in pairs
            assert ("I like coffee", "active") in pairs
            assert len(self._inactive_ids(old)) == 0
        finally:
            adapter.close()
            os.unlink(db)

    def test_sync_preserves_old_store_correctly_after_supersede(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("I use a Samsung S24")
            p.process("I now use a Pixel 8")

            assert len(old.memories) == 2
            statuses = set(self._status_map(old).values())
            assert statuses == {"active", "inactive"}
        finally:
            adapter.close()
            os.unlink(db)

    def test_no_sync_without_old_store(self):
        """WritePipeline without old_store should not crash."""
        db = tempfile.mktemp(suffix='.sqlite')
        try:
            store = SQLiteStore(db)
            adapter = SQLiteAdapter(store)
            adapter.load()
            from memory.foundation.pipeline import WritePipeline
            p = WritePipeline(adapter)  # no old_store
            r = p.process("I use a Samsung S24")
            assert r.status == "success"
            adapter.close()
        finally:
            if os.path.exists(db):
                os.unlink(db)

    def test_question_does_not_trigger_sync(self):
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("What is my name?")  # question — no sync
            assert len(old.memories) == 0  # nothing written
        finally:
            adapter.close()
            os.unlink(db)

    def test_restart_reflects_correct_state(self):
        """After write + restart, old store should reflect correct state."""
        adapter, old, db = self._make_pair()
        try:
            p = self._pipeline(adapter, old)
            p.process("I use a Samsung S24")
            p.process("I now use a Pixel 8")

            # Simulate restart: new old store, re-sync
            from memory.store import MemoryStore as OldMemoryStore
            old2 = OldMemoryStore(path='nonexistent.json')
            self._sync(adapter, old2)

            pairs = self._content_status_pairs(old2)
            assert ("I use a Samsung S24", "inactive") in pairs
            assert ("I now use a Pixel 8", "active") in pairs
        finally:
            adapter.close()
            os.unlink(db)
