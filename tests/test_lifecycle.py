"""Lifecycle tests — verifies intelligent memory lifecycle behavior.

Tests ADD, UPDATE, MERGE, IGNORE, INVALIDATE, SUPERSEDE operations
with entity isolation, temporal awareness, and contradiction handling.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from memory.foundation import WritePipeline, InMemoryStore, validate_memory
from memory.foundation.models import Memory, LifecycleState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store() -> InMemoryStore:
    return InMemoryStore()


@pytest.fixture
def pipeline(store: InMemoryStore) -> WritePipeline:
    return WritePipeline(store)


def run(pipeline: WritePipeline, msg: str):
    return pipeline.process(msg)


def active_memories(store: InMemoryStore) -> list[Memory]:
    return [m for m in store.get_active()]


def all_memories(store: InMemoryStore) -> list[Memory]:
    return store.load()


# ===========================================================================
# ADD
# ===========================================================================

class TestADD:
    def test_new_fact_created(self, pipeline, store):
        r = run(pipeline, "I like coffee")
        assert r.status == "success"
        assert len(r.memories) == 1
        assert r.results[0].action_taken == "created"
        assert len(active_memories(store)) == 1

    def test_different_entities_independent(self, pipeline, store):
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")
        self_mems = [m for m in active_memories(store) if m.entity.name == "self"]
        mom_mems = [m for m in active_memories(store) if m.entity.name == "mom"]
        assert len(self_mems) == 1
        assert len(mom_mems) == 1

    def test_different_domains_separate(self, pipeline, store):
        run(pipeline, "I like coffee")
        run(pipeline, "I am learning Python")
        domains = {m.domain for m in active_memories(store)}
        assert "interests" in domains
        assert "learning" in domains
        assert len(active_memories(store)) == 2


# ===========================================================================
# UPDATE / SUPERSEDE
# ===========================================================================

class TestSUPERSEDE:
    def test_value_change_supersedes(self, pipeline, store):
        run(pipeline, "I use an Oppo A53")
        r = run(pipeline, "I use a Samsung S24")
        assert r.results[0].action_taken == "superseded"
        active = active_memories(store)
        assert len(active) == 1
        assert "samsung" in active[0].content.lower()

    def test_correction_supersedes(self, pipeline, store):
        run(pipeline, "My name is Alice")
        r = run(pipeline, "Actually my name is Bob")
        assert r.results[0].action_taken == "superseded"
        active = active_memories(store)
        assert len(active) == 1
        assert "bob" in active[0].content.lower()

    def test_history_preserved(self, pipeline, store):
        r1 = run(pipeline, "I use an Oppo A53")
        r2 = run(pipeline, "I switched to Samsung S24")
        # Old memory should be inactive with superseded_by link
        old_id = r1.results[0].memory_id
        old = store.get_by_id(old_id)
        assert old is not None
        assert old.lifecycle.state == LifecycleState.INACTIVE.value
        assert old.lifecycle.superseded_by == r2.results[0].memory_id

    def test_same_value_ignored(self, pipeline, store):
        run(pipeline, "I like coffee")
        r = run(pipeline, "I like coffee")
        assert r.results[0].action_taken == "unchanged"
        assert len(active_memories(store)) == 1


# ===========================================================================
# MERGE (compatible addition)
# ===========================================================================

class TestMERGE:
    def test_compatible_attribute_added(self, pipeline, store):
        run(pipeline, "I use a Samsung S24")
        r = run(pipeline, "My Samsung S24 is black")
        # Should create a new compatible memory, not supersede
        assert r.results[0].action_taken in ("created", "merged")
        active = active_memories(store)
        # Both should be active
        assert len(active) >= 2

    def test_same_entity_different_attribute(self, pipeline, store):
        run(pipeline, "I am learning Python")
        r = run(pipeline, "I use VS Code for Python")
        assert r.results[0].action_taken in ("created", "merged")
        active = active_memories(store)
        assert len(active) >= 2


# ===========================================================================
# IGNORE (duplicate)
# ===========================================================================

class TestIGNORE:
    def test_exact_duplicate_ignored(self, pipeline, store):
        run(pipeline, "I like coffee")
        r = run(pipeline, "I like coffee")
        assert r.results[0].action_taken == "unchanged"
        assert len(active_memories(store)) == 1

    def test_paraphrase_duplicate_ignored(self, pipeline, store):
        run(pipeline, "I like coffee")
        r = run(pipeline, "coffee is something i like")
        # Same value "coffee" → should be ignored as duplicate
        assert r.results[0].action_taken == "unchanged"
        assert len(active_memories(store)) == 1

    def test_question_not_stored(self, pipeline, store):
        r = run(pipeline, "What is my name?")
        assert r.status == "question"
        assert len(active_memories(store)) == 0


# ===========================================================================
# INVALIDATE
# ===========================================================================

class TestINVALIDATE:
    def test_negation_invalidates(self, pipeline, store):
        r1 = run(pipeline, "I like coffee")
        old_id = r1.results[0].memory_id
        r2 = run(pipeline, "I don't like coffee anymore")
        assert r2.results[0].action_taken in ("deactivated", "superseded", "invalidated")
        # Old memory should be inactive
        old = store.get_by_id(old_id)
        assert old is not None
        assert old.lifecycle.state == LifecycleState.INACTIVE.value

    def test_forget_invalidates(self, pipeline, store):
        run(pipeline, "I like tea")
        r = run(pipeline, "Forget tea")
        assert r.results[0].action_taken == "deactivated"
        assert len(active_memories(store)) == 0


# ===========================================================================
# CONTRADICTION HANDLING
# ===========================================================================

class TestContradiction:
    def test_same_subject_different_value(self, pipeline, store):
        run(pipeline, "I like Java")
        r = run(pipeline, "I like Python")
        # Multi-value: both preferences are valid, both created
        assert r.results[0].action_taken == "created"
        active = active_memories(store)
        assert len(active) == 2
        values = {m.fact.value.lower() for m in active}
        assert "java" in values
        assert "python" in values

    def test_compatible_info_not_contradiction(self, pipeline, store):
        run(pipeline, "I use a Samsung S24")
        r = run(pipeline, "My Samsung S24 is black")
        # Should NOT supersede — compatible addition
        assert r.results[0].action_taken != "superseded"
        active = active_memories(store)
        assert len(active) >= 2

    def test_negation_overwrites(self, pipeline, store):
        r1 = run(pipeline, "I like Java")
        old_id = r1.results[0].memory_id
        r = run(pipeline, "I don't like Java anymore")
        assert r.results[0].action_taken in ("deactivated", "superseded", "invalidated")
        # Old memory should be inactive
        old = store.get_by_id(old_id)
        assert old is not None
        assert old.lifecycle.state == LifecycleState.INACTIVE.value


# ===========================================================================
# ENTITY ISOLATION
# ===========================================================================

class TestEntityIsolation:
    def test_self_vs_mom_independent(self, pipeline, store):
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")
        # Change user's preference — mom's should be unaffected
        r = run(pipeline, "I like green")
        # Multi-value: user can have multiple preferences
        self_mems = [m for m in active_memories(store) if m.entity.name == "self"]
        mom_mems = [m for m in active_memories(store) if m.entity.name == "mom"]
        assert len(self_mems) >= 1
        self_values = {m.fact.value.lower() for m in self_mems}
        assert "green" in self_values
        assert len(mom_mems) == 1
        assert "red" in mom_mems[0].content.lower()

    def test_third_party_isolated(self, pipeline, store):
        run(pipeline, "I like coffee")
        run(pipeline, "My friend Jake likes tea")
        run(pipeline, "I like tea")
        # User now likes tea — Jake's tea preference should be unaffected
        self_tea = [m for m in active_memories(store)
                    if m.entity.name == "self" and "tea" in m.content.lower()]
        jake_tea = [m for m in active_memories(store)
                    if m.entity.name == "jake" and "tea" in m.content.lower()]
        assert len(self_tea) == 1
        assert len(jake_tea) == 1

    def test_multiple_entities_independent(self, pipeline, store):
        run(pipeline, "I like blue")
        run(pipeline, "My mom likes red")
        run(pipeline, "My friend Jake likes green")
        run(pipeline, "I like red")
        # User added red — should not affect mom's red preference
        self_mems = [m for m in active_memories(store) if m.entity.name == "self"]
        mom_mems = [m for m in active_memories(store) if m.entity.name == "mom"]
        jake_mems = [m for m in active_memories(store) if m.entity.name == "jake"]
        self_values = {m.fact.value.lower() for m in self_mems}
        assert "red" in self_values
        assert len(mom_mems) == 1
        assert len(jake_mems) == 1


# ===========================================================================
# TEMPORAL BEHAVIOR
# ===========================================================================

class TestTemporal:
    def test_historical_not_overwrite_current(self, pipeline, store):
        run(pipeline, "I use a Samsung S24")
        r = run(pipeline, "I used to have an Oppo A53")
        # Historical statement should not supersede current phone
        samsung = [m for m in active_memories(store) if "samsung" in m.content.lower()]
        assert len(samsung) == 1

    def test_temporary_low_confidence(self, pipeline, store):
        r = run(pipeline, "I have a dentist appointment tomorrow")
        assert r.memories[0].temporal.is_persistent is False

    def test_persistent_by_default(self, pipeline, store):
        r = run(pipeline, "I like coffee")
        assert r.memories[0].temporal.is_persistent is True


# ===========================================================================
# UNCERTAINTY
# ===========================================================================

class TestUncertainty:
    def test_uncertain_low_confidence(self, pipeline, store):
        r = run(pipeline, "I might learn Rust someday")
        assert r.memories[0].confidence <= 0.5

    def test_certain_high_confidence(self, pipeline, store):
        r = run(pipeline, "I like coffee")
        assert r.memories[0].confidence >= 0.7

    def test_uncertain_not_supersede_certain(self, pipeline, store):
        run(pipeline, "I like coffee")
        r = run(pipeline, "I might like tea")
        # Uncertain should not supersede certain
        assert r.results[0].action_taken != "superseded"
        coffee = [m for m in active_memories(store) if "coffee" in m.content.lower()]
        assert len(coffee) == 1


# ===========================================================================
# MULTI-TURN SEQUENCES
# ===========================================================================

class TestMultiTurn:
    def test_phone_sequence(self, pipeline, store):
        # Turn 1
        r1 = run(pipeline, "I use an Oppo A53")
        assert r1.results[0].action_taken == "created"
        assert len(active_memories(store)) == 1

        # Turn 2
        r2 = run(pipeline, "I switched to Samsung S24")
        assert r2.results[0].action_taken == "superseded"
        active = active_memories(store)
        assert len(active) == 1
        assert "samsung" in active[0].content.lower()

        # Turn 3 — Oppo should be inactive
        oppo = [m for m in all_memories(store) if "oppo" in m.content.lower()]
        assert len(oppo) == 1
        assert oppo[0].lifecycle.state == LifecycleState.INACTIVE.value

    def test_preference_sequence(self, pipeline, store):
        r1 = run(pipeline, "I like Java")
        old_id = r1.results[0].memory_id
        r = run(pipeline, "I don't like Java anymore")
        assert r.results[0].action_taken in ("deactivated", "superseded", "invalidated")
        # Old memory should be inactive
        old = store.get_by_id(old_id)
        assert old is not None
        assert old.lifecycle.state == LifecycleState.INACTIVE.value

    def test_entity_isolation_sequence(self, pipeline, store):
        run(pipeline, "My mom likes pink")
        run(pipeline, "I like blue")
        r = run(pipeline, "I like pink")
        # User added pink — mom's should be unaffected
        self_mems = [m for m in active_memories(store) if m.entity.name == "self"]
        mom_mems = [m for m in active_memories(store) if m.entity.name == "mom"]
        self_values = {m.fact.value.lower() for m in self_mems}
        assert "pink" in self_values
        assert len(mom_mems) == 1
        assert "pink" in mom_mems[0].content.lower()


# ===========================================================================
# VALIDATION
# ===========================================================================

class TestValidation:
    def test_all_created_memories_valid(self, pipeline):
        messages = [
            "I like coffee",
            "My name is Alice",
            "I am learning Python",
            "I want to build a website",
            "I just got a new phone",
        ]
        for msg in messages:
            r = run(pipeline, msg)
            for m in r.memories:
                v = validate_memory(m)
                assert v.is_valid, f"Memory from '{msg}' invalid: {v.errors}"
