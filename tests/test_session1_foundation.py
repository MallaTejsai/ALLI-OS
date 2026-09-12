"""Comprehensive tests for ALLI v0.6 Session 1 Foundation.

Covers:
- MemoryType enum (episodic, semantic, procedural, temporary)
- Serialization/deserialization (to_dict/from_dict) for all models
- Round-trip equality
- Invalid data handling
- Missing required fields
- Single-value vs multi-value fact representation
- Entity model with aliases and references
- Validation of memory_type field
- Isolation from production memory
"""

import pytest
import json
from datetime import datetime, timezone, timedelta

from src.memory.foundation.models import (
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
from src.memory.foundation.storage import InMemoryStore
from src.memory.foundation.validation import (
    ValidationResult,
    validate_entity_ref,
    validate_event,
    validate_fact,
    validate_lifecycle,
    validate_memory,
    validate_provenance,
    validate_temporal,
)
from src.memory.foundation.ids import (
    generate_entity_id,
    generate_event_id,
    generate_memory_id,
    generate_relationship_id,
    generate_episode_id,
    reset_counters,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _utc_past_iso(days: int = 1) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).replace(
        microsecond=0
    ).isoformat()


def _utc_future_iso(days: int = 1) -> str:
    return (datetime.now(timezone.utc) + timedelta(days=days)).replace(
        microsecond=0
    ).isoformat()


def _make_entity_ref(name: str = "self", entity_type: str = "self") -> EntityRef:
    return EntityRef(name=name, entity_type=entity_type)


def _make_fact(
    fact_type: str = "learning",
    subject: str = "python",
    content: str = "I am learning Python",
) -> Fact:
    return Fact(
        fact_type=fact_type,
        subject=subject,
        predicate="is_learning",
        value="Python",
        content=content,
    )


def _make_temporal(
    valid_from: str | None = None,
    valid_until: str | None = None,
    is_persistent: bool = True,
) -> TemporalScope:
    return TemporalScope(
        valid_from=valid_from or _utc_now_iso(),
        valid_until=valid_until,
        is_persistent=is_persistent,
    )


def _make_provenance(
    source: str = "user_statement",
) -> Provenance:
    return Provenance(source=source)


def _make_lifecycle(
    state: str = "active",
) -> Lifecycle:
    return Lifecycle(state=state)


def _make_memory(
    id: str = "memory_001",
    entity_name: str = "self",
    fact_type: str = "learning",
    subject: str = "python",
    content: str = "I am learning Python",
    domain: str = "learning",
    lifecycle_state: str = "active",
    temporal: TemporalScope | None = None,
    confidence: float = 1.0,
    importance: str = "medium",
    memory_type: str = "semantic",
) -> Memory:
    return Memory(
        id=id,
        entity=_make_entity_ref(entity_name),
        fact=_make_fact(fact_type=fact_type, subject=subject, content=content),
        domain=domain,
        lifecycle=_make_lifecycle(lifecycle_state),
        provenance=_make_provenance(),
        temporal=temporal or _make_temporal(),
        confidence=confidence,
        importance=importance,
        memory_type=memory_type,
    )


# ---------------------------------------------------------------------------
# MemoryType enum tests
# ---------------------------------------------------------------------------


class TestMemoryType:
    def test_episodic(self) -> None:
        assert MemoryType.EPISODIC.value == "episodic"

    def test_semantic(self) -> None:
        assert MemoryType.SEMANTIC.value == "semantic"

    def test_procedural(self) -> None:
        assert MemoryType.PROCEDURAL.value == "procedural"

    def test_temporary(self) -> None:
        assert MemoryType.TEMPORARY.value == "temporary"

    def test_all_types_are_strings(self) -> None:
        for mt in MemoryType:
            assert isinstance(mt.value, str)

    def test_memory_type_in_memory_model(self) -> None:
        m = _make_memory(memory_type="episodic")
        assert m.memory_type == "episodic"

    def test_memory_type_default_is_semantic(self) -> None:
        m = _make_memory()
        assert m.memory_type == MemoryType.SEMANTIC.value

    def test_invalid_memory_type(self) -> None:
        m = _make_memory(memory_type="invalid")
        result = validate_memory(m)
        assert not result.is_valid
        assert any("memory_type" in e for e in result.errors)


# ---------------------------------------------------------------------------
# ImportanceLevel enum tests
# ---------------------------------------------------------------------------


class TestImportanceLevel:
    def test_critical(self) -> None:
        assert ImportanceLevel.CRITICAL.value == "critical"

    def test_high(self) -> None:
        assert ImportanceLevel.HIGH.value == "high"

    def test_medium(self) -> None:
        assert ImportanceLevel.MEDIUM.value == "medium"

    def test_low(self) -> None:
        assert ImportanceLevel.LOW.value == "low"

    def test_all_levels_are_strings(self) -> None:
        for level in ImportanceLevel:
            assert isinstance(level.value, str)

    def test_importance_in_memory_model(self) -> None:
        m = _make_memory(importance="high")
        assert m.importance == "high"

    def test_importance_default_is_medium(self) -> None:
        m = _make_memory()
        assert m.importance == ImportanceLevel.MEDIUM.value

    def test_invalid_importance(self) -> None:
        m = _make_memory(importance="urgent")
        result = validate_memory(m)
        assert not result.is_valid
        assert any("importance" in e for e in result.errors)

    def test_importance_serialization_round_trip(self) -> None:
        for level in ImportanceLevel:
            m = _make_memory(importance=level.value)
            d = m.to_dict()
            m2 = Memory.from_dict(d)
            assert m2.importance == level.value

    def test_importance_all_valid_values(self) -> None:
        for level in ImportanceLevel:
            m = _make_memory(importance=level.value)
            result = validate_memory(m)
            assert result.is_valid, f"importance={level.value} should be valid"


# ---------------------------------------------------------------------------
# Serialization/deserialization tests
# ---------------------------------------------------------------------------


class TestEntityRefSerialization:
    def test_to_dict(self) -> None:
        ref = EntityRef(name="Alice", entity_type="person", relation="friend", entity_id="e001")
        d = ref.to_dict()
        assert d["name"] == "Alice"
        assert d["entity_type"] == "person"
        assert d["relation"] == "friend"
        assert d["entity_id"] == "e001"

    def test_from_dict(self) -> None:
        d = {"name": "Bob", "entity_type": "person", "relation": None, "entity_id": None}
        ref = EntityRef.from_dict(d)
        assert ref.name == "Bob"
        assert ref.entity_type == "person"

    def test_round_trip(self) -> None:
        ref = EntityRef(name="Alice", entity_type="person")
        d = ref.to_dict()
        ref2 = EntityRef.from_dict(d)
        assert ref2.name == ref.name
        assert ref2.entity_type == ref.entity_type

    def test_from_dict_missing_optional_fields(self) -> None:
        d = {"name": "Charlie"}
        ref = EntityRef.from_dict(d)
        assert ref.name == "Charlie"
        assert ref.entity_type == EntityType.SELF.value
        assert ref.relation is None
        assert ref.entity_id is None


class TestFactSerialization:
    def test_to_dict(self) -> None:
        fact = Fact(fact_type="preference", subject="coffee", predicate="likes", value="coffee", content="I like coffee", polarity="positive")
        d = fact.to_dict()
        assert d["fact_type"] == "preference"
        assert d["polarity"] == "positive"

    def test_from_dict(self) -> None:
        d = {"fact_type": "learning", "subject": "python", "predicate": "is_learning", "value": "Python", "content": "I am learning Python", "polarity": None}
        fact = Fact.from_dict(d)
        assert fact.fact_type == "learning"
        assert fact.polarity is None

    def test_round_trip(self) -> None:
        fact = _make_fact()
        d = fact.to_dict()
        fact2 = Fact.from_dict(d)
        assert fact2.fact_type == fact.fact_type
        assert fact2.content == fact.content


class TestTemporalScopeSerialization:
    def test_to_dict(self) -> None:
        ts = TemporalScope(valid_from="2025-01-01T00:00:00+00:00", valid_until="2025-12-31T23:59:59+00:00", is_persistent=False, temporal_hint="tomorrow")
        d = ts.to_dict()
        assert d["valid_from"] == "2025-01-01T00:00:00+00:00"
        assert d["valid_until"] == "2025-12-31T23:59:59+00:00"
        assert d["is_persistent"] is False
        assert d["temporal_hint"] == "tomorrow"

    def test_from_dict(self) -> None:
        d = {"valid_from": "2025-01-01T00:00:00+00:00", "valid_until": None, "is_persistent": True, "temporal_hint": None}
        ts = TemporalScope.from_dict(d)
        assert ts.valid_from == "2025-01-01T00:00:00+00:00"
        assert ts.is_persistent is True

    def test_round_trip(self) -> None:
        ts = _make_temporal()
        d = ts.to_dict()
        ts2 = TemporalScope.from_dict(d)
        assert ts2.valid_from == ts.valid_from
        assert ts2.is_persistent == ts.is_persistent


class TestProvenanceSerialization:
    def test_to_dict(self) -> None:
        p = Provenance(source="inferred", extraction_method="llm", reason="test")
        d = p.to_dict()
        assert d["source"] == "inferred"
        assert d["extraction_method"] == "llm"

    def test_from_dict(self) -> None:
        d = {"source": "user_statement", "extraction_method": None, "message_id": None, "episode_id": None, "reason": None}
        p = Provenance.from_dict(d)
        assert p.source == "user_statement"

    def test_round_trip(self) -> None:
        p = _make_provenance("inferred")
        d = p.to_dict()
        p2 = Provenance.from_dict(d)
        assert p2.source == p.source


class TestLifecycleSerialization:
    def test_to_dict(self) -> None:
        lc = Lifecycle(state="active", superseded_by="m002", deactivated_at="2025-01-01T00:00:00+00:00")
        d = lc.to_dict()
        assert d["state"] == "active"
        assert d["superseded_by"] == "m002"

    def test_from_dict(self) -> None:
        d = {"state": "inactive", "superseded_by": "m001", "superseded_at": None, "deactivated_at": None, "expired_at": None}
        lc = Lifecycle.from_dict(d)
        assert lc.state == "inactive"
        assert lc.superseded_by == "m001"

    def test_round_trip(self) -> None:
        lc = _make_lifecycle()
        d = lc.to_dict()
        lc2 = Lifecycle.from_dict(d)
        assert lc2.state == lc.state


class TestMemorySerialization:
    def test_to_dict(self) -> None:
        m = _make_memory(memory_type="episodic")
        d = m.to_dict()
        assert d["id"] == "memory_001"
        assert d["memory_type"] == "episodic"
        assert "entity" in d
        assert "fact" in d
        assert "lifecycle" in d
        assert "provenance" in d
        assert "temporal" in d

    def test_from_dict(self) -> None:
        d = {
            "id": "mem_001",
            "entity": {"name": "self", "entity_type": "self", "relation": None, "entity_id": None},
            "fact": {"fact_type": "learning", "subject": "python", "predicate": "is_learning", "value": "Python", "content": "I am learning Python", "polarity": None},
            "domain": "learning",
            "lifecycle": {"state": "active", "superseded_by": None, "superseded_at": None, "deactivated_at": None, "expired_at": None},
            "provenance": {"source": "user_statement", "extraction_method": None, "message_id": None, "episode_id": None, "reason": None},
            "temporal": {"valid_from": "2025-01-01T00:00:00+00:00", "valid_until": None, "is_persistent": True, "temporal_hint": None},
            "confidence": 0.9,
            "importance": "high",
            "memory_type": "procedural",
        }
        m = Memory.from_dict(d)
        assert m.id == "mem_001"
        assert m.memory_type == "procedural"
        assert m.confidence == 0.9

    def test_round_trip(self) -> None:
        m = _make_memory(memory_type="temporary")
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        assert m2.id == m.id
        assert m2.memory_type == m.memory_type
        assert m2.entity.name == m.entity.name
        assert m2.fact.content == m.fact.content

    def test_round_trip_json(self) -> None:
        m = _make_memory()
        d = m.to_dict()
        json_str = json.dumps(d)
        d2 = json.loads(json_str)
        m2 = Memory.from_dict(d2)
        assert m2.id == m.id
        assert m2.memory_type == m.memory_type


class TestMemoryEventSerialization:
    def test_to_dict(self) -> None:
        e = MemoryEvent(id="evt_001", event_type="created", memory_id="mem_001", details={"key": "value"})
        d = e.to_dict()
        assert d["id"] == "evt_001"
        assert d["details"] == {"key": "value"}

    def test_from_dict(self) -> None:
        d = {"id": "evt_002", "event_type": "updated", "memory_id": "mem_002", "timestamp": "2025-01-01T00:00:00+00:00", "trigger": "user_statement", "details": {}}
        e = MemoryEvent.from_dict(d)
        assert e.id == "evt_002"
        assert e.event_type == "updated"

    def test_round_trip(self) -> None:
        e = MemoryEvent(id="evt_001", event_type="created", memory_id="mem_001")
        d = e.to_dict()
        e2 = MemoryEvent.from_dict(d)
        assert e2.id == e.id
        assert e2.event_type == e.event_type


class TestEpisodeSerialization:
    def test_to_dict(self) -> None:
        ep = Episode(id="ep_001", user_message="I like Python", entities_involved=[EntityRef(name="self")])
        d = ep.to_dict()
        assert d["id"] == "ep_001"
        assert len(d["entities_involved"]) == 1

    def test_from_dict(self) -> None:
        d = {
            "id": "ep_002",
            "user_message": "Test message",
            "timestamp": "2025-01-01T00:00:00+00:00",
            "entities_involved": [{"name": "self", "entity_type": "self", "relation": None, "entity_id": None}],
            "extracted_facts": [],
            "extracted_relationships": [],
            "memory_events": [],
            "provenance": {"source": "user_statement", "extraction_method": None, "message_id": None, "episode_id": None, "reason": None},
            "metadata": {},
        }
        ep = Episode.from_dict(d)
        assert ep.id == "ep_002"
        assert len(ep.entities_involved) == 1

    def test_round_trip(self) -> None:
        ep = Episode(id="ep_001", user_message="Test")
        d = ep.to_dict()
        ep2 = Episode.from_dict(d)
        assert ep2.id == ep.id
        assert ep2.user_message == ep.user_message


class TestRelationshipSerialization:
    def test_to_dict(self) -> None:
        rel = Relationship(
            id="rel_001",
            entity_a=EntityRef(name="Alice", entity_type="person"),
            entity_b=EntityRef(name="Bob", entity_type="person"),
            relation_type="friend",
        )
        d = rel.to_dict()
        assert d["id"] == "rel_001"
        assert d["relation_type"] == "friend"
        assert "entity_a" in d
        assert "entity_b" in d

    def test_from_dict(self) -> None:
        d = {
            "id": "rel_002",
            "entity_a": {"name": "Alice", "entity_type": "person", "relation": None, "entity_id": None},
            "entity_b": {"name": "Bob", "entity_type": "person", "relation": None, "entity_id": None},
            "relation_type": "sibling",
            "confidence": 0.9,
            "provenance": {"source": "user_statement", "extraction_method": None, "message_id": None, "episode_id": None, "reason": None},
            "temporal": {"valid_from": "2025-01-01T00:00:00+00:00", "valid_until": None, "is_persistent": True, "temporal_hint": None},
            "lifecycle": {"state": "active", "superseded_by": None, "superseded_at": None, "deactivated_at": None, "expired_at": None},
            "created_at": "2025-01-01T00:00:00+00:00",
            "updated_at": "2025-01-01T00:00:00+00:00",
        }
        rel = Relationship.from_dict(d)
        assert rel.id == "rel_002"
        assert rel.relation_type == "sibling"

    def test_round_trip(self) -> None:
        rel = Relationship(
            id="rel_001",
            entity_a=EntityRef(name="Alice"),
            entity_b=EntityRef(name="Bob"),
            relation_type="friend",
        )
        d = rel.to_dict()
        rel2 = Relationship.from_dict(d)
        assert rel2.id == rel.id
        assert rel2.entity_a.name == rel.entity_a.name


# ---------------------------------------------------------------------------
# Single-value vs multi-value fact representation tests
# ---------------------------------------------------------------------------


class TestFactSemantics:
    def test_single_current_value_superseded(self) -> None:
        """Phone: Samsung -> Oppo must support historical/superseded state."""
        old_phone = _make_memory(
            id="phone_old",
            fact_type="possession",
            subject="phone",
            content="I use Samsung",
            lifecycle_state="active",
        )
        new_phone = _make_memory(
            id="phone_new",
            fact_type="possession",
            subject="phone",
            content="I use Oppo",
            lifecycle_state="active",
        )
        old_phone.lifecycle.supersede("phone_new")
        assert old_phone.lifecycle.state == LifecycleState.INACTIVE.value
        assert old_phone.lifecycle.superseded_by == "phone_new"
        assert new_phone.lifecycle.state == LifecycleState.ACTIVE.value

    def test_multi_value_coexistence(self) -> None:
        """Likes: Python + Java must allow both to coexist."""
        like_python = _make_memory(
            id="like_python",
            fact_type="preference",
            subject="languages",
            content="I like Python",
            lifecycle_state="active",
        )
        like_java = _make_memory(
            id="like_java",
            fact_type="preference",
            subject="languages",
            content="I like Java",
            lifecycle_state="active",
        )
        store = InMemoryStore()
        store.add(like_python)
        store.add(like_java)
        assert store.count() == 2
        active = store.get_active()
        assert len(active) == 2


# ---------------------------------------------------------------------------
# Entity model tests
# ---------------------------------------------------------------------------


class TestEntityModel:
    def test_entity_with_aliases(self) -> None:
        e = Entity(id="e001", name="Alice", aliases=["Ali", "A"])
        assert e.matches_name("Alice")
        assert e.matches_name("alice")
        assert e.matches_name("Ali")
        assert e.matches_name("a")
        assert not e.matches_name("Bob")

    def test_entity_to_ref(self) -> None:
        e = Entity(id="e001", name="Alice", entity_type="person")
        ref = e.to_ref()
        assert ref.name == "Alice"
        assert ref.entity_type == "person"
        assert ref.entity_id == "e001"

    def test_entity_type_all_values(self) -> None:
        for et in EntityType:
            assert isinstance(et.value, str)

    def test_entity_ref_required_fields(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EntityRef(name="")
        with pytest.raises(ValueError, match="non-empty"):
            EntityRef(name="   ")


# ---------------------------------------------------------------------------
# Validation tests for new fields
# ---------------------------------------------------------------------------


class TestValidationNewFields:
    def test_valid_memory_type(self) -> None:
        for mt in MemoryType:
            m = _make_memory(memory_type=mt.value)
            result = validate_memory(m)
            assert result.is_valid, f"memory_type={mt.value} should be valid"

    def test_invalid_memory_type(self) -> None:
        m = _make_memory(memory_type="invalid")
        result = validate_memory(m)
        assert not result.is_valid
        assert any("memory_type" in e for e in result.errors)

    def test_importance_values(self) -> None:
        for imp in ["critical", "high", "medium", "low"]:
            m = _make_memory(importance=imp)
            result = validate_memory(m)
            assert result.is_valid, f"importance={imp} should be valid"

    def test_invalid_importance(self) -> None:
        m = _make_memory(importance="urgent")
        result = validate_memory(m)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Missing required fields tests
# ---------------------------------------------------------------------------


class TestMissingFields:
    def test_entity_ref_missing_name(self) -> None:
        with pytest.raises(ValueError):
            EntityRef(name="")

    def test_fact_missing_type(self) -> None:
        with pytest.raises(ValueError):
            Fact(fact_type="", subject="x", predicate="y", value="z", content="x")

    def test_fact_missing_content(self) -> None:
        with pytest.raises(ValueError):
            Fact(fact_type="x", subject="x", predicate="y", value="z", content="")

    def test_memory_missing_id(self) -> None:
        m = _make_memory(id="")
        result = validate_memory(m)
        assert not result.is_valid

    def test_memory_missing_domain(self) -> None:
        m = _make_memory(domain="")
        result = validate_memory(m)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# Storage interface tests
# ---------------------------------------------------------------------------


class TestStorageInterface:
    def test_in_memory_store_isolation(self) -> None:
        store1 = InMemoryStore()
        store2 = InMemoryStore()
        store1.add(_make_memory(id="m1"))
        assert store1.count() == 1
        assert store2.count() == 0

    def test_add_and_retrieve(self) -> None:
        store = InMemoryStore()
        m = _make_memory()
        store.add(m)
        retrieved = store.get_by_id("memory_001")
        assert retrieved is not None
        assert retrieved.id == "memory_001"

    def test_count(self) -> None:
        store = InMemoryStore()
        assert store.count() == 0
        store.add(_make_memory(id="m1"))
        store.add(_make_memory(id="m2"))
        assert store.count() == 2

    def test_get_active(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(id="m1", lifecycle_state="active"))
        store.add(_make_memory(id="m2", lifecycle_state="inactive"))
        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == "m1"

    def test_get_by_entity(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(entity_name="self"))
        store.add(_make_memory(id="m2", entity_name="mom"))
        self_memories = store.get_by_entity("self")
        assert len(self_memories) == 1

    def test_get_by_domain(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(domain="learning"))
        store.add(_make_memory(id="m2", domain="interests"))
        learning = store.get_by_domain("learning")
        assert len(learning) == 1


# ---------------------------------------------------------------------------
# Production memory isolation tests
# ---------------------------------------------------------------------------


class TestProductionMemoryIsolation:
    def test_new_models_do_not_modify_production(self) -> None:
        """Verify that creating new models does not affect production memory."""
        import hashlib
        import os

        prod_files = [
            "memory.json",
            "memory/conversation.json",
            "memory/long_term_memory.json",
        ]

        hashes_before = {}
        for f in prod_files:
            if os.path.exists(f):
                with open(f, "rb") as fh:
                    hashes_before[f] = hashlib.md5(fh.read()).hexdigest()

        # Create and manipulate foundation objects
        m = _make_memory()
        store = InMemoryStore()
        store.add(m)
        store.deactivate("memory_001")

        # Verify production files unchanged
        for f in prod_files:
            if os.path.exists(f):
                with open(f, "rb") as fh:
                    current_hash = hashlib.md5(fh.read()).hexdigest()
                    assert current_hash == hashes_before[f], f"Production file {f} was modified"


# ---------------------------------------------------------------------------
# ID generation tests
# ---------------------------------------------------------------------------


class TestIdGeneration:
    def setup_method(self) -> None:
        reset_counters()

    def test_entity_id_self(self) -> None:
        eid = generate_entity_id("self")
        assert eid == "entity_000"

    def test_entity_id_other(self) -> None:
        eid = generate_entity_id("Alice")
        assert eid == "entity_001"

    def test_memory_id(self) -> None:
        mid = generate_memory_id()
        assert mid.startswith("memory_")

    def test_event_id(self) -> None:
        eid = generate_event_id()
        assert eid.startswith("event_")

    def test_episode_id(self) -> None:
        eid = generate_episode_id()
        assert eid.startswith("episode_")

    def test_relationship_id(self) -> None:
        rid = generate_relationship_id()
        assert rid.startswith("rel_")

    def test_counters_increment(self) -> None:
        m1 = generate_memory_id()
        m2 = generate_memory_id()
        assert m1 != m2

    def test_reset_counters(self) -> None:
        for _ in range(5):
            generate_memory_id()
        reset_counters()
        m = generate_memory_id()
        assert m == "memory_001"


# ---------------------------------------------------------------------------
# Temporal state representation tests
# ---------------------------------------------------------------------------


class TestTemporalState:
    def test_persistent_fact(self) -> None:
        ts = TemporalScope(valid_from="2025-01-01T00:00:00+00:00", is_persistent=True)
        assert ts.is_current_at("2099-12-31T23:59:59+00:00")
        assert not ts.is_expired_at("2099-12-31T23:59:59+00:00")

    def test_temporary_fact(self) -> None:
        ts = TemporalScope(
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-12-31T23:59:59+00:00",
            is_persistent=False,
        )
        assert ts.is_current_at("2025-06-15T12:00:00+00:00")
        assert not ts.is_current_at("2026-01-01T00:00:00+00:00")

    def test_historical_fact(self) -> None:
        ts = TemporalScope(
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-06-01T00:00:00+00:00",
            is_persistent=True,
        )
        assert not ts.is_current_at("2025-12-31T00:00:00+00:00")
        assert ts.is_expired_at("2025-12-31T00:00:00+00:00")

    def test_future_fact(self) -> None:
        ts = TemporalScope(
            valid_from=_utc_future_iso(30),
            is_persistent=True,
        )
        assert not ts.is_current_at()
        assert ts.is_expired_at()


# ---------------------------------------------------------------------------
# Provenance distinction tests
# ---------------------------------------------------------------------------


class TestProvenanceDistinction:
    def test_user_asserted(self) -> None:
        p = Provenance(source="user_statement")
        assert p.is_user_stated()
        assert not p.is_inferred()

    def test_model_inferred(self) -> None:
        p = Provenance(source="inferred")
        assert p.is_inferred()
        assert not p.is_user_stated()

    def test_model_extraction(self) -> None:
        p = Provenance(source="model_extraction")
        assert not p.is_user_stated()
        assert not p.is_inferred()

    def test_default_is_user_statement(self) -> None:
        p = Provenance()
        assert p.source == "user_statement"


# ---------------------------------------------------------------------------
# Lifecycle state tests
# ---------------------------------------------------------------------------


class TestLifecycleStates:
    def test_proposed_to_active(self) -> None:
        lc = Lifecycle(state="proposed")
        lc.activate()
        assert lc.state == "active"
        assert lc.is_active

    def test_active_to_inactive(self) -> None:
        lc = Lifecycle(state="active")
        lc.deactivate()
        assert lc.state == "inactive"
        assert lc.is_inactive
        assert lc.deactivated_at is not None

    def test_active_to_expired(self) -> None:
        lc = Lifecycle(state="active")
        lc.expire()
        assert lc.state == "expired"
        assert lc.is_expired
        assert lc.expired_at is not None

    def test_supersede(self) -> None:
        lc = Lifecycle(state="active")
        lc.supersede("memory_002")
        assert lc.state == "inactive"
        assert lc.superseded_by == "memory_002"
        assert lc.superseded_at is not None


# ---------------------------------------------------------------------------
# Combinations tests
# ---------------------------------------------------------------------------


class TestCombinations:
    def test_episodic_memory_with_provenance(self) -> None:
        m = Memory(
            id="ep_mem_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="interaction", subject="chat", predicate="had_conversation", value="chat", content="Discussed Python"),
            domain="conversations",
            lifecycle=Lifecycle(state="active"),
            provenance=Provenance(source="user_statement"),
            temporal=TemporalScope(valid_from=_utc_now_iso()),
            memory_type="episodic",
        )
        assert m.memory_type == "episodic"
        assert m.provenance.is_user_stated()

    def test_procedural_memory(self) -> None:
        m = Memory(
            id="proc_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="skill", subject="python", predicate="knows_how_to", value="Python", content="Knows how to use Python"),
            domain="knowledge",
            lifecycle=Lifecycle(state="active"),
            provenance=Provenance(source="inferred"),
            temporal=TemporalScope(valid_from=_utc_now_iso()),
            memory_type="procedural",
        )
        assert m.memory_type == "procedural"
        assert m.provenance.is_inferred()

    def test_temporary_memory_with_expiry(self) -> None:
        m = Memory(
            id="temp_001",
            entity=EntityRef(name="self"),
            fact=Fact(fact_type="state", subject="hunger", predicate="feels", value="hungry", content="I am hungry"),
            domain="states",
            lifecycle=Lifecycle(state="active"),
            provenance=Provenance(source="user_statement"),
            temporal=TemporalScope(
                valid_from=_utc_now_iso(),
                valid_until=_utc_future_iso(1),
                is_persistent=False,
            ),
            memory_type="temporary",
        )
        assert m.memory_type == "temporary"
        assert not m.temporal.is_persistent


# ---------------------------------------------------------------------------
# Adversarial foundation tests
# ---------------------------------------------------------------------------


class TestAdversarialFoundation:
    def test_empty_string_fields(self) -> None:
        with pytest.raises(ValueError):
            EntityRef(name="")
        with pytest.raises(ValueError):
            Fact(fact_type="", subject="x", predicate="y", value="z", content="x")
        with pytest.raises(ValueError):
            Fact(fact_type="x", subject="x", predicate="y", value="z", content="")

    def test_invalid_enum_values(self) -> None:
        m = _make_memory(memory_type="invalid")
        result = validate_memory(m)
        assert not result.is_valid

        m2 = _make_memory(importance="urgent")
        result2 = validate_memory(m2)
        assert not result2.is_valid

    def test_confidence_out_of_range(self) -> None:
        m = _make_memory(confidence=1.5)
        result = validate_memory(m)
        assert not result.is_valid

        m2 = _make_memory(confidence=-0.1)
        result2 = validate_memory(m2)
        assert not result2.is_valid

    def test_unicode_content(self) -> None:
        m = _make_memory(content="私は日本語が好きです")
        assert m.content == "私は日本語が好きです"
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        assert m2.content == "私は日本語が好きです"

    def test_special_characters(self) -> None:
        m = _make_memory(content="I like C++ & Python <3")
        assert m.content == "I like C++ & Python <3"

    def test_long_content(self) -> None:
        long_content = "x" * 10000
        m = _make_memory(content=long_content)
        assert len(m.content) == 10000

    def test_concurrent_id_generation(self) -> None:
        import threading
        reset_counters()
        ids = []
        lock = threading.Lock()

        def generate_ids():
            for _ in range(10):
                mid = generate_memory_id()
                with lock:
                    ids.append(mid)

        threads = [threading.Thread(target=generate_ids) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 50
        assert len(set(ids)) == 50  # All unique
