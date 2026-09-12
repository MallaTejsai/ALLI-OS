"""Comprehensive tests for ALLI v0.6 foundation models, storage, validation, and IDs.

Covers:
- EntityRef, Fact, TemporalScope, Provenance, Lifecycle, Memory, MemoryEvent
- InMemoryStore operations
- Structural validation
- Identifier generation
"""

import pytest
from datetime import datetime, timezone, timedelta

from src.memory.foundation.models import (
    Entity,
    EntityRef,
    EntityType,
    Episode,
    Fact,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    EventType,
    Provenance,
    ProvenanceSource,
    Relationship,
    ScoredMemory,
    TemporalScope,
)
from src.memory.foundation.storage import InMemoryStore, _cosine_similarity
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
    )


# ---------------------------------------------------------------------------
# EntityRef tests
# ---------------------------------------------------------------------------


class TestEntityRef:
    def test_basic_creation(self) -> None:
        ref = EntityRef(name="Alice", entity_type="person")
        assert ref.name == "Alice"
        assert ref.entity_type == "person"
        assert ref.entity_id is None
        assert ref.relation is None

    def test_default_entity_type(self) -> None:
        ref = EntityRef(name="Bob")
        assert ref.entity_type == "self"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EntityRef(name="")

    def test_whitespace_name_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EntityRef(name="   ")

    def test_frozen(self) -> None:
        ref = EntityRef(name="Alice")
        with pytest.raises(AttributeError):
            ref.name = "Bob"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Fact tests
# ---------------------------------------------------------------------------


class TestFact:
    def test_basic_creation(self) -> None:
        fact = _make_fact()
        assert fact.fact_type == "learning"
        assert fact.subject == "python"
        assert fact.content == "I am learning Python"

    def test_empty_fact_type_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Fact(
                fact_type="",
                subject="x",
                predicate="y",
                value="z",
                content="something",
            )

    def test_empty_content_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Fact(
                fact_type="x",
                subject="x",
                predicate="y",
                value="z",
                content="",
            )

    def test_polarity_optional(self) -> None:
        fact = _make_fact()
        assert fact.polarity is None

    def test_frozen(self) -> None:
        fact = _make_fact()
        with pytest.raises(AttributeError):
            fact.content = "changed"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TemporalScope tests
# ---------------------------------------------------------------------------


class TestTemporalScope:
    def test_persistent_forever(self) -> None:
        ts = TemporalScope(
            valid_from="2025-01-01T00:00:00+00:00",
            is_persistent=True,
        )
        assert ts.is_current_at("2099-12-31T23:59:59+00:00")
        assert not ts.is_expired_at("2099-12-31T23:59:59+00:00")

    def test_temporary_with_valid_until(self) -> None:
        ts = TemporalScope(
            valid_from="2025-01-01T00:00:00+00:00",
            valid_until="2025-12-31T23:59:59+00:00",
            is_persistent=False,
        )
        assert ts.is_current_at("2025-06-15T12:00:00+00:00")
        assert not ts.is_current_at("2026-01-01T00:00:00+00:00")

    def test_before_valid_from(self) -> None:
        ts = TemporalScope(
            valid_from="2025-06-01T00:00:00+00:00",
        )
        assert not ts.is_current_at("2025-01-01T00:00:00+00:00")

    def test_current_timestamp_defaults_to_now(self) -> None:
        ts = TemporalScope(
            valid_from=_utc_past_iso(365),
            is_persistent=True,
        )
        assert ts.is_current_at()
        assert not ts.is_expired_at()


# ---------------------------------------------------------------------------
# Provenance tests
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_user_stated(self) -> None:
        p = Provenance(source="user_statement")
        assert p.is_user_stated()
        assert not p.is_inferred()

    def test_inferred(self) -> None:
        p = Provenance(source="inferred")
        assert p.is_inferred()
        assert not p.is_user_stated()

    def test_default_is_user_statement(self) -> None:
        p = Provenance()
        assert p.source == "user_statement"

    def test_frozen(self) -> None:
        p = Provenance()
        with pytest.raises(AttributeError):
            p.source = "inferred"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestLifecycle:
    def test_default_proposed(self) -> None:
        lc = Lifecycle()
        assert lc.state == "proposed"
        assert not lc.is_active

    def test_activate(self) -> None:
        lc = Lifecycle(state="proposed")
        lc.activate()
        assert lc.state == "active"
        assert lc.is_active

    def test_deactivate(self) -> None:
        lc = Lifecycle(state="active")
        lc.deactivate()
        assert lc.state == "inactive"
        assert lc.is_inactive
        assert lc.deactivated_at is not None

    def test_expire(self) -> None:
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
# Memory tests
# ---------------------------------------------------------------------------


class TestMemory:
    def test_basic_creation(self) -> None:
        m = _make_memory()
        assert m.id == "memory_001"
        assert m.entity_name == "self"
        assert m.fact_type == "learning"
        assert m.content == "I am learning Python"
        assert m.is_active
        assert not m.is_expired

    def test_matches_entity_and_subject(self) -> None:
        m = _make_memory()
        assert m.matches_entity_and_subject("self", "learning", "python")
        assert not m.matches_entity_and_subject("self", "interest", "python")
        assert not m.matches_entity_and_subject("other", "learning", "python")

    def test_domain(self) -> None:
        m = _make_memory(domain="interests")
        assert m.domain == "interests"

    def test_confidence_range(self) -> None:
        m = _make_memory(confidence=0.5)
        assert 0.0 <= m.confidence <= 1.0


# ---------------------------------------------------------------------------
# Entity tests
# ---------------------------------------------------------------------------


class TestEntity:
    def test_to_ref(self) -> None:
        e = Entity(id="entity_000", name="self", entity_type="self")
        ref = e.to_ref()
        assert ref.name == "self"
        assert ref.entity_id == "entity_000"

    def test_matches_name_exact(self) -> None:
        e = Entity(id="entity_000", name="Alice")
        assert e.matches_name("Alice")
        assert e.matches_name("alice")
        assert e.matches_name("ALICE")

    def test_matches_name_alias(self) -> None:
        e = Entity(id="entity_000", name="Alice", aliases=["Ali", "A"])
        assert e.matches_name("Ali")
        assert e.matches_name("a")

    def test_matches_name_no_match(self) -> None:
        e = Entity(id="entity_000", name="Alice")
        assert not e.matches_name("Bob")


# ---------------------------------------------------------------------------
# Relationship tests
# ---------------------------------------------------------------------------


class TestRelationship:
    def test_basic_creation(self) -> None:
        a = EntityRef(name="Alice", entity_type="person")
        b = EntityRef(name="Bob", entity_type="person")
        rel = Relationship(
            id="rel_001",
            entity_a=a,
            entity_b=b,
            relation_type="friend",
        )
        assert rel.id == "rel_001"
        assert rel.entity_a.name == "Alice"
        assert rel.entity_b.name == "Bob"
        assert rel.lifecycle.is_active


# ---------------------------------------------------------------------------
# MemoryEvent tests
# ---------------------------------------------------------------------------


class TestMemoryEvent:
    def test_basic_creation(self) -> None:
        e = MemoryEvent(
            id="event_001",
            event_type="created",
            memory_id="memory_001",
        )
        assert e.id == "event_001"
        assert e.event_type == "created"
        assert e.memory_id == "memory_001"


# ---------------------------------------------------------------------------
# Episode tests
# ---------------------------------------------------------------------------


class TestEpisode:
    def test_basic_creation(self) -> None:
        ep = Episode(
            id="episode_001",
            user_message="I am learning Python",
        )
        assert ep.id == "episode_001"
        assert ep.user_message == "I am learning Python"
        assert ep.entities_involved == []
        assert ep.extracted_facts == []


# ---------------------------------------------------------------------------
# ScoredMemory tests
# ---------------------------------------------------------------------------


class TestScoredMemory:
    def test_basic_creation(self) -> None:
        m = _make_memory()
        sm = ScoredMemory(memory=m, score=0.95)
        assert sm.memory.id == "memory_001"
        assert sm.score == 0.95
        assert sm.semantic_score == 0.0
        assert sm.domain_match is True
        assert sm.entity_match is True


# ---------------------------------------------------------------------------
# InMemoryStore tests
# ---------------------------------------------------------------------------


class TestInMemoryStore:
    def test_add_and_count(self) -> None:
        store = InMemoryStore()
        assert store.count() == 0
        m = _make_memory()
        store.add(m)
        assert store.count() == 1

    def test_load(self) -> None:
        store = InMemoryStore()
        m = _make_memory()
        store.add(m)
        memories = store.load()
        assert len(memories) == 1
        assert memories[0].id == "memory_001"

    def test_get_by_id_found(self) -> None:
        store = InMemoryStore()
        m = _make_memory()
        store.add(m)
        found = store.get_by_id("memory_001")
        assert found is not None
        assert found.id == "memory_001"

    def test_get_by_id_not_found(self) -> None:
        store = InMemoryStore()
        found = store.get_by_id("nonexistent")
        assert found is None

    def test_get_active(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(id="m1", lifecycle_state="active"))
        store.add(_make_memory(id="m2", lifecycle_state="inactive"))
        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == "m1"

    def test_get_active_excludes_expired(self) -> None:
        store = InMemoryStore()
        expired_temporal = _make_temporal(
            valid_from=_utc_past_iso(10),
            valid_until=_utc_past_iso(1),
        )
        store.add(_make_memory(id="m1", temporal=expired_temporal))
        active = store.get_active()
        assert len(active) == 0

    def test_get_by_entity(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(entity_name="self"))
        store.add(_make_memory(entity_name="self", id="m2"))
        store.add(_make_memory(entity_name="mom", id="m3"))
        self_memories = store.get_by_entity("self")
        assert len(self_memories) == 2

    def test_get_by_domain(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(domain="learning"))
        store.add(_make_memory(domain="learning", id="m2"))
        store.add(_make_memory(domain="interests", id="m3"))
        learning = store.get_by_domain("learning")
        assert len(learning) == 2

    def test_update(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory())
        updated = store.update("memory_001", {"importance": "high"})
        assert updated is not None
        assert updated.importance == "high"

    def test_update_not_found(self) -> None:
        store = InMemoryStore()
        result = store.update("nonexistent", {"importance": "high"})
        assert result is None

    def test_deactivate(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory())
        result = store.deactivate("memory_001")
        assert result is True
        assert store.get_by_id("memory_001").lifecycle.state == "inactive"

    def test_deactivate_not_found(self) -> None:
        store = InMemoryStore()
        result = store.deactivate("nonexistent")
        assert result is False

    def test_supersede(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(id="m1", lifecycle_state="active"))
        store.add(_make_memory(id="m2", lifecycle_state="active"))
        result = store.supersede("m1", "m2")
        assert result is True
        loser = store.get_by_id("m2")
        assert loser is not None
        assert loser.lifecycle.superseded_by == "m1"

    def test_supersede_not_found(self) -> None:
        store = InMemoryStore()
        result = store.supersede("m1", "nonexistent")
        assert result is False

    def test_save_replaces_all(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory(id="old"))
        new = _make_memory(id="new")
        store.save([new])
        assert store.count() == 1
        assert store.get_by_id("new") is not None

    def test_search_semantic(self) -> None:
        store = InMemoryStore()
        m1 = _make_memory(id="m1")
        m1.embedding = [1.0, 0.0, 0.0]
        m2 = _make_memory(id="m2")
        m2.embedding = [0.9, 0.1, 0.0]
        m3 = _make_memory(id="m3")
        m3.embedding = [0.0, 0.0, 1.0]
        store.add(m1)
        store.add(m2)
        store.add(m3)
        results = store.search_semantic([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        assert results[0].memory.id == "m1"
        assert results[0].score > results[1].score

    def test_search_semantic_no_embeddings(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory())
        results = store.search_semantic([1.0, 0.0], top_k=5)
        assert len(results) == 0

    def test_search_semantic_empty_query(self) -> None:
        store = InMemoryStore()
        store.add(_make_memory())
        results = store.search_semantic([], top_k=5)
        assert len(results) == 0


# ---------------------------------------------------------------------------
# Cosine similarity tests
# ---------------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0

    def test_opposite_vectors(self) -> None:
        assert abs(_cosine_similarity([1.0, 0.0], [-1.0, 0.0]) - (-1.0)) < 1e-10

    def test_empty_vectors(self) -> None:
        assert _cosine_similarity([], []) == 0.0

    def test_mismatched_lengths(self) -> None:
        assert _cosine_similarity([1.0], [1.0, 0.0]) == 0.0

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# ID generation tests
# ---------------------------------------------------------------------------


class TestIds:
    def setup_method(self) -> None:
        reset_counters()

    def test_generate_entity_id_self(self) -> None:
        eid = generate_entity_id("self")
        assert eid == "entity_000"

    def test_generate_entity_id_other(self) -> None:
        eid = generate_entity_id("Alice")
        assert eid == "entity_001"

    def test_generate_memory_id(self) -> None:
        mid = generate_memory_id()
        assert mid.startswith("memory_")

    def test_generate_event_id(self) -> None:
        eid = generate_event_id()
        assert eid.startswith("event_")

    def test_generate_episode_id(self) -> None:
        eid = generate_episode_id()
        assert eid.startswith("episode_")

    def test_generate_relationship_id(self) -> None:
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
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateEntityRef:
    def test_valid(self) -> None:
        ref = _make_entity_ref()
        result = validate_entity_ref(ref)
        assert result.is_valid
        assert result.errors == []

    def test_empty_name_raises_at_creation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            EntityRef(name="")

    def test_invalid_type(self) -> None:
        ref = EntityRef(name="Alice", entity_type="invalid")
        result = validate_entity_ref(ref)
        assert not result.is_valid
        assert any("entity_type" in e for e in result.errors)


class TestValidateFact:
    def test_valid(self) -> None:
        fact = _make_fact()
        result = validate_fact(fact)
        assert result.is_valid

    def test_empty_fact_type_raises_at_creation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Fact(fact_type="", subject="x", predicate="y", value="z", content="x")

    def test_empty_content_raises_at_creation(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            Fact(fact_type="x", subject="x", predicate="y", value="z", content="")

    def test_invalid_polarity(self) -> None:
        fact = Fact(
            fact_type="x", subject="x", predicate="y", value="z", content="x",
            polarity="invalid",
        )
        result = validate_fact(fact)
        assert not result.is_valid


class TestValidateTemporal:
    def test_valid_persistent(self) -> None:
        ts = _make_temporal()
        result = validate_temporal(ts)
        assert result.is_valid

    def test_valid_temporary(self) -> None:
        ts = _make_temporal(
            valid_until=_utc_future_iso(1),
        )
        result = validate_temporal(ts)
        assert result.is_valid

    def test_empty_valid_from(self) -> None:
        ts = TemporalScope(valid_from="")
        result = validate_temporal(ts)
        assert not result.is_valid

    def test_until_before_from(self) -> None:
        ts = TemporalScope(
            valid_from="2025-12-31T00:00:00+00:00",
            valid_until="2025-01-01T00:00:00+00:00",
        )
        result = validate_temporal(ts)
        assert not result.is_valid


class TestValidateProvenance:
    def test_valid_user_statement(self) -> None:
        p = _make_provenance("user_statement")
        result = validate_provenance(p)
        assert result.is_valid

    def test_valid_inferred(self) -> None:
        p = _make_provenance("inferred")
        result = validate_provenance(p)
        assert result.is_valid

    def test_invalid_source(self) -> None:
        p = Provenance(source="invalid")
        result = validate_provenance(p)
        assert not result.is_valid


class TestValidateLifecycle:
    def test_valid_proposed(self) -> None:
        lc = Lifecycle(state="proposed")
        result = validate_lifecycle(lc)
        assert result.is_valid

    def test_valid_active(self) -> None:
        lc = Lifecycle(state="active")
        result = validate_lifecycle(lc)
        assert result.is_valid

    def test_invalid_state(self) -> None:
        lc = Lifecycle(state="garbage")
        result = validate_lifecycle(lc)
        assert not result.is_valid

    def test_superseded_by_requires_inactive(self) -> None:
        lc = Lifecycle(state="active", superseded_by="memory_001")
        result = validate_lifecycle(lc)
        assert not result.is_valid


class TestValidateMemory:
    def test_valid(self) -> None:
        m = _make_memory()
        result = validate_memory(m)
        assert result.is_valid
        assert result.errors == []

    def test_empty_id(self) -> None:
        m = _make_memory(id="")
        result = validate_memory(m)
        assert not result.is_valid
        assert any("id" in e for e in result.errors)

    def test_empty_entity_name(self) -> None:
        """__post_init__ prevents creation with empty name."""
        with pytest.raises(ValueError, match="non-empty"):
            EntityRef(name="")

    def test_empty_domain(self) -> None:
        m = _make_memory(domain="")
        result = validate_memory(m)
        assert not result.is_valid

    def test_invalid_confidence_too_high(self) -> None:
        m = _make_memory(confidence=1.5)
        result = validate_memory(m)
        assert not result.is_valid

    def test_invalid_confidence_negative(self) -> None:
        m = _make_memory(confidence=-0.1)
        result = validate_memory(m)
        assert not result.is_valid

    def test_invalid_importance(self) -> None:
        m = _make_memory(importance="invalid")
        result = validate_memory(m)
        assert not result.is_valid

    def test_active_expired_cross_field(self) -> None:
        expired_temporal = _make_temporal(
            valid_from=_utc_past_iso(10),
            valid_until=_utc_past_iso(1),
        )
        m = _make_memory(temporal=expired_temporal)
        result = validate_memory(m)
        assert not result.is_valid
        assert any("expired" in e for e in result.errors)


class TestValidateEvent:
    def test_valid(self) -> None:
        e = MemoryEvent(id="event_001", event_type="created", memory_id="memory_001")
        result = validate_event(e)
        assert result.is_valid

    def test_empty_id(self) -> None:
        e = MemoryEvent(id="", event_type="created", memory_id="memory_001")
        result = validate_event(e)
        assert not result.is_valid

    def test_invalid_type(self) -> None:
        e = MemoryEvent(id="e", event_type="garbage", memory_id="m")
        result = validate_event(e)
        assert not result.is_valid

    def test_empty_memory_id(self) -> None:
        e = MemoryEvent(id="e", event_type="created", memory_id="")
        result = validate_event(e)
        assert not result.is_valid


# ---------------------------------------------------------------------------
# ValidationResult bool protocol
# ---------------------------------------------------------------------------


class TestValidationResult:
    def test_valid_is_truthy(self) -> None:
        v = ValidationResult(is_valid=True, errors=[])
        assert bool(v) is True

    def test_invalid_is_falsy(self) -> None:
        v = ValidationResult(is_valid=False, errors=["bad"])
        assert bool(v) is False
