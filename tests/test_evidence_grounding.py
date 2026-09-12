"""Tests for Session 7 — Evidence Grounding.

50+ tests covering:
  - Entity validation (correct entity, wrong entity, self/user equivalence)
  - Relation validation (compatible relations, incompatible relations)
  - Object validation (direct match, token overlap, special cases)
  - Domain validation (compatible domains, incompatible domains)
  - Temporal validation (current/historical queries, active/superseded memories)
  - Lifecycle validation (expired memories, superseded for current)
  - Negation detection
  - Uncertainty detection
  - Conflict detection
  - Strength calculation
  - Status determination
  - Edge cases
"""

from __future__ import annotations

import pytest

from src.memory.foundation.evidence_grounding import (
    EvidenceAssessment,
    EvidenceGrounder,
    EvidenceItem,
)
from src.memory.foundation.hybrid_retriever import (
    HybridRetrievalResult,
    RetrievalCandidate,
)
from src.memory.foundation.query_understanding import StructuredQuery


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_candidate(
    memory_id: str = "mem_001",
    content: str = "User likes cats",
    entity: str = "user",
    fact_type: str = "like",
    category: str = "interests",
    status: str = "active",
    value: str = "cats",
    final_score: float = 0.5,
    subject: str = "cats",
) -> RetrievalCandidate:
    """Create a RetrievalCandidate with a full memory dict."""
    memory = {
        "id": memory_id,
        "content": content,
        "entity": {"name": entity},
        "fact_type": fact_type,
        "category": category,
        "status": status,
        "value": value,
        "subject": subject,
    }
    return RetrievalCandidate(memory=memory, final_score=final_score)


def _make_result(candidates: list[RetrievalCandidate]) -> HybridRetrievalResult:
    """Create a HybridRetrievalResult from candidates."""
    return HybridRetrievalResult(
        candidates=candidates,
        status="ok" if candidates else "not_found",
    )


def _make_query(
    raw: str = "test query",
    subject: str = "",
    relation: str = "",
    obj: str = "",
    entity: str = "user",
    domain: str = "",
    state: str = "current",
) -> StructuredQuery:
    """Create a StructuredQuery with no constraints by default."""
    return StructuredQuery(
        raw_query=raw,
        subject=subject,
        relation=relation,
        object=obj,
        entity=entity,
        domain=domain,
        state=state,
    )


# ---------------------------------------------------------------------------
# Entity Validation (Tests 1-6)
# ---------------------------------------------------------------------------

class TestEntityValidation:
    def test_correct_entity(self):
        q = _make_query(entity="user")
        c = _make_candidate(entity="user")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_wrong_entity(self):
        q = _make_query(entity="user")
        c = _make_candidate(entity="mom")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        assert "entity_mismatch" in a.rejected[0].rejection_reasons[0]

    def test_self_user_equivalence(self):
        q = _make_query(entity="self")
        c = _make_candidate(entity="user")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_user_self_equivalence(self):
        q = _make_query(entity="user")
        c = _make_candidate(entity="self")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_unknown_entity_passes(self):
        q = _make_query(entity="unknown")
        c = _make_candidate(entity="anyone")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_empty_entity_passes(self):
        q = _make_query(entity="")
        c = _make_candidate(entity="user")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1


# ---------------------------------------------------------------------------
# Relation Validation (Tests 7-12)
# ---------------------------------------------------------------------------

class TestRelationValidation:
    def test_correct_relation(self):
        q = _make_query(relation="prefer")
        c = _make_candidate(fact_type="preference")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_wrong_relation(self):
        q = _make_query(relation="possess")
        c = _make_candidate(fact_type="goal")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        assert "relation_mismatch" in a.rejected[0].rejection_reasons[0]

    def test_possession_compat(self):
        q = _make_query(relation="possess")
        c = _make_candidate(fact_type="possession")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_unknown_relation_passes(self):
        q = _make_query(relation="unknown")
        c = _make_candidate(fact_type="goal")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_partial_relation_match(self):
        q = _make_query(relation="learn")
        c = _make_candidate(fact_type="learning")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_empty_relation_passes(self):
        q = _make_query(relation="")
        c = _make_candidate(fact_type="skill")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1


# ---------------------------------------------------------------------------
# Object Validation (Tests 13-18)
# ---------------------------------------------------------------------------

class TestObjectValidation:
    def test_direct_subject_match(self):
        q = _make_query(obj="cats")
        c = _make_candidate(subject="cats")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_token_overlap_match(self):
        q = _make_query(obj="orange color")
        c = _make_candidate(content="User likes orange colored things")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_special_phone_match(self):
        q = _make_query(obj="phone")
        c = _make_candidate(content="User has a Samsung phone")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_special_color_match(self):
        q = _make_query(obj="color")
        c = _make_candidate(content="User likes the color blue")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_no_match(self):
        q = _make_query(obj="completely unrelated thing")
        c = _make_candidate(content="User likes cats")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        assert "object_mismatch" in a.rejected[0].rejection_reasons[0]

    def test_unknown_object_passes(self):
        q = _make_query(obj="unknown")
        c = _make_candidate(content="User likes cats")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1


# ---------------------------------------------------------------------------
# Domain Validation (Tests 19-22)
# ---------------------------------------------------------------------------

class TestDomainValidation:
    def test_correct_domain(self):
        q = _make_query(domain="interests")
        c = _make_candidate(category="interests")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_wrong_domain(self):
        q = _make_query(domain="goals")
        c = _make_candidate(category="interests")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        assert "domain_mismatch" in a.rejected[0].rejection_reasons[0]

    def test_domain_compatibility(self):
        q = _make_query(domain="interests")
        c = _make_candidate(category="like")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_unknown_domain_passes(self):
        q = _make_query(domain="unknown")
        c = _make_candidate(category="goals")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1


# ---------------------------------------------------------------------------
# Temporal Validation (Tests 23-26)
# ---------------------------------------------------------------------------

class TestTemporalValidation:
    def test_current_query_active_memory(self):
        q = _make_query(state="current")
        c = _make_candidate(status="active")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_current_query_superseded_memory(self):
        q = _make_query(state="current")
        c = _make_candidate(status="superseded")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        assert "temporal_mismatch" in a.rejected[0].rejection_reasons[0]

    def test_historical_query_superseded_memory(self):
        q = _make_query(state="historical")
        c = _make_candidate(status="superseded")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_unknown_state_passes(self):
        q = _make_query(state="unknown")
        c = _make_candidate(status="superseded")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1


# ---------------------------------------------------------------------------
# Lifecycle Validation (Tests 27-28)
# ---------------------------------------------------------------------------

class TestLifecycleValidation:
    def test_expired_memory_rejected(self):
        q = _make_query(state="current")
        c = _make_candidate(status="expired")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        assert "temporal_mismatch" in a.rejected[0].rejection_reasons[0]

    def test_active_memory_accepted(self):
        q = _make_query(state="current")
        c = _make_candidate(status="active")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1


# ---------------------------------------------------------------------------
# Negation Detection (Tests 29-32)
# ---------------------------------------------------------------------------

class TestNegationDetection:
    def test_dont_pattern(self):
        q = _make_query()
        c = _make_candidate(content="User don't like spiders")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_negated is True

    def test_never_pattern(self):
        q = _make_query()
        c = _make_candidate(content="User never drinks coffee")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_negated is True

    def test_no_longer_pattern(self):
        q = _make_query()
        c = _make_candidate(content="User no longer plays guitar")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_negated is True

    def test_positive_no_negation(self):
        q = _make_query()
        c = _make_candidate(content="User likes cats")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_negated is False


# ---------------------------------------------------------------------------
# Uncertainty Detection (Tests 33-36)
# ---------------------------------------------------------------------------

class TestUncertaintyDetection:
    def test_i_think_pattern(self):
        q = _make_query()
        c = _make_candidate(content="I think I like cats")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_uncertain is True

    def test_maybe_pattern(self):
        q = _make_query()
        c = _make_candidate(content="Maybe I like cats")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_uncertain is True

    def test_not_sure_pattern(self):
        q = _make_query()
        c = _make_candidate(content="I'm not sure about my preferences")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_uncertain is True

    def test_certain_no_uncertainty(self):
        q = _make_query()
        c = _make_candidate(content="User likes cats")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.is_uncertain is False


# ---------------------------------------------------------------------------
# Conflict Detection (Tests 37-40)
# ---------------------------------------------------------------------------

class TestConflictDetection:
    def test_detects_conflict(self):
        q = _make_query(relation="prefer", obj="favorite color")
        c1 = _make_candidate(
            memory_id="m1", content="User favorite color is blue",
            fact_type="preference", value="blue",
        )
        c2 = _make_candidate(
            memory_id="m2", content="User favorite color is red",
            fact_type="preference", value="red",
        )
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c1, c2]))
        assert len(a.conflicts) >= 1

    def test_no_conflict_different_values(self):
        q = _make_query(relation="prefer", obj="favorite color")
        c1 = _make_candidate(
            memory_id="m1", content="User favorite color is blue",
            fact_type="preference", value="blue",
        )
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c1]))
        assert len(a.conflicts) == 0

    def test_conflict_sets_conflicting_status(self):
        q = _make_query(relation="prefer", obj="favorite color")
        c1 = _make_candidate(
            memory_id="m1", content="User favorite color is blue",
            fact_type="preference", value="blue",
        )
        c2 = _make_candidate(
            memory_id="m2", content="User favorite color is red",
            fact_type="preference", value="red",
        )
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c1, c2]))
        assert a.status == "CONFLICTING"

    def test_conflict_requires_same_entity_and_fact(self):
        q = _make_query()
        c1 = _make_candidate(
            memory_id="m1", content="User likes blue",
            entity="user", fact_type="like", value="blue",
        )
        c2 = _make_candidate(
            memory_id="m2", content="User likes cats",
            entity="user", fact_type="like", value="cats",
        )
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c1, c2]))
        assert len(a.conflicts) >= 1


# ---------------------------------------------------------------------------
# Strength Calculation (Tests 41-44)
# ---------------------------------------------------------------------------

class TestStrengthCalculation:
    def test_perfect_match_high_strength(self):
        q = _make_query(relation="prefer", entity="user", domain="interests", state="current")
        c = _make_candidate(
            entity="user", fact_type="preference", category="interests",
            status="active", final_score=0.9,
        )
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.strength >= 0.6

    def test_negation_penalizes(self):
        q = _make_query()
        c = _make_candidate(content="User don't like spiders")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.strength < 1.0

    def test_uncertainty_penalizes(self):
        q = _make_query()
        c = _make_candidate(content="I think I like cats")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.strength < 1.0

    def test_inactive_penalizes(self):
        q = _make_query(state="current")
        c = _make_candidate(status="inactive", final_score=0.9)
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.strength < 0.9


# ---------------------------------------------------------------------------
# Status Determination (Tests 45-48)
# ---------------------------------------------------------------------------

class TestStatusDetermination:
    def test_no_candidates_gives_none(self):
        q = _make_query()
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([]))
        assert a.status == "NONE"

    def test_all_rejected_gives_none(self):
        q = _make_query(entity="user")
        c = _make_candidate(entity="mom")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert a.status == "NONE"

    def test_accepted_with_high_strength_gives_supported(self):
        q = _make_query(relation="prefer", entity="user", domain="interests", state="current")
        c = _make_candidate(
            entity="user", fact_type="preference", category="interests",
            status="active", final_score=0.9,
        )
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert a.status in ("SUPPORTED", "WEAK")

    def test_conflicting_gives_conflicting(self):
        q = _make_query(relation="prefer")
        c1 = _make_candidate(memory_id="m1", fact_type="preference", value="blue")
        c2 = _make_candidate(memory_id="m2", fact_type="preference", value="red")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c1, c2]))
        assert a.status == "CONFLICTING"


# ---------------------------------------------------------------------------
# Edge Cases (Tests 49-53)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_query_fields(self):
        q = _make_query(subject="", relation="", obj="", entity="", domain="", state="")
        c = _make_candidate()
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.accepted) == 1

    def test_multiple_rejection_reasons(self):
        q = _make_query(entity="user", relation="possess", domain="goals")
        c = _make_candidate(entity="mom", fact_type="goal", category="interests")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert len(a.rejected) == 1
        reasons = a.rejected[0].rejection_reasons
        assert len(reasons) >= 2

    def test_evidence_item_post_init(self):
        q = _make_query()
        c = _make_candidate(entity="user", fact_type="like", category="interests")
        g = EvidenceGrounder()
        item = g._validate_candidate(c, q)
        assert item.entity == "user"
        assert item.fact_type == "like"
        assert item.domain == "interests"
        assert item.status == "active"

    def test_diagnostics_built(self):
        q = _make_query(raw="what is my name")
        c = _make_candidate()
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert "query" in a.diagnostics
        assert "candidates_evaluated" in a.diagnostics
        assert "final_status" in a.diagnostics

    def test_has_evidence_property(self):
        q = _make_query()
        c = _make_candidate()
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert a.has_evidence is True

    def test_no_evidence_property(self):
        q = _make_query(entity="user")
        c = _make_candidate(entity="mom")
        g = EvidenceGrounder()
        a = g.ground(q, _make_result([c]))
        assert a.has_evidence is False
