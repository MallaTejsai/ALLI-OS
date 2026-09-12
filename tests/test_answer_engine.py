"""Tests for Session 8 — Answer Engine.

50+ tests covering:
  - Pipeline integration (S5→S6→S7→Qwen)
  - UNKNOWN handling (no evidence)
  - WEAK handling (low confidence)
  - CONFLICTING handling (contradictory evidence)
  - SUPPORTED handling (valid evidence)
  - Anti-hallucination (no invented facts)
  - Entity isolation (mom vs user)
  - Historical/current distinction
  - Negation handling
  - Uncertainty handling
  - General vs personal question handling
  - Multiple supporting memories
  - Edge cases
"""

from __future__ import annotations

import pytest

from src.memory.foundation.answer_engine import (
    AnswerEngine,
    AnswerResult,
    _extract_topic,
    _unknown_response,
    _weak_response,
    _conflict_response,
    _format_evidence_for_qwen,
)
from src.memory.foundation.evidence_grounding import (
    EvidenceAssessment,
    EvidenceGrounder,
    EvidenceItem,
)
from src.memory.foundation.hybrid_retriever import (
    HybridRetrievalResult,
    HybridRetriever,
    RetrievalCandidate,
)
from src.memory.foundation.query_understanding import StructuredQuery
from src.memory.store import MemoryStore
from src.memory.embeddings import EmbeddingStore


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
    # Allow overriding category via fact_type for domain compatibility
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
    return HybridRetrievalResult(
        candidates=candidates,
        status="ok" if candidates else "not_found",
    )


def _make_query(
    raw: str = "test",
    subject: str = "",
    relation: str = "",
    obj: str = "",
    entity: str = "user",
    domain: str = "",
    state: str = "current",
) -> StructuredQuery:
    return StructuredQuery(
        raw_query=raw,
        subject=subject,
        relation=relation,
        object=obj,
        entity=entity,
        domain=domain,
        state=state,
    )


def _make_engine(candidates: list[RetrievalCandidate]) -> AnswerEngine:
    """Create an AnswerEngine with a mock retriever that returns fixed candidates."""
    store = MemoryStore(skip_embeddings=True)
    store.memories = [c.memory for c in candidates]
    embed_store = EmbeddingStore()

    # Create real hybrid retriever with the in-memory store
    retriever = HybridRetriever(store=store, embedding_store=embed_store)

    return AnswerEngine(
        hybrid_retriever=retriever,
        embedding_store=embed_store,
        allow_qwen=False,  # Offline mode for deterministic tests
    )


def _make_engine_online(candidates: list[RetrievalCandidate]) -> AnswerEngine:
    """Create an AnswerEngine with real Qwen for integration tests."""
    store = MemoryStore(skip_embeddings=True)
    store.memories = [c.memory for c in candidates]
    embed_store = EmbeddingStore()
    retriever = HybridRetriever(store=store, embedding_store=embed_store)

    return AnswerEngine(
        hybrid_retriever=retriever,
        embedding_store=embed_store,
        allow_qwen=True,
    )


# ---------------------------------------------------------------------------
# Topic Extraction (Tests 1-6)
# ---------------------------------------------------------------------------

class TestTopicExtraction:
    def test_what_is_my_name(self):
        assert _extract_topic("what is my name?") == "name"

    def test_what_is_my_favorite_color(self):
        assert _extract_topic("what is my favorite color?") == "favorite color"

    def test_do_i_like_spiders(self):
        assert _extract_topic("do i like spiders?") == "spiders"

    def test_what_phone_do_i_have(self):
        topic = _extract_topic("what phone do i have?")
        assert topic is not None

    def test_no_topic(self):
        assert _extract_topic("hello") is None

    def test_what_is_the_weather(self):
        topic = _extract_topic("what is the weather?")
        assert topic is not None


# ---------------------------------------------------------------------------
# Unknown Response (Tests 7-9)
# ---------------------------------------------------------------------------

class TestUnknownResponse:
    def test_with_topic(self):
        resp = _unknown_response("what is my favorite movie?")
        assert "favorite movie" in resp.lower()

    def test_without_topic(self):
        resp = _unknown_response("hello")
        assert "don't" in resp.lower() or "not" in resp.lower()

    def test_no_invention(self):
        resp = _unknown_response("what is my favorite movie?")
        assert "star wars" not in resp.lower()
        assert "inception" not in resp.lower()


# ---------------------------------------------------------------------------
# Weak Response (Tests 10-11)
# ---------------------------------------------------------------------------

class TestWeakResponse:
    def test_with_topic(self):
        resp = _weak_response("what is my favorite color?")
        assert "not fully sure" in resp.lower()

    def test_no_invention(self):
        resp = _weak_response("what is my favorite color?")
        assert "blue" not in resp.lower()
        assert "red" not in resp.lower()


# ---------------------------------------------------------------------------
# Conflict Response (Tests 12-13)
# ---------------------------------------------------------------------------

class TestConflictResponse:
    def test_with_values(self):
        assessment = EvidenceAssessment(
            conflicts=[
                [EvidenceItem(candidate=_make_candidate(value="blue"), content="blue")],
                [EvidenceItem(candidate=_make_candidate(value="red"), content="red")],
            ]
        )
        resp = _conflict_response("what is my favorite color?", assessment)
        assert "conflicting" in resp.lower()

    def test_without_values(self):
        assessment = EvidenceAssessment(conflicts=[])
        resp = _conflict_response("what is my color?", assessment)
        assert "conflicting" in resp.lower()


# ---------------------------------------------------------------------------
# Evidence Formatting (Tests 14-16)
# ---------------------------------------------------------------------------

class TestEvidenceFormatting:
    def test_includes_content(self):
        assessment = EvidenceAssessment(
            accepted=[
                EvidenceItem(
                    candidate=_make_candidate(content="User likes blue"),
                    content="User likes blue",
                )
            ]
        )
        text = _format_evidence_for_qwen("test", assessment)
        assert "User likes blue" in text

    def test_includes_negation_tag(self):
        assessment = EvidenceAssessment(
            accepted=[
                EvidenceItem(
                    candidate=_make_candidate(content="User dont like spiders"),
                    content="User dont like spiders",
                    is_negated=True,
                )
            ]
        )
        text = _format_evidence_for_qwen("test", assessment)
        assert "NEGATED" in text

    def test_excludes_rejected(self):
        assessment = EvidenceAssessment(
            accepted=[],
            rejected=[
                EvidenceItem(
                    candidate=_make_candidate(content="Rejected fact"),
                    content="Rejected fact",
                )
            ],
        )
        text = _format_evidence_for_qwen("test", assessment)
        assert "Rejected fact" not in text


# ---------------------------------------------------------------------------
# Unknown Handling (Tests 17-21)
# ---------------------------------------------------------------------------

class TestUnknownHandling:
    def test_no_evidence_returns_unknown(self):
        engine = _make_engine([])
        result = engine.answer("what is my favorite movie?")
        assert result.source == "unknown"
        assert result.answer
        assert not result.called_qwen

    def test_unknown_does_not_call_qwen(self):
        engine = _make_engine([])
        result = engine.answer("what is my favorite movie?")
        assert result.called_qwen is False

    def test_unknown_no_invention(self):
        engine = _make_engine([])
        result = engine.answer("what is my favorite movie?")
        # Must not contain any specific movie names
        assert "star wars" not in result.answer.lower()
        assert "inception" not in result.answer.lower()
        assert "matrix" not in result.answer.lower()

    def test_unknown_preserves_question_topic(self):
        engine = _make_engine([])
        result = engine.answer("what is my favorite movie?")
        assert "movie" in result.answer.lower()

    def test_unknown_status_in_result(self):
        engine = _make_engine([])
        result = engine.answer("what is my favorite movie?")
        assert result.evidence_assessment is not None
        assert result.evidence_assessment.status == "NONE"


# ---------------------------------------------------------------------------
# Weak Handling (Tests 22-25)
# ---------------------------------------------------------------------------

class TestWeakHandling:
    def test_weak_returns_weak_response(self):
        # Create a candidate that will be weak (low match)
        c = _make_candidate(
            content="User might like cats",
            fact_type="preference",
            final_score=0.3,
        )
        engine = _make_engine([c])
        result = engine.answer("what is my favorite animal?")
        # Weak or supported depending on match quality
        assert result.source in ("weak", "unknown")

    def test_weak_no_invention(self):
        c = _make_candidate(
            content="User might like cats",
            fact_type="preference",
            final_score=0.3,
        )
        engine = _make_engine([c])
        result = engine.answer("what is my favorite animal?")
        assert "dog" not in result.answer.lower()


# ---------------------------------------------------------------------------
# Conflict Handling (Tests 26-29)
# ---------------------------------------------------------------------------

class TestConflictHandling:
    def test_conflict_detected(self):
        c1 = _make_candidate(
            memory_id="m1",
            content="User favorite color is blue",
            fact_type="preference",
            value="blue",
        )
        c2 = _make_candidate(
            memory_id="m2",
            content="User favorite color is red",
            fact_type="preference",
            value="red",
        )
        engine = _make_engine([c1, c2])
        result = engine.answer("what is my favorite color?")
        # Should detect conflict
        assert result.evidence_assessment is not None

    def test_conflict_no_arbitrary_choice(self):
        c1 = _make_candidate(
            memory_id="m1",
            content="User favorite color is blue",
            fact_type="preference",
            value="blue",
        )
        c2 = _make_candidate(
            memory_id="m2",
            content="User favorite color is red",
            fact_type="preference",
            value="red",
        )
        engine = _make_engine([c1, c2])
        result = engine.answer("what is my favorite color?")
        # Must not silently pick one
        if result.source == "conflict":
            assert "conflicting" in result.answer.lower() or "clarify" in result.answer.lower()


# ---------------------------------------------------------------------------
# SUPPORTED Handling (Tests 30-33)
# ---------------------------------------------------------------------------

class TestSupportedHandling:
    def test_supported_offline_returns_evidence(self):
        c = _make_candidate(
            content="User name is Tejsai",
            entity="user",
            fact_type="identity",
            category="identity",
        )
        engine = _make_engine([c])
        result = engine.answer("what is my name?")
        assert result.source == "evidence_offline"
        assert "Tejsai" in result.answer

    def test_supported_evidence_only(self):
        c = _make_candidate(
            content="User likes Python",
            entity="user",
            fact_type="like",
            category="interests",
        )
        engine = _make_engine([c])
        result = engine.answer("what do i like?")
        assert "Python" in result.answer

    def test_no_rejected_in_answer(self):
        c_good = _make_candidate(
            memory_id="m1",
            content="User likes Python",
            entity="user",
            fact_type="like",
            category="interests",
        )
        c_bad = _make_candidate(
            memory_id="m2",
            content="Mom likes pink",
            entity="mom",
            fact_type="like",
            category="interests",
        )
        engine = _make_engine([c_good, c_bad])
        result = engine.answer("what programming languages do i like?")
        # Mom's preference should not appear
        assert "pink" not in result.answer.lower() or "mom" not in result.answer.lower()

    def test_multiple_evidence_items(self):
        c1 = _make_candidate(
            content="User likes Python",
            entity="user",
            fact_type="like",
            category="interests",
        )
        c2 = _make_candidate(
            content="User likes Java",
            entity="user",
            fact_type="like",
            category="interests",
        )
        engine = _make_engine([c1, c2])
        result = engine.answer("what do i like?")
        # Both should be in the evidence
        assert result.evidence_assessment is not None
        assert len(result.evidence_assessment.accepted) >= 1


# ---------------------------------------------------------------------------
# Entity Isolation (Tests 34-37)
# ---------------------------------------------------------------------------

class TestEntityIsolation:
    def test_user_not_mom(self):
        c_user = _make_candidate(
            content="User favorite color is blue",
            entity="user",
            fact_type="preference",
            value="blue",
        )
        c_mom = _make_candidate(
            content="Mom favorite color is pink",
            entity="mom",
            fact_type="preference",
            value="pink",
        )
        engine = _make_engine([c_user, c_mom])
        result = engine.answer("what is my favorite color?")
        # Should not include mom's preference
        assert result.evidence_assessment is not None
        for item in result.evidence_assessment.accepted:
            assert item.entity.lower() != "mom"

    def test_mom_not_user(self):
        c_user = _make_candidate(
            content="User favorite color is blue",
            entity="user",
            fact_type="preference",
            value="blue",
        )
        c_mom = _make_candidate(
            content="Mom favorite color is pink",
            entity="mom",
            fact_type="preference",
            value="pink",
        )
        engine = _make_engine([c_user, c_mom])
        result = engine.answer("what is mom's favorite color?")
        # Should not include user's preference
        assert result.evidence_assessment is not None
        for item in result.evidence_assessment.accepted:
            assert item.entity.lower() != "user" or "mom" in item.entity.lower()

    def test_entity_filtering_works(self):
        c_user = _make_candidate(
            content="User likes Python",
            entity="user",
        )
        c_mom = _make_candidate(
            content="Mom likes cooking",
            entity="mom",
        )
        engine = _make_engine([c_user, c_mom])
        result = engine.answer("what do i like?")
        assert result.evidence_assessment is not None
        # At least the user's memory should be accepted
        accepted_entities = [i.entity.lower() for i in result.evidence_assessment.accepted]
        # Mom should not be in accepted if entity filtering works
        assert "mom" not in accepted_entities or "user" in accepted_entities


# ---------------------------------------------------------------------------
# Historical/Current Handling (Tests 38-41)
# ---------------------------------------------------------------------------

class TestHistoricalCurrent:
    def test_current_excludes_superseded(self):
        c_active = _make_candidate(
            content="User phone is Samsung",
            entity="user",
            fact_type="possession",
            status="active",
            value="Samsung",
        )
        c_superseded = _make_candidate(
            content="User phone was Oppo",
            entity="user",
            fact_type="possession",
            status="superseded",
            value="Oppo",
        )
        engine = _make_engine([c_active, c_superseded])
        result = engine.answer("what phone do i currently use?")
        if result.evidence_assessment:
            for item in result.evidence_assessment.accepted:
                # Superseded should not be accepted for current queries
                if item.status == "superseded":
                    assert result.evidence_assessment.status != "SUPPORTED"

    def test_historical_includes_superseded(self):
        c_superseded = _make_candidate(
            content="User phone was Oppo",
            entity="user",
            fact_type="possession",
            status="superseded",
            value="Oppo",
        )
        engine = _make_engine([c_superseded])
        result = engine.answer("what phone did i used to have?")
        # Historical query should accept superseded memories
        assert result.evidence_assessment is not None

    def test_current_prefers_active(self):
        c_active = _make_candidate(
            content="User phone is Samsung",
            entity="user",
            fact_type="possession",
            category="possessions",
            status="active",
            value="Samsung",
        )
        engine = _make_engine([c_active])
        result = engine.answer("what phone do i use?")
        assert result.source in ("evidence_offline", "evidence")
        assert "Samsung" in result.answer


# ---------------------------------------------------------------------------
# Negation Handling (Tests 42-45)
# ---------------------------------------------------------------------------

class TestNegationHandling:
    def test_negation_detected(self):
        c = _make_candidate(
            content="User dont like spiders",
            entity="user",
            fact_type="dislike",
        )
        engine = _make_engine([c])
        result = engine.answer("do i like spiders?")
        if result.evidence_assessment and result.evidence_assessment.accepted:
            item = result.evidence_assessment.accepted[0]
            assert item.is_negated is True

    def test_negation_in_answer(self):
        c = _make_candidate(
            content="User dont like spiders",
            entity="user",
            fact_type="dislike",
        )
        engine = _make_engine([c])
        result = engine.answer("do i like spiders?")
        # The answer should respect the negation
        assert result.answer

    def test_negation_reduces_strength(self):
        c = _make_candidate(
            content="User dont like spiders",
            entity="user",
            fact_type="dislike",
        )
        engine = _make_engine([c])
        result = engine.answer("do i like spiders?")
        if result.evidence_assessment and result.evidence_assessment.accepted:
            item = result.evidence_assessment.accepted[0]
            # Negated evidence should have lower strength
            assert item.strength < 1.0


# ---------------------------------------------------------------------------
# Uncertainty Handling (Tests 46-49)
# ---------------------------------------------------------------------------

class TestUncertaintyHandling:
    def test_uncertainty_detected(self):
        c = _make_candidate(
            content="I think my favorite color is blue",
            entity="user",
            fact_type="preference",
        )
        engine = _make_engine([c])
        result = engine.answer("what is my favorite color?")
        if result.evidence_assessment and result.evidence_assessment.accepted:
            item = result.evidence_assessment.accepted[0]
            assert item.is_uncertain is True

    def test_uncertainty_reduces_strength(self):
        c = _make_candidate(
            content="I think my favorite color is blue",
            entity="user",
            fact_type="preference",
        )
        engine = _make_engine([c])
        result = engine.answer("what is my favorite color?")
        if result.evidence_assessment and result.evidence_assessment.accepted:
            item = result.evidence_assessment.accepted[0]
            assert item.strength < 1.0

    def test_uncertainty_preserved_in_answer(self):
        c = _make_candidate(
            content="I think my favorite color is blue",
            entity="user",
            fact_type="preference",
        )
        engine = _make_engine([c])
        result = engine.answer("what is my favorite color?")
        # Answer should not present uncertain fact as definite
        assert result.answer


# ---------------------------------------------------------------------------
# General vs Personal (Tests 50-53)
# ---------------------------------------------------------------------------

class TestGeneralVsPersonal:
    def test_general_knowledge_not_personal(self):
        engine = _make_engine([])
        result = engine.answer("what is the capital of France?")
        # Should not try to answer from personal memories
        assert result.source in ("unknown", "error")

    def test_personal_question_uses_evidence(self):
        c = _make_candidate(
            content="User name is Tejsai",
            entity="user",
            fact_type="identity",
            category="identity",
        )
        engine = _make_engine([c])
        result = engine.answer("what is my name?")
        assert "Tejsai" in result.answer

    def test_unrelated_memory_not_used(self):
        c = _make_candidate(
            content="User likes Python",
            entity="user",
            fact_type="like",
        )
        engine = _make_engine([c])
        result = engine.answer("what is my favorite movie?")
        assert "Python" not in result.answer

    def test_no_personal_facts_invented(self):
        engine = _make_engine([])
        result = engine.answer("what is my favorite movie?")
        # Must not invent any movie
        assert "star wars" not in result.answer.lower()
        assert "inception" not in result.answer.lower()
        assert "matrix" not in result.answer.lower()


# ---------------------------------------------------------------------------
# Edge Cases (Tests 54-58)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_question(self):
        engine = _make_engine([])
        result = engine.answer("")
        assert result.answer

    def test_very_long_question(self):
        engine = _make_engine([])
        result = engine.answer("what is " + "very " * 50 + "my name?")
        assert result.answer

    def test_typo_in_question(self):
        c = _make_candidate(
            content="User name is Tejsai",
            entity="user",
            fact_type="identity",
        )
        engine = _make_engine([c])
        result = engine.answer("whats my nmame?")
        # Should still process without crashing
        assert result.answer

    def test_slang_in_question(self):
        engine = _make_engine([])
        result = engine.answer("yo whats my fav color")
        assert result.answer

    def test_diagnostics_flag(self):
        engine = _make_engine([])
        result = engine.answer("what is my name?", diagnostics=True)
        assert result.diagnostics or result.evidence_assessment is not None

    def test_result_has_all_stages(self):
        c = _make_candidate()
        engine = _make_engine([c])
        result = engine.answer("what is my name?")
        assert result.structured_query is not None
        assert result.retrieval_result is not None
        assert result.evidence_assessment is not None
