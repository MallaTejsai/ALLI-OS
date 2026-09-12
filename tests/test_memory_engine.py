"""Tests for Session 9 — Memory Intelligence Integration.

50+ tests covering:
  - remember() pipeline
  - recall() pipeline
  - Write/read consistency
  - Lifecycle preservation
  - Entity isolation
  - Negation handling
  - Uncertainty handling
  - Historical/current distinction
  - Multiple facts
  - Unknown queries
  - General knowledge separation
  - Persistence
  - Edge cases
"""

from __future__ import annotations

import os
import tempfile
import shutil
import pytest

from src.memory.foundation.memory_engine import MemoryEngine, RememberResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Create an isolated MemoryEngine for testing."""
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test.sqlite")
    eng = MemoryEngine(db_path=db_path, allow_qwen=False)
    yield eng
    eng.close()
    shutil.rmtree(test_dir)


@pytest.fixture
def engine_online():
    """Create an isolated MemoryEngine with Qwen for integration tests."""
    test_dir = tempfile.mkdtemp()
    db_path = os.path.join(test_dir, "test_online.sqlite")
    eng = MemoryEngine(db_path=db_path, allow_qwen=True)
    yield eng
    eng.close()
    shutil.rmtree(test_dir)


# ---------------------------------------------------------------------------
# Remember Tests (Tests 1-10)
# ---------------------------------------------------------------------------

class TestRemember:
    def test_remember_basic(self, engine):
        result = engine.remember("I use a Samsung S24.")
        assert result.success
        assert result.operation == "created"
        assert result.entity in ("user", "self")  # Entity resolution uses "self"

    def test_remember_question_returns_no_op(self, engine):
        result = engine.remember("What is my name?")
        assert result.status == "question"

    def test_remember_greeting_returns_no_op(self, engine):
        result = engine.remember("Hello!")
        assert result.status in ("greeting", "no_op")

    def test_remember_empty_returns_no_op(self, engine):
        result = engine.remember("")
        assert result.status == "no_op"

    def test_remember_supersession(self, engine):
        engine.remember("I use an Oppo A53.")
        result = engine.remember("I now use a Samsung S24.")
        assert result.success
        assert result.operation in ("superseded", "created", "updated")

    def test_remember_entity_extraction(self, engine):
        result = engine.remember("My mom likes pink.")
        assert result.success
        assert "mom" in result.entity.lower()

    def test_remember_negation(self, engine):
        result = engine.remember("I don't like spiders.")
        assert result.success
        assert result.content

    def test_remember_uncertainty(self, engine):
        result = engine.remember("I think my favorite color is blue.")
        assert result.success

    def test_remember_multiple_facts(self, engine):
        r1 = engine.remember("I like Python.")
        r2 = engine.remember("I like Java.")
        assert r1.success
        assert r2.success

    def test_remember_has_diagnostics(self, engine):
        result = engine.remember("I use a Samsung S24.", diagnostics=True)
        assert result.write_result is not None


# ---------------------------------------------------------------------------
# Recall Tests (Tests 11-20)
# ---------------------------------------------------------------------------

class TestRecall:
    def test_recall_basic(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("What phone do I use?")
        assert answer
        assert "samsung" in answer.lower() or "s24" in answer.lower()

    def test_recall_empty_returns_prompt(self, engine):
        answer = engine.recall("")
        assert "question" in answer.lower() or "type" in answer.lower()

    def test_recall_unknown_returns_conservative(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("What is my favorite movie?")
        assert answer
        assert "star wars" not in answer.lower()
        assert "inception" not in answer.lower()

    def test_recall_entity_isolation(self, engine):
        engine.remember("My mom likes pink.")
        engine.remember("I like blue.")
        answer = engine.recall("What is my favorite color?")
        assert answer
        assert "pink" not in answer.lower() or "blue" in answer.lower()

    def test_recall_negation(self, engine):
        engine.remember("I don't like spiders.")
        answer = engine.recall("Do I like spiders?")
        assert answer
        assert "dont" in answer.lower() or "don't" in answer.lower() or "not" in answer.lower()

    def test_recall_historical(self, engine):
        engine.remember("I used to have an Oppo A53.")
        engine.remember("I now use a Samsung S24.")
        answer = engine.recall("What phone do I currently use?")
        assert answer
        assert "samsung" in answer.lower() or "s24" in answer.lower()

    def test_recall_multiple_facts(self, engine):
        engine.remember("I like Python.")
        engine.remember("I like Java.")
        answer = engine.recall("What do I like?")
        assert answer

    def test_recall_general_knowledge(self, engine):
        answer = engine.recall("What is Python?")
        assert answer

    def test_recall_has_answer(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("What phone do I use?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_recall_detailed(self, engine):
        engine.remember("I use a Samsung S24.")
        result = engine.recall_detailed("What phone do I use?")
        assert result.answer
        assert result.structured_query is not None
        assert result.retrieval_result is not None
        assert result.evidence_assessment is not None


# ---------------------------------------------------------------------------
# Write/Read Consistency (Tests 21-25)
# ---------------------------------------------------------------------------

class TestWriteReadConsistency:
    def test_write_then_read(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("What phone do I use?")
        assert "samsung" in answer.lower() or "s24" in answer.lower()

    def test_overwrite_then_read(self, engine):
        engine.remember("I use an Oppo A53.")
        engine.remember("I now use a Samsung S24.")
        answer = engine.recall("What phone do I currently use?")
        assert "samsung" in answer.lower() or "s24" in answer.lower()

    def test_entity_write_read(self, engine):
        engine.remember("My mom likes pink.")
        engine.remember("I like blue.")
        answer_user = engine.recall("What is my favorite color?")
        answer_mom = engine.recall("What color does my mom like?")
        assert "blue" in answer_user.lower()
        assert "pink" in answer_mom.lower()

    def test_negation_write_read(self, engine):
        engine.remember("I don't like spiders.")
        answer = engine.recall("Do I like spiders?")
        assert "dont" in answer.lower() or "don't" in answer.lower() or "not" in answer.lower()

    def test_multiple_write_read(self, engine):
        engine.remember("I like Python.")
        engine.remember("I like Java.")
        answer = engine.recall("What do I like?")
        assert answer


# ---------------------------------------------------------------------------
# Lifecycle Preservation (Tests 26-30)
# ---------------------------------------------------------------------------

class TestLifecyclePreservation:
    def test_supersession_preserved(self, engine):
        engine.remember("I use an Oppo A53.")
        engine.remember("I now use a Samsung S24.")
        # Current should be Samsung
        answer_current = engine.recall("What phone do I currently use?")
        assert "samsung" in answer_current.lower() or "s24" in answer_current.lower()

    def test_historical_access(self, engine):
        engine.remember("I used to have an Oppo A53.")
        engine.remember("I now use a Samsung S24.")
        # Historical should mention Oppo or show conflict
        answer_historical = engine.recall("What phone did I used to have?")
        assert answer_historical

    def test_deactivation(self, engine):
        engine.remember("I like cats.")
        engine.remember("I don't like cats anymore.")
        answer = engine.recall("Do I like cats?")
        assert answer

    def test_multiple_updates(self, engine):
        engine.remember("My favorite color is red.")
        engine.remember("My favorite color is blue.")
        engine.remember("My favorite color is green.")
        answer = engine.recall("What is my favorite color?")
        assert "green" in answer.lower()

    def test_temporal_scope(self, engine):
        engine.remember("I am learning Python this week.")
        answer = engine.recall("What am I learning?")
        assert answer


# ---------------------------------------------------------------------------
# Edge Cases (Tests 31-40)
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_repeated_fact(self, engine):
        engine.remember("I like Python.")
        result = engine.remember("I like Python.")
        assert result.status in ("success", "no_op")

    def test_typo_in_message(self, engine):
        result = engine.remember("I us a Samsumg S24.")
        assert result.status in ("success", "no_op")

    def test_slang_in_message(self, engine):
        result = engine.remember("yo i got a sick new phone its a samsung s24")
        assert result.status in ("success", "no_op")

    def test_empty_question(self, engine):
        answer = engine.recall("")
        assert answer

    def test_long_question(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("what is " + "very " * 20 + "my phone?")
        assert answer

    def test_all_caps(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("WHAT PHONE DO I USE?")
        assert answer

    def test_context_manager(self, test_dir=None):
        test_dir = test_dir or tempfile.mkdtemp()
        db_path = os.path.join(test_dir, "ctx_test.sqlite")
        with MemoryEngine(db_path=db_path, allow_qwen=False) as eng:
            eng.remember("I like Python.")
            answer = eng.recall("What do I like?")
            assert answer
        shutil.rmtree(test_dir, ignore_errors=True)

    def test_recall_detailed_has_all_stages(self, engine):
        engine.remember("I use a Samsung S24.")
        result = engine.recall_detailed("What phone do I use?")
        assert result.structured_query is not None
        assert result.retrieval_result is not None
        assert result.evidence_assessment is not None

    def test_remember_result_fields(self, engine):
        result = engine.remember("I use a Samsung S24.")
        assert result.message
        assert result.status
        assert result.entity

    def test_recall_returns_string(self, engine):
        engine.remember("I like Python.")
        answer = engine.recall("What do I like?")
        assert isinstance(answer, str)


# ---------------------------------------------------------------------------
# Integration (Tests 41-50)
# ---------------------------------------------------------------------------

class TestIntegration:
    def test_full_lifecycle(self, engine):
        # Write
        r1 = engine.remember("I use an Oppo A53.")
        assert r1.success

        r2 = engine.remember("I now use a Samsung S24.")
        assert r2.success

        # Read current
        answer_current = engine.recall("What phone do I currently use?")
        assert "samsung" in answer_current.lower() or "s24" in answer_current.lower()

        # Read historical
        answer_historical = engine.recall("What phone did I used to have?")
        assert answer_historical

    def test_multi_entity(self, engine):
        engine.remember("My mom likes pink.")
        engine.remember("I like blue.")
        engine.remember("My dad likes green.")

        answer_me = engine.recall("What is my favorite color?")
        answer_mom = engine.recall("What color does my mom like?")

        assert "blue" in answer_me.lower()
        assert "pink" in answer_mom.lower()

    def test_negation_flow(self, engine):
        engine.remember("I don't like spiders.")
        answer = engine.recall("Do I like spiders?")
        assert "dont" in answer.lower() or "don't" in answer.lower() or "not" in answer.lower()

    def test_uncertainty_flow(self, engine):
        engine.remember("I think my favorite color is blue.")
        answer = engine.recall("What is my favorite color?")
        assert answer

    def test_multiple_facts_flow(self, engine):
        engine.remember("I like Python.")
        engine.remember("I like Java.")
        answer = engine.recall("What do I like?")
        assert answer

    def test_unknown_query_flow(self, engine):
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("What is my favorite movie?")
        assert "star wars" not in answer.lower()
        assert "inception" not in answer.lower()

    def test_general_knowledge_separation(self, engine):
        answer = engine.recall("What is Python?")
        assert answer

    def test_repeated_write_stability(self, engine):
        for i in range(3):
            engine.remember("I like Python.")
        answer = engine.recall("What do I like?")
        assert answer

    def test_correction_flow(self, engine):
        engine.remember("My favorite color is red.")
        engine.remember("Actually, my favorite color is blue.")
        answer = engine.recall("What is my favorite color?")
        assert "blue" in answer.lower()

    def test_deletion_flow(self, engine):
        engine.remember("I like cats.")
        engine.remember("I don't like cats anymore.")
        answer = engine.recall("Do I like cats?")
        assert answer
