"""Comprehensive tests for ALLI v0.6 write pipeline.

Covers:
- Message understanding (classification, temporary, uncertain, correction, etc.)
- Entity resolution (self, possessive, named third-party)
- Fact extraction (multiple domains, multiple facts, polarity, confidence)
- Operation decisions (create, update, deactivate, supersede)
- Pipeline integration (full write path)
- Real-world scenarios
- Adversarial inputs
"""

import pytest

from src.memory.foundation.understanding import (
    MessageUnderstanding,
    MessageType,
    classify_message_type,
    detect_temporary,
    detect_uncertain,
    detect_third_party,
    detect_correction,
    detect_negation,
    detect_deletion,
    understand_message,
)
from src.memory.foundation.entity_resolution import (
    ResolvedEntity,
    resolve_entity,
)
from src.memory.foundation.extraction import (
    Action,
    ExtractionResult,
    MemoryOperation,
    PatternExtractor,
    _classify_domain,
    _extract_polarity,
    _extract_confidence,
    _split_multiple_facts,
)
from src.memory.foundation.operations import (
    OperationResult,
    apply_operation,
    _resolve_temporal,
)
from src.memory.foundation.pipeline import WritePipeline, WriteResult
from src.memory.foundation.storage import InMemoryStore
from src.memory.foundation.validation import validate_memory
from src.memory.foundation.models import (
    EntityRef,
    Fact,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    TemporalScope,
)
from src.memory.foundation.ids import reset_counters


@pytest.fixture(autouse=True)
def _reset_ids():
    """Reset ID counters before each test."""
    reset_counters()
    yield
    reset_counters()


@pytest.fixture
def store():
    return InMemoryStore()


@pytest.fixture
def pipeline(store):
    return WritePipeline(store)


# ===========================================================================
# UNDERSTANDING TESTS
# ===========================================================================


class TestClassifyMessageType:
    def test_statement(self) -> None:
        assert classify_message_type("I like coffee") == MessageType.STATEMENT

    def test_question_with_suffix(self) -> None:
        assert classify_message_type("What do I like?") == MessageType.QUESTION

    def test_question_with_starter(self) -> None:
        assert classify_message_type("What is my name") == MessageType.QUESTION

    def test_question_how(self) -> None:
        assert classify_message_type("How am I doing?") == MessageType.QUESTION

    def test_greeting(self) -> None:
        assert classify_message_type("Hello!") == MessageType.GREETING

    def test_greeting_hi(self) -> None:
        assert classify_message_type("Hi there") == MessageType.GREETING

    def test_greeting_hey(self) -> None:
        assert classify_message_type("Hey!") == MessageType.GREETING

    def test_command_forget(self) -> None:
        assert classify_message_type("Forget that I like coffee") == MessageType.COMMAND

    def test_command_remember(self) -> None:
        assert classify_message_type("Remember I like tea") == MessageType.COMMAND


class TestDetectTemporary:
    def test_tomorrow(self) -> None:
        is_temp, hint = detect_temporary("I have an exam tomorrow")
        assert is_temp is True
        assert "tomorrow" in (hint or "")

    def test_today(self) -> None:
        is_temp, hint = detect_temporary("I'm hungry today")
        assert is_temp is True

    def test_this_week(self) -> None:
        is_temp, hint = detect_temporary("I have a meeting this week")
        assert is_temp is True

    def test_right_now(self) -> None:
        is_temp, hint = detect_temporary("I'm tired right now")
        assert is_temp is True

    def test_persistent(self) -> None:
        is_temp, hint = detect_temporary("I like Python")
        assert is_temp is False
        assert hint is None

    def test_preparing_for(self) -> None:
        is_temp, hint = detect_temporary("I'm preparing for my exam tomorrow")
        assert is_temp is True


class TestDetectUncertain:
    def test_i_think(self) -> None:
        assert detect_uncertain("I think I might learn Rust") is True

    def test_maybe(self) -> None:
        assert detect_uncertain("Maybe I'll buy a new phone") is True

    def test_i_might(self) -> None:
        assert detect_uncertain("I might be interested in robotics") is True

    def test_not_sure(self) -> None:
        assert detect_uncertain("Not sure if I want to learn Java") is True

    def test_certain(self) -> None:
        assert detect_uncertain("I like coffee") is False

    def test_definitely(self) -> None:
        assert detect_uncertain("I definitely like Python") is False


class TestDetectThirdParty:
    def test_mom(self) -> None:
        assert detect_third_party("Mom likes blue") is True

    def test_my_friend(self) -> None:
        assert detect_third_party("My friend likes Java") is True

    def test_named(self) -> None:
        assert detect_third_party("Alice likes coffee") is True

    def test_self(self) -> None:
        assert detect_third_party("I like coffee") is False

    def test_brother(self) -> None:
        assert detect_third_party("My brother studies CS") is True


class TestDetectCorrection:
    def test_actually(self) -> None:
        assert detect_correction("Actually I prefer Java") is True

    def test_i_no_longer(self) -> None:
        assert detect_correction("I no longer use Samsung") is True

    def test_forget_that(self) -> None:
        assert detect_correction("Forget that I like coffee") is True

    def test_no_correction(self) -> None:
        assert detect_correction("I like coffee") is False


class TestDetectNegation:
    def test_dont_like(self) -> None:
        assert detect_negation("I don't like coffee") is True

    def test_do_not(self) -> None:
        assert detect_negation("I do not want that") is True

    def test_never(self) -> None:
        assert detect_negation("I never liked tea") is True

    def test_no_negation(self) -> None:
        assert detect_negation("I like coffee") is False


class TestDetectDeletion:
    def test_forget_that(self) -> None:
        assert detect_deletion("Forget that I like coffee") is True

    def test_remove_my(self) -> None:
        assert detect_deletion("Remove my coffee preference") is True

    def test_no_deletion(self) -> None:
        assert detect_deletion("I like coffee") is False


class TestUnderstandMessage:
    def test_basic_statement(self) -> None:
        u = understand_message("I like coffee")
        assert u.message_type == MessageType.STATEMENT
        assert u.is_temporary is False
        assert u.is_uncertain is False
        assert u.is_third_party is False

    def test_question(self) -> None:
        u = understand_message("What do I like?")
        assert u.message_type == MessageType.QUESTION

    def test_temporary(self) -> None:
        u = understand_message("I'm studying for my exam tomorrow")
        assert u.is_temporary is True

    def test_uncertain(self) -> None:
        u = understand_message("I think I might learn Rust")
        assert u.is_uncertain is True

    def test_third_party(self) -> None:
        u = understand_message("Mom likes blue")
        assert u.is_third_party is True

    def test_correction(self) -> None:
        u = understand_message("Actually I prefer Java")
        assert u.is_correction is True


# ===========================================================================
# ENTITY RESOLUTION TESTS
# ===========================================================================


class TestResolveEntity:
    def test_self_identification(self) -> None:
        r = resolve_entity("My name is Alice")
        assert r.entity.name == "self"
        assert r.entity.entity_type == "self"
        assert r.resolution_method == "self_identification"

    def test_self_i_am(self) -> None:
        r = resolve_entity("I am a student")
        assert r.entity.name == "self"

    def test_possessive_mom(self) -> None:
        r = resolve_entity("Mom likes blue")
        assert r.entity.name == "mom"
        assert r.entity.entity_type == "person"

    def test_possessive_my_friend(self) -> None:
        r = resolve_entity("My friend likes Java")
        assert r.entity.name == "friend"
        assert r.entity.entity_type == "person"

    def test_named_third_party(self) -> None:
        r = resolve_entity("Alice likes coffee")
        assert r.entity.name == "alice"
        assert r.entity.entity_type == "person"

    def test_self_reference(self) -> None:
        r = resolve_entity("I like coffee")
        assert r.entity.name == "self"
        assert r.resolution_method == "self_reference"

    def test_default_self(self) -> None:
        r = resolve_entity("learning Python")
        assert r.entity.name == "self"
        assert r.resolution_method == "default_self"

    def test_my_dad(self) -> None:
        r = resolve_entity("My dad works at Google")
        assert r.entity.name == "dad"
        assert r.entity.entity_type == "person"


# ===========================================================================
# EXTRACTION TESTS
# ===========================================================================


class TestClassifyDomain:
    def test_learning(self) -> None:
        assert _classify_domain("I am learning Python") == "learning"

    def test_preference(self) -> None:
        assert _classify_domain("I like coffee") == "interests"

    def test_goal(self) -> None:
        assert _classify_domain("I want to become a doctor") == "goals"

    def test_possession(self) -> None:
        assert _classify_domain("I have a MacBook") == "possessions"

    def test_identity(self) -> None:
        assert _classify_domain("My name is Alice") == "identity"

    def test_skill(self) -> None:
        assert _classify_domain("I can code in Python") == "knowledge"


class TestExtractPolarity:
    def test_positive(self) -> None:
        assert _extract_polarity("I like coffee") == "positive"

    def test_negative(self) -> None:
        assert _extract_polarity("I hate coffee") == "negative"

    def test_negated_positive(self) -> None:
        assert _extract_polarity("I don't like coffee") == "negative"

    def test_neutral(self) -> None:
        assert _extract_polarity("I drink coffee") is None


class TestExtractConfidence:
    def test_uncertain(self) -> None:
        assert _extract_confidence("I think I might", True) == 0.3

    def test_maybe(self) -> None:
        c = _extract_confidence("Maybe I'll learn Java", False)
        assert c <= 0.5

    def test_definite(self) -> None:
        c = _extract_confidence("I definitely like Python", False)
        assert c >= 0.9

    def test_default(self) -> None:
        c = _extract_confidence("I like coffee", False)
        assert 0.7 <= c <= 0.9


class TestSplitMultipleFacts:
    def test_single(self) -> None:
        assert len(_split_multiple_facts("I like coffee")) == 1

    def test_two_with_and(self) -> None:
        parts = _split_multiple_facts("I like coffee and I enjoy tea")
        assert len(parts) == 2

    def test_three(self) -> None:
        parts = _split_multiple_facts("I like coffee and tea and juice")
        assert len(parts) >= 2


class TestPatternExtractor:
    def setup_method(self) -> None:
        self.extractor = PatternExtractor()

    def test_learning(self) -> None:
        u = understand_message("I am learning Python")
        r = self.extractor.extract("I am learning Python", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.domain == "learning"
        assert op.fact_type == "learning"
        assert "python" in op.subject.lower()

    def test_preference(self) -> None:
        u = understand_message("I like coffee")
        r = self.extractor.extract("I like coffee", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.domain == "interests"
        assert op.polarity == "positive"

    def test_negative_preference(self) -> None:
        u = understand_message("I hate tea")
        r = self.extractor.extract("I hate tea", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.polarity == "negative"

    def test_goal(self) -> None:
        u = understand_message("I want to become a doctor")
        r = self.extractor.extract("I want to become a doctor", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.domain == "goals"

    def test_possession(self) -> None:
        u = understand_message("I have a MacBook")
        r = self.extractor.extract("I have a MacBook", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.domain == "possessions"

    def test_uncertain(self) -> None:
        u = understand_message("I think I might learn Rust")
        r = self.extractor.extract("I think I might learn Rust", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.is_uncertain is True
        assert op.confidence <= 0.5

    def test_temporary(self) -> None:
        u = understand_message("I'm studying for my exam tomorrow")
        r = self.extractor.extract("I'm studying for my exam tomorrow", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.is_temporary is True

    def test_multiple_facts(self) -> None:
        u = understand_message("I like coffee and I enjoy tea")
        r = self.extractor.extract("I like coffee and I enjoy tea", u)
        assert len(r.operations) >= 2

    def test_deletion(self) -> None:
        u = understand_message("Forget that I like coffee")
        r = self.extractor.extract("Forget that I like coffee", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.action == Action.DEACTIVATE

    def test_correction(self) -> None:
        u = understand_message("Actually I prefer Java now")
        r = self.extractor.extract("Actually I prefer Java now", u)
        assert len(r.operations) >= 1
        op = r.operations[0]
        assert op.action == Action.UPDATE


# ===========================================================================
# OPERATIONS TESTS
# ===========================================================================


class TestResolveTemporal:
    def test_persistent(self) -> None:
        ts = _resolve_temporal(True, None)
        assert ts.is_persistent is True
        assert ts.valid_until is None

    def test_temporary_tomorrow(self) -> None:
        ts = _resolve_temporal(False, "tomorrow")
        assert ts.is_persistent is False
        assert ts.valid_until is not None

    def test_temporary_this_week(self) -> None:
        ts = _resolve_temporal(False, "this week")
        assert ts.is_persistent is False
        assert ts.valid_until is not None


class TestApplyOperation:
    def test_create_new(self, store: InMemoryStore) -> None:
        op = MemoryOperation(
            action=Action.CREATE,
            domain="learning",
            fact_type="learning",
            content="I am learning Python",
            subject="python",
            value="Python",
        )
        result = apply_operation(op, store)
        assert result is not None
        assert result.action_taken == "created"
        assert result.memory is not None
        assert result.memory.entity.name == "self"
        assert result.memory.domain == "learning"
        assert store.count() == 1

    def test_create_duplicate_same_content(self, store: InMemoryStore) -> None:
        op = MemoryOperation(
            action=Action.CREATE,
            domain="learning",
            fact_type="learning",
            content="I am learning Python",
            subject="python",
            value="Python",
        )
        r1 = apply_operation(op, store)
        r2 = apply_operation(op, store)
        assert r2 is not None
        assert r2.action_taken == "unchanged"
        assert store.count() == 1

    def test_create_update_different_content(self, store: InMemoryStore) -> None:
        op1 = MemoryOperation(
            action=Action.CREATE,
            domain="learning",
            fact_type="learning",
            content="I am learning Python",
            subject="python",
            value="Python",
        )
        op2 = MemoryOperation(
            action=Action.CREATE,
            domain="learning",
            fact_type="learning",
            content="I am learning Java now",
            subject="python",
            value="Java",
        )
        r1 = apply_operation(op1, store)
        r2 = apply_operation(op2, store)
        assert r1 is not None
        assert r2 is not None
        assert r2.action_taken == "superseded"
        # Old memory should be inactive
        old = store.get_by_id(r1.memory_id)
        assert old is not None
        assert old.lifecycle.state == "inactive"

    def test_deactivate(self, store: InMemoryStore) -> None:
        op_create = MemoryOperation(
            action=Action.CREATE,
            domain="interests",
            fact_type="preference",
            content="I like coffee",
            subject="coffee",
            value="Coffee",
            polarity="positive",
        )
        apply_operation(op_create, store)
        op_delete = MemoryOperation(
            action=Action.DEACTIVATE,
            domain="interests",
            fact_type="preference",
            content="Forget that I like coffee",
            subject="coffee",
            value="Coffee",
        )
        result = apply_operation(op_delete, store)
        assert result is not None
        assert result.action_taken == "deactivated"
        assert store.get_active() == []

    def test_supersede(self, store: InMemoryStore) -> None:
        op1 = MemoryOperation(
            action=Action.CREATE,
            domain="learning",
            fact_type="learning",
            content="I am learning Python",
            subject="python",
            value="Python",
        )
        r1 = apply_operation(op1, store)
        op2 = MemoryOperation(
            action=Action.UPDATE,
            domain="learning",
            fact_type="learning",
            content="Actually I prefer Java now",
            subject="python",
            value="Java",
        )
        r2 = apply_operation(op2, store)
        assert r2 is not None
        assert r2.action_taken == "superseded"
        # Old should be inactive with superseded_by
        old = store.get_by_id(r1.memory_id)
        assert old is not None
        assert old.lifecycle.superseded_by == r2.memory_id


# ===========================================================================
# PIPELINE INTEGRATION TESTS
# ===========================================================================


class TestWritePipeline:
    def test_basic_statement(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I like coffee")
        assert r.status == "success"
        assert len(r.memories) == 1
        m = r.memories[0]
        assert m.entity.name == "self"
        assert m.domain == "interests"
        assert m.fact.polarity == "positive"

    def test_question_not_written(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("What do I like?")
        assert r.status == "question"
        assert len(r.memories) == 0

    def test_greeting_not_written(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("Hello!")
        assert r.status == "no_op"
        assert len(r.memories) == 0

    def test_learning(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I am learning Python")
        assert r.status == "success"
        assert len(r.memories) == 1
        m = r.memories[0]
        assert m.domain == "learning"
        assert "python" in m.fact.subject.lower()

    def test_goal(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I want to become a doctor")
        assert r.status == "success"
        m = r.memories[0]
        assert m.domain == "goals"

    def test_possession(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I have a MacBook")
        assert r.status == "success"
        m = r.memories[0]
        assert m.domain == "possessions"

    def test_entity_mom(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("Mom likes blue")
        assert r.status == "success"
        m = r.memories[0]
        assert m.entity.name == "mom"
        assert m.entity.entity_type == "person"

    def test_entity_friend(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("My friend likes Java")
        assert r.status == "success"
        m = r.memories[0]
        assert m.entity.name == "friend"
        assert m.entity.entity_type == "person"

    def test_correction(self, pipeline: WritePipeline, store: InMemoryStore) -> None:
        pipeline.process("I like coffee")
        r = pipeline.process("Actually I prefer tea now")
        assert r.status == "success"
        assert len(r.results) > 0

    def test_deletion(self, pipeline: WritePipeline, store: InMemoryStore) -> None:
        pipeline.process("I like coffee")
        r = pipeline.process("Forget that I like coffee")
        assert r.status == "success"
        assert len(r.results) > 0

    def test_negative_preference(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I hate spam")
        assert r.status == "success"
        m = r.memories[0]
        assert m.fact.polarity == "negative"

    def test_uncertain(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I think I might learn Rust")
        assert r.status == "success"
        m = r.memories[0]
        assert m.confidence <= 0.5

    def test_temporary(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I'm studying for my exam tomorrow")
        assert r.status == "success"
        m = r.memories[0]
        assert m.temporal.is_persistent is False

    def test_multiple_facts(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I like coffee and I enjoy tea")
        assert r.status == "success"
        assert len(r.memories) >= 2

    def test_events_produced(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I like coffee")
        assert len(r.events) == 1
        assert r.events[0].event_type == "created"

    def test_debug_trace(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I like coffee")
        assert "input" in r.debug_trace
        assert "understanding" in r.debug_trace
        assert "entity_resolution" in r.debug_trace
        assert "extraction" in r.debug_trace
        assert "persistence" in r.debug_trace

    def test_grammar_mistake(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("i am learn python")
        assert r.status == "success"
        m = r.memories[0]
        assert m.domain == "learning"

    def test_informal(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("learning rust")
        assert r.status == "success"
        m = r.memories[0]
        assert m.domain == "learning"

    def test_slang(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("yo i picked up some Go")
        assert r.status == "success"
        m = r.memories[0]
        assert m.domain == "learning"

    def test_identity(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("My name is Alice")
        assert r.status == "success"
        m = r.memories[0]
        assert m.domain == "identity"

    def test_memory_validated(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I like coffee")
        assert len(r.memories) == 1
        v = validate_memory(r.memories[0])
        assert v.is_valid

    def test_no_validation_errors(self, pipeline: WritePipeline) -> None:
        r = pipeline.process("I like coffee")
        assert r.validation_errors == []

    def test_store_populated(self, pipeline: WritePipeline, store: InMemoryStore) -> None:
        pipeline.process("I like coffee")
        pipeline.process("I am learning Python")
        assert store.count() == 2
