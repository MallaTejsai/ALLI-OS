"""Session 2 tests — Memory Understanding/Extraction.

Tests the complete Session 2 implementation:
- Fast Triage
- Message Understanding with Personal vs General detection
- Enhanced Extraction with temporal interpretation, negation, uncertainty
- Entity Resolution with coreference
- Multi-fact extraction
- General vs Personal boundary
"""

from __future__ import annotations

import pytest

from src.memory.foundation.triage import (
    TriageCategory,
    TriageResult,
    triage,
)
from src.memory.foundation.understanding import (
    ContentScope,
    MessageUnderstanding,
    MessageType,
    classify_message_type,
    detect_temporary,
    detect_uncertain,
    detect_third_party,
    detect_correction,
    detect_negation,
    detect_deletion,
    detect_personal_vs_general,
    understand_message,
)
from src.memory.foundation.extraction import (
    ExtractionResult,
    MemoryOperation,
    PatternExtractor,
    _classify_domain,
    _extract_subject_from_message,
    _split_multiple_facts,
    _interpret_temporal_context,
    _extract_negation_context,
    _extract_uncertainty_context,
)
from src.memory.foundation.entity_resolution import (
    CoreferenceChain,
    CoreferenceResolver,
    ResolvedEntity,
    resolve_entity,
    resolve_coreferences,
)


# ============================================================================
# A. Fast Triage Tests
# ============================================================================

class TestFastTriage:
    """Test fast triage classification."""

    def test_greeting_hello(self):
        result = triage("Hello!")
        assert result.category == TriageCategory.GREETING
        assert result.has_memory_content is False
        assert result.has_instruction is False
        assert result.confidence >= 0.9

    def test_greeting_hi(self):
        result = triage("Hi there!")
        assert result.category == TriageCategory.GREETING

    def test_greeting_yo(self):
        result = triage("Yo!")
        assert result.category == TriageCategory.GREETING

    def test_greeting_good_morning(self):
        result = triage("Good morning!")
        assert result.category == TriageCategory.GREETING

    def test_greeting_whats_up(self):
        result = triage("What's up!")
        assert result.category == TriageCategory.GREETING

    def test_acknowledgement_ok(self):
        result = triage("Ok")
        assert result.category == TriageCategory.ACKNOWLEDGEMENT
        assert result.has_memory_content is False

    def test_acknowledgement_sure(self):
        result = triage("Sure!")
        assert result.category == TriageCategory.ACKNOWLEDGEMENT

    def test_acknowledgement_got_it(self):
        result = triage("Got it")
        assert result.category == TriageCategory.ACKNOWLEDGEMENT

    def test_acknowledgement_thanks(self):
        result = triage("Thanks!")
        assert result.category == TriageCategory.ACKNOWLEDGEMENT

    def test_acknowledgement_cool(self):
        result = triage("Cool")
        assert result.category == TriageCategory.ACKNOWLEDGEMENT

    def test_general_knowledge_what_is(self):
        result = triage("What is Python?")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE
        assert result.has_memory_content is False

    def test_general_knowledge_how_does(self):
        result = triage("How does recursion work?")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_general_knowledge_who_is(self):
        result = triage("Who is Albert Einstein?")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_question_personal(self):
        result = triage("What is my favorite color?")
        assert result.category == TriageCategory.QUESTION
        assert result.has_memory_content is True

    def test_question_ending(self):
        result = triage("Do you like coffee?")
        assert result.category == TriageCategory.QUESTION

    def test_memory_statement_i_like(self):
        result = triage("I like coffee")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_memory_statement_i_learn(self):
        result = triage("I am learning Python")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_memory_statement_i_have(self):
        result = triage("I have a MacBook Pro")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_memory_statement_with_instruction(self):
        result = triage("Remember that I like tea")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True
        assert result.has_instruction is True

    def test_memory_statement_forget(self):
        result = triage("Forget about my old phone")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_instruction is True

    def test_conversational_bye(self):
        result = triage("Bye!")
        assert result.category == TriageCategory.CONVERSATIONAL

    def test_conversational_lets_talk(self):
        result = triage("Let's chat")
        assert result.category == TriageCategory.CONVERSATIONAL

    def test_malformed_empty(self):
        result = triage("")
        assert result.category == TriageCategory.MALFORMED

    def test_malformed_single_char(self):
        result = triage("x")
        assert result.category == TriageCategory.MALFORMED

    def test_malformed_only_punctuation(self):
        result = triage("???")
        assert result.category == TriageCategory.MALFORMED

    def test_malformed_only_numbers(self):
        result = triage("12345")
        assert result.category == TriageCategory.MALFORMED

    def test_mixed_conversational_with_memory(self):
        result = triage("bro I like coffee")
        assert result.category == TriageCategory.MIXED
        assert result.has_memory_content is True

    def test_mixed_love_with_memory(self):
        result = triage("lol I love programming")
        assert result.category == TriageCategory.MIXED
        assert result.has_memory_content is True

    def test_general_python(self):
        result = triage("Tell me about Python")
        assert result.category == TriageCategory.QUESTION

    def test_general_algorithm(self):
        result = triage("What is an algorithm?")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_question_tell_me_about(self):
        result = triage("Tell me about yourself?")
        assert result.category == TriageCategory.QUESTION


# ============================================================================
# B. Message Understanding Tests
# ============================================================================

class TestMessageUnderstanding:
    """Test message understanding classification."""

    def test_statement_classification(self):
        result = understand_message("I like coffee")
        assert result.message_type == MessageType.STATEMENT

    def test_question_classification(self):
        result = understand_message("Do you like coffee?")
        assert result.message_type == MessageType.QUESTION

    def test_command_classification(self):
        result = understand_message("Remember that I like tea")
        assert result.message_type == MessageType.COMMAND

    def test_greeting_classification(self):
        result = understand_message("Hello!")
        assert result.message_type == MessageType.GREETING

    def test_temporary_detection(self):
        result = understand_message("I'm learning Python for now")
        assert result.is_temporary is True

    def test_uncertain_detection(self):
        result = understand_message("I think I like coffee")
        assert result.is_uncertain is True

    def test_third_party_detection(self):
        result = understand_message("My friend likes tea")
        assert result.is_third_party is True

    def test_correction_detection(self):
        result = understand_message("Actually, I like tea, not coffee")
        assert result.is_correction is True

    def test_negation_detection(self):
        result = understand_message("I don't like coffee")
        assert result.is_negation is True

    def test_deletion_detection(self):
        result = understand_message("Forget about my old phone")
        assert result.is_deletion is True

    def test_temporal_hint_past(self):
        result = understand_message("I used to like coffee")
        assert result.temporal_hint is not None

    def test_personal_content(self):
        result = understand_message("I like coffee")
        assert result.is_personal is True
        assert result.is_general is False

    def test_general_content(self):
        result = understand_message("What is Python?")
        assert result.is_general is True
        assert result.is_personal is False


# ============================================================================
# C. Temporal Interpretation Tests
# ============================================================================

class TestTemporalInterpretation:
    """Test temporal context interpretation."""

    def test_historical_used_to(self):
        context = _interpret_temporal_context("I used to like coffee")
        assert context == "historical"

    def test_historical_was(self):
        context = _interpret_temporal_context("I was learning Python")
        assert context == "historical"

    def test_historical_before(self):
        context = _interpret_temporal_context("Before I liked tea")
        assert context == "historical"

    def test_current_now(self):
        context = _interpret_temporal_context("I currently like coffee")
        assert context == "current"

    def test_current_i_am(self):
        context = _interpret_temporal_context("I am learning Python")
        assert context == "current"

    def test_future_will(self):
        context = _interpret_temporal_context("I will learn Rust")
        assert context == "future"

    def test_future_going_to(self):
        context = _interpret_temporal_context("I'm going to learn Go")
        assert context == "future"

    def test_future_plan_to(self):
        context = _interpret_temporal_context("I plan to learn C++")
        assert context == "future"

    def test_stopped_no_longer(self):
        context = _interpret_temporal_context("I no longer like coffee")
        assert context == "stopped"

    def test_stopped_dont_anymore(self):
        context = _interpret_temporal_context("I don't like coffee anymore")
        assert context == "stopped"

    def test_stopped_quit(self):
        context = _interpret_temporal_context("I quit smoking")
        assert context == "stopped"


# ============================================================================
# D. Negation Handling Tests
# ============================================================================

class TestNegationHandling:
    """Test negation context extraction."""

    def test_explicit_negation_dont(self):
        context = _extract_negation_context("I don't like coffee")
        assert context["is_negated"] is True
        assert context["negation_type"] == "explicit"

    def test_explicit_negation_never(self):
        context = _extract_negation_context("I never drink tea")
        assert context["is_negated"] is True

    def test_explicit_negation_not(self):
        context = _extract_negation_context("I am not interested in Java")
        assert context["is_negated"] is True

    def test_no_negation(self):
        context = _extract_negation_context("I like coffee")
        assert context["is_negated"] is False

    def test_negation_scope(self):
        context = _extract_negation_context("I don't like coffee")
        assert context["negation_scope"] is not None


# ============================================================================
# E. Uncertainty Handling Tests
# ============================================================================

class TestUncertaintyHandling:
    """Test uncertainty context extraction."""

    def test_uncertain_i_think(self):
        context = _extract_uncertainty_context("I think I like coffee")
        assert context["is_uncertain"] is True
        assert context["uncertainty_level"] == "medium"

    def test_uncertain_maybe(self):
        context = _extract_uncertainty_context("Maybe I want tea")
        assert context["is_uncertain"] is True
        assert context["uncertainty_type"] == "speculative"

    def test_uncertain_not_sure(self):
        context = _extract_uncertainty_context("I'm not sure if I like coffee")
        assert context["is_uncertain"] is True
        assert context["uncertainty_level"] == "high"

    def test_hedged_i_might(self):
        context = _extract_uncertainty_context("I might want coffee")
        assert context["is_uncertain"] is True
        assert context["uncertainty_type"] == "hedged"

    def test_no_uncertainty(self):
        context = _extract_uncertainty_context("I like coffee")
        assert context["is_uncertain"] is False

    def test_uncertain_i_guess(self):
        context = _extract_uncertainty_context("I guess I like tea")
        assert context["is_uncertain"] is True
        assert context["uncertainty_level"] == "high"

    def test_uncertain_probably(self):
        context = _extract_uncertainty_context("Probably I like coffee")
        assert context["is_uncertain"] is True


# ============================================================================
# F. Multi-fact Extraction Tests
# ============================================================================

class TestMultiFactExtraction:
    """Test splitting messages into multiple facts."""

    def test_single_fact(self):
        facts = _split_multiple_facts("I like coffee")
        assert len(facts) == 1
        assert facts[0] == "I like coffee"

    def test_two_facts_with_and(self):
        facts = _split_multiple_facts("I like coffee and I like tea")
        assert len(facts) == 2

    def test_two_facts_with_also(self):
        facts = _split_multiple_facts("I like coffee also I like tea")
        assert len(facts) == 2

    def test_three_facts(self):
        facts = _split_multiple_facts("I like coffee and I like tea and I like juice")
        assert len(facts) == 3

    def test_fact_with_plus(self):
        facts = _split_multiple_facts("I like coffee plus I like tea")
        assert len(facts) == 2

    def test_no_split_without_conjunction(self):
        facts = _split_multiple_facts("I really like coffee")
        assert len(facts) == 1


# ============================================================================
# G. General vs Personal Boundary Tests
# ============================================================================

class TestGeneralVsPersonal:
    """Test general vs personal content detection."""

    def test_personal_i_like(self):
        is_personal, is_general = detect_personal_vs_general("I like coffee")
        assert is_personal is True
        assert is_general is False

    def test_personal_my_phone(self):
        is_personal, is_general = detect_personal_vs_general("My phone is broken")
        assert is_personal is True
        assert is_general is False

    def test_general_what_is(self):
        is_personal, is_general = detect_personal_vs_general("What is Python?")
        assert is_general is True
        assert is_personal is False

    def test_general_who_is(self):
        is_personal, is_general = detect_personal_vs_general("Who is Albert Einstein?")
        assert is_general is True
        assert is_personal is False

    def test_general_how_does(self):
        is_personal, is_general = detect_personal_vs_general("How does recursion work?")
        assert is_general is True
        assert is_personal is False

    def test_personal_experience_with_topic(self):
        is_personal, is_general = detect_personal_vs_general("I use Python for web development")
        assert is_personal is True
        assert is_general is False


# ============================================================================
# H. Entity Resolution Tests
# ============================================================================

class TestEntityResolution:
    """Test entity resolution."""

    def test_self_identification(self):
        result = resolve_entity("My name is John")
        assert result.entity.name == "self"
        assert result.resolution_method == "self_identification"
        assert result.confidence == 1.0

    def test_possessive_third_party(self):
        result = resolve_entity("My mom likes tea")
        assert result.entity.name == "mom"
        assert result.entity.entity_type == "person"
        assert result.resolution_method == "possessive_third_party"

    def test_possessive_third_party_with_name(self):
        result = resolve_entity("My friend Jake likes coffee")
        assert result.entity.name == "jake"
        assert result.entity.entity_type == "person"

    def test_named_third_party(self):
        result = resolve_entity("Alice likes coffee")
        assert result.entity.name == "alice"
        assert result.entity.entity_type == "person"
        assert result.resolution_method == "named_third_party"

    def test_self_reference(self):
        result = resolve_entity("I like coffee")
        assert result.entity.name == "self"
        assert result.resolution_method == "self_reference"

    def test_default_self(self):
        result = resolve_entity("Likes coffee")
        assert result.entity.name == "self"
        assert result.resolution_method == "default_self"


# ============================================================================
# I. Coreference Resolution Tests
# ============================================================================

class TestCoreferenceResolution:
    """Test coreference resolution."""

    def test_resolver_initialization(self):
        resolver = CoreferenceResolver()
        assert resolver.chains == {}
        assert resolver.mention_order == []
        assert resolver.current_entity is None

    def test_add_mention(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        assert "alice" in resolver.chains
        assert resolver.current_entity == "alice"

    def test_resolve_pronoun(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        resolved = resolver.resolve_pronoun("she", "Alice likes coffee")
        assert resolved == "alice"

    def test_resolve_demonstrative(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("the new phone", "phone", "object")
        resolved = resolver.resolve_demonstrative("this", "This is great")
        assert resolved == "the new phone"

    def test_resolve_relative(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        resolved = resolver.resolve_relative("who", "Alice, who likes coffee")
        assert resolved == "alice"

    def test_reset(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        resolver.reset()
        assert resolver.chains == {}
        assert resolver.mention_order == []

    def test_get_all_mentions(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        resolver.add_mention("Bob", "bob", "person")
        mentions = resolver.get_all_mentions()
        assert "alice" in mentions
        assert "bob" in mentions

    def test_get_mention_count(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        resolver.add_mention("she", "alice", "person")
        count = resolver.get_mention_count("alice")
        assert count == 2

    def test_resolve_coreferences_in_message(self):
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        resolutions = resolve_coreferences("She likes coffee", resolver)
        assert len(resolutions) > 0


# ============================================================================
# J. Domain Classification Tests
# ============================================================================

class TestDomainClassification:
    """Test domain classification for extraction."""

    def test_learning_domain(self):
        domain = _classify_domain("I am learning Python")
        assert domain == "learning"

    def test_interests_domain(self):
        domain = _classify_domain("I like coffee")
        assert domain == "interests"

    def test_goals_domain(self):
        domain = _classify_domain("I aim to become a developer")
        assert domain == "goals"

    def test_possessions_domain(self):
        domain = _classify_domain("I have a MacBook Pro")
        assert domain == "possessions"

    def test_identity_domain(self):
        domain = _classify_domain("My name is John")
        assert domain == "identity"

    def test_knowledge_domain(self):
        domain = _classify_domain("I know Python")
        assert domain == "knowledge"

    def test_general_domain(self):
        domain = _classify_domain("Something happened")
        assert domain == "general"


# ============================================================================
# K. Subject Extraction Tests
# ============================================================================

class TestSubjectExtraction:
    """Test subject extraction from messages."""

    def test_subject_from_learning(self):
        subject = _extract_subject_from_message("I am learning Python", "learning")
        assert subject is not None
        assert len(subject) > 0

    def test_subject_from_interests(self):
        subject = _extract_subject_from_message("I like coffee", "interests")
        assert subject is not None
        assert len(subject) > 0

    def test_subject_from_possessions(self):
        subject = _extract_subject_from_message("I have a MacBook Pro", "possessions")
        assert subject is not None
        assert len(subject) > 0

    def test_subject_from_identity(self):
        subject = _extract_subject_from_message("My name is John", "identity")
        assert subject is not None
        assert len(subject) > 0


# ============================================================================
# L. Extraction Pipeline Tests
# ============================================================================

class TestExtractionPipeline:
    """Test the complete extraction pipeline."""

    def test_simple_extraction(self):
        extractor = PatternExtractor()
        understanding = understand_message("I like coffee")
        result = extractor.extract("I like coffee", understanding)
        assert isinstance(result, ExtractionResult)
        assert len(result.operations) > 0

    def test_extraction_with_temporal(self):
        extractor = PatternExtractor()
        understanding = understand_message("I used to like coffee")
        result = extractor.extract("I used to like coffee", understanding)
        assert result.temporal_hint is not None

    def test_extraction_with_negation(self):
        extractor = PatternExtractor()
        understanding = understand_message("I don't like coffee")
        result = extractor.extract("I don't like coffee", understanding)
        assert result.operations[0].polarity == "negative"

    def test_extraction_with_uncertainty(self):
        extractor = PatternExtractor()
        understanding = understand_message("I think I like coffee")
        result = extractor.extract("I think I like coffee", understanding)
        assert result.is_uncertain is True

    def test_extraction_multi_fact(self):
        extractor = PatternExtractor()
        understanding = understand_message("I like coffee and I like tea")
        result = extractor.extract("I like coffee and I like tea", understanding)
        assert len(result.operations) >= 2

    def test_extraction_deletion(self):
        extractor = PatternExtractor()
        understanding = understand_message("Forget about my old phone")
        result = extractor.extract("Forget about my old phone", understanding)
        assert result.is_deletion is True
        assert result.operations[0].action == "DEACTIVATE"

    def test_extraction_correction(self):
        extractor = PatternExtractor()
        understanding = understand_message("Actually, I like tea, not coffee")
        result = extractor.extract("Actually, I like tea, not coffee", understanding)
        assert result.is_correction is True
        assert result.operations[0].action == "UPDATE"


# ============================================================================
# M. Confidence Extraction Tests
# ============================================================================

class TestConfidenceExtraction:
    """Test confidence level extraction."""

    def test_low_confidence_uncertain(self):
        from src.memory.foundation.extraction import _extract_confidence
        confidence = _extract_confidence("I think I like coffee", True)
        assert confidence == 0.3

    def test_medium_confidence_maybe(self):
        from src.memory.foundation.extraction import _extract_confidence
        confidence = _extract_confidence("Maybe I like coffee", False)
        assert confidence == 0.5

    def test_high_confidence_definitely(self):
        from src.memory.foundation.extraction import _extract_confidence
        confidence = _extract_confidence("I definitely like coffee", False)
        assert confidence == 0.95

    def test_default_confidence(self):
        from src.memory.foundation.extraction import _extract_confidence
        confidence = _extract_confidence("I like coffee", False)
        assert confidence == 0.8


# ============================================================================
# N. Polarity Extraction Tests
# ============================================================================

class TestPolarityExtraction:
    """Test polarity extraction from messages."""

    def test_positive_polarity(self):
        from src.memory.foundation.extraction import _extract_polarity
        polarity = _extract_polarity("I love coffee")
        assert polarity == "positive"

    def test_negative_polarity(self):
        from src.memory.foundation.extraction import _extract_polarity
        polarity = _extract_polarity("I hate coffee")
        assert polarity == "negative"

    def test_neutral_polarity(self):
        from src.memory.foundation.extraction import _extract_polarity
        polarity = _extract_polarity("I drink coffee")
        assert polarity is None


# ============================================================================
# O. Adversarial / Edge Case Tests
# ============================================================================

class TestAdversarialEdgeCases:
    """Test adversarial and edge cases."""

    def test_empty_message_triage(self):
        result = triage("")
        assert result.category == TriageCategory.MALFORMED

    def test_very_long_message(self):
        long_msg = "I like coffee " * 100
        result = triage(long_msg)
        assert result.category in [
            TriageCategory.MEMORY_STATEMENT,
            TriageCategory.MIXED,
            TriageCategory.CONVERSATIONAL,
        ]

    def test_unicode_message(self):
        result = triage("I like café")
        assert result.category in [
            TriageCategory.MEMORY_STATEMENT,
            TriageCategory.CONVERSATIONAL,
        ]

    def test_special_characters(self):
        result = triage("I like coffee!!!")
        assert result.category == TriageCategory.MEMORY_STATEMENT

    def test_multiple_questions(self):
        result = triage("What is Python? What is Java?")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_mixed_personal_general(self):
        result = triage("I use Python and what is Java?")
        assert result.category in [
            TriageCategory.MIXED,
            TriageCategory.MEMORY_STATEMENT,
            TriageCategory.QUESTION,
        ]

    def test_temporary_with_uncertainty(self):
        understanding = understand_message("I think I'm learning Python for now")
        assert understanding.is_temporary is True
        assert understanding.is_uncertain is True

    def test_negation_with_uncertainty(self):
        understanding = understand_message("I don't think I like coffee")
        assert understanding.is_negation is True


# ============================================================================
# P. Integration Tests
# ============================================================================

class TestIntegration:
    """Test integration of Session 2 components."""

    def test_triage_to_understanding(self):
        """Test pipeline from triage to understanding."""
        triage_result = triage("I like coffee")
        understanding = understand_message("I like coffee")
        
        assert triage_result.has_memory_content is True
        assert understanding.is_personal is True
        assert understanding.message_type == MessageType.STATEMENT

    def test_triage_to_extraction(self):
        """Test pipeline from triage to extraction."""
        extractor = PatternExtractor()
        understanding = understand_message("I like coffee")
        result = extractor.extract("I like coffee", understanding)
        
        assert len(result.operations) > 0
        assert result.operations[0].domain == "interests"

    def test_entity_to_extraction(self):
        """Test entity resolution with extraction."""
        entity = resolve_entity("My friend Jake likes coffee")
        extractor = PatternExtractor()
        understanding = understand_message("My friend Jake likes coffee")
        result = extractor.extract("My friend Jake likes coffee", understanding)
        
        assert entity.entity.name == "jake"
        assert entity.entity.entity_type == "person"

    def test_full_pipeline(self):
        """Test complete Session 2 pipeline."""
        # Triage
        triage_result = triage("I like coffee")
        assert triage_result.has_memory_content is True
        
        # Understanding
        understanding = understand_message("I like coffee")
        assert understanding.is_personal is True
        
        # Entity resolution
        entity = resolve_entity("I like coffee")
        assert entity.entity.name == "self"
        
        # Extraction
        extractor = PatternExtractor()
        result = extractor.extract("I like coffee", understanding)
        assert len(result.operations) > 0

    def test_temporal_pipeline(self):
        """Test temporal interpretation through pipeline."""
        understanding = understand_message("I used to like coffee")
        extractor = PatternExtractor()
        result = extractor.extract("I used to like coffee", understanding)
        
        assert result.temporal_hint is not None

    def test_negation_pipeline(self):
        """Test negation handling through pipeline."""
        understanding = understand_message("I don't like coffee")
        extractor = PatternExtractor()
        result = extractor.extract("I don't like coffee", understanding)
        
        assert result.operations[0].polarity == "negative"

    def test_coreference_pipeline(self):
        """Test coreference resolution through pipeline."""
        resolver = CoreferenceResolver()
        resolver.add_mention("Alice", "alice", "person")
        
        entity = resolve_entity("Alice likes coffee")
        assert entity.entity.name == "alice"
        
        resolutions = resolve_coreferences("She likes tea", resolver)
        assert len(resolutions) > 0


# ============================================================================
# P. Unseen Behavioral Certification (FIX 1-4)
# ============================================================================

class TestFix1PersonalPronouns:
    """FIX 1: Personal triage pronouns (we/us/our)."""

    def test_we_statement_memory(self):
        """'We...' can contain personal memory content."""
        result = triage("We moved into our new apartment yesterday")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_our_statement_memory(self):
        """'Our...' can contain personal memory content."""
        result = triage("Our team won the championship last night")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_us_statement_memory(self):
        """'Us...' can occur in personal statements."""
        result = triage("Us graduates should celebrate together")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_we_planning_memory(self):
        """'We are planning...' is personal memory."""
        result = triage("We are planning a trip to Japan next summer")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True

    def test_our_pet_memory(self):
        """'Our pet...' is personal memory."""
        result = triage("Our cat learned a new trick yesterday")
        assert result.category == TriageCategory.MEMORY_STATEMENT
        assert result.has_memory_content is True


class TestFix2GeneralKnowledgePatterns:
    """FIX 2: General knowledge question patterns (why/who/how much)."""

    def test_why_factual(self):
        """'Why is the sky blue' → general_knowledge."""
        result = triage("Why is the sky blue")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_who_factual(self):
        """'Who invented the telephone' → general_knowledge."""
        result = triage("Who invented the telephone")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_how_much_factual(self):
        """'How much does a python weigh' → general_knowledge."""
        result = triage("How much does a python weigh")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE

    def test_why_personal(self):
        """'Why do I like Java' → question (personal)."""
        result = triage("Why do I like Java")
        assert result.category == TriageCategory.QUESTION
        assert result.has_memory_content is True

    def test_who_personal(self):
        """'Who is my mom' → question (personal)."""
        result = triage("Who is my mom")
        assert result.category == TriageCategory.QUESTION
        assert result.has_memory_content is True

    def test_how_much_personal(self):
        """'How much do I weigh' → question (personal)."""
        result = triage("How much do I weigh")
        assert result.category == TriageCategory.QUESTION
        assert result.has_memory_content is True

    def test_why_our_personal(self):
        """'Why do we need this' → question (personal)."""
        result = triage("Why do we need this")
        assert result.category == TriageCategory.QUESTION
        assert result.has_memory_content is True

    def test_who_was_factual(self):
        """'Who was the first president' → general_knowledge."""
        result = triage("Who was the first president")
        assert result.category == TriageCategory.GENERAL_KNOWLEDGE


class TestFix3PersonalVsGeneral:
    """FIX 3: Personal vs general default for factual world statements."""

    def test_great_wall_general(self):
        """'The Great Wall of China is very long' → general."""
        is_personal, is_general = detect_personal_vs_general(
            "The Great Wall of China is very long"
        )
        assert is_personal is False
        assert is_general is True

    def test_water_boils_general(self):
        """'Water boils at 100 degrees Celsius' → general."""
        is_personal, is_general = detect_personal_vs_general(
            "Water boils at 100 degrees Celsius"
        )
        assert is_personal is False
        assert is_general is True

    def test_python_created_general(self):
        """'Python was created by Guido van Rossum' → general."""
        is_personal, is_general = detect_personal_vs_general(
            "Python was created by Guido van Rossum"
        )
        assert is_personal is False
        assert is_general is True

    def test_triangle_general(self):
        """'A triangle has three sides' → general."""
        is_personal, is_general = detect_personal_vs_general(
            "A triangle has three sides"
        )
        assert is_personal is False
        assert is_general is True

    def test_atom_general(self):
        """'An atom contains protons and neutrons' → general."""
        is_personal, is_general = detect_personal_vs_general(
            "An atom contains protons and neutrons"
        )
        assert is_personal is False
        assert is_general is True

    def test_i_visited_personal(self):
        """'I visited the Great Wall last summer' → personal."""
        is_personal, is_general = detect_personal_vs_general(
            "I visited the Great Wall of China last summer"
        )
        assert is_personal is True
        assert is_general is False

    def test_we_moved_personal(self):
        """'We moved into our new apartment' → personal."""
        is_personal, is_general = detect_personal_vs_general(
            "We moved into our new apartment"
        )
        assert is_personal is True
        assert is_general is False

    def test_weather_unknown(self):
        """'The weather is nice today' → unknown (casual observation)."""
        is_personal, is_general = detect_personal_vs_general(
            "The weather is nice today"
        )
        assert is_personal is False
        assert is_general is False

    def test_music_enjoyable_unknown(self):
        """'Music is enjoyable' → unknown (opinion)."""
        is_personal, is_general = detect_personal_vs_general(
            "Music is enjoyable"
        )
        assert is_personal is False
        assert is_general is False

    def test_cars_faster_unknown(self):
        """'Cars are faster than bicycles' → unknown (no clear signal)."""
        is_personal, is_general = detect_personal_vs_general(
            "Cars are faster than bicycles"
        )
        assert is_personal is False
        assert is_general is False


class TestFix4Completeness:
    """FIX 4: Structural completeness check."""

    def test_incomplete_missing_object(self):
        """'I like' → incomplete (missing object)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("I like")
        result = extractor.extract("I like", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.INCOMPLETE

    def test_incomplete_missing_complement(self):
        """'My favorite is' → incomplete (missing complement)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("My favorite is")
        result = extractor.extract("My favorite is", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.INCOMPLETE

    def test_incomplete_missing_goal(self):
        """'I want to' → incomplete (missing goal)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("I want to")
        result = extractor.extract("I want to", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.INCOMPLETE

    def test_ambiguous_it_nice(self):
        """'It is nice' → ambiguous (vague subject)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("It is nice")
        result = extractor.extract("It is nice", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.AMBIGUOUS

    def test_ambiguous_that_good(self):
        """'That was good' → ambiguous (vague subject)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("That was good")
        result = extractor.extract("That was good", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.AMBIGUOUS

    def test_complete_preference(self):
        """'I love my mom' → complete (clear preference)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("I love my mom")
        result = extractor.extract("I love my mom", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.COMPLETE

    def test_complete_usage(self):
        """'I use Python every day' → complete (clear usage)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("I use Python every day")
        result = extractor.extract("I use Python every day", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.COMPLETE

    def test_complete_goal(self):
        """'I want to learn Japanese' → complete (clear goal)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("I want to learn Japanese")
        result = extractor.extract("I want to learn Japanese", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.COMPLETE

    def test_complete_factual(self):
        """'The Earth orbits the Sun' → complete (factual statement)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("The Earth orbits the Sun")
        result = extractor.extract("The Earth orbits the Sun", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.COMPLETE

    def test_complete_identity(self):
        """'My dog is named Rex' → complete (clear identity)."""
        from src.memory.foundation.extraction import check_completeness, CompletenessLevel
        extractor = PatternExtractor()
        understanding = understand_message("My dog is named Rex")
        result = extractor.extract("My dog is named Rex", understanding)
        assert len(result.operations) > 0
        level = check_completeness(result.operations[0])
        assert level == CompletenessLevel.COMPLETE
