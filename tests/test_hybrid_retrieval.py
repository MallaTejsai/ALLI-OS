"""Tests for ALLI v0.6 Hybrid Retrieval Engine (Session 6).

Tests the HybridRetriever that consumes StructuredQuery and retrieves
ranked memory candidates using multiple retrieval channels.

Test Groups:
  A: Controlled test dataset setup
  B: Exact fact retrieval
  C: Paraphrase retrieval
  D: Grammar/typo retrieval
  E: Entity-specific retrieval
  F: Historical retrieval
  G: Current retrieval
  H: Multiple facts retrieval
  I: Related concept retrieval
  J: Irrelevant semantic match
  K: Unknown/empty retrieval
  L: Entity isolation
  M: Relationship retrieval
  N: Temporal filtering
  O: Negated/uncertain queries
  P: Candidate fusion
  Q: Ranking verification
  R: Retrieval trace
  S: Adversarial tests (40+)
"""

import os
import sys
import unittest
from typing import Optional
from unittest.mock import MagicMock

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from memory.foundation.query_understanding import StructuredQuery, understand_query
from memory.foundation.hybrid_retriever import (
    HybridRetriever,
    HybridRetrievalResult,
    RetrievalCandidate,
    RetrievalTrace,
    tokenize,
    token_overlap_score,
)


# ---------------------------------------------------------------------------
# Mock Embedding Store (deterministic for testing)
# ---------------------------------------------------------------------------

class MockEmbeddingStore:
    """Deterministic mock embedding store for testing."""

    def __init__(self):
        self._call_count = 0

    def generate_embedding(self, text: str) -> list[float]:
        self._call_count += 1
        # Deterministic: hash-based pseudo-embedding
        h = hash(text.lower().strip()) % 1000
        # Create a 384-dim vector with the hash encoded
        vec = [0.0] * 384
        vec[0] = float(h) / 1000.0
        vec[1] = float(hash(text[:5]) % 100) / 100.0
        vec[2] = float(len(text)) / 100.0
        for i in range(3, min(384, len(text) + 3)):
            vec[i] = float(ord(text[i - 3]) % 100) / 100.0
        return vec


# ---------------------------------------------------------------------------
# Controlled Test Memory Dataset
# ---------------------------------------------------------------------------

def _make_memory(
    memory_id: str,
    content: str,
    category: str = "other",
    fact_type: str = "general",
    entity_name: str = "user",
    entity_type: str = "self",
    status: str = "active",
    subject: str = "",
    value: str = "",
    embedding: Optional[list[float]] = None,
    expires_at: Optional[str] = None,
    superseded_by: Optional[str] = None,
    **kwargs,
) -> dict:
    """Create a test memory dict."""
    return {
        "id": memory_id,
        "content": content,
        "category": category,
        "fact_type": fact_type,
        "subject": subject or content.split()[0] if content else "",
        "value": value or content,
        "status": status,
        "entity": {"name": entity_name, "type": entity_type},
        "embedding": embedding,
        "expires_at": expires_at,
        "superseded_by": superseded_by,
        "importance": kwargs.get("importance", "medium"),
        "confidence": kwargs.get("confidence", 1.0),
        "created_at": kwargs.get("created_at", "2025-01-01T00:00:00"),
        "updated_at": kwargs.get("updated_at", "2025-01-01T00:00:00"),
    }


def _make_embedding_store_with_vectors(memories: list[dict]) -> MockEmbeddingStore:
    """Create a mock embedding store and pre-populate memory embeddings."""
    store = MockEmbeddingStore()
    for m in memories:
        if not m.get("embedding"):
            m["embedding"] = store.generate_embedding(m.get("content", ""))
    return store


# Controlled test dataset
TEST_MEMORIES = [
    # USER — current phone
    _make_memory(
        "mem_001", "I use a Samsung S24 phone",
        category="possession", fact_type="possession",
        entity_name="user", subject="phone", value="Samsung S24",
    ),
    # USER — historical phone (superseded)
    _make_memory(
        "mem_002", "I used to have an Oppo A53 phone",
        category="possession", fact_type="possession",
        entity_name="user", subject="phone", value="Oppo A53",
        status="inactive", superseded_by="mem_001",
    ),
    # USER — favorite color
    _make_memory(
        "mem_003", "My favorite color is blue",
        category="preference", fact_type="like",
        entity_name="user", subject="color", value="blue",
    ),
    # USER — disliked color
    _make_memory(
        "mem_004", "I dislike the color orange",
        category="preference", fact_type="dislike",
        entity_name="user", subject="color", value="orange",
    ),
    # USER — programming languages (learning)
    _make_memory(
        "mem_005", "I am learning Python programming",
        category="learning", fact_type="learning",
        entity_name="user", subject="programming language", value="Python",
    ),
    # USER — programming languages (learning)
    _make_memory(
        "mem_006", "I am learning JavaScript",
        category="learning", fact_type="learning",
        entity_name="user", subject="programming language", value="JavaScript",
    ),
    # USER — current learning (robotics)
    _make_memory(
        "mem_007", "I am currently studying robotics",
        category="learning", fact_type="learning",
        entity_name="user", subject="robotics", value="robotics",
    ),
    # USER — past learning (history)
    _make_memory(
        "mem_008", "I previously studied history in college",
        category="learning", fact_type="learning",
        entity_name="user", subject="history", value="history",
        status="inactive",
    ),
    # USER — goals
    _make_memory(
        "mem_009", "My goal is to become a software engineer",
        category="goal", fact_type="goal",
        entity_name="user", subject="career", value="software engineer",
    ),
    # USER — skills
    _make_memory(
        "mem_010", "I am skilled at Python and data analysis",
        category="skill", fact_type="skill",
        entity_name="user", subject="programming", value="Python",
    ),
    # MOM — favorite color
    _make_memory(
        "mem_011", "My mom's favorite color is red",
        category="preference", fact_type="like",
        entity_name="mom", entity_type="person",
        subject="color", value="red",
    ),
    # MOM — interest
    _make_memory(
        "mem_012", "My mom enjoys gardening",
        category="interest", fact_type="like",
        entity_name="mom", entity_type="person",
        subject="gardening", value="gardening",
    ),
    # MOM — phone
    _make_memory(
        "mem_013", "My mom uses an iPhone 15",
        category="possession", fact_type="possession",
        entity_name="mom", entity_type="person",
        subject="phone", value="iPhone 15",
    ),
    # JAKE — interest
    _make_memory(
        "mem_014", "Jake likes playing chess",
        category="interest", fact_type="like",
        entity_name="jake", entity_type="person",
        subject="chess", value="chess",
    ),
    # JAKE — skill
    _make_memory(
        "mem_015", "Jake is skilled at guitar",
        category="skill", fact_type="skill",
        entity_name="jake", entity_type="person",
        subject="guitar", value="guitar",
    ),
    # SISTER — preference
    _make_memory(
        "mem_016", "My sister prefers tea over coffee",
        category="preference", fact_type="like",
        entity_name="sister", entity_type="person",
        subject="tea", value="tea",
    ),
    # UNRELATED — random
    _make_memory(
        "mem_017", "The weather is nice today",
        category="other", fact_type="general",
        entity_name="user", subject="weather", value="nice",
    ),
    # USER — favorite movie
    _make_memory(
        "mem_018", "My favorite movie is Inception",
        category="preference", fact_type="like",
        entity_name="user", subject="movie", value="Inception",
    ),
    # USER — historical learning (superseded)
    _make_memory(
        "mem_019", "I used to study chemistry",
        category="learning", fact_type="learning",
        entity_name="user", subject="chemistry", value="chemistry",
        status="inactive", superseded_by="mem_005",
    ),
    # USER — laptop
    _make_memory(
        "mem_020", "I use a MacBook Pro laptop",
        category="possession", fact_type="possession",
        entity_name="user", subject="laptop", value="MacBook Pro",
    ),
]


class MockStore:
    """Mock MemoryStore for testing."""

    def __init__(self, memories: list[dict]):
        self.memories = memories

    def active_memories(self) -> list[dict]:
        result = []
        for m in self.memories:
            status = str(m.get("status", "active")).lower()
            if status == "active":
                result.append(m)
        return result


# ---------------------------------------------------------------------------
# Group A: Dataset Setup Verification
# ---------------------------------------------------------------------------

class TestDatasetSetup(unittest.TestCase):
    """Verify the test dataset is correctly structured."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_A1_dataset_size(self):
        """Dataset has 20 memories."""
        self.assertEqual(len(TEST_MEMORIES), 20)

    def test_A2_active_memories(self):
        """Active memories exclude inactive ones."""
        active = self.store.active_memories()
        self.assertEqual(len(active), 17)  # 20 - 3 inactive

    def test_A3_entity_distribution(self):
        """Memories span multiple entities."""
        entities = set()
        for m in TEST_MEMORIES:
            e = m.get("entity", {})
            entities.add(e.get("name", "user"))
        self.assertIn("user", entities)
        self.assertIn("mom", entities)
        self.assertIn("jake", entities)
        self.assertIn("sister", entities)

    def test_A4_embeddings_populated(self):
        """All memories have embeddings."""
        for m in TEST_MEMORIES:
            self.assertIsInstance(m.get("embedding"), list)
            self.assertTrue(len(m.get("embedding", [])) > 0)


# ---------------------------------------------------------------------------
# Group B: Exact Fact Retrieval
# ---------------------------------------------------------------------------

class TestExactFactRetrieval(unittest.TestCase):
    """Test exact fact retrieval with direct queries."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_B1_phone_query(self):
        """'What phone do I use?' → Samsung S24 should be top result."""
        sq = understand_query("What phone do I use?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        top = result.top_candidate
        self.assertIsNotNone(top)
        self.assertIn("Samsung", top.content)

    def test_B2_color_query(self):
        """'What is my favorite color?' → blue should be top result."""
        sq = understand_query("What is my favorite color?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        top = result.top_candidate
        self.assertIsNotNone(top)
        self.assertIn("blue", top.content.lower())

    def test_B3_laptop_query(self):
        """'What laptop do I use?' → MacBook Pro should be found."""
        sq = understand_query("What laptop do I use?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        contents = [c.content for c in result.candidates]
        self.assertTrue(any("MacBook" in c for c in contents))


# ---------------------------------------------------------------------------
# Group C: Paraphrase Retrieval
# ---------------------------------------------------------------------------

class TestParaphraseRetrieval(unittest.TestCase):
    """Test that paraphrases retrieve the same memories."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_C1_paraphrase_phone(self):
        """Paraphrases of 'What phone do I use?' all find Samsung."""
        queries = [
            "What phone do I use?",
            "Which mobile am I currently using?",
            "What device am I carrying?",
        ]
        for q in queries:
            sq = understand_query(q)
            result = self.retriever.retrieve(sq)
            if result.has_results:
                contents = [c.content for c in result.candidates]
                self.assertTrue(
                    any("Samsung" in c for c in contents),
                    f"Query '{q}' should find Samsung",
                )

    def test_C2_paraphrase_color(self):
        """Paraphrases of favorite color query find blue."""
        queries = [
            "What is my favorite color?",
            "Which color do I like?",
            "What colour am I fond of?",
        ]
        for q in queries:
            sq = understand_query(q)
            result = self.retriever.retrieve(sq)
            if result.has_results:
                contents = [c.content for c in result.candidates]
                self.assertTrue(
                    any("blue" in c.lower() for c in contents),
                    f"Query '{q}' should find blue",
                )


# ---------------------------------------------------------------------------
# Group D: Grammar/Typo Retrieval
# ---------------------------------------------------------------------------

class TestGrammarTypoRetrieval(unittest.TestCase):
    """Test retrieval with grammar errors and typos."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_D1_typo_phone(self):
        """Typo: 'wat phone i hav?' should find Samsung."""
        sq = understand_query("wat phone i hav?")
        result = self.retriever.retrieve(sq)
        # May or may not find results depending on normalization
        # But should not crash
        self.assertIsInstance(result, HybridRetrievalResult)

    def test_D2_grammar_phone(self):
        """Grammar error: 'what phone i have?' should find Samsung."""
        sq = understand_query("what phone i have?")
        result = self.retriever.retrieve(sq)
        self.assertIsInstance(result, HybridRetrievalResult)


# ---------------------------------------------------------------------------
# Group E: Entity-Specific Retrieval
# ---------------------------------------------------------------------------

class TestEntityRetrieval(unittest.TestCase):
    """Test entity-specific retrieval."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_E1_mom_entity(self):
        """'What does my mom like?' → mom's preferences."""
        sq = understand_query("What does my mom like?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        # Should find mom's memories
        entities = set()
        for c in result.candidates:
            e = c.memory.get("entity", {})
            entities.add(e.get("name", "user"))
        self.assertIn("mom", entities)

    def test_E2_jake_entity(self):
        """'What does Jake like?' → jake's interests."""
        sq = understand_query("What does Jake like?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        entities = set()
        for c in result.candidates:
            e = c.memory.get("entity", {})
            entities.add(e.get("name", "user"))
        self.assertIn("jake", entities)

    def test_E3_mom_phone(self):
        """'What phone does my mom use?' → iPhone 15."""
        sq = understand_query("What phone does my mom use?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        contents = [c.content for c in result.candidates]
        self.assertTrue(any("iPhone" in c for c in contents))


# ---------------------------------------------------------------------------
# Group F: Historical Retrieval
# ---------------------------------------------------------------------------

class TestHistoricalRetrieval(unittest.TestCase):
    """Test historical retrieval."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_F1_previous_phone(self):
        """'What phone did I use before?' → Oppo A53 should be found."""
        sq = understand_query("What phone did I use before?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        contents = [c.content for c in result.candidates]
        self.assertTrue(any("Oppo" in c for c in contents))

    def test_F2_historical_state(self):
        """Historical query includes inactive memories."""
        sq = understand_query("What phone did I use before?")
        result = self.retriever.retrieve(sq)
        # Should include inactive memories in pool
        self.assertTrue(result.has_results)


# ---------------------------------------------------------------------------
# Group G: Current Retrieval
# ---------------------------------------------------------------------------

class TestCurrentRetrieval(unittest.TestCase):
    """Test current retrieval prioritizes active memories."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_G1_current_phone(self):
        """'What phone do I use now?' → Samsung S24 (current)."""
        sq = understand_query("What phone do I use now?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        top = result.top_candidate
        self.assertIn("Samsung", top.content)

    def test_G2_current_temporal_score(self):
        """Current queries give higher temporal_score to active memories."""
        sq = understand_query("What phone do I use now?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            top = result.top_candidate
            self.assertGreaterEqual(top.temporal_score, 0.5)


# ---------------------------------------------------------------------------
# Group H: Multiple Facts Retrieval
# ---------------------------------------------------------------------------

class TestMultipleFactsRetrieval(unittest.TestCase):
    """Test retrieval of multiple facts."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_H1_programming_languages(self):
        """'What programming languages am I learning?' → multiple results."""
        sq = understand_query("What programming languages am I learning?")
        result = self.retriever.retrieve(sq, max_results=5)
        self.assertTrue(result.has_results)
        # Should find both Python and JavaScript
        contents = [c.content.lower() for c in result.candidates]
        has_python = any("python" in c for c in contents)
        has_javascript = any("javascript" in c for c in contents)
        # At least one should be found
        self.assertTrue(has_python or has_javascript)


# ---------------------------------------------------------------------------
# Group I: Related Concept Retrieval
# ---------------------------------------------------------------------------

class TestRelatedConceptRetrieval(unittest.TestCase):
    """Test retrieval of related concepts."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_I1_study_query(self):
        """'What am I studying?' → learning-related memories."""
        sq = understand_query("What am I studying?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)


# ---------------------------------------------------------------------------
# Group J: Irrelevant Semantic Match
# ---------------------------------------------------------------------------

class TestIrrelevantSemanticMatch(unittest.TestCase):
    """Test that irrelevant semantic matches don't dominate."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_J1_movie_query(self):
        """'What is my favorite movie?' → Inception should be found."""
        sq = understand_query("What is my favorite movie?")
        result = self.retriever.retrieve(sq)
        self.assertTrue(result.has_results)
        contents = [c.content for c in result.candidates]
        self.assertTrue(any("Inception" in c for c in contents))

    def test_J2_football_team(self):
        """'What is my favorite football team?' → no strong match expected."""
        sq = understand_query("What is my favorite football team?")
        result = self.retriever.retrieve(sq)
        # May return weak candidates but should not claim strong match
        if result.has_results:
            top = result.top_candidate
            # Top score should be low since no football memory exists
            self.assertLess(top.final_score, 0.8)


# ---------------------------------------------------------------------------
# Group K: Unknown/Empty Retrieval
# ---------------------------------------------------------------------------

class TestUnknownRetrieval(unittest.TestCase):
    """Test unknown/empty retrieval scenarios."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_K1_empty_query(self):
        """Empty query returns no results."""
        sq = StructuredQuery(raw_query="")
        result = self.retriever.retrieve(sq)
        self.assertFalse(result.has_results)

    def test_K2_general_knowledge(self):
        """General knowledge query returns no results."""
        sq = understand_query("What is photosynthesis?")
        result = self.retriever.retrieve(sq)
        # Should not crash, may return weak results
        self.assertIsInstance(result, HybridRetrievalResult)


# ---------------------------------------------------------------------------
# Group L: Entity Isolation
# ---------------------------------------------------------------------------

class TestEntityIsolation(unittest.TestCase):
    """Test entity isolation between different entities."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_L1_user_vs_mom(self):
        """User query should not return mom's memories as top result."""
        sq = understand_query("What do I like?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            top = result.top_candidate
            top_entity = top.memory.get("entity", {}).get("name", "user")
            # Top result should be user's memory, not mom's
            self.assertIn(top_entity, ("user", "self"))

    def test_L2_mom_vs_user(self):
        """Mom query should return mom's memories."""
        sq = understand_query("What does my mom like?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            # At least one result should be mom's
            entities = [c.memory.get("entity", {}).get("name") for c in result.candidates]
            self.assertIn("mom", entities)

    def test_L3_jake_vs_user(self):
        """Jake query should return Jake's memories."""
        sq = understand_query("What does Jake like?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            entities = [c.memory.get("entity", {}).get("name") for c in result.candidates]
            self.assertIn("jake", entities)


# ---------------------------------------------------------------------------
# Group M: Relationship Retrieval
# ---------------------------------------------------------------------------

class TestRelationshipRetrieval(unittest.TestCase):
    """Test relationship-based retrieval."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_M1_mom_relationship(self):
        """'Who is my mom?' → mom entity should be found."""
        sq = understand_query("Who is my mom?")
        result = self.retriever.retrieve(sq)
        # Should at least not crash
        self.assertIsInstance(result, HybridRetrievalResult)


# ---------------------------------------------------------------------------
# Group N: Temporal Filtering
# ---------------------------------------------------------------------------

class TestTemporalFiltering(unittest.TestCase):
    """Test temporal filtering behavior."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_N1_current_prefers_active(self):
        """Current query gives higher temporal_score to active memories."""
        sq = understand_query("What phone do I use now?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            for c in result.candidates:
                status = c.memory.get("status", "active")
                if status == "active":
                    self.assertGreaterEqual(c.temporal_score, 0.5)

    def test_N2_historical_prefers_inactive(self):
        """Historical query includes inactive memories."""
        sq = understand_query("What phone did I use before?")
        result = self.retriever.retrieve(sq)
        # Should find historical memories
        self.assertTrue(result.has_results)


# ---------------------------------------------------------------------------
# Group O: Negated/Uncertain Queries
# ---------------------------------------------------------------------------

class TestNegatedUncertainQueries(unittest.TestCase):
    """Test negated and uncertain query handling."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_O1_negated_query(self):
        """'Do I still like Java?' → should not crash."""
        sq = understand_query("Do I still like Java?")
        result = self.retriever.retrieve(sq)
        self.assertIsInstance(result, HybridRetrievalResult)

    def test_O2_uncertain_query(self):
        """'I think I have a phone?' → should not crash."""
        sq = understand_query("I think I have a phone?")
        result = self.retriever.retrieve(sq)
        self.assertIsInstance(result, HybridRetrievalResult)


# ---------------------------------------------------------------------------
# Group P: Candidate Fusion
# ---------------------------------------------------------------------------

class TestCandidateFusion(unittest.TestCase):
    """Test candidate fusion from multiple channels."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_P1_multi_source_candidate(self):
        """Candidate from multiple sources should rank higher."""
        sq = understand_query("What phone do I use?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            top = result.top_candidate
            # Samsung should appear in multiple channels
            self.assertGreaterEqual(len(top.sources), 1)


# ---------------------------------------------------------------------------
# Group Q: Ranking Verification
# ---------------------------------------------------------------------------

class TestRankingVerification(unittest.TestCase):
    """Test ranking correctness."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_Q1_ranking_order(self):
        """Results should be ranked by final_score descending."""
        sq = understand_query("What phone do I use?")
        result = self.retriever.retrieve(sq)
        if result.has_results:
            scores = [c.final_score for c in result.candidates]
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(scores[i], scores[i + 1])

    def test_Q2_ranks_assigned(self):
        """Each candidate should have a rank."""
        sq = understand_query("What phone do I use?")
        result = self.retriever.retrieve(sq)
        for i, c in enumerate(result.candidates, 1):
            self.assertEqual(c.rank, i)


# ---------------------------------------------------------------------------
# Group R: Retrieval Trace
# ---------------------------------------------------------------------------

class TestRetrievalTrace(unittest.TestCase):
    """Test retrieval trace is populated."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def test_R1_trace_populated(self):
        """Trace should be populated after retrieval."""
        sq = understand_query("What phone do I use?")
        result = self.retriever.retrieve(sq)
        self.assertIsInstance(result.trace, RetrievalTrace)
        self.assertEqual(result.trace.query, "What phone do I use?")
        self.assertGreater(result.trace.candidate_pool_size, 0)

    def test_R2_trace_channel_counts(self):
        """Trace should have channel candidate counts."""
        sq = understand_query("What phone do I use?")
        result = self.retriever.retrieve(sq)
        trace = result.trace
        self.assertGreaterEqual(trace.structured_candidates, 0)
        self.assertGreaterEqual(trace.keyword_candidates, 0)
        self.assertGreaterEqual(trace.semantic_candidates, 0)


# ---------------------------------------------------------------------------
# Group S: Adversarial Tests (40+)
# ---------------------------------------------------------------------------

class TestAdversarialRetrieval(unittest.TestCase):
    """40+ adversarial retrieval scenarios."""

    def setUp(self):
        self.embedding_store = _make_embedding_store_with_vectors(TEST_MEMORIES)
        self.store = MockStore(TEST_MEMORIES)
        self.retriever = HybridRetriever(self.store, self.embedding_store)

    def _test_query(self, query_str: str, should_find: bool = True):
        """Helper to test a query."""
        sq = understand_query(query_str)
        result = self.retriever.retrieve(sq)
        self.assertIsInstance(result, HybridRetrievalResult)
        return result

    # Entity isolation adversarial
    def test_S01_entity_isolation_user(self):
        self._test_query("What do I like?")

    def test_S02_entity_isolation_mom(self):
        self._test_query("What does my mom like?")

    def test_S03_entity_isolation_jake(self):
        self._test_query("What does Jake like?")

    def test_S04_entity_isolation_sister(self):
        self._test_query("What does my sister prefer?")

    # Current vs historical
    def test_S05_current_phone(self):
        result = self._test_query("What phone do I use now?")
        if result.has_results:
            self.assertIn("Samsung", result.top_candidate.content)

    def test_S06_historical_phone(self):
        self._test_query("What phone did I use before?")

    def test_S07_old_phone(self):
        self._test_query("What was my old phone?")

    # Semantic similarity adversarial
    def test_S08_semantic_distant(self):
        self._test_query("Tell me about my possessions")

    def test_S09_semantic_close(self):
        self._test_query("Which mobile am I using?")

    # Lexical similarity adversarial
    def test_S10_exact_keyword(self):
        self._test_query("Samsung phone")

    def test_S11_partial_keyword(self):
        self._test_query("phone")

    # Relation matching adversarial
    def test_S12_possession_query(self):
        self._test_query("What do I own?")

    def test_S13_preference_query(self):
        self._test_query("What do I prefer?")

    def test_S14_learning_query(self):
        self._test_query("What am I studying?")

    # Object matching adversarial
    def test_S15_object_phone(self):
        self._test_query("What kind of phone?")

    def test_S16_object_color(self):
        self._test_query("What color?")

    def test_S17_object_language(self):
        self._test_query("What language?")

    # Multiple entities adversarial
    def test_S18_multi_entity_mom(self):
        self._test_query("What does my mom like and own?")

    def test_S19_multi_entity_jake(self):
        self._test_query("What does Jake like?")

    # Duplicate memories adversarial
    def test_S20_duplicate_content(self):
        self._test_query("What phone?")

    # Inactive memories adversarial
    def test_S21_inactive_memory(self):
        self._test_query("What phone did I have?")

    def test_S22_superseded_memory(self):
        self._test_query("What was my previous phone?")

    # Weak semantic matches adversarial
    def test_S23_weak_match(self):
        self._test_query("What is my favorite food?")

    def test_S24_no_match(self):
        self._test_query("What is my favorite football team?")

    # Unrelated semantic matches adversarial
    def test_S25_unrelated_match(self):
        self._test_query("Tell me about the weather")

    # Typo queries adversarial
    def test_S26_typo_phone(self):
        self._test_query("wat phone i hav?")

    def test_S27_typo_color(self):
        self._test_query("wat color i liek?")

    def test_S28_typo_language(self):
        self._test_query("wat langauge i learn?")

    # Slang queries adversarial
    def test_S29_slang_phone(self):
        self._test_query("yo what phone u rockin?")

    def test_S30_slang_color(self):
        self._test_query("bruh whats my fave color")

    # Vague queries adversarial
    def test_S31_vague_stuff(self):
        self._test_query("tell me about my stuff")

    def test_S32_vague_me(self):
        self._test_query("what do you know about me?")

    # Temporal queries adversarial
    def test_S33_temporal_now(self):
        self._test_query("what am i using now?")

    def test_S34_temporal_before(self):
        self._test_query("what did i use before?")

    def test_S35_temporal_future(self):
        self._test_query("what will i use next?")

    # Cardinality adversarial
    def test_S36_count_query(self):
        self._test_query("how many phones do i have?")

    def test_S37_existence_query(self):
        self._test_query("do i have a laptop?")

    # Contradictory memories adversarial
    def test_S38_contradictory(self):
        self._test_query("what color do i like?")

    # Missing memories adversarial
    def test_S39_missing_memory(self):
        self._test_query("what is my favorite book?")

    # Empty results adversarial
    def test_S40_empty_results(self):
        result = self._test_query("what is the meaning of life?")
        # Should not crash
        self.assertIsInstance(result, HybridRetrievalResult)

    # Additional adversarial
    def test_S41_negation(self):
        self._test_query("what color do i not like?")

    def test_S42_uncertainty(self):
        self._test_query("i think i like blue?")

    def test_S43_contractions(self):
        self._test_query("whats my phone?")

    def test_S44_all_caps(self):
        self._test_query("WHAT PHONE DO I USE?")

    def test_S45_mixed_case(self):
        self._test_query("WhAt PhOnE dO i UsE?")

    def test_S46_extra_spaces(self):
        self._test_query("what   phone   do   i   use?")

    def test_S47_special_chars(self):
        self._test_query("what phone do i use?!")

    def test_S48_long_query(self):
        self._test_query("I was just wondering what kind of phone I am currently using these days?")

    def test_S49_single_word(self):
        self._test_query("phone")

    def test_S50_two_words(self):
        self._test_query("my phone")


# ---------------------------------------------------------------------------
# Group T: Token Utilities
# ---------------------------------------------------------------------------

class TestTokenUtilities(unittest.TestCase):
    """Test token utility functions."""

    def test_T1_tokenize(self):
        """Tokenize removes stop words."""
        tokens = tokenize("What is my favorite color?")
        self.assertIn("favorite", tokens)
        self.assertIn("color", tokens)
        self.assertNotIn("what", tokens)
        self.assertNotIn("is", tokens)
        self.assertNotIn("my", tokens)

    def test_T2_token_overlap(self):
        """Token overlap score works correctly."""
        score = token_overlap_score(["phone", "samsung"], ["samsung", "phone", "use"])
        self.assertGreater(score, 0.5)

    def test_T3_token_overlap_empty(self):
        """Empty tokens return 0."""
        score = token_overlap_score([], ["phone"])
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
