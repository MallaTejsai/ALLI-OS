"""Phase 5: Entity-scoped retrieval tests.

Tests that memory queries resolve entities, apply deterministic filtering
(entity/category/temporal) BEFORE semantic ranking, and return structured
retrieval results. No Ollama dependency for deterministic logic tests.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.query_entity import QueryEntity, resolve_query_entity
from memory.query_intent import QueryIntentDetector, QueryIntent
from memory.retriever import MemoryRetriever, RetrievalResult
from memory.store import MemoryStore
from memory.embeddings import EmbeddingStore


def make_retrieval_stack(tmpdir: str):
    """Create memory stack with skip_embeddings for offline tests."""
    path = os.path.join(tmpdir, "long_term_memory.json")
    embedding_store = EmbeddingStore()
    store = MemoryStore(path=path, embedding_store=embedding_store, skip_embeddings=True)
    store.load(ensure_embeddings=False)
    intent_detector = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(store, embedding_store, intent_detector=intent_detector)
    return store, retriever, intent_detector


def _add_memory(store, category, content, entity_name="user", entity_type="self",
                fact_type=None, value=None, subject=None):
    """Helper to add a memory with entity fields directly to the store."""
    return store.add_memory(
        category=category,
        content=content,
        fact_type=fact_type,
        value=value,
        subject=subject,
        entity_name=entity_name,
        entity_type=entity_type,
    )


# ===========================================================================
# Test Group 1: QueryEntity resolution (from query_entity.py)
# ===========================================================================

class TestQueryEntityResolution(unittest.TestCase):
    """Unit tests for resolve_query_entity — no embeddings required."""

    def test_my_favourite_colour_resolves_to_user(self):
        entity = resolve_query_entity("what is my favourite colour?")
        self.assertEqual(entity.name, "user")
        self.assertEqual(entity.entity_type, "self")
        self.assertTrue(entity.is_resolved)

    def test_i_like_resolves_to_user(self):
        entity = resolve_query_entity("what do i like?")
        self.assertEqual(entity.name, "user")
        self.assertTrue(entity.is_resolved)

    def test_moms_favourite_colour_resolves_to_mom(self):
        entity = resolve_query_entity("what is mom's favourite colour?")
        self.assertEqual(entity.name, "mom")
        self.assertEqual(entity.entity_type, "person")
        self.assertTrue(entity.is_resolved)

    def test_my_moms_name_resolves_to_mom(self):
        entity = resolve_query_entity("what is my mom's name?")
        self.assertEqual(entity.name, "mom")
        self.assertTrue(entity.is_resolved)

    def test_dads_birthday_resolves_to_dad(self):
        entity = resolve_query_entity("when is dad's birthday?")
        self.assertEqual(entity.name, "dad")
        self.assertTrue(entity.is_resolved)

    def test_their_favourite_colour_unresolved(self):
        entity = resolve_query_entity("what is their favourite colour?")
        self.assertTrue(entity.is_unresolved)

    def test_his_name_unresolved(self):
        entity = resolve_query_entity("what is his name?")
        self.assertTrue(entity.is_unresolved)

    def test_my_friend_name_resolves_to_friend(self):
        entity = resolve_query_entity("what is my friend's name?")
        self.assertEqual(entity.name, "friend")
        self.assertTrue(entity.is_resolved)

    def test_empty_query_resolves_to_user(self):
        entity = resolve_query_entity("")
        self.assertEqual(entity.name, "user")

    def test_none_query_resolves_to_user(self):
        entity = resolve_query_entity(None)
        self.assertEqual(entity.name, "user")


# ===========================================================================
# Test Group 2: QueryIntent entity/cardinality/temporal fields
# ===========================================================================

class TestQueryIntentPhase5Fields(unittest.TestCase):
    """Test that QueryIntent includes Phase 5 fields via cue-only path."""

    def setUp(self):
        self.detector = QueryIntentDetector()

    def test_my_preferences_has_entity_user(self):
        """'What are my preferences?' matches preference cue → entity=user."""
        intent = self.detector.detect("What are my preferences?")
        self.assertIsNotNone(intent.entity)
        self.assertEqual(intent.entity.name, "user")
        self.assertTrue(intent.is_personal)
        self.assertTrue(intent.is_memory_read)

    def test_their_preferences_has_unresolved_entity(self):
        """'What are their preferences?' matches preference cue → entity unresolved."""
        intent = self.detector.detect("What are their preferences?")
        self.assertIsNotNone(intent.entity)
        self.assertTrue(intent.entity.is_unresolved)

    def test_plural_query_detected(self):
        intent = self.detector.detect("What are my goals?")
        self.assertEqual(intent.cardinality, "plural")

    def test_singular_query_detected(self):
        intent = self.detector.detect("What is my name?")
        self.assertEqual(intent.cardinality, "singular")

    def test_temporal_scope_current(self):
        intent = self.detector.detect("What am I currently learning?")
        self.assertEqual(intent.temporal_scope, "current")

    def test_general_knowledge_not_personal(self):
        intent = self.detector.detect("What is the capital of France?")
        self.assertFalse(intent.is_personal)


# ===========================================================================
# Test Group 3: Entity-scoped filtering in retriever
# ===========================================================================

class TestEntityScopedFiltering(unittest.TestCase):
    """Test that entity filtering happens BEFORE semantic ranking."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store, self.retriever, self.detector = make_retrieval_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_user_entity_filters_to_user_memories(self):
        """Query for 'my preferences' should only return memories with entity='user'."""
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue", subject="colour")
        _add_memory(self.store, "preference", "User likes red",
                    entity_name="mom", value="red", subject="colour")

        intent = self.detector.detect("What are my preferences?")
        filtered = self.retriever._filter_by_entity(
            self.store.active_memories(), intent
        )
        self.assertEqual(len(filtered), 1)
        self.assertIn("blue", filtered[0]["content"].lower())

    def test_unresolved_entity_returns_empty(self):
        """Unresolved entity (their) should return no results."""
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue", subject="colour")

        intent = self.detector.detect("What are their preferences?")
        filtered = self.retriever._filter_by_entity(
            self.store.active_memories(), intent
        )
        self.assertEqual(len(filtered), 0)

    def test_no_entity_returns_all(self):
        """Query without entity should return all active memories."""
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue", subject="colour")
        _add_memory(self.store, "preference", "Mom likes red",
                    entity_name="mom", value="red", subject="colour")

        intent = QueryIntent(
            primary="preference",
            score=0.8,
            is_memory_read=True,
            is_general_knowledge=False,
            entity=None,
        )
        filtered = self.retriever._filter_by_entity(
            self.store.active_memories(), intent
        )
        self.assertEqual(len(filtered), 2)


# ===========================================================================
# Test Group 4: Entity scope never overridden by embedding similarity
# ===========================================================================

class TestEntityScopeVsSimilarity(unittest.TestCase):
    """Core Phase 5 test: embedding similarity must NEVER override entity scope.

    Even if a user's memory has a very high embedding similarity to a query
    about another entity, only the target entity's memories should be returned.
    """

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store, self.retriever, self.detector = make_retrieval_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_entity_scope_overrides_similarity(self):
        """Mom query returns only mom's memories even if user's are similar."""
        _add_memory(self.store, "preference", "Mom likes purple",
                    entity_name="mom", value="purple", subject="colour")
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue", subject="colour")

        intent = self.detector.detect("What are their preferences?")
        # With "their" (unresolved), the entity filtering should block ALL
        # memories because we can't determine whose preferences to return
        filtered = self.retriever._filter_by_entity(
            self.store.active_memories(), intent
        )
        # No memories should be returned — entity is unresolved
        self.assertEqual(len(filtered), 0)

    def test_user_query_excludes_mom_memories(self):
        """'my preferences' should never include mom's memories."""
        _add_memory(self.store, "preference", "Mom likes purple",
                    entity_name="mom", value="purple", subject="colour")
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue", subject="colour")

        intent = self.detector.detect("What are my preferences?")
        filtered = self.retriever._filter_by_entity(
            self.store.active_memories(), intent
        )
        # Only user's memory should remain
        self.assertEqual(len(filtered), 1)
        self.assertIn("blue", filtered[0]["content"].lower())
        # Mom's memory must not be present
        for m in filtered:
            self.assertNotIn("purple", m.get("content", "").lower())


# ===========================================================================
# Test Group 5: Temporal filtering
# ===========================================================================

class TestTemporalFiltering(unittest.TestCase):
    """Expired memories should not appear in current queries."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store, self.retriever, self.detector = make_retrieval_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_expired_memory_excluded_from_current_query(self):
        """Expired memories are excluded unless query asks for history."""
        # Add an active memory
        _add_memory(self.store, "task", "Take out trash",
                    entity_name="user", fact_type="task")

        # Add a memory and manually set it expired
        mem = _add_memory(self.store, "task", "Old expired task",
                          entity_name="user", fact_type="task")
        yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        self.store.memories = [
            {**m, "expires_at": yesterday} if m.get("id") == mem["id"] else m
            for m in self.store.memories
        ]
        self.store.save()

        intent = self.detector.detect("What tasks do I have?")
        filtered = self.retriever._filter_by_entity(
            self.store.active_memories(), intent
        )
        for m in filtered:
            self.assertNotEqual(m.get("content"), "Old expired task")


# ===========================================================================
# Test Group 6: RetrievalResult structured output
# ===========================================================================

class TestRetrievalResult(unittest.TestCase):
    """Test the RetrievalResult structured output."""

    def test_found_status(self):
        result = RetrievalResult(
            status="found",
            selected=[{"content": "test"}],
        )
        self.assertTrue(result.has_results)
        self.assertFalse(result.is_ambiguous)

    def test_not_found_status(self):
        result = RetrievalResult(status="not_found")
        self.assertFalse(result.has_results)
        self.assertFalse(result.is_ambiguous)

    def test_ambiguous_status(self):
        result = RetrievalResult(status="ambiguous")
        self.assertTrue(result.is_ambiguous)

    def test_unresolved_entity_status(self):
        result = RetrievalResult(status="unresolved_entity")
        self.assertFalse(result.has_results)

    def test_not_personal_status(self):
        result = RetrievalResult(status="not_personal")
        self.assertFalse(result.has_results)


# ===========================================================================
# Test Group 7: Entity in memory store
# ===========================================================================

class TestMemoryStoreEntity(unittest.TestCase):
    """Test that memories are stored with correct entity fields."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        path = os.path.join(self._tmpdir.name, "long_term_memory.json")
        embedding_store = EmbeddingStore()
        self.store = MemoryStore(path=path, embedding_store=embedding_store, skip_embeddings=True)
        self.store.load(ensure_embeddings=False)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_default_entity_is_user(self):
        mem = _add_memory(self.store, "preference", "User likes blue",
                          value="blue", subject="colour")
        self.assertEqual(mem["entity"]["name"], "user")
        self.assertEqual(mem["entity"]["type"], "self")

    def test_explicit_entity_name(self):
        mem = _add_memory(self.store, "preference", "Mom likes red",
                          entity_name="mom", entity_type="person",
                          value="red", subject="colour")
        self.assertEqual(mem["entity"]["name"], "mom")
        self.assertEqual(mem["entity"]["type"], "person")


# ===========================================================================
# Test Group 8: Compound query preserves entity scope
# ===========================================================================

class TestCompoundQueryEntityScope(unittest.TestCase):
    """Compound queries (e.g., 'skills and goals') preserve entity scope."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store, self.retriever, self.detector = make_retrieval_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_compound_query_preserves_entity(self):
        """Entity scope should be passed to sub-intents in compound queries."""
        _add_memory(self.store, "skill", "User knows Python",
                    entity_name="user", value="python")
        _add_memory(self.store, "goal", "User wants to learn Rust",
                    entity_name="user", value="rust")
        _add_memory(self.store, "skill", "Mom knows French",
                    entity_name="mom", value="french")

        intent = self.detector.detect("What are my skills and goals?")
        # Verify entity is set on the intent
        self.assertIsNotNone(intent.entity)
        self.assertEqual(intent.entity.name, "user")

        # Entity filtering should exclude mom's memories
        all_active = self.store.active_memories()
        filtered = self.retriever._filter_by_entity(all_active, intent)
        # Should have 2 memories (both user's) - not mom's
        self.assertEqual(len(filtered), 2)
        for m in filtered:
            self.assertEqual(m["entity"]["name"], "user")
        contents = [m["content"].lower() for m in filtered]
        self.assertTrue(any("python" in c for c in contents))
        self.assertTrue(any("rust" in c for c in contents))
        # Mom's memory must not be present
        self.assertFalse(any("french" in c for c in contents))


# ===========================================================================
# Test Group 9: Debug diagnostics include entity info
# ===========================================================================

class TestDebugDiagnostics(unittest.TestCase):
    """Retrieval debug output should include entity diagnostics."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store, self.retriever, self.detector = make_retrieval_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_debug_includes_entity(self):
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue")
        self.retriever.retrieve_relevant_memories(
            "What are my preferences?"
        )
        debug = self.retriever.last_debug
        self.assertIn("entity", debug)
        self.assertIn("cardinality", debug)
        self.assertIn("is_personal", debug)
        self.assertIn("filter_diagnostics", debug)


# ===========================================================================
# Test Group 10: Edge cases
# ===========================================================================

class TestEdgeCases(unittest.TestCase):
    """Edge cases for entity-scoped retrieval."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store, self.retriever, self.detector = make_retrieval_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_empty_query_returns_nothing(self):
        result = self.retriever.retrieve_relevant_memories("")
        self.assertEqual(len(result), 0)

    def test_whitespace_only_query_returns_nothing(self):
        result = self.retriever.retrieve_relevant_memories("   ")
        self.assertEqual(len(result), 0)

    def test_general_knowledge_returns_empty(self):
        result = self.retriever.retrieve_relevant_memories("What is 2+2?")
        self.assertEqual(len(result), 0)

    def test_non_memory_read_returns_empty(self):
        result = self.retriever.retrieve_relevant_memories("How are you?")
        self.assertEqual(len(result), 0)

    def test_retrieve_structured_not_personal(self):
        result = self.retriever.retrieve_structured("How are you?")
        self.assertEqual(result.status, "not_personal")

    def test_retrieve_structured_unresolved_entity(self):
        _add_memory(self.store, "preference", "User likes blue",
                    entity_name="user", value="blue")
        result = self.retriever.retrieve_structured(
            "What are their preferences?"
        )
        self.assertEqual(result.status, "unresolved_entity")


if __name__ == "__main__":
    unittest.main()
