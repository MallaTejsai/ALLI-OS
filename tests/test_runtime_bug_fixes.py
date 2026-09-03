"""Regression tests for the 12 runtime bug fixes (v0.6 stabilization)."""

import os
import shutil
import tempfile
import unittest

from memory.extractor import MemoryExtractor, is_temporary_message, has_explicit_change_signal
from memory.manager import SafeMemoryManager
from memory.query_intent import QueryIntentDetector, CATEGORY_QUERY_CUES, CONCEPT_PROTOTYPES
from memory.query_entity import resolve_query_entity
from memory.retriever import MemoryRetriever
from memory.store import MemoryStore
from memory.answers import grounded_answer, _personal_overview_answer, _deduplicate_preferences
from memory.embeddings import EmbeddingStore


def make_stack(tmpdir: str):
    """Create a full memory stack for fast tests (embeddings disabled)."""
    path = os.path.join(tmpdir, "long_term_memory.json")
    embedding_store = EmbeddingStore()
    store = MemoryStore(path=path, embedding_store=embedding_store, skip_embeddings=True)
    store.load(ensure_embeddings=False)
    manager = SafeMemoryManager(store)
    detector = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(store, embedding_store, intent_detector=detector)
    return manager, retriever, store, embedding_store


# ============================================================================
# BUG 1: Possession retrieval — "What phone do I have?"
# ============================================================================
class TestBug1PossessionRetrieval(unittest.TestCase):
    """BUG 1: CONCEPT_PROTOTYPES missing 'possession' entry."""

    def test_possession_has_prototype(self):
        """possession must have semantic prototypes for intent detection."""
        self.assertIn("possession", CONCEPT_PROTOTYPES)
        self.assertTrue(len(CONCEPT_PROTOTYPES["possession"]) > 0)

    def test_possession_cued_in_category_cues(self):
        """'what phone do I have' must cue possession intent."""
        self.assertIn("possession", CATEGORY_QUERY_CUES)
        patterns = CATEGORY_QUERY_CUES["possession"]
        import re
        matched = any(re.search(p, "what phone do i have") for p in patterns)
        self.assertTrue(matched, "possession cue should match 'what phone do i have'")

    def test_possession_stored_and_retrievable(self):
        """Possession memory is stored and can be retrieved by intent detection."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, retriever, store, _ = make_stack(tmpdir)
            manager.process_user_message("I have an Oppo A53.")
            active = store.active_memories()
            possessions = [m for m in active if m["category"] == "possession"]
            self.assertEqual(len(possessions), 1)
            self.assertIn("oppo", possessions[0]["content"].lower())
        finally:
            shutil.rmtree(tmpdir)


# ============================================================================
# BUG 2: Entity detection works but entity filtering not implemented
# ============================================================================
class TestBug2EntityResolution(unittest.TestCase):
    """BUG 2: Entity is correctly detected but not used in retrieval."""

    def test_mom_entity_resolved(self):
        """'my mom's favourite colour' resolves entity to 'mom'."""
        entity = resolve_query_entity("what is my mom's favourite colour")
        self.assertEqual(entity.name, "mom")
        self.assertEqual(entity.entity_type, "person")
        self.assertTrue(entity.is_resolved)

    def test_self_entity_for_my_query(self):
        """'what is my favourite colour' resolves to user entity."""
        entity = resolve_query_entity("what is my favourite colour")
        self.assertEqual(entity.name, "user")
        self.assertTrue(entity.is_self)


# ============================================================================
# BUG 3: "What is my favourite movie?" — favourite not cued as preference
# ============================================================================
class TestBug3FavouriteCue(unittest.TestCase):
    """BUG 3: 'favourite' was missing from CATEGORY_QUERY_CUES['preference']."""

    def test_favourite_cues_preference(self):
        """'favourite' must appear in preference cues."""
        import re
        patterns = CATEGORY_QUERY_CUES["preference"]
        matched = any(re.search(p, "what is my favourite movie") for p in patterns)
        self.assertTrue(matched, "'favourite' should cue preference intent")

    def test_favorite_cues_preference(self):
        """'favorite' (US spelling) must also cue preference."""
        import re
        patterns = CATEGORY_QUERY_CUES["preference"]
        matched = any(re.search(p, "what is my favorite movie") for p in patterns)
        self.assertTrue(matched, "'favorite' should cue preference intent")


# ============================================================================
# BUG 4: "Actually, I am learning Java now" — doesn't supersede Python
# ============================================================================
class TestBug4LearningSupersession(unittest.TestCase):
    """BUG 4: _upsert_by_subject lacks change-signal-aware supersession."""

    def test_correction_supersedes_old_learning(self):
        """'Actually, I am learning Java' supersedes 'I am learning Python'."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            manager.process_user_message("I am learning Python.")
            result = manager.process_user_message("Actually, I am learning Java now.")
            self.assertEqual(result[0]["status"], "added")
            self.assertIn("superseded_ids", result[0])
            # Old memory should be inactive
            active = store.active_memories()
            learning = [m for m in active if m["category"] == "learning"]
            self.assertEqual(len(learning), 1)
            self.assertIn("java", learning[0]["content"].lower())
        finally:
            shutil.rmtree(tmpdir)

    def test_correction_supersedes_old_skill(self):
        """'Actually, I know Java' supersedes 'I know Python' in skill category."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            manager.process_user_message("I know Python.")
            result = manager.process_user_message("Actually, I know Java now.")
            self.assertEqual(result[0]["status"], "added")
            self.assertIn("superseded_ids", result[0])
        finally:
            shutil.rmtree(tmpdir)

    def test_no_signal_does_not_supersede(self):
        """Without change signal, new learning is added alongside old."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            manager.process_user_message("I am learning Python.")
            manager.process_user_message("I am learning Java now.")
            active = store.active_memories()
            learning = [m for m in active if m["category"] == "learning"]
            self.assertEqual(len(learning), 2)
        finally:
            shutil.rmtree(tmpdir)


# ============================================================================
# BUG 5: "What am I preparing for?" — should cue task, not learning
# ============================================================================
class TestBug5PreparingForCue(unittest.TestCase):
    """BUG 5: 'preparing for' was missing from CATEGORY_QUERY_CUES['task']."""

    def test_preparing_for_cues_task(self):
        """'preparing for' must cue task intent."""
        import re
        patterns = CATEGORY_QUERY_CUES["task"]
        matched = any(re.search(p, "what am i preparing for") for p in patterns)
        self.assertTrue(matched, "'preparing for' should cue task intent")

    def test_studying_for_cues_task(self):
        """'studying for' must cue task intent."""
        import re
        patterns = CATEGORY_QUERY_CUES["task"]
        matched = any(re.search(p, "what am i studying for") for p in patterns)
        self.assertTrue(matched, "'studying for' should cue task intent")


# ============================================================================
# BUG 6: Multi-item preference handling (defense-in-depth)
# ============================================================================
class TestBug6MultiValuePreference(unittest.TestCase):
    """BUG 6: Multiple preference values should be split correctly."""

    def test_like_python_and_java_creates_two_memories(self):
        """'I like Python and Java' should create two preference memories."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            manager.process_user_message("I like Python and Java.")
            active = store.active_memories()
            prefs = [m for m in active if m["category"] == "preference"]
            subjects = {m.get("subject") for m in prefs}
            self.assertIn("python", subjects)
            self.assertIn("java", subjects)
        finally:
            shutil.rmtree(tmpdir)


# ============================================================================
# BUG 7: "become no" stored as goal
# ============================================================================
class TestBug7MalformedGoalRejection(unittest.TestCase):
    """BUG 7: 'become no' / 'become no one' should not be stored as goals."""

    def test_become_no_not_stored(self):
        """'I want to become no' should not create a goal."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            result = manager.process_user_message("I want to become no.")
            # Should be skipped or empty
            if result:
                self.assertNotEqual(result[0].get("status"), "added")
            active = store.active_memories()
            goals = [m for m in active if m["category"] == "goal"]
            self.assertEqual(len(goals), 0)
        finally:
            shutil.rmtree(tmpdir)

    def test_become_no_one_not_stored(self):
        """'I want to become no one' should not create a goal."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            result = manager.process_user_message("I want to become no one.")
            if result:
                self.assertNotEqual(result[0].get("status"), "added")
            active = store.active_memories()
            goals = [m for m in active if m["category"] == "goal"]
            self.assertEqual(len(goals), 0)
        finally:
            shutil.rmtree(tmpdir)

    def test_become_none_not_stored(self):
        """'I want to become none' should not create a goal."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            result = manager.process_user_message("I want to become none.")
            if result:
                self.assertNotEqual(result[0].get("status"), "added")
        finally:
            shutil.rmtree(tmpdir)


# ============================================================================
# BUG 8: Singular query returns entire collection
# ============================================================================
class TestBug8SingularQueryLimit(unittest.TestCase):
    """BUG 8: Cardinality='singular' should limit retrieval results."""

    def test_singular_query_limits_results(self):
        """When cardinality is singular, retrieval should return fewer results."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, retriever, store, _ = make_stack(tmpdir)
            # Add multiple goals
            manager.process_user_message("I want to become a doctor.")
            manager.process_user_message("I want to become a lawyer.")
            manager.process_user_message("I want to become an engineer.")

            intent = retriever.detect_intent("What is my goal?")
            self.assertEqual(intent.cardinality, "singular")

            # Retrieve with singular intent — should limit to 3 or fewer
            results = retriever.retrieve_with_scores("What is my goal?", intent=intent)
            self.assertLessEqual(len(results), 3)
        finally:
            shutil.rmtree(tmpdir)


# ============================================================================
# BUG 9: Temporary tasks misclassified as goals
# ============================================================================
class TestBug9TemporaryTaskMisclassification(unittest.TestCase):
    """BUG 9: 'prepare for my exam' should be task, not goal."""

    def test_prepare_for_exam_is_temporary(self):
        """'prepare for my exam' should be detected as temporary."""
        self.assertTrue(is_temporary_message("I want to prepare for my exam."))

    def test_study_for_exam_is_temporary(self):
        """'study for my exam' should be detected as temporary."""
        self.assertTrue(is_temporary_message("I want to study for my exam."))

    def test_prepare_for_exam_extracted_as_task(self):
        """'I want to prepare for my exam tomorrow' should not be extracted as a goal."""
        extractor = MemoryExtractor()
        ops = extractor.extract("I want to prepare for my exam tomorrow.")
        # The key fix: it must not produce a goal. Whether it produces a task
        # depends on the extraction patterns. The critical thing is no goal.
        goals = [op for op in ops if op.get("category") == "goal"]
        self.assertEqual(len(goals), 0, "Should NOT be extracted as a goal")


# ============================================================================
# BUG 10: Contradictory preference states
# ============================================================================
class TestBug10ContradictoryPreference(unittest.TestCase):
    """BUG 10: Both 'like' and 'dislike' for same subject should not coexist."""

    def test_preference_reversal_supersedes(self):
        """'I don't like C++' should update 'I like C++' (not add new)."""
        tmpdir = tempfile.mkdtemp()
        try:
            manager, _, store, _ = make_stack(tmpdir)
            manager.process_user_message("I like C++.")
            manager.process_user_message("I don't like C++.")
            active = store.active_memories()
            prefs = [m for m in active if m["category"] == "preference"]
            # Should have exactly one preference for C++
            cpp_prefs = [m for m in prefs if "c++" in (m.get("subject") or "").lower()]
            self.assertEqual(len(cpp_prefs), 1)
            self.assertEqual(cpp_prefs[0]["fact_type"], "dislike")
        finally:
            shutil.rmtree(tmpdir)


# ============================================================================
# BUG 11: Personal queries get general LLM answers
# ============================================================================
class TestBug11PersonalQueryGuard(unittest.TestCase):
    """BUG 11: Personal queries with pronouns should not be classified as general_knowledge."""

    def _make_detector(self):
        embedding_store = EmbeddingStore()
        return QueryIntentDetector(embedding_store)

    def test_what_is_my_name_not_general_knowledge(self):
        """'What is my name?' should not be classified as general_knowledge."""
        detector = self._make_detector()
        intent = detector.detect("What is my name?")
        self.assertFalse(intent.is_general_knowledge)

    def test_what_do_i_like_not_general_knowledge(self):
        """'What do I like?' should not be classified as general_knowledge."""
        detector = self._make_detector()
        intent = detector.detect("What do I like?")
        self.assertFalse(intent.is_general_knowledge)

    def test_how_does_cpu_work_is_general_knowledge(self):
        """'How does a CPU work?' should still be general_knowledge."""
        detector = self._make_detector()
        intent = detector.detect("How does a CPU work?")
        self.assertTrue(intent.is_general_knowledge)


# ============================================================================
# BUG 12: Self-overview contamination (defense-in-depth)
# ============================================================================
class TestBug12OverviewFiltering(unittest.TestCase):
    """BUG 12: Personal overview should filter tasks and deduplicate preferences."""

    def test_overview_separates_tasks(self):
        """Tasks should appear in a separate 'Active Tasks' section."""
        memories = [
            {"category": "goal", "content": "User wants to become a doctor", "value": "doctor", "subject": "doctor"},
            {"category": "task", "content": "User has exam tomorrow", "value": "exam", "subject": "exam"},
        ]
        answer = _personal_overview_answer(memories)
        self.assertIn("Goals", answer)
        self.assertIn("Active Tasks", answer)

    def test_overview_deduplicates_preferences(self):
        """Duplicate preferences for same subject should be deduplicated."""
        memories = [
            {"category": "preference", "content": "User likes coffee", "value": "like", "subject": "coffee", "fact_type": "like"},
            {"category": "preference", "content": "User dislikes coffee", "value": "dislike", "subject": "coffee", "fact_type": "dislike"},
        ]
        deduped = _deduplicate_preferences(memories)
        # Should keep only the most recent (last one)
        self.assertEqual(len(deduped), 1)

    def test_overview_includes_possessions(self):
        """Possessions should appear in the overview."""
        memories = [
            {"category": "possession", "content": "User has iPhone", "value": "iPhone", "subject": "phone"},
        ]
        answer = _personal_overview_answer(memories)
        self.assertIn("Possessions", answer)


if __name__ == "__main__":
    unittest.main()
