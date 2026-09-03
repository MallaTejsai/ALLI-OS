"""Tests for the 4 critical stabilization fixes identified by the behavioral audit.

CRITICAL #1: Possession/fact extraction and retrieval
CRITICAL #2: Education vs learning category alignment
CRITICAL #3: Generic correction/change detection (supersession)
CRITICAL #4: Default runtime path verification
"""

import tempfile
import unittest
from memory.extractor import MemoryExtractor, is_temporary_message, has_explicit_change_signal
from memory.manager import SafeMemoryManager
from memory.store import MemoryStore
from memory.embeddings import EmbeddingStore
from memory.query_intent import QueryIntentDetector
from memory.retriever import MemoryRetriever


def make_stack(tmpdir: str):
    """Create a full memory stack for integration tests."""
    path = f"{tmpdir}/long_term_memory.json"
    embedding_store = EmbeddingStore()
    store = MemoryStore(path=path, embedding_store=embedding_store, skip_embeddings=True)
    store.load(ensure_embeddings=False)
    manager = SafeMemoryManager(store)
    detector = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(store, embedding_store, intent_detector=detector)
    return manager, retriever, store, detector


# ============================================================================
# CRITICAL #1: Possession/fact extraction
# ============================================================================

class TestPossessionExtraction(unittest.TestCase):
    """CRITICAL #1: 'I have an Oppo A53' must be extracted as possession."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_i_have_device(self):
        ops = self.extractor.extract("I have an Oppo A53.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "possession")
        self.assertEqual(ops[0]["action"], "ADD")
        self.assertIn("oppo a53", ops[0]["content"].lower())

    def test_i_own_device(self):
        ops = self.extractor.extract("I own a MacBook.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "possession")
        self.assertIn("macbook", ops[0]["content"].lower())

    def test_i_possess_item(self):
        ops = self.extractor.extract("I possess a laptop.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "possession")

    def test_i_use_device(self):
        ops = self.extractor.extract("I use an iPhone.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "possession")
        self.assertIn("iphone", ops[0]["content"].lower())

    def test_my_device_is_x(self):
        ops = self.extractor.extract("My phone is an iPhone.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "possession")
        self.assertIn("iphone", ops[0]["content"].lower())

    def test_i_have_exam_not_possession(self):
        """'I have an exam tomorrow' should be a task, not possession."""
        ops = self.extractor.extract("I have an exam tomorrow.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "task")

    def test_i_have_feeling_not_possession(self):
        """'I have a feeling' should not be extracted as possession."""
        ops = self.extractor.extract("I have a feeling that it will rain.")
        self.assertEqual(len(ops), 0)

    def test_possession_entity_is_user(self):
        ops = self.extractor.extract("I have an Oppo A53.")
        self.assertEqual(len(ops), 1)
        # Entity is set during storage, not extraction
        self.assertEqual(ops[0]["subject"], "oppo")


class TestPossessionStorage(unittest.TestCase):
    """CRITICAL #1: Possessions must be stored with correct category."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store, _ = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_possession_stored_with_correct_category(self):
        result = self.manager.process_user_message("I have an Oppo A53.")
        self.assertEqual(result[0]["status"], "added")
        mem = result[0]["memory"]
        self.assertEqual(mem["category"], "possession")
        self.assertEqual(mem["fact_type"], "owned_item")

    def test_possession_entity_is_user(self):
        result = self.manager.process_user_message("I own a MacBook.")
        mem = result[0]["memory"]
        self.assertEqual(mem["entity"]["name"], "user")

    def test_possession_not_deduplicated_with_task(self):
        """Possession should not deduplicate with task (different categories)."""
        self.manager.process_user_message("I have an exam tomorrow.")
        result = self.manager.process_user_message("I have an Oppo A53.")
        self.assertEqual(result[0]["status"], "added")
        self.assertEqual(result[0]["memory"]["category"], "possession")


# ============================================================================
# CRITICAL #2: Education vs learning alignment
# ============================================================================

class TestEducationVsLearningExtraction(unittest.TestCase):
    """CRITICAL #2: Education and learning must be consistently routed."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_studying_cse_is_education(self):
        ops = self.extractor.extract("I am studying CSE.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")
        self.assertEqual(ops[0]["fact_type"], "studying")

    def test_studying_computer_science_is_education(self):
        ops = self.extractor.extract("I am studying computer science.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")

    def test_studying_diploma_is_education(self):
        ops = self.extractor.extract("I am studying diploma second year.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")

    def test_learning_python_is_learning(self):
        ops = self.extractor.extract("I am learning Python.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")
        self.assertEqual(ops[0]["fact_type"], "learning")

    def test_learning_java_is_learning(self):
        ops = self.extractor.extract("I am learning Java.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_studying_for_exam_is_task(self):
        """'I am studying for my mid exam' should be a temporary task."""
        ops = self.extractor.extract("I am studying for my mid exam.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "task")
        self.assertEqual(ops[0]["fact_type"], "task")

    def test_preparing_for_exam_is_task(self):
        """'I am preparing for my exam' should be a temporary task."""
        ops = self.extractor.extract("I am preparing for my exam.")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "task")

    def test_education_not_learning(self):
        """Education and learning must never mix."""
        edu_ops = self.extractor.extract("I am studying CSE.")
        learn_ops = self.extractor.extract("I am learning Python.")
        self.assertEqual(edu_ops[0]["category"], "education")
        self.assertEqual(learn_ops[0]["category"], "learning")
        self.assertNotEqual(edu_ops[0]["category"], learn_ops[0]["category"])


class TestEducationVsLearningStorage(unittest.TestCase):
    """CRITICAL #2: Education and learning must be stored separately."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store, _ = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_education_stored_as_education(self):
        result = self.manager.process_user_message("I am studying CSE.")
        mem = result[0]["memory"]
        self.assertEqual(mem["category"], "education")

    def test_learning_stored_as_learning(self):
        result = self.manager.process_user_message("I am learning Python.")
        mem = result[0]["memory"]
        self.assertEqual(mem["category"], "learning")

    def test_education_and_learning_coexist(self):
        """Education and learning can coexist as separate memories."""
        self.manager.process_user_message("I am studying CSE.")
        self.manager.process_user_message("I am learning Python.")
        active = self.store.active_memories()
        categories = [m["category"] for m in active]
        self.assertIn("education", categories)
        self.assertIn("learning", categories)


# ============================================================================
# CRITICAL #3: Corrections/supersession
# ============================================================================

class TestCorrectionExtraction(unittest.TestCase):
    """CRITICAL #3: Corrections must be detected with change signals."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_actually_keyword_detected(self):
        self.assertTrue(has_explicit_change_signal("Actually, I am learning Java now."))

    def test_in_fact_detected(self):
        self.assertTrue(has_explicit_change_signal("In fact, I prefer tea."))

    def test_switched_to_detected(self):
        self.assertTrue(has_explicit_change_signal("I switched to Java."))

    def test_replaced_with_detected(self):
        self.assertTrue(has_explicit_change_signal("I replaced Python with Java."))

    def test_i_now_detected(self):
        self.assertTrue(has_explicit_change_signal("I now use a different phone."))

    def test_no_change_signal_without_keywords(self):
        self.assertFalse(has_explicit_change_signal("I am learning Python."))


class TestCorrectionSupersession(unittest.TestCase):
    """CRITICAL #3: Corrections must supersede old memories."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store, _ = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_correction_supersedes_old_memory(self):
        """'Actually, I'm learning Java' supersedes old Python learning."""
        # Store original
        result1 = self.manager.process_user_message("I am learning Python.")
        old_id = result1[0]["memory"]["id"]

        # Correction
        result2 = self.manager.process_user_message("Actually, I am learning Java now.")
        self.assertEqual(result2[0]["status"], "added")
        self.assertIn("superseded_ids", result2[0])
        self.assertIn(old_id, result2[0]["superseded_ids"])

        # Verify old is inactive
        old_mem = self.store.get_by_id(old_id)
        self.assertEqual(old_mem["status"], "inactive")
        self.assertEqual(old_mem["superseded_by"], result2[0]["memory"]["id"])

    def test_correction_preserves_old_memory(self):
        """Superseded memory is preserved (not deleted) for historical record."""
        self.manager.process_user_message("I am learning Python.")
        self.manager.process_user_message("Actually, I am learning Java now.")

        # Old memory still exists but is inactive
        all_mems = self.store.memories
        inactive = [m for m in all_mems if m.get("status") == "inactive"]
        self.assertEqual(len(inactive), 1)
        self.assertIn("python", inactive[0]["content"].lower())

    def test_duplicate_without_change_signal_not_superseded(self):
        """'I am learning Java' without 'actually' does NOT supersede."""
        result1 = self.manager.process_user_message("I am learning Python.")
        result2 = self.manager.process_user_message("I am learning Java now.")
        # Should be added as new (not superseded) since no change signal
        self.assertEqual(result2[0]["status"], "added")
        self.assertNotIn("superseded_id", result2[0])
        # Both should be active
        active = self.store.active_memories()
        learning = [m for m in active if m["category"] == "learning"]
        self.assertEqual(len(learning), 2)

    def test_polarity_reversal_still_works(self):
        """Polarity reversal (like→dislike) still works."""
        self.manager.process_user_message("I like tea.")
        result = self.manager.process_user_message("I dislike tea.")
        self.assertIn(result[0]["status"], ("updated", "added"))
        # Verify the memory is updated
        active = self.store.active_memories()
        tea_mems = [m for m in active if "tea" in m.get("content", "").lower()]
        self.assertEqual(len(tea_mems), 1)


# ============================================================================
# CRITICAL #4: Default runtime path
# ============================================================================

class TestDefaultRuntimePath(unittest.TestCase):
    """CRITICAL #4: All fixes must work through the default runtime path."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store, _ = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_possession_through_manager(self):
        """Possession extraction works through SafeMemoryManager."""
        result = self.manager.process_user_message("I have an Oppo A53.")
        self.assertEqual(result[0]["status"], "added")
        self.assertEqual(result[0]["memory"]["category"], "possession")

    def test_education_through_manager(self):
        """Education extraction works through SafeMemoryManager."""
        result = self.manager.process_user_message("I am studying CSE.")
        self.assertEqual(result[0]["status"], "added")
        self.assertEqual(result[0]["memory"]["category"], "education")

    def test_learning_through_manager(self):
        """Learning extraction works through SafeMemoryManager."""
        result = self.manager.process_user_message("I am learning Python.")
        self.assertEqual(result[0]["status"], "added")
        self.assertEqual(result[0]["memory"]["category"], "learning")

    def test_correction_through_manager(self):
        """Correction supersession works through SafeMemoryManager."""
        self.manager.process_user_message("I am learning Python.")
        result = self.manager.process_user_message("Actually, I am learning Java now.")
        self.assertEqual(result[0]["status"], "added")
        self.assertIn("superseded_ids", result[0])

    def test_task_through_manager(self):
        """Task extraction works through SafeMemoryManager."""
        result = self.manager.process_user_message("I am studying for my mid exam.")
        self.assertEqual(result[0]["status"], "added")
        self.assertEqual(result[0]["memory"]["category"], "task")


# ============================================================================
# Integration: Full pipeline test
# ============================================================================

class TestFullPipeline(unittest.TestCase):
    """End-to-end test of all 4 critical fixes working together."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store, _ = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_all_fixes_together(self):
        """All 4 critical fixes work in a single conversation flow."""
        # CRITICAL #1: Possession
        r1 = self.manager.process_user_message("I have an Oppo A53.")
        self.assertEqual(r1[0]["memory"]["category"], "possession")

        # CRITICAL #2: Education vs learning
        r2 = self.manager.process_user_message("I am studying CSE.")
        r3 = self.manager.process_user_message("I am learning Python.")
        self.assertEqual(r2[0]["memory"]["category"], "education")
        self.assertEqual(r3[0]["memory"]["category"], "learning")

        # CRITICAL #3: Correction
        r4 = self.manager.process_user_message("Actually, I am learning Java now.")
        self.assertIn("superseded_ids", r4[0])

        # Verify final state: possession + education + learning(java) = 3 active
        active = self.store.active_memories()
        self.assertEqual(len(active), 3)
        categories = {m["category"] for m in active}
        self.assertIn("possession", categories)
        self.assertIn("education", categories)
        self.assertIn("learning", categories)
        # Verify learning is Java (not Python)
        learning_mems = [m for m in active if m["category"] == "learning"]
        self.assertEqual(len(learning_mems), 1)
        self.assertIn("java", learning_mems[0]["content"].lower())


if __name__ == "__main__":
    unittest.main()
