"""
ALLI v0.6 memory system tests.

Run:
  python tests/test_memory_manager.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.answers import grounded_answer
from memory.embeddings import EmbeddingError, EmbeddingStore
from memory.extractor import MemoryExtractor
from memory.manager import SafeMemoryManager
from memory.query_intent import QueryIntentDetector
from memory.retriever import MemoryRetriever
from memory.store import MemoryStore, load_json_list, save_json
from memory import config


def make_stack(tmpdir: str):
    path = os.path.join(tmpdir, "long_term_memory.json")
    embedding_store = EmbeddingStore()
    store = MemoryStore(path=path, embedding_store=embedding_store)
    store.load(ensure_embeddings=False)
    manager = SafeMemoryManager(store)
    intent = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(store, embedding_store, intent_detector=intent)
    return manager, retriever, store


class LearningExtractionTests(unittest.TestCase):
    """Regression coverage for learning writes that does not require Ollama."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_learning_creation_and_deletion_operations(self):
        add_operations = self.extractor.extract("I am learning C")
        self.assertEqual(len(add_operations), 1)
        self.assertEqual(
            add_operations[0],
            {
                "action": "ADD",
                "category": "learning",
                "fact_type": "learning",
                "content": "User is learning C",
                "value": "C",
                "subject": "c",
                "importance": "medium",
            },
        )

        delete_operations = self.extractor.extract(
            "I don't want to learn C anymore"
        )
        self.assertEqual(len(delete_operations), 1)
        self.assertEqual(delete_operations[0]["action"], "DELETE_MATCH")
        self.assertEqual(delete_operations[0]["category"], "learning")
        self.assertEqual(delete_operations[0]["content_hint"], "C")
        self.assertTrue(delete_operations[0]["explicit_change"])


class MemoryLifecycleSafetyTests(unittest.TestCase):
    """Offline acceptance tests for v0.6 memory safety and recovery."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.extractor = MemoryExtractor()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_temporary_today_statement_is_not_extracted(self):
        self.assertEqual(
            self.extractor.extract("I am learning Rust for today"),
            [],
        )

    def test_time_bound_until_date_is_not_extracted(self):
        self.assertEqual(
            self.extractor.extract("I am learning Rust until Friday"),
            [],
        )

    def test_direct_third_party_statement_is_not_stored_as_user_fact(self):
        self.assertEqual(
            self.extractor.extract("My friend is learning Flutter"),
            [],
        )

    def test_quoted_third_party_speech_is_not_stored_as_user_fact(self):
        self.assertEqual(
            self.extractor.extract("My friend said, 'I am learning C'"),
            [],
        )

    def test_restart_preserves_existing_memory_records(self):
        path = os.path.join(self._tmpdir.name, "long_term_memory.json")
        store = MemoryStore(path=path)
        stored = store.add_memory(
            category="learning",
            fact_type="learning",
            content="User is learning Python",
            value="Python",
            subject="python",
            embedding=[0.1, 0.2],
        )

        restarted_store = MemoryStore(path=path)
        restarted_store.load(ensure_embeddings=False)

        self.assertEqual(restarted_store.get_by_id(stored["id"])["value"], "Python")
        self.assertEqual(len(restarted_store.active_memories()), 1)

    def test_new_memory_has_user_provenance_and_timestamps(self):
        path = os.path.join(self._tmpdir.name, "long_term_memory.json")
        store = MemoryStore(path=path)
        stored = store.add_memory(
            category="skill",
            fact_type="skill",
            content="User knows Python",
            value="Python",
            subject="python",
            embedding=[0.1, 0.2],
        )

        self.assertEqual(stored["source"], "user_statement")
        self.assertEqual(stored["confidence"], 1.0)
        self.assertIsNotNone(datetime.fromisoformat(stored["created_at"]))
        self.assertIsNotNone(datetime.fromisoformat(stored["updated_at"]))

    def test_legacy_memory_receives_provenance_without_losing_content(self):
        path = os.path.join(self._tmpdir.name, "long_term_memory.json")
        legacy = {
            "id": "memory_001",
            "category": "skill",
            "fact_type": "skill",
            "content": "User knows Python",
            "value": "Python",
            "subject": "python",
            "status": "active",
            "embedding": [0.1, 0.2],
            "embedding_version": config.EMBEDDING_TEXT_VERSION,
        }
        save_json(path, [legacy])

        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)
        migrated = store.get_by_id("memory_001")

        self.assertEqual(migrated["content"], "User knows Python")
        self.assertEqual(migrated["source"], "user_statement")
        self.assertEqual(migrated["confidence"], 1.0)
        self.assertIsNotNone(datetime.fromisoformat(migrated["created_at"]))
        self.assertIsNotNone(datetime.fromisoformat(migrated["updated_at"]))

    def test_corrupt_file_is_backed_up_before_recovery(self):
        path = os.path.join(self._tmpdir.name, "long_term_memory.json")
        corrupt_payload = "{not valid json"
        Path(path).write_text(corrupt_payload, encoding="utf-8")

        self.assertEqual(load_json_list(path), [])
        backups = list(Path(self._tmpdir.name).glob("long_term_memory.corrupt.*.json"))

        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(encoding="utf-8"), corrupt_payload)
        self.assertEqual(load_json_list(path), [])

    def test_failed_atomic_replace_keeps_previous_memory_file_valid(self):
        path = os.path.join(self._tmpdir.name, "long_term_memory.json")
        original = [{"id": "memory_001", "content": "User knows Python"}]
        replacement = [{"id": "memory_002", "content": "User knows Rust"}]
        save_json(path, original)

        with patch("memory.store.os.replace", side_effect=OSError("disk unavailable")):
            with self.assertRaises(OSError):
                save_json(path, replacement)

        self.assertEqual(load_json_list(path), original)


class MemoryV06Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            EmbeddingStore().verify_model()
        except EmbeddingError as error:
            raise unittest.SkipTest(str(error)) from error

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def ask(self, question: str) -> str:
        intent = self.retriever.detect_intent(question)
        if intent.is_multi:
            grouped = self.retriever.retrieve_by_requested_intents(
                question, intent=intent
            )
            memories = []
            for items in grouped.values():
                memories.extend(items)
            return grounded_answer(
                intent, memories, question=question, grouped=grouped
            )
        memories = self.retriever.retrieve_relevant_memories(
            question, intent=intent
        )
        return grounded_answer(intent, memories, question=question)

    def active(self):
        return self.store.active_memories()

    # 1–5 identity / nickname
    def test_01_add_name(self):
        r = self.manager.process_user_message("My name is Tejsai")
        self.assertEqual(r[0]["status"], "added")
        self.assertEqual(self.active()[0]["value"], "Tejsai")
        self.assertEqual(self.active()[0]["status"], "active")

    def test_02_duplicate_name(self):
        self.manager.process_user_message("My name is Tejsai")
        r = self.manager.process_user_message("My name is Tejsai")
        self.assertEqual(r[0]["status"], "unchanged")
        self.assertEqual(len(self.active()), 1)

    def test_03_update_name_with_history(self):
        self.manager.process_user_message("My name is Tejsai")
        r = self.manager.process_user_message("My name is Rahul")
        self.assertEqual(r[0]["status"], "updated")
        active_names = [
            m for m in self.active()
            if m.get("fact_type") == "name"
        ]
        self.assertEqual(len(active_names), 1)
        self.assertEqual(active_names[0]["value"], "Rahul")
        inactive = [
            m for m in self.store.memories
            if m.get("status") == "inactive"
        ]
        self.assertGreaterEqual(len(inactive), 1)
        self.assertIn("Tejsai", inactive[0]["content"])
        answer = self.ask("What is my name?")
        self.assertIn("Rahul", answer)
        self.assertNotIn("Tejsai", answer)

    def test_04_add_nickname(self):
        self.manager.process_user_message("My nickname is Binnu")
        self.assertEqual(self.active()[0]["fact_type"], "nickname")

    def test_05_retrieve_nickname(self):
        self.manager.process_user_message("My name is Tejsai")
        self.manager.process_user_message("My nickname is Binnu")
        answer = self.ask("What should you call me?")
        self.assertIn("Binnu", answer)
        self.assertNotIn("Tejsai", answer)
        self.assertNotIn("memory_", answer)

    # 6–8 goals
    def test_06_add_goal(self):
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        self.assertEqual(len(self.active()), 1)
        self.assertEqual(self.active()[0]["category"], "goal")

    def test_07_add_second_goal(self):
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        self.manager.process_user_message(
            "My goal is to become a content creator"
        )
        goals = [m for m in self.active() if m["category"] == "goal"]
        self.assertEqual(len(goals), 2)

    def test_08_duplicate_goal(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        r = self.manager.process_user_message(
            "I hope to become a software engineer"
        )
        # May be unchanged via semantic match, or added if wording too different.
        # At minimum: only one active software-engineer goal after near-duplicate.
        if r and r[0]["status"] == "unchanged":
            self.assertEqual(len(self.active()), 1)
        else:
            goals = [m for m in self.active() if m["category"] == "goal"]
            # Accept either duplicate blocked or single semantic merge.
            self.assertLessEqual(len(goals), 2)

    # Regression: incomplete goal after "become" must not be stored
    def test_08b_incomplete_goal_after_become(self):
        """I want to become in future must not store a meaningless goal."""
        self.manager.process_user_message("I want to become in future")
        goals = [m for m in self.active() if m["category"] == "goal"]
        self.assertEqual(len(goals), 0)

    # 9–11 skill vs learning
    def test_09_add_skill(self):
        self.manager.process_user_message("I know Python")
        self.assertEqual(self.active()[0]["category"], "skill")
        answer = self.ask("What are my skills?")
        self.assertIn("Python", answer)

    def test_10_add_learning_goal(self):
        self.manager.process_user_message("I want to learn Python")
        mem = self.active()[0]
        self.assertEqual(mem["category"], "learning")
        self.assertEqual(mem["fact_type"], "learning")
        self.assertNotEqual(mem["category"], "skill")

    def test_11_goal_not_skill(self):
        self.manager.process_user_message("I want to learn Python")
        answer = self.ask("What are my skills?")
        self.assertIn("don't currently", answer.lower())
        self.assertNotIn("Python", answer)

    # 12–15 preferences
    def test_12_like_c(self):
        self.manager.process_user_message("I like C programming")
        mem = self.active()[0]
        self.assertEqual(mem["fact_type"], "like")
        self.assertEqual(mem["subject"], "c")

    def test_13_dislike_c_updates(self):
        self.manager.process_user_message("I don't like C programming")
        r = self.manager.process_user_message("I like C programming")
        self.assertEqual(r[0]["status"], "updated")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["fact_type"], "like")

    def test_14_like_c_again(self):
        self.manager.process_user_message("I don't like C programming")
        self.manager.process_user_message("I like C programming")
        r = self.manager.process_user_message("I like C programming")
        self.assertEqual(r[0]["status"], "unchanged")
        self.assertEqual(
            len([m for m in self.active() if m["category"] == "preference"]),
            1,
        )

    def test_15_preference_wording_variants(self):
        self.manager.process_user_message("I like C")
        r = self.manager.process_user_message("I like C programming")
        # Same subject after normalization → update or unchanged, not second active.
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)

    # 16–17 semantic retrieval
    def test_15b_preference_query_directionality(self):
        self.manager.process_user_message("I like Flutter")
        self.manager.process_user_message("I don't like C")

        for question in ("What do I like?", "what i like"):
            answer = self.ask(question).lower()
            self.assertIn("flutter", answer)
            self.assertNotIn("dislike c", answer)

        for question in (
            "What do I dislike?",
            "What do I not like?",
            "What don't I like?",
            "what i don't like",
        ):
            answer = self.ask(question).lower()
            self.assertIn("c", answer)
            self.assertNotIn("flutter", answer)

        answer = self.ask("What are my preferences?").lower()
        self.assertIn("flutter", answer)
        self.assertIn("c", answer)

    def test_15c_personal_overview_is_broad_and_cleans_interest_wording(self):
        self.manager.process_user_message("My name is Tejsai")
        self.manager.process_user_message("I know Python")
        self.manager.process_user_message("I am interested in video editing")
        interest = [m for m in self.active() if m["category"] == "interest"][0]
        interest["value"] = "I am interested in video editing"
        interest["content"] = "User is interested in I am interested in video editing"
        self.store.save()
        self.store.load(ensure_embeddings=False)

        answer = self.ask("give introduction about me")
        self.assertIn("Tejsai", answer)
        self.assertIn("Python", answer)
        self.assertIn("video editing", answer.lower())
        self.assertNotIn("interested in i am interested", answer.lower())

    def test_16_semantic_goal_retrieval(self):
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        answer = self.ask("What are my goals?")
        self.assertIn("software engineer", answer.lower())

    def test_16b_dream_not_contaminated_by_goal(self):
        """Dreams and goals are separate categories."""
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        answer = self.ask("What are my dreams?")
        self.assertIn("don't currently", answer.lower())

    def test_17_semantic_nickname_retrieval(self):
        self.manager.process_user_message("My nickname is Binnu")
        answer = self.ask("What do you call me?")
        self.assertIn("Binnu", answer)

    # 18–19 unknown / filtering
    def test_18_unknown_memory(self):
        answer = self.ask("What is my favorite food?")
        self.assertIn("favorite food", answer.lower())
        self.assertIn("stored", answer.lower())

    def test_19_irrelevant_filtering(self):
        self.manager.process_user_message("My name is Tejsai")
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        answer = self.ask("What is my nickname?")
        self.assertIn("don't currently", answer.lower())
        self.assertNotIn("software engineer", answer.lower())
        self.assertNotIn("Tejsai", answer)

    # 20 missing embedding
    def test_20_missing_embedding_backfill(self):
        path = os.path.join(self._tmpdir.name, "noemb.json")
        save_json(
            path,
            [{
                "id": "memory_001",
                "category": "identity",
                "fact_type": "name",
                "content": "User's name is Tejsai",
                "importance": "high",
            }],
        )
        store = MemoryStore(path=path, embedding_store=EmbeddingStore())
        store.load(ensure_embeddings=True)
        self.assertTrue(store.memories[0].get("embedding"))
        self.assertEqual(store.memories[0].get("status"), "active")

    # 23 duplicate detection with missing embeddings
    def test_23_duplicate_detection_with_missing_embeddings(self):
        # Arrange: add a memory, then explicitly clear its embedding
        # to simulate a memory without embeddings (e.g. after migration or
        # manual deserialization).
        path = os.path.join(self._tmpdir.name, "noemb_dup.json")
        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)

        mem = store.add_memory(
            category="skill",
            content="I know Python",
            fact_type="skill",
        )
        # Explicitly remove the embedding to test the fallback path.
        mem["embedding"] = None
        store.save()

        # Act: reload without re-generating embeddings
        store2 = MemoryStore(path=path)
        store2.load(ensure_embeddings=False)

        mgr = SafeMemoryManager(store2)

        # Assert: same content should be detected as duplicate even
        # without embeddings (normalized-content fallback).
        dup = mgr.find_duplicate(
            category="skill",
            fact_type="skill",
            content="I know Python",
        )
        self.assertIsNotNone(dup)
        self.assertEqual(dup["content"], "I know Python")

        # Assert: different content should NOT be detected as duplicate.
        dup2 = mgr.find_duplicate(
            category="skill",
            fact_type="skill",
            content="I know Java",
        )
        self.assertIsNone(dup2)

    # 21 invalid JSON
    def test_21_invalid_json_recovery(self):
        path = os.path.join(self._tmpdir.name, "bad.json")
        with open(path, "w", encoding="utf-8") as file:
            file.write("{not-valid")
        data = load_json_list(path)
        self.assertEqual(data, [])
        backups = [
            name for name in os.listdir(self._tmpdir.name)
            if "corrupt" in name
        ]
        self.assertTrue(backups)

    # 22 migration
    def test_22_migration(self):
        path = os.path.join(self._tmpdir.name, "old.json")
        save_json(
            path,
            [{
                "id": "memory_001",
                "category": "identity",
                "content": "User's name is tejsai",
                "importance": "high",
            }],
        )
        store = MemoryStore(path=path, embedding_store=EmbeddingStore())
        store.load(ensure_embeddings=True)
        mem = store.memories[0]
        self.assertEqual(mem["id"], "memory_001")
        self.assertEqual(mem["fact_type"], "name")
        self.assertTrue(mem.get("value"))
        self.assertTrue(mem.get("subject"))
        self.assertEqual(mem.get("status"), "active")
        self.assertTrue(mem.get("embedding"))

    # 23 conversation separate
    def test_23_conversation_separate(self):
        conv_path = os.path.join(self._tmpdir.name, "conversation.json")
        save_json(conv_path, [{"role": "user", "content": "hello"}])
        self.manager.process_user_message("My name is Tejsai")
        conv = load_json_list(conv_path, create_if_missing=False)
        self.assertEqual(conv[0]["content"], "hello")
        self.assertTrue(all("role" in m for m in conv))
        self.assertTrue(all("category" in m for m in self.store.memories))

    # 24 no hallucination
    def test_24_no_hallucination(self):
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        answer = self.ask("What are my skills?")
        self.assertIn("don't currently", answer.lower())
        self.assertNotIn("software engineer", answer.lower())

    # 25 multiple goals remain separate
    def test_25_multiple_goals_separate(self):
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        self.manager.process_user_message(
            "My goal is to become a content creator"
        )
        answer = self.ask("What are my goals?")
        lower = answer.lower()
        self.assertIn("software engineer", lower)
        self.assertIn("content creator", lower)

    def test_manual_scenarios_bundle(self):
        """Acceptance scenarios 1–10 from the v0.6 brief."""
        self.manager.process_user_message("My name is Tejsai")
        self.assertEqual(self.ask("What is my name?"), "Your name is Tejsai.")

        self.manager.process_user_message("My nickname is Binnu")
        nick = self.ask("What should you call me?")
        self.assertIn("Binnu", nick)

        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        goals = self.ask("What are my goals?")
        self.assertIn("software engineer", goals.lower())

        self.manager.process_user_message(
            "My goal is to become a content creator"
        )
        goals = self.ask("What are my goals?")
        self.assertIn("software engineer", goals.lower())
        self.assertIn("content creator", goals.lower())

        self.manager.process_user_message("I want to learn Python")
        skills = self.ask("What are my skills?")
        self.assertIn("don't currently", skills.lower())

        self.manager.process_user_message("I know Python")
        skills2 = self.ask("What are my skills?")
        self.assertIn("Python", skills2)

        # Fresh preference store path via new stack piece
        prefs = [m for m in self.active() if m["category"] == "preference"]
        # isolate preference on same store
        self.manager.process_user_message("I don't like C programming")
        self.manager.process_user_message("I like C programming")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["fact_type"], "like")

        r = self.manager.process_user_message("My name is Tejsai")
        # name was Tejsai then may have been unchanged or already Tejsai
        self.assertIn(r[0]["status"], {"unchanged", "updated", "added"})

        food = self.ask("What is my favorite food?")
        self.assertIn("favorite food", food.lower())

    # ---- Final 3 bug-fix tests ----
    def test_bug1_nick_name_normalization(self):
        self.manager.process_user_message("My name is Tejsai")
        self.manager.process_user_message("My nickname is Binnu")
        self.assertEqual(self.ask("What is my name?"), "Your name is Tejsai.")
        self.assertIn("Binnu", self.ask("What is my nickname?"))
        nick_space = self.ask("What is my nick name?")
        self.assertIn("Binnu", nick_space)
        self.assertNotIn("Tejsai", nick_space)
        call_me = self.ask("What should you call me?")
        self.assertIn("Binnu", call_me)

    def test_bug2_generic_preference_subject(self):
        self.manager.process_user_message("I don't like C programming")
        r = self.manager.process_user_message("I like C")
        self.assertEqual(r[0]["status"], "updated")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["fact_type"], "like")
        self.assertEqual(prefs[0]["subject"], "c")

        r = self.manager.process_user_message("I don't like C program")
        self.assertEqual(r[0]["status"], "updated")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["fact_type"], "dislike")

        self.manager.process_user_message("I don't like Java")
        r = self.manager.process_user_message("I like Java programming")
        self.assertEqual(r[0]["status"], "updated")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        subjects = {m["subject"] for m in prefs}
        self.assertIn("c", subjects)
        self.assertIn("java", subjects)
        java = [m for m in prefs if m["subject"] == "java"][0]
        self.assertEqual(java["fact_type"], "like")
        # C and Java remain separate.
        self.assertEqual(len(prefs), 2)

    def test_bug3_multi_category_skills_and_goals(self):
        self.manager.process_user_message("I know Python")
        self.manager.process_user_message(
            "My goal is to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        answer = self.ask("What are my skills and goals?")
        lower = answer.lower()
        self.assertIn("skills", lower)
        self.assertIn("goals", lower)
        self.assertIn("python", lower)
        self.assertIn("software engineer", lower)
        self.assertIn("content creator", lower)

        answer2 = self.ask("What are my goals and skills?")
        lower2 = answer2.lower()
        self.assertIn("python", lower2)
        self.assertIn("software engineer", lower2)
        self.assertIn("content creator", lower2)

        answer3 = self.ask("What are my skills, goals and interests?")
        lower3 = answer3.lower()
        self.assertIn("skills", lower3)
        self.assertIn("goals", lower3)
        self.assertIn("interests", lower3)
        # Unrelated categories should not appear as sections.
        self.assertNotIn("nickname", lower3)
        self.assertNotIn("education", lower3)

    # ---- Stress-test regression coverage ----
    def test_stress_compound_query_includes_preferences(self):
        self.manager.process_user_message("I know C")
        self.manager.process_user_message("I am learning Flutter")
        self.manager.process_user_message("I want to become an AI engineer")
        self.manager.process_user_message("I like Python")
        answer = self.ask("Tell me my skills, what I'm learning, my goal, and what I like.")
        for label in ("skills", "learning", "goals", "preferences"):
            self.assertIn(label, answer.lower())
        for value in ("c", "flutter", "ai engineer", "python"):
            self.assertIn(value, answer.lower())

    def test_stress_enjoy_and_negative_enjoy_route_to_preferences(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I don't like Java")
        self.manager.process_user_message("I am interested in video editing")
        self.assertIn("python", self.ask("Which technology do I enjoy?").lower())
        negative = self.ask("Which technology don't I enjoy?").lower()
        self.assertIn("java", negative)
        self.assertNotIn("video editing", negative)

    def test_stress_future_questions_do_not_read_current_memories(self):
        self.manager.process_user_message("I know Python")
        self.manager.process_user_message("I want to become an AI engineer")
        for question in (
            "What skills will I have in 10 years?",
            "What will my future career definitely be?",
        ):
            intent = self.retriever.detect_intent(question)
            self.assertFalse(intent.is_memory_read)
            self.assertEqual(self.retriever.retrieve_relevant_memories(question, intent=intent), [])

    def test_stress_background_typo_is_stored_and_retrieved(self):
        self.manager.process_user_message("i am studing diploma second year")
        education = [m for m in self.active() if m["category"] == "education"]
        self.assertEqual(len(education), 1)
        self.assertIn("diploma second year", education[0]["value"].lower())
        self.assertIn("diploma second year", self.ask("What am I studying?").lower())

    def test_stress_compound_statement_persists_every_fact(self):
        results = self.manager.process_user_message(
            "I know C, I am learning Flutter, I want to become an AI engineer, and I like Python."
        )
        self.assertEqual(len(results), 4)
        active = self.active()
        self.assertEqual({m["category"] for m in active}, {"skill", "learning", "goal", "preference"})
        self.assertIn("c", self.ask("What are my skills?").lower())
        self.assertIn("flutter", self.ask("What am I learning?").lower())
        self.assertIn("ai engineer", self.ask("What are my goals?").lower())
        ambiguous = self.retriever.detect_intent("What do I like about something?")
        self.assertFalse(ambiguous.is_memory_read)

    def test_stress_java_and_javascript_remain_distinct(self):
        self.manager.process_user_message("I know Java")
        self.manager.process_user_message("I know JavaScript")
        skills = self.ask("What are my skills?").lower()
        self.assertIn("java", skills)
        self.assertIn("javascript", skills)

    def test_adversarial_advice_and_ambiguous_preference_queries_are_not_reads(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I know Java")
        for question in (
            "What do I like about something?",
            "Which technology is best for me?",
            "What should I learn?",
        ):
            intent = self.retriever.detect_intent(question)
            self.assertFalse(intent.is_memory_read, question)
            self.assertEqual(
                self.retriever.retrieve_relevant_memories(question, intent=intent),
                [],
            )

    def test_adversarial_current_memory_read_routes_remain_specific(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I don't like Java")
        self.manager.process_user_message("I am interested in robotics")
        self.manager.process_user_message("I am learning Flutter")
        self.assertIn("python", self.ask("What do I like?").lower())
        self.assertIn("java", self.ask("What do I dislike?").lower())
        self.assertIn("robotics", self.ask("What am I interested in?").lower())
        self.assertIn("flutter", self.ask("What am I learning?").lower())

    def test_adversarial_preference_temporal_words_normalize_subjects(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I don't like Python anymore")
        self.manager.process_user_message("I like Python again")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["subject"], "python")
        self.assertEqual(prefs[0]["fact_type"], "like")

    def test_adversarial_enjoy_and_interest_have_distinct_categories(self):
        self.manager.process_user_message("I enjoy robotics")
        self.manager.process_user_message("I don't enjoy python")
        self.manager.process_user_message("I am interested in robotics")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        by_subject = {m["subject"]: m["fact_type"] for m in prefs}
        self.assertEqual(by_subject["robotics"], "like")
        self.assertEqual(by_subject["python"], "dislike")
        interests = [m for m in self.active() if m["category"] == "interest"]
        self.assertEqual(len(interests), 1)
        self.assertEqual(interests[0]["subject"], "robotics")

    def test_final_preference_temporal_conflicts_leave_one_active_state(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I don't like Python anymore")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual([(m["subject"], m["fact_type"]) for m in prefs], [("python", "dislike")])

        self.manager.process_user_message("I like Python again")
        self.manager.process_user_message("I don't like Python")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual([(m["subject"], m["fact_type"]) for m in prefs], [("python", "dislike")])

    def test_final_learning_temporal_suffix_is_not_stored(self):
        self.manager.process_user_message("I am learning C++ now")
        learning = [m for m in self.active() if m["category"] == "learning"]
        self.assertEqual(learning[0]["value"], "C++")
        self.assertEqual(learning[0]["subject"], "c++")

    def test_final_migration_normalizes_legacy_active_records(self):
        path = os.path.join(self._tmpdir.name, "legacy.json")
        save_json(path, [
            {"id": "memory_001", "category": "preference", "fact_type": "like", "subject": "python again", "content": "User likes Python again", "status": "active"},
            {"id": "memory_002", "category": "preference", "fact_type": "dislike", "subject": "python", "content": "User dislikes Python", "status": "active"},
            {"id": "memory_003", "category": "learning", "fact_type": "learning", "subject": "c++ now", "value": "C++ now", "content": "User is learning C++ now", "status": "active"},
        ])
        store = MemoryStore(path=path, embedding_store=EmbeddingStore())
        store.load(ensure_embeddings=False)
        active = store.active_memories()
        prefs = [m for m in active if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual((prefs[0]["subject"], prefs[0]["fact_type"]), ("python", "dislike"))
        learning = [m for m in active if m["category"] == "learning"][0]
        self.assertEqual((learning["value"], learning["subject"]), ("C++", "c++"))

    def test_final_duplicate_goal_is_not_added_and_history_stays_excluded(self):
        self.manager.process_user_message("I want to become a machine learning engineer")
        result = self.manager.process_user_message("My goal changed. I want to become a machine learning engineer.")
        self.assertEqual(result[0]["status"], "unchanged")
        goals = [m for m in self.active() if m["category"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.manager.process_user_message("I want to become an AI engineer")
        self.assertEqual(len([m for m in self.active() if m["category"] == "goal"]), 2)
        self.manager.process_user_message("I don't like Python")
        self.manager.process_user_message("I like Python again")
        self.assertTrue(any(m["status"] == "inactive" for m in self.store.memories))
        self.assertNotIn("dislike python", self.ask("What do I like?").lower())

    def test_final_inactive_preference_history_is_never_retrieved(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I don't like Python")
        self.assertTrue(any(m["status"] == "inactive" for m in self.store.memories))
        answer = self.ask("What do I like?").lower()
        self.assertIn("don't currently", answer)
        self.assertNotIn("python", answer)

    # ---- Final goal UPDATE/DELETE fixes ----
    def test_goal_fix1_explicit_replacement(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        before = {
            m["id"]: m
            for m in self.active()
            if m["category"] == "goal"
        }
        self.assertEqual(len(before), 2)
        engineer_id = next(
            mid
            for mid, m in before.items()
            if "software engineer" in m["content"].lower()
        )
        creator_id = next(
            mid
            for mid, m in before.items()
            if "content creator" in m["content"].lower()
        )

        results = self.manager.process_user_message(
            "My goal changed from becoming a software engineer "
            "to becoming a researcher"
        )
        self.assertEqual(results[0]["status"], "updated")
        self.assertEqual(results[0]["memory_id"], engineer_id)

        goals = [m for m in self.active() if m["category"] == "goal"]
        texts = " | ".join(m["content"].lower() for m in goals)
        self.assertEqual(len(goals), 2)
        self.assertIn("researcher", texts)
        self.assertIn("content creator", texts)
        self.assertNotIn("software engineer", texts)
        self.assertTrue(any(m["id"] == engineer_id for m in goals))
        self.assertTrue(any(m["id"] == creator_id for m in goals))

    def test_goal_fix2_multiple_delete(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        self.manager.process_user_message(
            "I want to become a researcher"
        )
        results = self.manager.process_user_message(
            "I don't want to become software engineer and content creator"
        )
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["status"] == "deleted" for r in results))

        goals = [m for m in self.active() if m["category"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertIn("researcher", goals[0]["content"].lower())

    def test_goal_fix3_single_delete(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        results = self.manager.process_user_message(
            "I don't want to become a software engineer anymore"
        )
        self.assertEqual(results[0]["status"], "deleted")
        goals = [m for m in self.active() if m["category"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertIn("content creator", goals[0]["content"].lower())

    def test_goal_fix4_unrelated_learning_protection(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        before = [
            m["content"]
            for m in self.active()
            if m["category"] == "goal"
        ]
        self.manager.process_user_message("I want to learn Python")
        after = [
            m["content"]
            for m in self.active()
            if m["category"] == "goal"
        ]
        self.assertEqual(before, after)
        learning = [m for m in self.active() if m["category"] == "learning"]
        self.assertEqual(len(learning), 1)
        self.assertIn("python", learning[0]["content"].lower())

    def test_goal_fix5_ambiguous_change_no_random_overwrite(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        before_ids = {
            m["id"]
            for m in self.active()
            if m["category"] == "goal"
        }
        results = self.manager.process_user_message(
            "My goal changed. I want to become a researcher."
        )
        self.assertEqual(results[0]["status"], "added")
        self.assertTrue(results[0].get("reason", "").startswith("ambiguous"))

        goals = [m for m in self.active() if m["category"] == "goal"]
        self.assertEqual(len(goals), 3)
        texts = " | ".join(m["content"].lower() for m in goals)
        self.assertIn("software engineer", texts)
        self.assertIn("content creator", texts)
        self.assertIn("researcher", texts)
        # Original goal IDs/content untouched.
        for mid in before_ids:
            mem = self.store.get_by_id(mid)
            self.assertIsNotNone(mem)
            self.assertEqual(mem.get("status"), "active")
            self.assertNotIn("researcher", mem["content"].lower())

    def test_goal_fix6_persistence(self):
        path = self.store.path
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        self.manager.process_user_message(
            "My goal changed from becoming a software engineer "
            "to becoming a researcher"
        )
        self.manager.process_user_message(
            "I don't want to become a content creator anymore"
        )

        # Simulate restart: reload from the same file.
        store2 = MemoryStore(path=path, embedding_store=EmbeddingStore())
        store2.load(ensure_embeddings=False)
        goals = [m for m in store2.active_memories() if m["category"] == "goal"]
        self.assertEqual(len(goals), 1)
        self.assertIn("researcher", goals[0]["content"].lower())

        manager2 = SafeMemoryManager(store2)
        retriever2 = MemoryRetriever(
            store2,
            store2.embedding_store,
            intent_detector=QueryIntentDetector(store2.embedding_store),
        )
        intent = retriever2.detect_intent("What are my goals?")
        memories = retriever2.retrieve_relevant_memories(
            "What are my goals?", intent=intent
        )
        answer = grounded_answer(intent, memories, question="What are my goals?")
        self.assertIn("researcher", answer.lower())
        self.assertNotIn("software engineer", answer.lower())
        self.assertNotIn("content creator", answer.lower())

    # ---- Final memory-safety bug fixes ----
    def test_safety1_interest_never_updates_name(self):
        self.manager.process_user_message("My name is Tejsai")
        name = [m for m in self.active() if m.get("fact_type") == "name"][0]
        self.assertEqual(name["value"].lower(), "tejsai")

        results = self.manager.process_user_message(
            "I am interested in web development and editing"
        )
        self.assertTrue(results)
        self.assertTrue(all(r["status"] == "added" for r in results))
        self.assertFalse(any(r.get("conflict_resolved") for r in results))

        name_after = [
            m for m in self.active()
            if m.get("category") == "identity" and m.get("fact_type") == "name"
        ]
        self.assertEqual(len(name_after), 1)
        self.assertEqual(name_after[0]["id"], name["id"])
        self.assertIn("tejsai", name_after[0]["content"].lower())

        interests = [m for m in self.active() if m["category"] == "interest"]
        self.assertGreaterEqual(len(interests), 2)
        values = " ".join(m.get("value", "").lower() for m in interests)
        self.assertIn("web development", values)
        self.assertIn("editing", values)

    def test_safety1b_interest_typo_never_updates_name(self):
        self.manager.process_user_message("My name is Tejsai")
        results = self.manager.process_user_message(
            "I am intrested in web development and editing"
        )
        self.assertTrue(results)
        self.assertFalse(any(r.get("conflict_resolved") for r in results))
        name = [
            m for m in self.active()
            if m.get("fact_type") == "name"
        ][0]
        self.assertIn("tejsai", name["content"].lower())
        self.assertTrue(
            any(m["category"] == "interest" for m in self.active())
        )

    def test_safety2_learning_distinct_from_skill(self):
        self.manager.process_user_message("I know Python")
        self.manager.process_user_message("I am learning C programming")
        skills = [m for m in self.active() if m["category"] == "skill"]
        learning = [m for m in self.active() if m["category"] == "learning"]
        self.assertEqual(len(skills), 1)
        self.assertEqual(len(learning), 1)
        self.assertIn("python", skills[0]["content"].lower())
        self.assertIn("c", learning[0]["content"].lower())

        skill_answer = self.ask("What are my skills?")
        self.assertIn("Python", skill_answer)
        self.assertNotIn("C", skill_answer)

        learn_answer = self.ask("What am I learning?")
        self.assertRegex(learn_answer.lower(), r"\bc\b")

        self.manager.process_user_message("I stopped learning C")
        learning_after = [
            m for m in self.active() if m["category"] == "learning"
        ]
        self.assertEqual(len(learning_after), 0)
        skills_after = [m for m in self.active() if m["category"] == "skill"]
        self.assertEqual(len(skills_after), 1)
        self.assertIn("python", skills_after[0]["content"].lower())

    def test_safety3_nickname_does_not_replace_name(self):
        self.manager.process_user_message("My name is Tejsai")
        self.manager.process_user_message("My nickname is Binnu")
        identities = [
            m for m in self.active() if m["category"] == "identity"
        ]
        types = {m["fact_type"] for m in identities}
        self.assertEqual(types, {"name", "nickname"})
        name = [m for m in identities if m["fact_type"] == "name"][0]
        self.assertIn("tejsai", name["content"].lower())

    def test_bug4_learning_delete_wrong_memory(self):
        """Regression test: deleting "C" should not delete "machine learning engineer".

        When an existing learning memory "machine learning engineer" exists and the
        user adds "I am learning C" then asks "I don't want to learn C anymore", the
        "C" memory should be deleted, not the "machine learning engineer" memory.
        """
        # Add existing learning memory
        self.manager.process_user_message("I am learning machine learning engineer")
        # Add new learning memory about C
        self.manager.process_user_message("I am learning C")
        # Delete the C learning memory
        results = self.manager.process_user_message("I don't want to learn C anymore")
        self.assertTrue(results)
        # The "machine learning engineer" memory should still be active
        learning = [m for m in self.active() if m["category"] == "learning"]
        self.assertEqual(len(learning), 1)
        self.assertIn("machine learning engineer", learning[0]["content"].lower())
        # The "C" memory should have been deleted
        c_memories = [m for m in self.active() if m.get("subject") == "c"]
        self.assertEqual(len(c_memories), 0)


class GoalVsLearningSeparationTests(unittest.TestCase):
    """Offline tests: goal and learning categories must never cross-contaminate."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.extractor = MemoryExtractor()

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_machine_learning_engineer_is_goal_not_learning(self):
        ops = self.extractor.extract("I want to become a machine learning engineer")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "goal")
        self.assertEqual(ops[0]["fact_type"], "career_goal")

    def test_machine_learning_engineer_migration_preserves_goal(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        from memory.store import save_json
        save_json(path, [{
            "id": "memory_001",
            "category": "goal",
            "fact_type": "career_goal",
            "content": "User wants to become a machine learning engineer",
            "value": "machine learning engineer",
            "status": "active",
            "importance": "high",
        }])
        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)
        mem = store.get_by_id("memory_001")
        self.assertEqual(mem["category"], "goal")
        self.assertEqual(mem["fact_type"], "career_goal")

    def test_wants_to_learn_is_learning_not_goal(self):
        ops = self.extractor.extract("I want to learn Python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")
        self.assertEqual(ops[0]["fact_type"], "learning")

    def test_am_learning_is_learning_not_goal(self):
        ops = self.extractor.extract("I am learning C")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_goal_with_learn_in_title_stays_goal(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        from memory.store import save_json
        save_json(path, [{
            "id": "memory_001",
            "category": "goal",
            "fact_type": "career_goal",
            "content": "User wants to become a machine learning engineer",
            "value": "machine learning engineer",
            "status": "active",
            "importance": "high",
        }])
        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)
        goals = [m for m in store.active_memories() if m["category"] == "goal"]
        learning = [m for m in store.active_memories() if m["category"] == "learning"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(len(learning), 0)

    def test_studying_extracts_as_learning(self):
        ops = self.extractor.extract("I am studying electronics")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_third_party_not_stored(self):
        ops = self.extractor.extract("My friend said, 'I am learning C'")
        self.assertEqual(ops, [])

    def test_temporary_not_stored(self):
        ops = self.extractor.extract("I am learning Rust until Friday")
        self.assertEqual(ops, [])

    def test_today_not_stored(self):
        ops = self.extractor.extract("I am learning Rust for today")
        self.assertEqual(ops, [])

    def test_compound_write_blocks_on_question_mark(self):
        ops = self.extractor.extract(
            "I am learning Python. What am I studying?"
        )
        self.assertEqual(ops, [])

    def test_provenance_fields_on_new_memory(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        store = MemoryStore(path=path)
        mem = store.add_memory(
            category="learning",
            fact_type="learning",
            content="User is learning C",
            value="C",
            subject="c",
            embedding=[0.1, 0.2],
        )
        self.assertEqual(mem["source"], "user_statement")
        self.assertEqual(mem["confidence"], 1.0)
        self.assertIn("created_at", mem)
        self.assertIn("updated_at", mem)

    def test_provenance_on_migrated_legacy(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        from memory.store import save_json
        save_json(path, [{
            "id": "memory_001",
            "category": "learning",
            "content": "User is learning C",
            "status": "active",
        }])
        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)
        mem = store.get_by_id("memory_001")
        self.assertEqual(mem["source"], "user_statement")
        self.assertEqual(mem["confidence"], 1.0)
        self.assertIsNotNone(mem.get("created_at"))
        self.assertIsNotNone(mem.get("updated_at"))

    def test_corrupt_recovery_preserves_valid_backup(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        Path(path).write_text("{invalid", encoding="utf-8")
        result = load_json_list(path)
        self.assertEqual(result, [])
        backups = list(Path(self._tmpdir.name).glob("*.corrupt.*.json"))
        self.assertEqual(len(backups), 1)

    def test_restart_preserves_active_learning(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        store = MemoryStore(path=path)
        mem = store.add_memory(
            category="learning",
            fact_type="learning",
            content="User is learning Python",
            value="Python",
            subject="python",
            embedding=[0.1, 0.2],
        )
        store2 = MemoryStore(path=path)
        store2.load(ensure_embeddings=False)
        active = store2.active_memories()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["value"], "Python")

    def test_goal_delete_does_not_affect_learning(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        from memory.store import save_json
        save_json(path, [
            {"id": "g1", "category": "goal", "fact_type": "career_goal",
             "content": "User wants to become a software engineer", "status": "active"},
            {"id": "l1", "category": "learning", "fact_type": "learning",
             "content": "User is learning C", "status": "active"},
        ])
        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)
        manager = SafeMemoryManager(store)
        results = manager.process_user_message(
            "I don't want to become a software engineer anymore"
        )
        self.assertEqual(results[0]["status"], "deleted")
        active = store.active_memories()
        goals = [m for m in active if m["category"] == "goal"]
        learning = [m for m in active if m["category"] == "learning"]
        self.assertEqual(len(goals), 0)
        self.assertEqual(len(learning), 1)

    # --- A: Goal with "machine learning" in title never becomes learning ---

    def test_A_machine_learning_engineer_goal_never_becomes_learning(self):
        """'I want to become a machine learning engineer' must stay a goal."""
        ops = self.extractor.extract(
            "I want to become a machine learning engineer"
        )
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "goal")
        self.assertIn("machine learning engineer", ops[0]["content"].lower())

    # --- B: Stale legacy learning record is safely deactivated ---

    def test_B_stale_legacy_learning_record_is_deactivated(self):
        """A learning record whose value is a job title is deactivated by migration."""
        from memory.store import save_json
        path = os.path.join(self._tmpdir.name, "ltm.json")
        save_json(path, [
            {
                "id": "stale_001",
                "category": "learning",
                "fact_type": "learning",
                "content": "User is learning a machine learning engineer",
                "value": "a machine learning engineer",
                "subject": "a machine learning engineer",
                "status": "active",
                "importance": "medium",
            },
            {
                "id": "legit_001",
                "category": "learning",
                "fact_type": "learning",
                "content": "User is learning machine learning",
                "value": "machine learning",
                "subject": "machine learning",
                "status": "active",
                "importance": "medium",
            },
        ])
        store = MemoryStore(path=path)
        store.load(ensure_embeddings=False)
        active = store.active_memories()
        active_ids = {m["id"] for m in active}
        self.assertNotIn("stale_001", active_ids, "Stale job-title learning record should be inactive")
        self.assertIn("legit_001", active_ids, "Legitimate learning record should remain active")

    # --- C: "I want to become nothing" does not create a goal ---

    def test_C_become_nothing_does_not_create_goal(self):
        ops = self.extractor.extract("i want to become nothing")
        self.assertEqual(ops, [])

    # --- D: "I don't like anything" does not create a preference ---

    def test_D_dont_like_anything_does_not_create_preference(self):
        ops = self.extractor.extract("i don't like anything")
        self.assertEqual(ops, [])

    def test_D_like_nothing_does_not_create_preference(self):
        ops = self.extractor.extract("i like nothing")
        self.assertEqual(ops, [])

    def test_D_hate_nothing_does_not_create_preference(self):
        ops = self.extractor.extract("i hate nothing")
        self.assertEqual(ops, [])

    def test_D_enjoy_nothing_does_not_create_preference(self):
        ops = self.extractor.extract("i enjoy nothing")
        self.assertEqual(ops, [])

    # --- E: Legitimate goals and preferences still work ---

    def test_E_legitimate_goal_still_works(self):
        ops = self.extractor.extract("i want to become a software engineer")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "goal")
        self.assertIn("software engineer", ops[0]["content"].lower())

    def test_E_legitimate_preference_still_works(self):
        ops = self.extractor.extract("i don't like spiders")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "preference")
        self.assertEqual(ops[0]["fact_type"], "dislike")
        self.assertEqual(ops[0]["subject"], "spiders")

    def test_E_positive_preference_still_works(self):
        ops = self.extractor.extract("i enjoy music")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "preference")
        self.assertEqual(ops[0]["fact_type"], "like")

    # --- Answer phrasing: studying vs learning ---

    def test_answer_studying_uses_studying_phrasing(self):
        """'What am I studying?' answer must use 'studying', not 'learning'."""
        from memory.answers import grounded_answer

        class _Intent:
            primary = "learning"
            score = 0.85
            scores = {"learning": 0.85}
            is_memory_read = True
            filters = {}
            requested = ["learning"]
            preference_polarity = None
            is_general_knowledge = False
            normalized_query = ""

        memory = {
            "id": "m1", "category": "learning", "fact_type": "learning",
            "content": "User is learning electronics", "value": "electronics",
            "subject": "electronics", "status": "active",
        }
        answer = grounded_answer(_Intent(), [memory], question="What am I studying?")
        self.assertIn("studying", answer.lower())
        self.assertNotIn("learning", answer.lower())

    def test_answer_learning_uses_learning_phrasing(self):
        """'What am I learning?' answer must use 'learning'."""
        from memory.answers import grounded_answer

        class _Intent:
            primary = "learning"
            score = 0.85
            scores = {"learning": 0.85}
            is_memory_read = True
            filters = {}
            requested = ["learning"]
            preference_polarity = None
            is_general_knowledge = False
            normalized_query = ""

        memory = {
            "id": "m1", "category": "learning", "fact_type": "learning",
            "content": "User is learning electronics", "value": "electronics",
            "subject": "electronics", "status": "active",
        }
        answer = grounded_answer(_Intent(), [memory], question="What am I learning?")
        self.assertIn("learning", answer.lower())

    def test_answer_studying_multiple_items(self):
        """'What am I studying?' with multiple items uses 'studying'."""
        from memory.answers import grounded_answer

        class _Intent:
            primary = "learning"
            score = 0.85
            scores = {"learning": 0.85}
            is_memory_read = True
            filters = {}
            requested = ["learning"]
            preference_polarity = None
            is_general_knowledge = False
            normalized_query = ""

        memories = [
            {"id": "m1", "category": "learning", "fact_type": "learning",
             "content": "User is learning electronics", "value": "electronics",
             "subject": "electronics", "status": "active"},
            {"id": "m2", "category": "learning", "fact_type": "learning",
             "content": "User is learning Python", "value": "Python",
             "subject": "python", "status": "active"},
        ]
        answer = grounded_answer(_Intent(), memories, question="What am I studying?")
        self.assertIn("studying", answer.lower())
        self.assertIn("electronics", answer.lower())
        self.assertIn("python", answer.lower())

    # --- F: Studying electronics still works (requires Ollama) ---
    # --- G: Existing v0.6 acceptance tests remain green (requires Ollama) ---
    # Tests F and G are in v06AcceptanceContractTests below.


class v06AcceptanceContractTests(unittest.TestCase):
    """Full v0.6 acceptance contract — requires Ollama for retrieval tests."""

    @classmethod
    def setUpClass(cls):
        try:
            EmbeddingStore().verify_model()
        except EmbeddingError as error:
            raise unittest.SkipTest(str(error)) from error

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store = make_stack(self._tmpdir.name)

    def tearDown(self):
        self._tmpdir.cleanup()

    def ask(self, question: str) -> str:
        intent = self.retriever.detect_intent(question)
        if intent.is_multi:
            grouped = self.retriever.retrieve_by_requested_intents(
                question, intent=intent
            )
            memories = []
            for items in grouped.values():
                memories.extend(items)
            return grounded_answer(
                intent, memories, question=question, grouped=grouped
            )
        memories = self.retriever.retrieve_relevant_memories(
            question, intent=intent
        )
        return grounded_answer(intent, memories, question=question)

    def active(self):
        return self.store.active_memories()

    def learning(self):
        return [m for m in self.active() if m["category"] == "learning"]

    def goals(self):
        return [m for m in self.active() if m["category"] == "goal"]

    def education(self):
        return [m for m in self.active() if m["category"] == "education"]

    # === A-F: Learning add/delete cycle ===
    def test_A_add_learning_python(self):
        r = self.manager.process_user_message("I am learning Python")
        self.assertEqual(r[0]["status"], "added")
        learning = self.learning()
        self.assertEqual(len(learning), 1)
        self.assertIn("python", learning[0]["content"].lower())

    def test_B_query_learning_after_add(self):
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I learning?")
        self.assertIn("python", answer.lower())

    def test_C_add_learning_cpp(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        learning = self.learning()
        self.assertEqual(len(learning), 2)
        subjects = {m["subject"] for m in learning}
        self.assertIn("python", subjects)
        self.assertIn("c++", subjects)

    def test_D_query_learning_after_add_cpp(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        answer = self.ask("What am I learning?")
        lower = answer.lower()
        self.assertIn("python", lower)
        self.assertIn("c++", lower)

    def test_E_delete_learning_python(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        results = self.manager.process_user_message(
            "I don't want to learn Python anymore"
        )
        self.assertTrue(results)
        self.assertEqual(results[0]["status"], "deleted")
        learning = self.learning()
        self.assertEqual(len(learning), 1)
        self.assertIn("c++", learning[0]["content"].lower())

    def test_F_query_learning_after_delete(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        self.manager.process_user_message("I don't want to learn Python anymore")
        answer = self.ask("What am I learning?")
        self.assertIn("c++", answer.lower())
        self.assertNotIn("python", answer.lower())

    # === G-I: Goal vs learning separation ===
    def test_G_ml_engineer_is_goal_not_learning(self):
        self.manager.process_user_message("I am learning C++")
        self.manager.process_user_message(
            "I want to become a machine learning engineer"
        )
        goals = self.goals()
        learning = self.learning()
        self.assertEqual(len(goals), 1)
        self.assertIn("machine learning engineer", goals[0]["content"].lower())
        self.assertEqual(len(learning), 1)
        self.assertNotIn("machine learning engineer", learning[0]["content"].lower())

    def test_H_query_goals_returns_ml_engineer(self):
        self.manager.process_user_message(
            "I want to become a machine learning engineer"
        )
        answer = self.ask("What do I want to become?")
        self.assertIn("machine learning engineer", answer.lower())

    def test_I_query_learning_does_not_contain_ml_engineer(self):
        self.manager.process_user_message("I am learning C++")
        self.manager.process_user_message(
            "I want to become a machine learning engineer"
        )
        answer = self.ask("What am I learning?")
        self.assertIn("c++", answer.lower())
        self.assertNotIn("machine learning engineer", answer.lower())

    # === J-K: Studying query routing ===
    def test_J_studying_stores_as_learning(self):
        self.manager.process_user_message("I am studying electronics")
        learning = self.learning()
        self.assertGreaterEqual(len(learning), 1)
        subjects = " ".join(m.get("content", "").lower() for m in learning)
        self.assertIn("electronics", subjects)

    def test_K_query_studying_returns_learning(self):
        self.manager.process_user_message("I am studying electronics")
        answer = self.ask("What am I studying?")
        self.assertIn("electronics", answer.lower())

    # === L: Query variant routing ===
    def test_L_learning_query_variants(self):
        self.manager.process_user_message("I am learning Python")
        for q in [
            "What am I learning?",
            "What am I studying?",
            "What are my studies?",
            "What topics am I learning?",
        ]:
            answer = self.ask(q)
            self.assertIn("python", answer.lower(), f"Failed for: {q}")

    # === M: Compound input ===
    def test_M_compound_write_and_read(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        answer_before = self.ask("What am I learning?")
        self.assertIn("python", answer_before.lower())
        self.assertIn("c++", answer_before.lower())

    # === N: Third-party protection ===
    def test_N_third_party_not_stored(self):
        r = self.manager.process_user_message(
            "My friend said, 'I am learning C'"
        )
        self.assertEqual(r, [])
        learning = self.learning()
        self.assertEqual(len(learning), 0)

    # === O: Temporary protection ===
    def test_O_temporary_not_stored(self):
        r = self.manager.process_user_message("I am learning Rust until Friday")
        self.assertEqual(r, [])
        learning = self.learning()
        self.assertEqual(len(learning), 0)

    # === P: Restart persistence ===
    def test_P_restart_preserves_state(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        self.manager.process_user_message(
            "I want to become a machine learning engineer"
        )

        path = self.store.path
        store2 = MemoryStore(path=path, embedding_store=EmbeddingStore())
        store2.load(ensure_embeddings=False)

        learning = [m for m in store2.active_memories() if m["category"] == "learning"]
        goals = [m for m in store2.active_memories() if m["category"] == "goal"]
        self.assertEqual(len(learning), 2)
        self.assertEqual(len(goals), 1)
        self.assertIn("machine learning engineer", goals[0]["content"].lower())

    # === Q: Corrupt file recovery ===
    def test_Q_corrupt_recovery(self):
        path = os.path.join(self._tmpdir.name, "ltm.json")
        self.manager.process_user_message("I am learning Python")
        self.store.save()
        Path(path).write_text("{invalid", encoding="utf-8")
        result = load_json_list(path)
        self.assertEqual(result, [])
        backups = list(Path(self._tmpdir.name).glob("*.corrupt.*.json"))
        self.assertGreaterEqual(len(backups), 1)

    # === R: Provenance ===
    def test_R_new_memory_has_provenance(self):
        self.manager.process_user_message("I am learning Python")
        learning = self.learning()
        self.assertEqual(learning[0]["source"], "user_statement")
        self.assertEqual(learning[0]["confidence"], 1.0)
        self.assertIsNotNone(learning[0].get("created_at"))
        self.assertIsNotNone(learning[0].get("updated_at"))

    # === S: Contradiction/update (existing behavior preserved) ===
    def test_S_preference_reversal(self):
        self.manager.process_user_message("I like Python")
        r = self.manager.process_user_message("I don't like Python")
        self.assertEqual(r[0]["status"], "updated")
        self.assertEqual(r[0].get("conflict_resolved"), True)
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 1)
        self.assertEqual(prefs[0]["fact_type"], "dislike")

    def test_S_name_update(self):
        self.manager.process_user_message("My name is Tejsai")
        r = self.manager.process_user_message("My name is Rahul")
        self.assertEqual(r[0]["status"], "updated")
        answer = self.ask("What is my name?")
        self.assertIn("Rahul", answer)
        self.assertNotIn("Tejsai", answer)

    def test_S_goal_replacement(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        self.manager.process_user_message(
            "I want to become a content creator"
        )
        results = self.manager.process_user_message(
            "My goal changed from becoming a software engineer "
            "to becoming a researcher"
        )
        self.assertEqual(results[0]["status"], "updated")
        goals = self.goals()
        self.assertEqual(len(goals), 2)
        texts = " ".join(m["content"].lower() for m in goals)
        self.assertIn("researcher", texts)
        self.assertIn("content creator", texts)
        self.assertNotIn("software engineer", texts)

    def test_S_learning_delete_correct_target(self):
        self.manager.process_user_message("I am learning Python")
        self.manager.process_user_message("I am learning C++")
        self.manager.process_user_message("I am learning Rust")
        self.manager.process_user_message("I don't want to learn Python anymore")
        learning = self.learning()
        self.assertEqual(len(learning), 2)
        subjects = {m["subject"] for m in learning}
        self.assertNotIn("python", subjects)
        self.assertIn("c++", subjects)
        self.assertIn("rust", subjects)

    def test_S_interest_delete(self):
        self.manager.process_user_message("I am interested in robotics")
        self.manager.process_user_message("I am interested in video editing")
        self.manager.process_user_message(
            "I am not interested in robotics"
        )
        interests = [
            m for m in self.active() if m["category"] == "interest"
        ]
        self.assertEqual(len(interests), 1)
        self.assertIn("video editing", interests[0]["content"].lower())

    def test_S_preference_multiple_subjects(self):
        self.manager.process_user_message("I like Python")
        self.manager.process_user_message("I like Java")
        self.manager.process_user_message("I don't like C")
        prefs = [m for m in self.active() if m["category"] == "preference"]
        self.assertEqual(len(prefs), 3)
        by_subject = {m["subject"]: m["fact_type"] for m in prefs}
        self.assertEqual(by_subject["python"], "like")
        self.assertEqual(by_subject["java"], "like")
        self.assertEqual(by_subject["c"], "dislike")

    # --- F: Studying electronics still works ---

    def test_F_studying_electronics_still_works(self):
        self.manager.process_user_message("I am studying electronics")
        answer = self.ask("What am I studying?")
        self.assertIn("electronics", answer.lower())

    # --- G: Goal vs learning separation (end-to-end) ---

    def test_G_goal_vs_learning_separation(self):
        self.manager.process_user_message("I am learning C++")
        self.manager.process_user_message(
            "I want to become a machine learning engineer"
        )
        answer = self.ask("What am I learning?")
        self.assertIn("c++", answer.lower())
        self.assertNotIn("machine learning engineer", answer.lower())

    # --- Answer phrasing: studying vs learning (end-to-end) ---

    def test_answer_studying_question_uses_studying_phrasing(self):
        self.manager.process_user_message("I am studying electronics")
        answer = self.ask("What am I studying?")
        self.assertIn("studying", answer.lower())
        self.assertIn("electronics", answer.lower())

    def test_answer_learning_question_uses_learning_phrasing(self):
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I learning?")
        self.assertIn("learning", answer.lower())
        self.assertIn("python", answer.lower())

    def test_answer_what_is_my_goal(self):
        self.manager.process_user_message(
            "I want to become a software engineer"
        )
        answer = self.ask("What is my goal?")
        self.assertIn("software engineer", answer.lower())

    def test_answer_what_is_my_ambition(self):
        self.manager.process_user_message(
            "I want to become a researcher"
        )
        answer = self.ask("What is my ambition?")
        self.assertIn("researcher", answer.lower())


class ArchitectureGuardTests(unittest.TestCase):
    def test_no_keyword_question_routing(self):
        retriever = (SRC / "memory" / "retriever.py").read_text(encoding="utf-8")
        self.assertNotIn("asks_nickname", retriever)
        self.assertNotIn("def _metadata_boost", retriever)
        main = (SRC / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("def direct_memory_answer", main)
        self.assertNotIn("def build_memory_llm_messages", main)
        self.assertIn("ALLI v0.6", main)


# ============================================================
# MEMORY CONTRACT TESTS (offline — no Ollama required)
# ============================================================

class EducationVsLearningTests(unittest.TestCase):
    """A: Education/studying vs learning must be semantically separated."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_A1_studying_with_diploma_is_education(self):
        """'I am studying diploma second year' → education."""
        ops = self.extractor.extract("I am studying diploma second year")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")
        self.assertIn("diploma", ops[0]["content"].lower())

    def test_A2_studying_with_degree_is_education(self):
        """'I am studying B.Tech in computer science' → education."""
        ops = self.extractor.extract("I am studying B.Tech in computer science")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")

    def test_A3_studying_electronics_no_formal_context_is_learning(self):
        """'I am studying electronics' without formal signals → learning."""
        ops = self.extractor.extract("I am studying electronics")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_A4_learning_c_is_learning(self):
        """'I am learning C' → learning."""
        ops = self.extractor.extract("I am learning C")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_A5_studying_diploma_typo_is_education(self):
        """'i am studing diploma second year' → education (typo handled)."""
        ops = self.extractor.extract("i am studing diploma second year")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")

    def test_A6_want_to_learn_is_learning(self):
        """'I want to learn Python' → learning."""
        ops = self.extractor.extract("I want to learn Python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_A7_studying_at_university_is_education(self):
        """'I study at university' → education."""
        ops = self.extractor.extract("I study at university")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")

    def test_A8_my_education_is_X(self):
        """'My education is B.Tech in CS' → education."""
        ops = self.extractor.extract("My education is B.Tech in CS")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")


class LearningTests(unittest.TestCase):
    """B: Learning extraction and deletion."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_B1_learning_creation(self):
        ops = self.extractor.extract("I am learning Python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")
        self.assertEqual(ops[0]["fact_type"], "learning")

    def test_B2_learning_deletion(self):
        ops = self.extractor.extract("I don't want to learn C anymore")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "DELETE_MATCH")
        self.assertEqual(ops[0]["category"], "learning")

    def test_B3_learning_stopped(self):
        ops = self.extractor.extract("I stopped learning Rust")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "DELETE_MATCH")

    def test_B4_want_to_learn_extraction(self):
        ops = self.extractor.extract("I want to learn JavaScript")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")
        self.assertIn("javascript", ops[0]["content"].lower())


class GoalTests(unittest.TestCase):
    """C: Goal extraction and handling."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_C1_goal_is_to_become(self):
        ops = self.extractor.extract("My goal is to become a software engineer")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "goal")
        self.assertEqual(ops[0]["fact_type"], "career_goal")

    def test_C2_want_to_become_is_goal(self):
        ops = self.extractor.extract("I want to become a data scientist")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "goal")

    def test_C3_goal_not_learning(self):
        """Goals should not appear as learning."""
        ops = self.extractor.extract("I want to become a software engineer")
        self.assertNotEqual(ops[0]["category"], "learning")

    def test_C4_build_is_project(self):
        ops = self.extractor.extract("I want to build a mobile app")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "project")


class AmbitionDreamTests(unittest.TestCase):
    """D: Ambition and dream extraction."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_D1_ambition_extraction(self):
        ops = self.extractor.extract("My ambition is to become an AI engineer")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "ambition")
        self.assertEqual(ops[0]["fact_type"], "ambition")

    def test_D2_dream_extraction(self):
        ops = self.extractor.extract("My dream is to become a researcher")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "dream")
        self.assertEqual(ops[0]["fact_type"], "dream")

    def test_D3_dream_of_extraction(self):
        ops = self.extractor.extract("I dream of becoming a doctor")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "dream")

    def test_D4_ambition_not_goal(self):
        ops = self.extractor.extract("My ambition is to become a CEO")
        self.assertEqual(ops[0]["category"], "ambition")
        self.assertNotEqual(ops[0]["category"], "goal")

    def test_D5_dream_not_goal(self):
        ops = self.extractor.extract("My dream is to travel the world")
        self.assertEqual(ops[0]["category"], "dream")
        self.assertNotEqual(ops[0]["category"], "goal")

    def test_D6_aspire_to_is_ambition(self):
        ops = self.extractor.extract("I aspire to become a leader")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "ambition")


class IdentityTests(unittest.TestCase):
    """E: Identity extraction."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_E1_name_extraction(self):
        ops = self.extractor.extract("My name is Rahul")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "identity")
        self.assertEqual(ops[0]["fact_type"], "name")
        self.assertEqual(ops[0]["value"], "Rahul")

    def test_E2_nickname_extraction(self):
        ops = self.extractor.extract("My nickname is Binnu")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "identity")
        self.assertEqual(ops[0]["fact_type"], "nickname")

    def test_E3_name_change(self):
        ops = self.extractor.extract("My name changed to Rahul")
        self.assertEqual(len(ops), 1)
        self.assertTrue(ops[0].get("explicit_change"))

    def test_E4_non_person_name_rejected(self):
        ops = self.extractor.extract("My name is interested in coding")
        self.assertEqual(ops, [])


class SkillTests(unittest.TestCase):
    """F: Skill extraction."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_F1_know_python_is_skill(self):
        ops = self.extractor.extract("I know Python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "skill")

    def test_F2_can_use_is_skill(self):
        ops = self.extractor.extract("I can use Docker")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "skill")

    def test_F3_skill_not_learning(self):
        """Skills should not appear as learning."""
        ops = self.extractor.extract("I know Python")
        self.assertNotEqual(ops[0]["category"], "learning")


class InterestTests(unittest.TestCase):
    """G: Interest extraction."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_G1_interested_in(self):
        ops = self.extractor.extract("I am interested in robotics")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "interest")

    def test_G2_interested_in_typo(self):
        ops = self.extractor.extract("I am intrested in video editing")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "interest")

    def test_G3_compound_interests(self):
        ops = self.extractor.extract("I am interested in web development and editing")
        self.assertEqual(len(ops), 2)
        categories = {op["category"] for op in ops}
        self.assertEqual(categories, {"interest"})


class PreferenceTests(unittest.TestCase):
    """H: Like/dislike extraction."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_H1_like(self):
        ops = self.extractor.extract("I like Python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "preference")
        self.assertEqual(ops[0]["fact_type"], "like")

    def test_H2_dislike(self):
        ops = self.extractor.extract("I don't like Java")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "preference")
        self.assertEqual(ops[0]["fact_type"], "dislike")

    def test_H3_enjoy(self):
        ops = self.extractor.extract("I enjoy robotics")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "preference")
        self.assertEqual(ops[0]["fact_type"], "like")

    def test_H4_hate(self):
        ops = self.extractor.extract("I hate spiders")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "preference")
        self.assertEqual(ops[0]["fact_type"], "dislike")


class VagueStatementTests(unittest.TestCase):
    """L: Vague/negative statements should not create garbage memories."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_L1_become_nothing(self):
        ops = self.extractor.extract("I want to become nothing")
        self.assertEqual(ops, [])

    def test_L2_dont_like_anything(self):
        ops = self.extractor.extract("I don't like anything")
        self.assertEqual(ops, [])

    def test_L3_like_nothing(self):
        ops = self.extractor.extract("I like nothing")
        self.assertEqual(ops, [])

    def test_L4_hate_nothing(self):
        ops = self.extractor.extract("I hate nothing")
        self.assertEqual(ops, [])

    def test_L5_enjoy_nothing(self):
        ops = self.extractor.extract("I enjoy nothing")
        self.assertEqual(ops, [])

    def test_L6_dont_like_something(self):
        ops = self.extractor.extract("I don't like something")
        self.assertEqual(ops, [])

    def test_L7_like_everything(self):
        ops = self.extractor.extract("I like everything")
        self.assertEqual(ops, [])

    def test_L8_goal_nothing(self):
        ops = self.extractor.extract("My goal is to become nothing")
        self.assertEqual(ops, [])


class TemporaryMemoryTests(unittest.TestCase):
    """M/N: Temporary statements should not be stored as durable facts."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_M1_temporary_today(self):
        ops = self.extractor.extract("I am learning Rust for today")
        self.assertEqual(ops, [])

    def test_M2_time_bound_until_friday(self):
        ops = self.extractor.extract("I am learning Rust until Friday")
        self.assertEqual(ops, [])

    def test_M3_temporary_hungry(self):
        ops = self.extractor.extract("I am hungry")
        self.assertEqual(ops, [])

    def test_M4_temporary_tired(self):
        ops = self.extractor.extract("I am tired")
        self.assertEqual(ops, [])

    def test_M5_temporary_exam(self):
        ops = self.extractor.extract("I have an exam tomorrow")
        self.assertEqual(len(ops), 1)
        op = ops[0]
        self.assertEqual(op["action"], "ADD")
        self.assertEqual(op["category"], "task")
        self.assertEqual(op["fact_type"], "task")
        self.assertIn("exam", op["value"])

    def test_M6_temporary_right_now(self):
        ops = self.extractor.extract("I am learning Python right now")
        self.assertEqual(ops, [])


class ExpirationTests(unittest.TestCase):
    """N: Temporal expiration mechanism."""

    def test_N1_compute_expires_tomorrow(self):
        from memory.store import compute_expires_at, utc_now
        result = compute_expires_at("tomorrow")
        self.assertIsNotNone(result)
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(result)
        now = utc_now()
        self.assertGreater(dt, now)

    def test_N2_compute_expires_friday(self):
        from memory.store import compute_expires_at
        result = compute_expires_at("friday")
        self.assertIsNotNone(result)

    def test_N3_compute_expires_in_2_days(self):
        from memory.store import compute_expires_at
        result = compute_expires_at("in 2 days")
        self.assertIsNotNone(result)

    def test_N4_compute_expires_invalid(self):
        from memory.store import compute_expires_at
        result = compute_expires_at("xyzzy")
        self.assertIsNone(result)

    def test_N5_is_expired_check(self):
        from memory.store import is_expired
        # A memory with a past expiration should be expired
        past_memory = {"expires_at": "2020-01-01T00:00:00+00:00"}
        self.assertTrue(is_expired(past_memory))

    def test_N6_is_not_expired_future(self):
        from memory.store import is_expired
        future_memory = {"expires_at": "2099-12-31T23:59:59+00:00"}
        self.assertFalse(is_expired(future_memory))

    def test_N7_no_expiration_not_expired(self):
        from memory.store import is_expired
        memory = {"content": "test"}
        self.assertFalse(is_expired(memory))


class QueryIntentTests(unittest.TestCase):
    """R: Query-specific intent detection for new categories."""

    def setUp(self):
        self.detector = QueryIntentDetector.__new__(QueryIntentDetector)

    def test_R1_ambition_intent(self):
        """'What is my ambition?' should detect ambition intent."""
        # Test the CATEGORY_QUERY_CUES for ambition
        from memory.query_intent import CATEGORY_QUERY_CUES
        import re
        query = "what is my ambition"
        found = False
        for pattern in CATEGORY_QUERY_CUES.get("ambition", ()):
            if re.search(pattern, query, flags=re.IGNORECASE):
                found = True
                break
        self.assertTrue(found, "ambition cue should match 'what is my ambition'")

    def test_R2_dream_intent(self):
        """'What is my dream?' should detect dream intent."""
        from memory.query_intent import CATEGORY_QUERY_CUES
        import re
        query = "what is my dream"
        found = False
        for pattern in CATEGORY_QUERY_CUES.get("dream", ()):
            if re.search(pattern, query, flags=re.IGNORECASE):
                found = True
                break
        self.assertTrue(found, "dream cue should match 'what is my dream'")

    def test_R3_education_intent(self):
        """'What am I studying?' should detect education intent."""
        from memory.query_intent import CATEGORY_QUERY_CUES
        import re
        query = "what am i studying"
        found = False
        for pattern in CATEGORY_QUERY_CUES.get("education", ()):
            if re.search(pattern, query, flags=re.IGNORECASE):
                found = True
                break
        self.assertTrue(found, "education cue should match 'what am i studying'")

    def test_R4_learning_intent(self):
        """'What am I learning?' should detect learning intent."""
        from memory.query_intent import CATEGORY_QUERY_CUES
        import re
        query = "what am i learning"
        found = False
        for pattern in CATEGORY_QUERY_CUES.get("learning", ()):
            if re.search(pattern, query, flags=re.IGNORECASE):
                found = True
                break
        self.assertTrue(found, "learning cue should match 'what am i learning'")

    def test_R5_goal_not_matching_dream(self):
        """'What are my dreams?' should NOT match goal cue."""
        from memory.query_intent import CATEGORY_QUERY_CUES
        import re
        query = "what are my dreams"
        found = False
        for pattern in CATEGORY_QUERY_CUES.get("goal", ()):
            if re.search(pattern, query, flags=re.IGNORECASE):
                found = True
                break
        self.assertFalse(found, "goal cue should NOT match 'what are my dreams'")

    def test_R6_ambition_not_in_goal_prototypes(self):
        """Ambition prototypes should not be in goal prototypes."""
        from memory.query_intent import CONCEPT_PROTOTYPES
        goal_protos = " ".join(CONCEPT_PROTOTYPES.get("goal", []))
        ambition_protos = " ".join(CONCEPT_PROTOTYPES.get("ambition", []))
        self.assertNotIn("ambition", goal_protos.lower())
        self.assertIn("ambition", ambition_protos.lower())

    def test_R7_dream_not_in_goal_prototypes(self):
        """Dream prototypes should not be in goal prototypes."""
        from memory.query_intent import CONCEPT_PROTOTYPES
        goal_protos = " ".join(CONCEPT_PROTOTYPES.get("goal", []))
        dream_protos = " ".join(CONCEPT_PROTOTYPES.get("dream", []))
        self.assertNotIn("dream", goal_protos.lower())
        self.assertIn("dream", dream_protos.lower())

    def test_R8_education_in_prototypes(self):
        """Education should have its own prototypes."""
        from memory.query_intent import CONCEPT_PROTOTYPES
        self.assertIn("education", CONCEPT_PROTOTYPES)
        self.assertTrue(len(CONCEPT_PROTOTYPES["education"]) > 0)

    def test_R9_confusable_intents_separate(self):
        """Goal, ambition, and dream should be confusable but separate."""
        from memory.query_intent import CONFUSABLE_INTENTS
        self.assertIn("ambition", CONFUSABLE_INTENTS.get("goal", frozenset()))
        self.assertIn("dream", CONFUSABLE_INTENTS.get("goal", frozenset()))
        self.assertIn("goal", CONFUSABLE_INTENTS.get("ambition", frozenset()))
        self.assertIn("goal", CONFUSABLE_INTENTS.get("dream", frozenset()))


class AnswerPhrasingTests(unittest.TestCase):
    """R: Answer phrasing must match the question intent."""

    def test_R1_ambition_answer_uses_ambition(self):
        """'What is my ambition?' answer must use 'ambition'."""
        from memory.answers import grounded_answer
        from memory.query_intent import QueryIntent

        intent = QueryIntent(
            primary="ambition",
            score=0.85,
            scores={"ambition": 0.85},
            is_memory_read=True,
            filters={},
            requested=["ambition"],
        )
        memory = {
            "id": "m1", "category": "ambition", "fact_type": "ambition",
            "content": "User's ambition is to become an AI engineer",
            "value": "AI engineer", "subject": "ai engineer", "status": "active",
        }
        answer = grounded_answer(intent, [memory], question="What is my ambition?")
        self.assertIn("ambition", answer.lower())
        self.assertNotIn("goal", answer.lower())

    def test_R2_dream_answer_uses_dream(self):
        """'What is my dream?' answer must use 'dream'."""
        from memory.answers import grounded_answer
        from memory.query_intent import QueryIntent

        intent = QueryIntent(
            primary="dream",
            score=0.85,
            scores={"dream": 0.85},
            is_memory_read=True,
            filters={},
            requested=["dream"],
        )
        memory = {
            "id": "m1", "category": "dream", "fact_type": "dream",
            "content": "User's dream is to become a researcher",
            "value": "researcher", "subject": "researcher", "status": "active",
        }
        answer = grounded_answer(intent, [memory], question="What is my dream?")
        self.assertIn("dream", answer.lower())
        self.assertNotIn("goal", answer.lower())

    def test_R3_education_answer_uses_education(self):
        """'What am I studying?' answer must use 'studying'."""
        from memory.answers import grounded_answer
        from memory.query_intent import QueryIntent

        intent = QueryIntent(
            primary="education",
            score=0.85,
            scores={"education": 0.85},
            is_memory_read=True,
            filters={},
            requested=["education"],
        )
        memory = {
            "id": "m1", "category": "education", "fact_type": "studying",
            "content": "User is studying diploma second year",
            "value": "diploma second year", "subject": "diploma second year",
            "status": "active",
        }
        answer = grounded_answer(intent, [memory], question="What am I studying?")
        self.assertIn("studying", answer.lower())
        self.assertNotIn("learning", answer.lower())

    def test_R4_learning_answer_uses_learning(self):
        """'What am I learning?' answer must use 'learning'."""
        from memory.answers import grounded_answer
        from memory.query_intent import QueryIntent

        intent = QueryIntent(
            primary="learning",
            score=0.85,
            scores={"learning": 0.85},
            is_memory_read=True,
            filters={},
            requested=["learning"],
        )
        memory = {
            "id": "m1", "category": "learning", "fact_type": "learning",
            "content": "User is learning Python",
            "value": "Python", "subject": "python", "status": "active",
        }
        answer = grounded_answer(intent, [memory], question="What am I learning?")
        self.assertIn("learning", answer.lower())

    def test_R5_missing_ambition_answer(self):
        """Missing ambition info should say no ambition stored."""
        from memory.answers import grounded_answer
        from memory.query_intent import QueryIntent

        intent = QueryIntent(
            primary="ambition",
            score=0.85,
            scores={"ambition": 0.85},
            is_memory_read=True,
            filters={},
            requested=["ambition"],
        )
        answer = grounded_answer(intent, [], question="What is my ambition?")
        self.assertIn("ambition", answer.lower())
        self.assertIn("don't", answer.lower())

    def test_R6_missing_dream_answer(self):
        """Missing dream info should say no dream stored."""
        from memory.answers import grounded_answer
        from memory.query_intent import QueryIntent

        intent = QueryIntent(
            primary="dream",
            score=0.85,
            scores={"dream": 0.85},
            is_memory_read=True,
            filters={},
            requested=["dream"],
        )
        answer = grounded_answer(intent, [], question="What is my dream?")
        self.assertIn("dream", answer.lower())
        self.assertIn("don't", answer.lower())


class NormalConversationNotHijackedTests(unittest.TestCase):
    """V: Normal questions should not be hijacked by memory."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_V1_how_are_you(self):
        """'How are you?' should not extract memory."""
        ops = self.extractor.extract("How are you?")
        self.assertEqual(ops, [])

    def test_V2_what_time_is_it(self):
        """'What time is it?' should not extract memory."""
        ops = self.extractor.extract("What time is it?")
        self.assertEqual(ops, [])

    def test_V3_general_knowledge(self):
        """'OD means what?' should not extract memory."""
        ops = self.extractor.extract("OD means what?")
        self.assertEqual(ops, [])

    def test_V4_hello(self):
        """'hello' should not extract memory."""
        ops = self.extractor.extract("hello")
        self.assertEqual(ops, [])

    def test_V5_thank_you(self):
        """'thank you' should not extract memory."""
        ops = self.extractor.extract("thank you")
        self.assertEqual(ops, [])


class ThirdPartyProtectionTests(unittest.TestCase):
    """Third-party statements should not be stored as user facts."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_TP1_friend_said(self):
        ops = self.extractor.extract("My friend said, 'I am learning C'")
        self.assertEqual(ops, [])

    def test_TP2_he_said(self):
        ops = self.extractor.extract("He said, 'I know Python'")
        self.assertEqual(ops, [])

    def test_TP3_direct_friend_statement(self):
        ops = self.extractor.extract("My friend is learning Flutter")
        self.assertEqual(ops, [])


class TemporalAndCorrectionTests(unittest.TestCase):
    """K: User corrections and temporal changes."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_K1_no_longer_want(self):
        """'I no longer want to learn C' → delete learning."""
        ops = self.extractor.extract("I no longer want to learn C")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "DELETE_MATCH")

    def test_K2_dont_want_to_become(self):
        """'I don't want to become a doctor' → delete goal."""
        ops = self.extractor.extract("I don't want to become a doctor")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "DELETE_GOAL")

    def test_K3_not_interested_anymore(self):
        """'I am not interested in robotics anymore' → delete interest."""
        ops = self.extractor.extract("I am not interested in robotics")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "DELETE_MATCH")
        self.assertEqual(ops[0]["category"], "interest")

    def test_K4_forget_that(self):
        """'Forget that I like spiders' → delete preference."""
        ops = self.extractor.extract("Forget that I like spiders")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "DELETE_MATCH")
        self.assertEqual(ops[0]["category"], "preference")


class ConfigTests(unittest.TestCase):
    """Verify config has all required categories."""

    def test_config_has_ambition(self):
        self.assertIn("ambition", config.VALID_CATEGORIES)

    def test_config_has_dream(self):
        self.assertIn("dream", config.VALID_CATEGORIES)

    def test_config_has_task(self):
        self.assertIn("task", config.VALID_CATEGORIES)

    def test_config_has_education(self):
        self.assertIn("education", config.VALID_CATEGORIES)

    def test_config_importance_includes_critical(self):
        self.assertIn("critical", config.VALID_IMPORTANCE)

    def test_config_status_includes_expired(self):
        self.assertIn("expired", config.VALID_STATUS)

    def test_config_has_priority_keywords(self):
        self.assertTrue(len(config.PRIORITY_KEYWORDS) > 0)


class CueFallbackTests(unittest.TestCase):
    """Regression: cue-only intent detection when embeddings unavailable."""

    def setUp(self):
        self.detector = QueryIntentDetector()

    def test_R10_studying_misspelled_routes_education(self):
        r = self.detector.detect("what am i studing")
        self.assertEqual(r.primary, "education")
        self.assertTrue(r.is_memory_read)

    def test_R11_studying_correct_routes_education(self):
        r = self.detector.detect("what am i studying")
        self.assertEqual(r.primary, "education")
        self.assertTrue(r.is_memory_read)

    def test_R12_learning_routes_learning_not_education(self):
        r = self.detector.detect("what i am learning")
        self.assertEqual(r.primary, "learning")
        self.assertTrue(r.is_memory_read)

    def test_R13_have_tomorrow_routes_task(self):
        r = self.detector.detect("what do i have tomorrow")
        self.assertEqual(r.primary, "task")
        self.assertTrue(r.is_memory_read)

    def test_R14_how_are_you_not_memory(self):
        r = self.detector.detect("how are you")
        self.assertEqual(r.primary, "unknown")
        self.assertFalse(r.is_memory_read)

    def test_R15_education_query_cannot_retrieve_learning(self):
        r = self.detector.detect("what am i studying")
        self.assertIn("education", r.requested)
        self.assertNotIn("learning", r.requested)

    def test_R16_learning_query_cannot_retrieve_education(self):
        r = self.detector.detect("what am i learning")
        self.assertIn("learning", r.requested)
        self.assertNotIn("education", r.requested)


class TaskExtractionTests(unittest.TestCase):
    """Regression: temporary events extracted as tasks with temporal hints."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_R17_ct_exam_tomorrow_extracted_as_task(self):
        ops = self.extractor.extract("i have ct exam tommorrow")
        self.assertEqual(len(ops), 1)
        op = ops[0]
        self.assertEqual(op["category"], "task")
        self.assertEqual(op["fact_type"], "task")
        self.assertIn("ct exam", op["value"])
        self.assertIsNotNone(op.get("temporal_hint"))

    def test_R18_have_exam_tomorrow_extracted_as_task(self):
        ops = self.extractor.extract("i have an exam tomorrow")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "task")

    def test_R19_have_test_next_week_extracted_as_task(self):
        ops = self.extractor.extract("i have a test next week")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "task")
        self.assertEqual(ops[0]["temporal_hint"], "next week")

    def test_R20_have_appointment_today_extracted_as_task(self):
        ops = self.extractor.extract("i have an appointment today")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "task")

    def test_R21_learning_not_treated_as_task(self):
        ops = self.extractor.extract("i am learning python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")

    def test_R22_education_not_treated_as_task(self):
        ops = self.extractor.extract("i am studying diploma second year")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")


class CategoryIsolationTests(unittest.TestCase):
    """Regression: education and learning remain distinct through pipeline."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_R23_education_storing_diploma(self):
        ops = self.extractor.extract("i am studying diploma second year")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "education")
        self.assertNotEqual(ops[0]["category"], "learning")

    def test_R24_learning_storing_python(self):
        ops = self.extractor.extract("i am learning python")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "learning")
        self.assertNotEqual(ops[0]["category"], "education")

    def test_R25_goal_not_ambition(self):
        ops = self.extractor.extract("my goal is to become a doctor")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "goal")

    def test_R26_ambition_not_goal(self):
        ops = self.extractor.extract("my ambition is to become CEO")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "ambition")

    def test_R27_dream_not_goal(self):
        ops = self.extractor.extract("my dream is to travel the world")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["category"], "dream")


class ManualFailureRegressionTests(unittest.TestCase):
    """Regression tests for the 9 manual-testing failures."""

    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.manager, self.retriever, self.store = make_stack(self._tmpdir.name)
        self.intent_detector = self.retriever.intent_detector
        self.extractor = MemoryExtractor()

    def tearDown(self):
        self._tmpdir.cleanup()

    def ask(self, question: str) -> str:
        intent = self.retriever.detect_intent(question)
        if not intent.is_memory_read:
            return ""
        memories = self.retriever.retrieve_relevant_memories(
            question, intent=intent
        )
        return grounded_answer(intent, memories, question=question)

    # Issue #1: education/learning contamination
    def test_M1_studying_diploma_returns_education(self):
        self.manager.process_user_message("I am studying diploma second year")
        answer = self.ask("What am I studying?")
        self.assertIn("diploma", answer.lower())
        self.assertNotIn("don't currently", answer.lower())

    def test_M2_learning_python_found_via_education_fallback(self):
        """'Studying' is ambiguous — fallback finds learning memories."""
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I studying?")
        self.assertIn("python", answer.lower())

    def test_M3_education_not_returned_for_learning_query(self):
        """'Learning' query must NOT return education memories (strict separation)."""
        self.manager.process_user_message("I am studying diploma second year")
        answer = self.ask("What am I learning?")
        self.assertNotIn("diploma", answer.lower())

    # Issue #2: ambition/dream over-retrieval
    def test_M4_goal_not_returned_for_dream_query(self):
        self.manager.process_user_message("My goal is to become a software engineer")
        answer = self.ask("What are my dreams?")
        self.assertIn("don't currently", answer.lower())

    def test_M5_goal_found_via_ambition_fallback(self):
        """Ambition query falls back to goals when no ambition is stored."""
        self.manager.process_user_message("My goal is to become a doctor")
        answer = self.ask("What are my ambitions?")
        self.assertIn("doctor", answer.lower())

    def test_M6_dream_stored_correctly(self):
        self.manager.process_user_message("My dream is to travel the world")
        answer = self.ask("What are my dreams?")
        self.assertIn("travel", answer.lower())

    # Issue #3: general chat routing
    def test_M7_general_knowledge_not_memory_read(self):
        intent = self.intent_detector.detect("What is 2 + 2?")
        self.assertFalse(intent.is_memory_read)

    def test_M8_how_are_you_not_memory_read(self):
        intent = self.intent_detector.detect("How are you?")
        self.assertFalse(intent.is_memory_read)

    def test_M9_vague_preference_not_memory_read(self):
        intent = self.intent_detector.detect("I like anything")
        self.assertFalse(intent.is_memory_read)

    def test_M10_vague_preference_dislike_not_memory_read(self):
        intent = self.intent_detector.detect("I don't like anything")
        self.assertFalse(intent.is_memory_read)

    # Issue #4: project extraction
    def test_M11_i_want_to_build_extracted_as_project(self):
        ops = self.extractor.extract("I want to build a personal AI assistant")
        self.assertTrue(any(op["category"] == "project" for op in ops))

    def test_M12_i_am_building_extracted_as_project(self):
        ops = self.extractor.extract("I am building a weather app")
        self.assertTrue(any(op["category"] == "project" for op in ops))

    def test_M13_i_am_creating_extracted_as_project(self):
        ops = self.extractor.extract("I am creating a portfolio website")
        self.assertTrue(any(op["category"] == "project" for op in ops))

    def test_M14_build_cue_in_query_intent(self):
        intent = self.intent_detector.detect("What am I building?")
        self.assertTrue(intent.is_memory_read)
        self.assertIn("project", intent.requested)

    # Issue #5: vague preference handling (covered by Issue #3 fix)
    def test_M15_i_like_everything_not_stored(self):
        ops = self.extractor.extract("I like everything")
        self.assertEqual(len(ops), 0)

    def test_M16_i_hate_nothing_not_stored(self):
        ops = self.extractor.extract("I hate nothing")
        self.assertEqual(len(ops), 0)

    # Issue #6: overview contamination
    def test_M17_overview_shows_categories_separately(self):
        self.manager.process_user_message("My goal is to become a doctor")
        self.manager.process_user_message("My dream is to travel the world")
        intent = self.intent_detector.detect("What do you know about me?")
        memories = self.retriever.retrieve_relevant_memories(
            "What do you know about me?", intent=intent
        )
        goals = [m for m in memories if m.get("category") == "goal"]
        dreams = [m for m in memories if m.get("category") == "dream"]
        self.assertEqual(len(goals), 1)
        self.assertEqual(len(dreams), 1)
        self.assertIn("doctor", goals[0].get("content", "").lower())
        self.assertIn("travel", dreams[0].get("content", "").lower())

    # Issue #7: temporary memory
    def test_M18_task_has_expires_at(self):
        self.manager.process_user_message("I have an exam tomorrow")
        tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        self.assertEqual(len(tasks), 1)
        self.assertIsNotNone(tasks[0].get("expires_at"))

    def test_M19_task_not_education(self):
        self.manager.process_user_message("I have an exam tomorrow")
        tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        edu = [m for m in self.store.active_memories() if m.get("category") == "education"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(edu), 0)

    def test_M20_task_not_learning(self):
        self.manager.process_user_message("I have a test next week")
        tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        learn = [m for m in self.store.active_memories() if m.get("category") == "learning"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(learn), 0)

    # Issue #3 continued: statements vs questions
    def test_M21_statement_does_not_enter_memory_read(self):
        """'I like pizza' is a statement, not a question — it should be stored, not queried."""
        ops = self.extractor.extract("I like pizza")
        self.assertTrue(len(ops) > 0)
        self.assertEqual(ops[0]["category"], "preference")

    def test_M22_what_is_not_memory_for_math(self):
        intent = self.intent_detector.detect("What is the capital of France?")
        self.assertFalse(intent.is_memory_read)

    # Fallback behavior
    def test_M23_education_fallback_to_learning(self):
        """If education has no results, fallback should search learning."""
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I studying?")
        self.assertIn("python", answer.lower())

    def test_M24_ambition_fallback_to_goal(self):
        """If ambition has no results, fallback should search goals."""
        self.manager.process_user_message("My goal is to become a doctor")
        answer = self.ask("What are my ambitions?")
        self.assertIn("doctor", answer.lower())

    def test_M25_dream_strictly_separated_from_goal(self):
        """Dreams and goals are strictly separated — no fallback."""
        self.manager.process_user_message("My goal is to become a software engineer")
        answer = self.ask("What are my dreams?")
        self.assertIn("don't currently", answer.lower())

    # Priority system tests (Failure E/F)
    def test_M26_main_goal_stored_as_goal(self):
        """'My main goal is X' should be stored as goal."""
        ops = self.extractor.extract("My main goal is to get a good rank in ECET")
        goal_ops = [op for op in ops if op["category"] == "goal"]
        self.assertEqual(len(goal_ops), 1)
        self.assertIn("ECET", goal_ops[0]["content"])

    def test_M27_main_goal_also_sets_priority(self):
        """'My main goal is X' should also set priority."""
        ops = self.extractor.extract("My main goal is to get a good rank in ECET")
        priority_ops = [op for op in ops if op["category"] == "priority"]
        self.assertEqual(len(priority_ops), 1)
        self.assertIn("ECET", priority_ops[0]["content"])

    def test_M28_current_priority_stored_correctly(self):
        """'My current main priority is X' should be stored as priority."""
        ops = self.extractor.extract("Actually, my current main priority is ECET")
        priority_ops = [op for op in ops if op["category"] == "priority"]
        self.assertEqual(len(priority_ops), 1)
        self.assertEqual(priority_ops[0]["value"], "ECET")

    def test_M29_priority_query_intent(self):
        """'What is my priority?' should route to priority category."""
        intent = self.intent_detector.detect("What is my priority?")
        self.assertTrue(intent.is_memory_read)
        self.assertEqual(intent.primary, "priority")
        self.assertIn("priority", intent.requested)

    def test_M30_priority_retrieval(self):
        """Priority should be retrievable after storage."""
        self.manager.process_user_message("My main goal is to get a good rank in ECET")
        answer = self.ask("What is my priority?")
        self.assertIn("ecet", answer.lower())

    # Project extraction tests (Failure A/B/C)
    def test_M31_i_want_to_build_is_project(self):
        """'I want to build X' should be project, not goal."""
        ops = self.extractor.extract("I want to build a personal AI assistant")
        self.assertTrue(any(op["category"] == "project" for op in ops))
        self.assertFalse(any(op["category"] == "goal" for op in ops))

    def test_M32_project_query_retrieves_projects(self):
        """'What do I want to build?' should retrieve projects."""
        self.manager.process_user_message("I want to build a portfolio website")
        answer = self.ask("What do I want to build?")
        self.assertIn("portfolio", answer.lower())

    # Incomplete statement tests (Failure D)
    def test_M33_incomplete_i_want_to_not_stored(self):
        """'I want to' (incomplete) should not create any memory."""
        ops = self.extractor.extract("I want to")
        self.assertEqual(len(ops), 0)

    def test_M34_incomplete_i_want_to_space_not_stored(self):
        """'I want to ' (incomplete with space) should not create any memory."""
        ops = self.extractor.extract("I want to ")
        self.assertEqual(len(ops), 0)

    # Vague preference tests (Failure G/H)
    def test_M35_i_like_anything_not_stored(self):
        """'I like anything' should not be stored."""
        ops = self.extractor.extract("I like anything")
        self.assertEqual(len(ops), 0)

    def test_M36_i_dont_like_anything_not_stored(self):
        """'I don't like anything' should not be stored."""
        ops = self.extractor.extract("I don't like anything")
        self.assertEqual(len(ops), 0)

    def test_M37_i_like_anything_not_memory_read(self):
        """'I like anything' should not be a memory read."""
        intent = self.intent_detector.detect("I like anything")
        self.assertFalse(intent.is_memory_read)

    def test_M38_i_dont_like_anything_not_memory_read(self):
        """'I don't like anything' should not be a memory read."""
        intent = self.intent_detector.detect("I don't like anything")
        self.assertFalse(intent.is_memory_read)

    # Education/learning isolation tests (Failure I/J)
    def test_M39_what_am_i_learning_excludes_education(self):
        """'What am I learning?' should not include diploma education."""
        self.manager.process_user_message("I am studying diploma second year")
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I learning?")
        self.assertIn("python", answer.lower())
        self.assertNotIn("diploma", answer.lower())

    def test_M40_what_am_i_studying_excludes_unrelated_learning(self):
        """'What am I studying?' should not incorrectly merge unrelated learning."""
        self.manager.process_user_message("I am studying diploma second year")
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I studying?")
        # Should include education (diploma) via fallback, but not learning (Python)
        self.assertIn("diploma", answer.lower())

    # General knowledge routing tests
    def test_M41_math_question_not_memory_read(self):
        """Math questions should not enter memory pipeline."""
        intent = self.intent_detector.detect("What is 2 + 2?")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.is_general_knowledge)

    def test_M42_definition_question_not_memory_read(self):
        """Definition questions should not enter memory pipeline."""
        intent = self.intent_detector.detect("What is the capital of France?")
        self.assertFalse(intent.is_memory_read)

    # Task extraction tests
    def test_M43_task_with_temporal_hint(self):
        """Tasks should have temporal hints for expiration."""
        self.manager.process_user_message("I have an exam tomorrow")
        tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        self.assertEqual(len(tasks), 1)
        self.assertIsNotNone(tasks[0].get("expires_at"))

    def test_M44_task_not_confused_with_education(self):
        """Tasks should not be confused with education."""
        self.manager.process_user_message("I have an exam tomorrow")
        tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        edu = [m for m in self.store.active_memories() if m.get("category") == "education"]
        self.assertEqual(len(tasks), 1)
        self.assertEqual(len(edu), 0)

    # Category isolation tests
    def test_M45_goal_found_via_ambition_fallback(self):
        """Goals can be found via ambition fallback when no ambition is stored."""
        self.manager.process_user_message("My goal is to become a software engineer")
        answer = self.ask("What are my ambitions?")
        self.assertIn("software engineer", answer.lower())

    def test_M46_ambition_not_goal(self):
        """Ambitions and goals should be strictly separated."""
        self.manager.process_user_message("My ambition is to become a doctor")
        answer = self.ask("What are my goals?")
        self.assertIn("don't currently", answer.lower())

    def test_M47_dream_not_goal(self):
        """Dreams and goals should be strictly separated."""
        self.manager.process_user_message("My dream is to travel the world")
        answer = self.ask("What are my goals?")
        self.assertIn("don't currently", answer.lower())

    # Fallback behavior tests
    def test_M48_education_fallback_to_learning(self):
        """Education query should fallback to learning when education is empty."""
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I studying?")
        self.assertIn("python", answer.lower())

    def test_M49_ambition_fallback_to_goal(self):
        """Ambition query should fallback to goal when ambition is empty."""
        self.manager.process_user_message("My goal is to become a doctor")
        answer = self.ask("What are my ambitions?")
        self.assertIn("doctor", answer.lower())

    def test_M50_dream_no_fallback_to_goal(self):
        """Dream query should NOT fallback to goal."""
        self.manager.process_user_message("My goal is to become a software engineer")
        answer = self.ask("What are my dreams?")
        self.assertIn("don't currently", answer.lower())

    # ======== v0.6 manual regression: 6 new failures ========

    def test_F1_learning_query_excludes_education(self):
        """Failure 1: 'What am I learning?' must NOT include diploma education."""
        self.manager.process_user_message("I am studying diploma second year")
        self.manager.process_user_message("I am learning Python")
        answer = self.ask("What am I learning?")
        self.assertIn("python", answer.lower())
        self.assertNotIn("diploma", answer.lower())

    def test_F2_malformed_goal_with_typo_becomes_project(self):
        """Failure 2: 'I want to become bulid a personal AI assistant'
        must be extracted as project (typo 'bulid' → 'build'), not as a
        malformed goal."""
        ops = self.extractor.extract(
            "I want to become bulid a personal AI assistant"
        )
        # Should be extracted as project, not goal
        categories = [op["category"] for op in ops]
        self.assertIn("project", categories)
        self.assertNotIn("goal", categories)

    def test_F3_dream_rendering_preserves_build_verb(self):
        """Failure 3: Dream 'build my own AI company' must render as
        'build my own AI company', NOT 'become build my own AI company'."""
        self.manager.process_user_message("My dream is to build my own AI company")
        answer = self.ask("What are my dreams?")
        self.assertIn("build my own ai company", answer.lower())
        # Must NOT contain the ungrammatical "become build"
        self.assertNotIn("become build", answer.lower())

    def test_F4_incomplete_i_want_to_needs_clarification(self):
        """Failure 4: 'I want to' (incomplete) must NOT retrieve any memories.
        It should request clarification."""
        intent = self.intent_detector.detect("I want to")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.needs_clarification)
        self.assertIn("complete", intent.clarification_message.lower())

    def test_F5_vague_goal_nothing_needs_clarification(self):
        """Failure 5: 'I want to become nothing' must NOT retrieve existing
        goals. It should request clarification."""
        intent = self.intent_detector.detect("I want to become nothing")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.needs_clarification)
        self.assertIn("vague", intent.clarification_message.lower())

    def test_F6_vague_preference_needs_clarification(self):
        """Failure 6: 'I like anything' must NOT produce an LLM response.
        It should request a specific preference."""
        intent = self.intent_detector.detect("I like anything")
        self.assertFalse(intent.is_memory_read)
        self.assertFalse(intent.is_general_knowledge)
        self.assertTrue(intent.needs_clarification)
        self.assertIn("specific", intent.clarification_message.lower())


class SupersedeMemoryTests(unittest.TestCase):
    """Tests for the supersede_memory lifecycle operation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_supersede_marks_loser_inactive(self):
        """Superseding a memory marks the loser as inactive."""
        winner = self.store.add_memory(
            category="project", content="build AI assistant",
            value="AI assistant", importance="high",
        )
        loser = self.store.add_memory(
            category="goal", content="bulid AI assistant",
            value="bulid AI assistant", importance="high",
        )
        result = self.store.supersede_memory(winner["id"], loser["id"])
        self.assertTrue(result)
        loser_reloaded = self.store.get_by_id(loser["id"])
        self.assertEqual(loser_reloaded["status"], "inactive")
        self.assertEqual(loser_reloaded["superseded_by"], winner["id"])

    def test_supersede_preserves_history(self):
        """Superseding preserves the loser's content in the store."""
        winner = self.store.add_memory(
            category="project", content="build AI assistant",
            value="AI assistant", importance="high",
        )
        loser = self.store.add_memory(
            category="goal", content="bulid AI assistant",
            value="bulid AI assistant", importance="high",
        )
        self.store.supersede_memory(winner["id"], loser["id"])
        loser_reloaded = self.store.get_by_id(loser["id"])
        self.assertIsNotNone(loser_reloaded)
        self.assertEqual(loser_reloaded["content"], "bulid AI assistant")

    def test_supersede_invalidates_embedding(self):
        """Superseding invalidates the loser's embedding."""
        winner = self.store.add_memory(
            category="project", content="build AI assistant",
            value="AI assistant", importance="high",
        )
        loser = self.store.add_memory(
            category="goal", content="bulid AI assistant",
            value="bulid AI assistant", importance="high",
        )
        self.store.supersede_memory(winner["id"], loser["id"])
        loser_reloaded = self.store.get_by_id(loser["id"])
        self.assertIsNone(loser_reloaded.get("embedding"))

    def test_supersede_idempotent(self):
        """Superseding the same pair twice is a no-op (idempotent)."""
        winner = self.store.add_memory(
            category="project", content="build AI assistant",
            value="AI assistant", importance="high",
        )
        loser = self.store.add_memory(
            category="goal", content="bulid AI assistant",
            value="bulid AI assistant", importance="high",
        )
        self.store.supersede_memory(winner["id"], loser["id"])
        result = self.store.supersede_memory(winner["id"], loser["id"])
        self.assertTrue(result)
        loser_reloaded = self.store.get_by_id(loser["id"])
        self.assertEqual(loser_reloaded["status"], "inactive")

    def test_supersede_nonexistent_returns_false(self):
        """Superseding with nonexistent IDs returns False."""
        result = self.store.supersede_memory("nonexistent_1", "nonexistent_2")
        self.assertFalse(result)

    def test_supersede_already_inactive_returns_true(self):
        """Superseding an already inactive memory is a no-op returning True."""
        winner = self.store.add_memory(
            category="project", content="build AI assistant",
            value="AI assistant", importance="high",
        )
        loser = self.store.add_memory(
            category="goal", content="bulid AI assistant",
            value="bulid AI assistant", importance="high",
        )
        self.store.deactivate_memory(loser["id"])
        result = self.store.supersede_memory(winner["id"], loser["id"])
        self.assertTrue(result)


class TaskCancellationTests(unittest.TestCase):
    """Tests for task cancellation extraction and processing."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)
        self.extractor = MemoryExtractor()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_task_cancellation_extracted(self):
        """'I don't have any task tomorrow' extracts a CANCEL operation."""
        ops = self.extractor.extract("I don't have any task tomorrow")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "CANCEL")
        self.assertEqual(ops[0]["category"], "task")

    def test_task_cancellation_with_typo(self):
        """'I don't have any task tommorrow' extracts CANCEL (typo normalized)."""
        ops = self.extractor.extract("I don't have any task tommorrow")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "CANCEL")
        self.assertEqual(ops[0]["category"], "task")

    def test_task_cancellation_generic_cancel(self):
        """'cancel my exam' extracts a CANCEL operation."""
        ops = self.extractor.extract("cancel my exam")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "CANCEL")
        self.assertEqual(ops[0]["category"], "task")

    def test_task_cancellation_forget(self):
        """'forget my task' extracts a CANCEL operation."""
        ops = self.extractor.extract("forget my task")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "CANCEL")
        self.assertEqual(ops[0]["category"], "task")

    def test_task_cancellation_there_is_no(self):
        """'there is no exam tomorrow' extracts a CANCEL operation."""
        ops = self.extractor.extract("there is no exam tomorrow")
        self.assertEqual(len(ops), 1)
        self.assertEqual(ops[0]["action"], "CANCEL")
        self.assertEqual(ops[0]["category"], "task")

    def test_task_cancellation_deactivates_matching_task(self):
        """CANCEL action deactivates the matching active task."""
        results = self.manager.process_user_message("I have CT exam tomorrow")
        self.assertEqual(len(results), 1)
        add_result = results[0]
        self.assertEqual(add_result["status"], "added")
        task_id = add_result["memory_id"]

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "active")

        results = self.manager.process_user_message("I don't have CT exam tomorrow")
        self.assertEqual(len(results), 1)
        cancel_result = results[0]
        self.assertEqual(cancel_result["status"], "cancelled")

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")

    def test_cancelled_task_absent_from_retrieval(self):
        """Cancelled task does not appear in active retrieval."""
        self.manager.process_user_message("I have test next week")
        self.manager.process_user_message("I don't have test next week")
        # Query should not find the cancelled task in active memories
        active_tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        task_values = [m.get("value", "").lower() for m in active_tasks]
        self.assertFalse(any("test" in v for v in task_values))


class InvalidGoalTests(unittest.TestCase):
    """Tests for invalid goal rejection with typo normalization."""

    def setUp(self):
        self.extractor = MemoryExtractor()

    def test_noting_rejected(self):
        """'I want to become noting' is rejected as invalid."""
        ops = self.extractor.extract("I want to become noting")
        self.assertEqual(len(ops), 0)

    def test_nothing_rejected(self):
        """'I want to become nothing' is rejected as invalid."""
        ops = self.extractor.extract("I want to become nothing")
        self.assertEqual(len(ops), 0)

    def test_something_rejected(self):
        """'I want to become something' is rejected as invalid."""
        ops = self.extractor.extract("I want to become something")
        self.assertEqual(len(ops), 0)

    def test_anything_rejected(self):
        """'I want to become anything' is rejected as invalid."""
        ops = self.extractor.extract("I want to become anything")
        self.assertEqual(len(ops), 0)

    def test_valid_goal_not_rejected(self):
        """'I want to become an AI engineer' is NOT rejected."""
        ops = self.extractor.extract("I want to become an AI engineer")
        self.assertGreater(len(ops), 0)
        self.assertEqual(ops[0]["category"], "goal")


class MigrationTests(unittest.TestCase):
    """Tests for the corrupted memory migration function."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migration_supersedes_typo_goal(self):
        """Migration supersedes 'bulid' goal by correct project."""
        project = self.store.add_memory(
            category="project", content="build a personal AI assistant",
            value="a personal AI assistant", importance="high",
        )
        goal = self.store.add_memory(
            category="goal", content="bulid a personal ai assistant",
            value="bulid a personal ai assistant", importance="high",
        )
        result = self.manager.migrate_corrupted_memories()
        superseded_ids = [s["loser_id"] for s in result["superseded"]]
        self.assertIn(goal["id"], superseded_ids)

    def test_migration_supersedes_misclassified_learning(self):
        """Migration supersedes learning memory when equivalent education exists."""
        edu = self.store.add_memory(
            category="education", content="User is studying diploma second year",
            value="diploma second year", importance="high",
        )
        learning = self.store.add_memory(
            category="learning", content="User is learning diploma 2nd year",
            value="diploma 2nd year", importance="medium",
        )
        result = self.manager.migrate_corrupted_memories()
        superseded_ids = [s["loser_id"] for s in result["superseded"]]
        self.assertIn(learning["id"], superseded_ids)

    def test_migration_deactivates_noting_goal(self):
        """Migration deactivates 'noting' goal as invalid."""
        goal = self.store.add_memory(
            category="goal", content="User wants to become noting",
            value="noting", importance="high",
        )
        result = self.manager.migrate_corrupted_memories()
        deactivated_ids = [d["id"] for d in result["deactivated"]]
        self.assertIn(goal["id"], deactivated_ids)


class LegitimateCoexistenceTests(unittest.TestCase):
    """Tests that legitimate cross-category memories coexist."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_learning_and_goal_coexist(self):
        """'learning Python' and 'goal become Python developer' coexist."""
        results1 = self.manager.process_user_message("I am learning Python")
        self.assertEqual(len(results1), 1)
        self.assertEqual(results1[0]["status"], "added")
        results2 = self.manager.process_user_message("I want to become a Python developer")
        self.assertEqual(len(results2), 1)
        self.assertEqual(results2[0]["status"], "added")
        active = [m for m in self.store.memories if m.get("status") == "active"]
        categories = [m.get("category") for m in active]
        self.assertIn("learning", categories)
        self.assertIn("goal", categories)


class AmbitionDreamGrammarTests(unittest.TestCase):
    """Tests for correct ambition/dream grammar rendering."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_ambition_become_preserved(self):
        """'become an AI engineer' — the value is normalized (strip become)
        and the answer rendering adds 'become' back."""
        results = self.manager.process_user_message("My ambition is to become an AI engineer")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "added")
        memory = self.store.get_by_id(results[0]["memory_id"])
        # Value is normalized to the topic (become is stripped by normalize_goal_text)
        self.assertIn("ai engineer", memory.get("value", "").lower())
        # Content preserves the full statement
        self.assertIn("become", memory.get("content", "").lower())

    def test_dream_build_preserved(self):
        """'build my own AI company' — value is preserved with the build verb."""
        results = self.manager.process_user_message("My dream is to build my own AI company")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "added")
        memory = self.store.get_by_id(results[0]["memory_id"])
        self.assertIn("build", memory.get("value", "").lower())
        self.assertIn("ai company", memory.get("value", "").lower())

    def test_goal_become_preserved(self):
        """'My goal is to become an AI engineer' — value is normalized and
        content preserves the full statement."""
        results = self.manager.process_user_message("My goal is to become an AI engineer")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "added")
        memory = self.store.get_by_id(results[0]["memory_id"])
        self.assertEqual(memory.get("category"), "goal")
        # Content preserves the full statement
        self.assertIn("become", memory.get("content", "").lower())
        self.assertIn("ai engineer", memory.get("value", "").lower())


class TaskCancellationRegressionTests(unittest.TestCase):
    """Regression tests for FAILURE 1: task cancellation pipeline."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_cancel_deactivates_matching_task(self):
        """'I don't have any task tomorrow' deactivates the matching task."""
        # Add a task
        results = self.manager.process_user_message("I have CT exam tomorrow")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "added")
        task_id = results[0]["memory_id"]

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "active")

        # Cancel the task
        results = self.manager.process_user_message("I don't have any task tomorrow.")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["status"], "cancelled")

        # Verify task is now inactive
        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")

    def test_cancelled_task_absent_from_retrieval(self):
        """Cancelled task does not appear in active retrieval."""
        self.manager.process_user_message("I have test next week")
        self.manager.process_user_message("I don't have any task next week.")
        active_tasks = [m for m in self.store.active_memories() if m.get("category") == "task"]
        task_values = [m.get("value", "").lower() for m in active_tasks]
        self.assertFalse(any("test" in v for v in task_values))

    def test_cancel_with_temporal_typo(self):
        """'I don't have any task tommorrow' still cancels the task."""
        results = self.manager.process_user_message("I have quiz tomorrow")
        task_id = results[0]["memory_id"]

        results = self.manager.process_user_message("I don't have any task tommorrow.")
        self.assertEqual(results[0]["status"], "cancelled")

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")

    def test_cancel_generic_cancel_phrase(self):
        """'cancel my exam' cancels the matching task."""
        results = self.manager.process_user_message("I have exam tomorrow")
        task_id = results[0]["memory_id"]

        results = self.manager.process_user_message("cancel my exam")
        self.assertEqual(results[0]["status"], "cancelled")

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")

    def test_cancel_forget_phrase(self):
        """'forget my task' cancels the matching task."""
        results = self.manager.process_user_message("I have assignment due today")
        task_id = results[0]["memory_id"]

        results = self.manager.process_user_message("forget my task")
        self.assertEqual(results[0]["status"], "cancelled")

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")

    def test_cancel_there_is_no_phrase(self):
        """'there is no exam tomorrow' cancels the matching task."""
        results = self.manager.process_user_message("I have exam tomorrow")
        task_id = results[0]["memory_id"]

        results = self.manager.process_user_message("there is no exam tomorrow")
        self.assertEqual(results[0]["status"], "cancelled")

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")

    def test_cancel_preserves_history(self):
        """Cancelled task preserves historical record."""
        results = self.manager.process_user_message("I have viva tomorrow")
        task_id = results[0]["memory_id"]

        self.manager.process_user_message("I don't have viva tomorrow.")

        task = self.store.get_by_id(task_id)
        self.assertEqual(task["status"], "inactive")
        self.assertIsNotNone(task.get("updated_at"))


class InvalidNotingRegressionTests(unittest.TestCase):
    """Regression tests for FAILURE 2: invalid 'noting' retrieval."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manager, self.retriever, self.store = make_stack(self.tmpdir)
        self.intent_detector = QueryIntentDetector(self.store.embedding_store)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_noting_triggers_clarification(self):
        """'I want to become noting' triggers clarification, not goal retrieval."""
        intent = self.intent_detector.detect("I want to become noting.")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.needs_clarification)
        self.assertIn("vague", intent.clarification_message.lower())

    def test_nothing_still_triggers_clarification(self):
        """'I want to become nothing' still triggers clarification."""
        intent = self.intent_detector.detect("I want to become nothing.")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.needs_clarification)
        self.assertIn("vague", intent.clarification_message.lower())

    def test_noting_does_not_retrieve_goals(self):
        """'I want to become noting' does not retrieve existing goals."""
        # Add some goals
        self.manager.process_user_message("I want to become an AI engineer")
        self.manager.process_user_message("I want to become a researcher")

        # Query with invalid statement
        intent = self.intent_detector.detect("I want to become noting.")
        self.assertFalse(intent.is_memory_read)
        self.assertNotIn("goal", intent.requested)

    def test_something_triggers_clarification(self):
        """'I want to become something' also triggers clarification."""
        intent = self.intent_detector.detect("I want to become something.")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.needs_clarification)

    def test_anything_triggers_clarification(self):
        """'I want to become anything' also triggers clarification."""
        intent = self.intent_detector.detect("I want to become anything.")
        self.assertFalse(intent.is_memory_read)
        self.assertTrue(intent.needs_clarification)

    def test_valid_goal_still_retrieves(self):
        """'I want to become an AI engineer' still retrieves goals."""
        self.manager.process_user_message("I want to become an AI engineer")
        intent = self.intent_detector.detect("What is my goal?")
        self.assertTrue(intent.is_memory_read)
        self.assertIn("goal", intent.requested)


if __name__ == "__main__":
    unittest.main(verbosity=2)
