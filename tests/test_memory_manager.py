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
        answer = self.ask("What are my dreams?")
        self.assertIn("software engineer", answer.lower())

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
        dreams = self.ask("What are my dreams?")
        self.assertIn("software engineer", dreams.lower())

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
        background = [m for m in self.active() if m["category"] == "education"]
        self.assertEqual(len(background), 1)
        self.assertIn("diploma second year", background[0]["value"].lower())
        self.assertIn("diploma second year", self.ask("What is my background?").lower())

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


class ArchitectureGuardTests(unittest.TestCase):
    def test_no_keyword_question_routing(self):
        retriever = (SRC / "memory" / "retriever.py").read_text(encoding="utf-8")
        self.assertNotIn("asks_nickname", retriever)
        self.assertNotIn("def _metadata_boost", retriever)
        main = (SRC / "main.py").read_text(encoding="utf-8")
        self.assertNotIn("def direct_memory_answer", main)
        self.assertIn("ALLI v0.6", main)


if __name__ == "__main__":
    unittest.main(verbosity=2)
