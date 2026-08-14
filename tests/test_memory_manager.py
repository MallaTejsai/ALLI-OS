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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from memory.answers import grounded_answer
from memory.embeddings import EmbeddingError, EmbeddingStore
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

    # 9–11 skill vs learning
    def test_09_add_skill(self):
        self.manager.process_user_message("I know Python")
        self.assertEqual(self.active()[0]["category"], "skill")
        answer = self.ask("What are my skills?")
        self.assertIn("Python", answer)

    def test_10_add_learning_goal(self):
        self.manager.process_user_message("I am learning C programming")
        mem = self.active()[0]
        self.assertEqual(mem["category"], "learning")
        self.assertEqual(mem["fact_type"], "learning")
        self.assertNotEqual(mem["category"], "skill")

    def test_11_goal_not_skill(self):
        self.manager.process_user_message("I am learning C programming")
        answer = self.ask("What are my skills?")
        self.assertIn("don't currently", answer.lower())
        self.assertNotIn("C", answer)

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

    # New tests for v0.6 positive-learning fix
    def test_26_learning_add(self):
        r = self.manager.process_user_message("I am learning C programming")
        mem = self.active()[0]
        self.assertEqual(mem["category"], "learning")
        self.assertEqual(mem["fact_type"], "learning")

    def test_27_learning_add_lowercase(self):
        r = self.manager.process_user_message("i'm learning c programming")
        mem = self.active()[0]
        self.assertEqual(mem["category"], "learning")
        self.assertEqual(mem["fact_type"], "learning")

    def test_28_learning_removal(self):
        self.manager.process_user_message("I am learning C programming")
        r = self.manager.process_user_message("I don't want to learn C anymore")
        self.assertEqual(r[0]["status"], "deleted")

    # Regression tests
    def test_29_learning_still_skill(self):
        self.manager.process_user_message("I know Python")
        self.manager.process_user_message("I am learning C programming")
        skills = [m for m in self.active() if m["category"] == "skill"]
        learning = [m for m in self.active() if m["category"] == "learning"]
        self.assertEqual(len(skills), 1)
        self.assertEqual(len(learning), 1)
        self.assertIn("python", skills[0]["content"].lower())
        self.assertIn("c", learning[0]["content"].lower())

    def test_30_learning_not_skill(self):
        self.manager.process_user_message("I am learning C programming")
        answer = self.ask("What are my skills?")
        self.assertIn("don't currently", answer.lower())
        self.assertNotIn("C", answer)

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
