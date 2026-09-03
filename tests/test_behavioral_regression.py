"""End-to-end behavioral regression tests for ALLI v0.6.

Tests the complete pipeline: write → store → query → retrieve → answer.
Uses the real memory stack with Ollama embeddings.
"""

import os
import sys
import tempfile
import shutil

# Ensure correct path
SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("OLLAMA_HOST", "http://172.25.80.1:11434")

from memory.store import MemoryStore
from memory.embeddings import EmbeddingStore, EmbeddingError
from memory.query_intent import QueryIntentDetector
from memory.retriever import MemoryRetriever
from memory.manager import SafeMemoryManager
from memory.answers import grounded_answer
from memory.extractor import MemoryExtractor


def make_stack(tmpdir: str):
    """Create a fresh memory stack with real embeddings."""
    path = os.path.join(tmpdir, "long_term_memory.json")
    embedding_store = EmbeddingStore()
    try:
        embedding_store.verify_model()
    except EmbeddingError:
        print("[SKIP] Ollama not available")
        return None
    store = MemoryStore(path=path, embedding_store=embedding_store)
    store.load(ensure_embeddings=True)
    manager = SafeMemoryManager(store)
    detector = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(store, embedding_store, intent_detector=detector)
    return manager, retriever, store


def write_and_query(stack, statements: list[str], query: str, label: str):
    """Write statements then query. Returns (answer, debug_info)."""
    manager, retriever, store = stack
    for stmt in statements:
        results = manager.process_user_message(stmt)
        for r in results:
            print(f"  WRITE: {stmt!r} → {r.get('status')} ({r.get('memory_id', '')[:8]})")

    intent = retriever.detect_intent(query)
    entity = getattr(intent, "entity", None)
    entity_name = getattr(entity, "name", None) if entity else None
    print(f"  INTENT: primary={intent.primary}, entity={entity_name}, "
          f"is_memory_read={intent.is_memory_read}, cardinality={intent.cardinality}")

    if intent.is_memory_read:
        if intent.is_multi:
            grouped = retriever.retrieve_by_requested_intents(query, intent=intent)
            relevant = []
            for items in grouped.values():
                relevant.extend(items)
            answer = grounded_answer(intent, relevant, question=query, grouped=grouped)
        else:
            relevant = retriever.retrieve_relevant_memories(query, intent=intent)
            answer = grounded_answer(intent, relevant, question=query)
    else:
        answer = None  # Would go to general LLM
        relevant = []

    debug = retriever.last_debug
    selected = debug.get("selected", [])
    print(f"  RETRIEVED: {len(relevant)} memories, {len(selected)} selected")
    for s in selected[:3]:
        print(f"    → {s.get('id', '?')[:8]}: {s.get('content', '?')[:60]}")

    return answer, relevant


def run_all_tests():
    """Run all behavioral regression scenarios."""
    tmpdir = tempfile.mkdtemp()
    results = {}

    try:
        stack = make_stack(tmpdir)
        if stack is None:
            print("Ollama not available, cannot run behavioral tests")
            return

        # =========================================================================
        # SCENARIO A: POSSESSION
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO A: POSSESSION")
        print("=" * 60)
        answer, _ = write_and_query(stack,
            ["I have an Oppo A53."],
            "What phone do I have?",
            "possession"
        )
        print(f"  ANSWER: {answer}")
        results["A_possession"] = answer and "oppo" in answer.lower()

        # =========================================================================
        # SCENARIO B: ENTITY ISOLATION
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO B: ENTITY ISOLATION")
        print("=" * 60)
        # First store both memories
        manager, retriever, store = stack
        manager.process_user_message("My favourite colour is black.")
        manager.process_user_message("My mom's favourite colour is blue.")

        # Query user
        answer_user, _ = write_and_query(stack, [],
            "What is my favourite colour?",
            "entity_user"
        )
        print(f"  USER ANSWER: {answer_user}")

        # Query mom
        answer_mom, _ = write_and_query(stack, [],
            "What is my mom's favourite colour?",
            "entity_mom"
        )
        print(f"  MOM ANSWER: {answer_mom}")

        user_ok = answer_user and "black" in answer_user.lower()
        mom_ok = answer_mom and "blue" in answer_mom.lower()
        results["B_entity_isolation"] = user_ok and mom_ok

        # =========================================================================
        # SCENARIO C: UNKNOWN MEMORY / GROUNDING
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO C: UNKNOWN MEMORY / GROUNDING")
        print("=" * 60)
        answer_movie, _ = write_and_query(stack, [],
            "What is my favourite movie?",
            "unknown_movie"
        )
        print(f"  MOVIE ANSWER: {answer_movie}")
        # Should NOT contain unrelated memory like "robotics"
        grounding_ok = answer_movie and "robotics" not in answer_movie.lower()
        missing_ok = answer_movie and ("don't" in answer_movie.lower() or "not stored" in answer_movie.lower() or "favorite" in answer_movie.lower())
        results["C_grounding"] = grounding_ok and missing_ok

        # =========================================================================
        # SCENARIO D: GOAL SEMANTICS
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO D: GOAL SEMANTICS")
        print("=" * 60)
        manager.process_user_message("My goal is to become a doctor.")
        manager.process_user_message("My goal is to become a software engineer.")

        answer_goals, _ = write_and_query(stack, [],
            "What are my goals?",
            "goals_plural"
        )
        print(f"  GOALS PLURAL: {answer_goals}")

        answer_goal, _ = write_and_query(stack, [],
            "What is my goal?",
            "goals_singular"
        )
        print(f"  GOAL SINGULAR: {answer_goal}")

        goals_plural_ok = answer_goals and ("doctor" in answer_goals.lower() or "software engineer" in answer_goals.lower())
        results["D_goal_semantics"] = goals_plural_ok

        # =========================================================================
        # SCENARIO E: CORRECTION / SUPERCESSION
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO E: CORRECTION / SUPERCESSION")
        print("=" * 60)
        # Fresh stack for this test
        tmpdir2 = tempfile.mkdtemp()
        stack2 = make_stack(tmpdir2)
        if stack2:
            manager2, retriever2, store2 = stack2
            manager2.process_user_message("I am learning Python.")
            manager2.process_user_message("Actually, I am learning Java now.")

            answer_learn, _ = write_and_query(stack2, [],
                "What am I learning?",
                "correction"
            )
            print(f"  LEARNING ANSWER: {answer_learn}")

            # Check active memories
            active = store2.active_memories()
            learning = [m for m in active if m["category"] == "learning"]
            print(f"  ACTIVE LEARNING: {len(learning)}")
            for m in learning:
                print(f"    → {m['content'][:60]} (status={m.get('status', 'active')})")

            # Python should be inactive, Java active
            python_active = any("python" in m["content"].lower() and m.get("status") != "inactive" for m in learning)
            java_active = any("java" in m["content"].lower() and m.get("status") != "inactive" for m in learning)
            results["E_correction"] = java_active and not python_active
            shutil.rmtree(tmpdir2)

        # =========================================================================
        # SCENARIO F: TEMPORARY TASK
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO F: TEMPORARY TASK")
        print("=" * 60)
        manager.process_user_message("I want to prepare for my mid exam.")

        answer_task, _ = write_and_query(stack, [],
            "What am I preparing for?",
            "task"
        )
        print(f"  TASK ANSWER: {answer_task}")
        results["F_temporary_task"] = answer_task and "exam" in answer_task.lower()

        # =========================================================================
        # SCENARIO G: PREFERENCE RETRIEVAL
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO G: PREFERENCE RETRIEVAL")
        print("=" * 60)
        manager.process_user_message("I like Python and Java.")
        manager.process_user_message("I don't like Java and C++.")

        answer_likes, _ = write_and_query(stack, [],
            "What programming languages do I like?",
            "preference_like"
        )
        print(f"  LIKES ANSWER: {answer_likes}")

        answer_dislikes, _ = write_and_query(stack, [],
            "What programming languages don't I like?",
            "preference_dislike"
        )
        print(f"  DISLIKES ANSWER: {answer_dislikes}")

        results["G_preference"] = (
            answer_likes is not None and
            answer_dislikes is not None and
            "don't" not in (answer_likes or "").lower()[:20]
        )

        # =========================================================================
        # SCENARIO H: GENERAL VS PERSONAL
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO H: GENERAL VS PERSONAL")
        print("=" * 60)
        intent_general = retriever.detect_intent("What is Python?")
        print(f"  GENERAL INTENT: primary={intent_general.primary}, "
              f"is_general_knowledge={intent_general.is_general_knowledge}, "
              f"is_memory_read={intent_general.is_memory_read}")

        intent_personal = retriever.detect_intent("What am I learning?")
        print(f"  PERSONAL INTENT: primary={intent_personal.primary}, "
              f"is_general_knowledge={intent_personal.is_general_knowledge}, "
              f"is_memory_read={intent_personal.is_memory_read}")

        results["H_general_vs_personal"] = (
            intent_general.is_general_knowledge and
            not intent_personal.is_general_knowledge and
            intent_personal.is_memory_read
        )

        # =========================================================================
        # SCENARIO I: SELF OVERVIEW
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO I: SELF OVERVIEW")
        print("=" * 60)
        answer_overview, _ = write_and_query(stack, [],
            "Tell me about myself.",
            "overview"
        )
        print(f"  OVERVIEW: {answer_overview[:200] if answer_overview else 'None'}...")
        results["I_overview"] = answer_overview and len(answer_overview) > 50

        # =========================================================================
        # SCENARIO J: RESTART PERSISTENCE
        # =========================================================================
        print("\n" + "=" * 60)
        print("SCENARIO J: RESTART PERSISTENCE")
        print("=" * 60)
        # Store a memory, then create a new stack pointing to the same file
        manager.process_user_message("My name is TestUser.")
        store.save()

        # Simulate restart by creating new stack with same path
        path = os.path.join(tmpdir, "long_term_memory.json")
        embedding_store2 = EmbeddingStore()
        store3 = MemoryStore(path=path, embedding_store=embedding_store2)
        store3.load(ensure_embeddings=True)
        active = store3.active_memories()
        name_mems = [m for m in active if m["category"] == "identity" and m.get("fact_type") == "name"]
        print(f"  PERSISTENCE: {len(name_mems)} name memories after restart")
        results["J_persistence"] = len(name_mems) > 0

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "=" * 60)
    print("BEHAVIORAL TEST RESULTS")
    print("=" * 60)
    passed = 0
    failed = 0
    for scenario, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  {scenario}: {status}")
        if ok:
            passed += 1
        else:
            failed += 1

    print(f"\n  TOTAL: {passed} passed, {failed} failed")
    return results


if __name__ == "__main__":
    run_all_tests()
