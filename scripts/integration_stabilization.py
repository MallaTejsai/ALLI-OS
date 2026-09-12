"""Integration test for v0.6 stabilization bugs."""
import os
import sys
import tempfile

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from memory.answers import grounded_answer
from memory.embeddings import EmbeddingStore
from memory.extractor import MemoryExtractor
from memory.manager import SafeMemoryManager
from memory.query_intent import QueryIntentDetector
from memory.retriever import MemoryRetriever
from memory.store import MemoryStore


def run_scenario(name: str, steps: list[tuple[str, str]]) -> None:
    """steps: list of (user_message, expected_substring_in_answer or '')"""
    print(f"\n=== {name} ===")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "ltm.json")
    es = EmbeddingStore()
    store = MemoryStore(path=path, embedding_store=es, skip_embeddings=False)
    store.load(ensure_embeddings=True)
    mgr = SafeMemoryManager(store)
    det = QueryIntentDetector(es)
    ret = MemoryRetriever(store, es, intent_detector=det)

    for msg, expect in steps:
        from memory.extractor import looks_like_question

        if not looks_like_question(msg):
            mgr.process_user_message(msg)
        intent = det.detect(msg)
        if intent.is_memory_read:
            mems = ret.retrieve_relevant_memories(msg, intent=intent)
            answer = grounded_answer(intent, mems, question=msg)
        else:
            answer = f"[NOT MEMORY READ: {intent.primary}]"
        print(f"  Q/A: {msg!r}")
        print(f"       intent={intent.primary} entity={getattr(intent.entity, 'name', None)} unresolved={getattr(intent.entity, 'is_unresolved', None)}")
        print(f"       -> {answer[:120]}")
        if expect and expect.lower() not in answer.lower():
            print(f"       FAIL expected substring: {expect!r}")


def main() -> None:
    run_scenario(
        "Possession",
        [
            ("I have an Oppo A53.", ""),
            ("What phone do I have?", "oppo"),
        ],
    )
    run_scenario(
        "Entity scope",
        [
            ("My favourite colour is black.", ""),
            ("My mom's favourite colour is blue.", ""),
            ("What is my mom's favourite colour?", "blue"),
            ("What is my favourite colour?", "black"),
        ],
    )
    run_scenario(
        "Unknown personal",
        [
            ("What is my favourite movie?", "don't"),
        ],
    )
    run_scenario(
        "Learning correction",
        [
            ("I am learning Python.", ""),
            ("Actually, I am learning Java now.", ""),
            ("What am I learning?", "java"),
        ],
    )
    run_scenario(
        "Temporary task",
        [
            ("I want to prepare for my mid exam.", ""),
            ("What am I preparing for?", "exam"),
        ],
    )
    run_scenario(
        "Multi-value preference",
        [
            ("I like Python and Java.", ""),
            ("What programming languages do I like?", "python"),
        ],
    )


if __name__ == "__main__":
    main()
