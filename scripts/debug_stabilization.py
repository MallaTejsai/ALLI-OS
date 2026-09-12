"""Temporary diagnostic script for v0.6 stabilization."""
import os
import sys
import tempfile

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, os.path.abspath(SRC))

from memory.query_entity import resolve_query_entity
from memory.query_intent import QueryIntentDetector
from memory.embeddings import EmbeddingStore
from memory.store import MemoryStore
from memory.manager import SafeMemoryManager
from memory.retriever import MemoryRetriever
from memory.answers import grounded_answer
from memory.extractor import MemoryExtractor


def main() -> None:
    queries = [
        "What phone do I have?",
        "Which phone am I using?",
        "What is my mom's favourite colour?",
        "What is my favourite colour?",
        "What is my favourite movie?",
        "What am I preparing for?",
        "What programming languages do I like?",
        "What is my goal?",
        "What are my goals?",
    ]
    print("=== ENTITY RESOLUTION ===")
    for q in queries:
        e = resolve_query_entity(q.lower())
        print(f"{q!r} -> name={e.name!r} unresolved={e.is_unresolved}")

    print("\n=== INTENT (full detect with embeddings) ===")
    es = EmbeddingStore()
    det = QueryIntentDetector(es)
    for q in queries:
        intent = det.detect(q)
        print(
            f"{q!r} -> primary={intent.primary!r} memory_read={intent.is_memory_read} "
            f"general={intent.is_general_knowledge} card={intent.cardinality}"
        )

    print("\n=== POSSESSION PIPELINE ===")
    tmpdir = tempfile.mkdtemp()
    path = os.path.join(tmpdir, "ltm.json")
    store = MemoryStore(path=path, embedding_store=es, skip_embeddings=True)
    store.load(ensure_embeddings=False)
    mgr = SafeMemoryManager(store)
    ret = MemoryRetriever(store, es, intent_detector=det)

    mgr.process_user_message("I have an Oppo A53.")
    intent = det._cue_only_fallback("what phone do i have?")
    intent.entity = resolve_query_entity("what phone do i have?")
    mems = ret.retrieve_relevant_memories("What phone do I have?", intent=intent)
    print("memories:", len(mems), [m.get("content") for m in mems])
    print("answer:", grounded_answer(intent, mems, question="What phone do I have?"))

    print("\n=== EXTRACTION SAMPLES ===")
    ex = MemoryExtractor()
    for msg in [
        "My favourite colour is black.",
        "My mom's favourite colour is blue.",
        "I want to prepare for my mid exam.",
        "I am learning Python.",
        "I want to become no",
        "I like Python and Java.",
    ]:
        ops = ex.extract(msg)
        print(msg, "->", [(o.get("category"), o.get("entity_name"), o.get("subject"), o.get("value")) for o in ops])


if __name__ == "__main__":
    main()
