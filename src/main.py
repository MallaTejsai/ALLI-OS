"""
ALLI v0.6 — Personal AI assistant with complete memory management.

Architecture:
  WRITE: fact statement → extract → duplicate/contradiction → add/update/unchanged
  READ:  personal question → intent → semantic+category retrieval → grounded answer
  CHAT:  general knowledge → Ollama without forced personal memories

Python owns memory decisions. Embeddings search. Qwen speaks when needed.
"""

from __future__ import annotations

import os
import sys

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import ollama

from memory.ollama_connection import create_ollama_client, check_ollama_health

ollama_client, ollama_endpoint = create_ollama_client()

from memory import (
    CHAT_MODEL,
    EmbeddingError,
    EmbeddingStore,
    MemoryRetriever,
    MemoryStore,
    SafeMemoryManager,
    load_json_list,
    save_json,
)
from memory import config
from memory.answers import grounded_answer
from memory.extractor import looks_like_question
from memory.foundation.understanding import classify_message_type, MessageType
from memory.query_intent import QueryIntentDetector
from memory.foundation.pipeline import WritePipeline
from memory.foundation.adapter import PersistentStore, SQLiteAdapter, sync_sqlite_to_old_store
from memory.foundation.sqlite_store import SQLiteStore
from memory.foundation.query_understanding import understand_query
from memory.foundation.hybrid_retriever import HybridRetriever
from memory.foundation.answer_engine import AnswerEngine


STOP_WORDS = {
    "about", "also", "and", "are", "can", "could", "does", "for", "from",
    "have", "how", "into", "is", "learn", "learning", "like", "more", "my",
    "need", "should", "that", "the", "their", "they", "this", "user", "want",
    "what", "when", "where", "which", "with", "would", "you", "your",
}


def meaningful_words(text: str) -> set[str]:
    import re

    words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    return {
        word
        for word in words
        if len(word) >= 3 and word not in STOP_WORDS
    }


def is_follow_up_question(current_question: str, previous_user_message: str) -> bool:
    current_lower = current_question.lower().strip()
    follow_up_starts = (
        "and ",
        "also ",
        "what about",
        "why ",
        "how ",
        "can you explain",
        "tell me more",
        "give me an example",
        "what do you mean",
        "then ",
    )
    if current_lower.startswith(follow_up_starts):
        return True
    return bool(
        meaningful_words(current_question)
        & meaningful_words(previous_user_message)
    )


def select_conversation_context(conversation: list) -> list:
    chat_messages = [
        message
        for message in conversation
        if (
            message.get("role") in {"user", "assistant"}
            and isinstance(message.get("content"), str)
        )
    ]
    if not chat_messages:
        return []

    current_question = chat_messages[-1]
    previous_user_messages = [
        message
        for message in chat_messages[:-1]
        if message["role"] == "user"
    ]
    if not previous_user_messages:
        return [current_question]

    previous_question = previous_user_messages[-1]["content"]
    if not is_follow_up_question(current_question["content"], previous_question):
        return [current_question]

    return chat_messages[-config.RECENT_MESSAGE_LIMIT :]


GENERAL_SYSTEM_PROMPT = (
    "You are ALLI, a helpful local personal AI assistant. "
    "Always reply in English. "
    "Do not invent personal facts about the user."
)


def build_general_chat_messages(conversation: list, question: str) -> list:
    recent_messages = select_conversation_context(conversation)
    messages = [{"role": "system", "content": GENERAL_SYSTEM_PROMPT}]
    if recent_messages:
        messages.extend(recent_messages)
    else:
        messages.append({"role": "user", "content": question})
    return messages


def verify_chat_model(client: ollama.Client, model_name: str) -> None:
    try:
        listed = client.list()
    except Exception as error:
        raise RuntimeError(f"Ollama is unavailable: {error}") from error

    names = set()
    models = getattr(listed, "models", None)
    if models is None and isinstance(listed, dict):
        models = listed.get("models", [])
    for item in models or []:
        name = getattr(item, "model", None) or getattr(item, "name", None)
        if name is None and isinstance(item, dict):
            name = item.get("model") or item.get("name")
        if name:
            names.add(str(name))
            names.add(str(name).split(":")[0])

    short = model_name.split(":")[0]
    if model_name not in names and short not in names:
        raise RuntimeError(
            f"Chat model '{model_name}' is not installed.\n"
            f"Install it with: ollama pull {model_name}"
        )


def memory_debug_enabled() -> bool:
    env = os.environ.get("ALLI_MEMORY_DEBUG", "").strip().lower()
    return config.MEMORY_DEBUG or env in {"1", "true", "yes", "on"}


def print_memory_debug(retriever: MemoryRetriever, intent, answer_source: str) -> None:
    debug = retriever.last_debug or {}
    print("\n[Memory Debug]")
    print(f"  Query: {debug.get('query')}")
    print("  → query embedding generated")
    print(
        f"  → detected semantic category/fact type: "
        f"{intent.primary} (score={intent.score:.3f})"
    )
    top_scores = sorted(
        (intent.scores or {}).items(),
        key=lambda item: item[1],
        reverse=True,
    )[:5]
    print(f"  → intent scores: {top_scores}")
    # Phase 5: entity and cardinality diagnostics
    entity = intent.entity
    if entity:
        entity_desc = f"{entity.name}" if entity.is_resolved else "unresolved"
        if entity.relation_to_user:
            entity_desc += f" ({entity.relation_to_user})"
        print(f"  → entity: {entity_desc} (type={entity.entity_type})")
    else:
        print(f"  → entity: none")
    print(f"  → cardinality: {intent.cardinality}")
    print(f"  → is_personal: {intent.is_personal}")
    # Filter diagnostics
    filter_diag = debug.get("filter_diagnostics", {})
    if filter_diag:
        print(f"  → filter: {filter_diag.get('total_active', '?')} active → {filter_diag.get('after_entity_filter', '?')} after entity filter")
    print(f"  → retrieved candidates: {len(debug.get('candidates', []))}")
    for item in debug.get("selected", []):
        print(
            f"     selected {item.get('id')}: "
            f"sim={item.get('similarity', 0):.3f} "
            f"final={item.get('final_score', 0):.3f} "
            f"| {item.get('content')}"
        )
    print(f"  → final answer source: {answer_source}")


def main() -> None:
    os.makedirs(config.MEMORY_DIR, exist_ok=True)
    os.makedirs(config.BACKUP_DIR, exist_ok=True)

    print("=" * 40)
    print("           ALLI v0.6")
    print("=" * 40)

    # Health check with retry for startup delays
    health_ok, health_msg = check_ollama_health(ollama_client, ollama_endpoint)
    if not health_ok:
        print(f"\n{health_msg}")
        return

    embedding_store = EmbeddingStore(client=ollama_client)
    embeddings_ok = True
    try:
        embedding_store.verify_model()
    except EmbeddingError as error:
        embeddings_ok = False
        print(f"[Warning] {error}")

    # Old JSON store for backward compatibility and retrieval
    store = MemoryStore(
        path=config.LONG_TERM_MEMORY_FILE,
        embedding_store=embedding_store,
    )
    store.load(ensure_embeddings=embeddings_ok)

    # New v0.6 SQLite persistent storage
    sqlite_db_path = os.path.join(config.MEMORY_DIR, "v06_dev.sqlite")
    sqlite_store = SQLiteStore(sqlite_db_path)
    sqlite_adapter = SQLiteAdapter(sqlite_store)
    sqlite_adapter.load()

    # Keep old adapter for compatibility (not used for writes)
    persistent_store = PersistentStore(store)
    persistent_store.load()

    # Sync SQLite memories into old store so retriever can find them
    sync_sqlite_to_old_store(sqlite_adapter, store)

    # Keep old manager for retrieval compatibility (not used for writes)
    manager = SafeMemoryManager(store)
    intent_detector = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(
        store,
        embedding_store,
        intent_detector=intent_detector,
    )

    # New v0.6 HybridRetriever — consumes StructuredQuery from Session 5
    hybrid_retriever = HybridRetriever(store, embedding_store)

    # New v0.6 AnswerEngine — orchestrates S5→S6→S7→Qwen pipeline
    answer_engine = AnswerEngine(
        hybrid_retriever=hybrid_retriever,
        embedding_store=embedding_store,
        ollama_client=ollama_client,
    )

    conversation = load_json_list(config.CONVERSATION_FILE)

    # New v0.6 WritePipeline — uses SQLite storage, syncs old store after writes
    write_pipeline = WritePipeline(sqlite_adapter, old_store=store)

    if not embeddings_ok:
        print(
            "Memory embeddings are unavailable because the embedding model "
            "is not installed/running."
        )

    try:
        verify_chat_model(ollama_client, CHAT_MODEL)
    except RuntimeError as error:
        print(f"\nALLI: {error}")
        return

    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nALLI: Goodbye!")
            break

        if not question:
            print("\nALLI: Please type a message.")
            continue

        conversation.append({"role": "user", "content": question})
        answer = None
        answer_source = "ollama"
        intent = None

        # ==========================================
        # MEMORY WRITE PIPELINE (new v0.6 foundation)
        # ==========================================
        write_result = None
        msg_type = classify_message_type(question)
        if not looks_like_question(question) and msg_type != MessageType.QUESTION:
            try:
                write_result = write_pipeline.process(question)
                for r in write_result.results:
                    action = r.action_taken
                    memory_id = r.memory_id
                    if action == "created":
                        print(f"[Memory Added] {memory_id}")
                    elif action == "superseded":
                        print(f"[Memory Superseded] {memory_id}")
                    elif action == "updated":
                        print(f"[Memory Updated] {memory_id}")
                    elif action == "deactivated":
                        print(f"[Memory Deactivated] {memory_id}")
                    elif action == "invalidated":
                        print(f"[Memory Invalidated] {memory_id}")
                    elif action == "unchanged":
                        print(f"[Memory Unchanged] {memory_id}")
            except EmbeddingError as error:
                print(f"[Warning] Memory update skipped: {error}")
            except Exception as error:
                print(f"[Memory Pipeline Error] {error}")

        # ==========================================
        # MEMORY READ vs GENERAL CHAT
        # ==========================================
        intent = retriever.detect_intent(question)

        # A message that (a) produced memory writes and (b) is not a question
        # is a statement the user just taught us. Acknowledge the write — do
        # NOT route it through the read pipeline (the intent detector can
        # misclassify negation statements like "I don't want to learn Rust
        # anymore." as memory reads, which would answer with a confusing
        # "I don't have that information stored").
        is_question_like = (
            msg_type == MessageType.QUESTION or looks_like_question(question)
        )
        if write_result and write_result.memories and not is_question_like:
            memory = write_result.memories[0]
            status = write_result.status
            content = memory.content
            if status == "no_op":
                answer = "I already have that stored."
            elif status == "success":
                action = write_result.results[0].action_taken if write_result.results else "created"
                if action == "superseded":
                    answer = f"Got it — I updated that memory: {content}"
                elif action in ("deactivated", "invalidated"):
                    answer = "Okay, I'll forget that."
                else:
                    answer = f"Got it — I'll remember that: {content}"
            else:
                answer = "Okay."
            answer_source = "memory_write_ack"

        # Vague/incomplete statements need clarification, not an LLM response.
        elif intent.needs_clarification:
            answer = intent.clarification_message
            answer_source = "clarification"
        # General knowledge / advice / vague statements never touch memory.
        elif intent.is_general_knowledge or not intent.is_memory_read:
            messages = build_general_chat_messages(conversation, question)
            print("\nALLI: Thinking...")
            try:
                response = ollama_client.chat(model=CHAT_MODEL, messages=messages)
                answer = response["message"]["content"]
                answer_source = "ollama_general"
            except Exception as error:
                print(f"\nALLI: I could not contact Ollama. {error}")
                save_json(config.CONVERSATION_FILE, conversation)
                continue
        elif intent.is_memory_read:
            # Use new v0.6 AnswerEngine for personal memory questions
            print("\nALLI: Thinking...")
            try:
                result = answer_engine.answer(question)
                answer = result.answer
                answer_source = f"answer_engine_{result.source}"

                if memory_debug_enabled() and result.evidence_assessment:
                    ea = result.evidence_assessment
                    print(f"[Evidence] status={ea.status} accepted={len(ea.accepted)} rejected={len(ea.rejected)}")
                    if result.called_qwen:
                        print("[Evidence] Qwen called with validated evidence only")
            except Exception as error:
                # Fallback to old path if AnswerEngine fails
                print(f"[AnswerEngine Error] {error}, using fallback")
                if intent.is_multi:
                    grouped = retriever.retrieve_by_requested_intents(
                        question,
                        intent=intent,
                    )
                    relevant = []
                    for items in grouped.values():
                        relevant.extend(items)
                    answer = grounded_answer(
                        intent,
                        relevant,
                        question=question,
                        grouped=grouped,
                    )
                else:
                    relevant = retriever.retrieve_relevant_memories(
                        question,
                        intent=intent,
                    )
                    answer = grounded_answer(intent, relevant, question=question)
                answer_source = "grounded_memories_fallback"

        else:
            # Normal knowledge / conversation — do not force memory retrieval.
            messages = build_general_chat_messages(conversation, question)
            print("\nALLI: Thinking...")
            try:
                response = ollama_client.chat(model=CHAT_MODEL, messages=messages)
                answer = response["message"]["content"]
                answer_source = "ollama_general"
            except Exception as error:
                print(f"\nALLI: I could not contact Ollama. {error}")
                save_json(config.CONVERSATION_FILE, conversation)
                continue

        conversation.append({"role": "assistant", "content": answer})
        save_json(config.CONVERSATION_FILE, conversation)

        print("\nALLI:")
        print(answer)

    # Cleanup SQLite connection
    sqlite_adapter.close()


if __name__ == "__main__":
    main()
