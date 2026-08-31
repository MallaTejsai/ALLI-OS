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
from ollama import chat

ollama_client = ollama.Client(host=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))

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
from memory.answers import grounded_answer, llm_memory_system_prompt
from memory.extractor import looks_like_question
from memory.query_intent import QueryIntentDetector


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


def build_memory_llm_messages(
    question: str,
    relevant_memories: list,
) -> list:
    messages = [{"role": "system", "content": llm_memory_system_prompt()}]
    if relevant_memories:
        memory_lines = [
            (
                f"- category={memory.get('category')}, "
                f"fact_type={memory.get('fact_type')}, "
                f"value={memory.get('value')}, "
                f"content={memory.get('content')}"
            )
            for memory in relevant_memories
        ]
        messages.append({
            "role": "system",
            "content": (
                "Answer only from these supplied memories:\n"
                + "\n".join(memory_lines)
            ),
        })
    else:
        messages.append({
            "role": "system",
            "content": (
                "No relevant memories were supplied. "
                "Say that the requested personal information is not stored."
            ),
        })
    messages.append({"role": "user", "content": question})
    return messages


def verify_chat_model(model_name: str) -> None:
    try:
        listed = __import__("ollama").list()
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

    embedding_store = EmbeddingStore(client=ollama_client)
    embeddings_ok = True
    try:
        embedding_store.verify_model()
    except EmbeddingError as error:
        embeddings_ok = False
        print(f"[Warning] {error}")

    store = MemoryStore(
        path=config.LONG_TERM_MEMORY_FILE,
        embedding_store=embedding_store,
    )
    store.load(ensure_embeddings=embeddings_ok)

    manager = SafeMemoryManager(store)
    intent_detector = QueryIntentDetector(embedding_store)
    retriever = MemoryRetriever(
        store,
        embedding_store,
        intent_detector=intent_detector,
    )

    conversation = load_json_list(config.CONVERSATION_FILE)

    print("=" * 40)
    print("           ALLI v0.6")
    print("=" * 40)
    if not embeddings_ok:
        print(
            "Memory embeddings are unavailable because the embedding model "
            "is not installed/running."
        )

    try:
        verify_chat_model(CHAT_MODEL)
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
        # MEMORY WRITE PIPELINE (statements only)
        # ==========================================
        write_results = []
        if not looks_like_question(question):
            try:
                write_results = manager.process_user_message(question)
                for result in write_results:
                    status = result.get("status")
                    memory_id = result.get("memory_id", "")
                    if status == "added":
                        print(f"[Memory Added] {memory_id}")
                    elif status == "updated":
                        if result.get("conflict_resolved"):
                            print(f"[Memory Conflict Resolved] {memory_id}")
                        else:
                            print(f"[Memory Updated] {memory_id}")
                    elif status == "deleted":
                        print(f"[Memory Deleted] {memory_id}")
                    elif status == "unchanged":
                        print(f"[Memory Unchanged] {memory_id}")
            except EmbeddingError as error:
                print(f"[Warning] Memory update skipped: {error}")
            except Exception as error:
                print(f"[Memory Manager Error] {error}")

        # ==========================================
        # MEMORY READ vs GENERAL CHAT
        # ==========================================
        intent = retriever.detect_intent(question)

        if intent.is_memory_read:
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
            answer_source = "grounded_memories"

            if memory_debug_enabled():
                debug = retriever.last_debug or {}
                print(
                    f"[Memory Search] {len(debug.get('candidates', []))} candidates"
                )
                for item in debug.get("selected", []):
                    print(f"[Memory Selected] {item.get('id')}")
                print_memory_debug(retriever, intent, answer_source)

        elif write_results:
            # After storing a fact, give a short grounded acknowledgement
            # without inventing extra personal details.
            memory = write_results[0].get("memory") or {}
            status = write_results[0].get("status")
            content = memory.get("content", "")
            if status == "unchanged":
                answer = "I already have that stored."
            elif status == "updated":
                answer = f"Got it — I updated that memory: {content}"
            elif status == "added":
                answer = f"Got it — I'll remember that: {content}"
            else:
                answer = "Okay."
            answer_source = "memory_write_ack"

        else:
            # Normal knowledge / conversation — do not force memory retrieval.
            messages = build_general_chat_messages(conversation, question)
            print("\nALLI: Thinking...")
            try:
                response = chat(model=CHAT_MODEL, messages=messages)
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


if __name__ == "__main__":
    main()
