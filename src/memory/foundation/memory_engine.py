"""Memory Intelligence Engine for ALLI v0.6.

Combines all memory subsystems into one coherent public API:

    engine.remember(message)   — write pipeline
    engine.recall(question)    — read pipeline

This is an integration layer. It does NOT reimplement:
  - Message understanding (Session 2)
  - Entity resolution (Session 2)
  - Fact extraction (Session 2)
  - Lifecycle operations (Session 3)
  - Persistent storage (Session 4)
  - Query understanding (Session 5)
  - Hybrid retrieval (Session 6)
  - Evidence grounding (Session 7)
  - Answer generation (Session 8)

Scope:
  - Orchestrate existing components
  - Provide clean public API
  - Maintain single source of truth
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .pipeline import WritePipeline, WriteResult
from .answer_engine import AnswerEngine, AnswerResult
from .hybrid_retriever import HybridRetriever
from .evidence_grounding import EvidenceGrounder
from .query_understanding import understand_query
from .adapter import SQLiteAdapter
from .sqlite_store import SQLiteStore
from .storage import MemoryStore as FoundationMemoryStore
from ..embeddings import EmbeddingStore
from ..store import MemoryStore as OldMemoryStore
from ..query_intent import QueryIntentDetector


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "qwen2.5:7b"
_DEFAULT_DB_NAME = "v06_memory_engine.sqlite"


# ---------------------------------------------------------------------------
# Remember Result
# ---------------------------------------------------------------------------

@dataclass
class RememberResult:
    """Result from engine.remember(message)."""

    message: str = ""
    status: str = ""  # "success", "question", "greeting", "no_op", "error"
    operation: str = ""  # "created", "superseded", "updated", "deactivated", "unchanged"
    memory_id: str = ""
    entity: str = ""
    fact_summary: str = ""
    lifecycle_state: str = ""
    content: str = ""

    # Full pipeline result (for diagnostics)
    write_result: Optional[WriteResult] = None
    diagnostics: dict = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.status == "success"


# ---------------------------------------------------------------------------
# Memory Engine
# ---------------------------------------------------------------------------

class MemoryEngine:
    """Unified memory intelligence with remember() and recall() APIs.

    Usage:
        engine = MemoryEngine(db_path="/path/to/sqlite")
        engine.remember("I use a Samsung S24.")
        answer = engine.recall("What phone do I use?")
        print(answer)  # "You use a Samsung S24 phone."

    The engine orchestrates:
      - Write: understanding → extraction → lifecycle → persistence
      - Read:  query understanding → hybrid retrieval → evidence grounding → answer
    """

    def __init__(
        self,
        *,
        db_path: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        allow_qwen: bool = True,
    ):
        """
        Args:
            db_path: Path to SQLite database. If None, uses default location.
            model: Ollama model name for answer generation.
            allow_qwen: Whether to call Qwen for answer generation.
        """
        # Storage
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..", "..", "..", "memory",
                _DEFAULT_DB_NAME,
            )
            db_path = os.path.normpath(db_path)

        self._db_path = db_path
        self._sqlite_store = SQLiteStore(db_path)
        self._sqlite_adapter = SQLiteAdapter(self._sqlite_store)
        self._sqlite_adapter.load()

        # Old-format store for backward compatibility with retriever
        self._old_store = OldMemoryStore(skip_embeddings=True)
        self._sync_to_old_store()

        # Shared Ollama client (auto-discovered endpoint)
        from ..ollama_connection import create_ollama_client
        self._ollama_client, self._ollama_endpoint = create_ollama_client()

        # Embeddings
        self._embedding_store = EmbeddingStore(client=self._ollama_client)

        # Intent detector for general knowledge routing
        self._intent_detector = QueryIntentDetector(self._embedding_store)

        # Write pipeline
        self._write_pipeline = WritePipeline(self._sqlite_adapter)

        # Retrieval
        self._hybrid_retriever = HybridRetriever(
            store=self._old_store,
            embedding_store=self._embedding_store,
        )

        # Answer engine
        self._answer_engine = AnswerEngine(
            hybrid_retriever=self._hybrid_retriever,
            embedding_store=self._embedding_store,
            model=model,
            allow_qwen=allow_qwen,
            ollama_client=self._ollama_client,
        )

        # Evidence grounder (for direct access if needed)
        self._grounder = EvidenceGrounder()

    def _sync_to_old_store(self) -> None:
        """Sync foundation memories to old-format store for retriever."""
        foundation_memories = self._sqlite_adapter.load()
        self._old_store.memories = []
        for mem in foundation_memories:
            state = mem.lifecycle.state
            if hasattr(state, "value"):
                state = state.value
            self._old_store.memories.append({
                "id": mem.id,
                "content": mem.fact.content,
                "entity": {"name": mem.entity.name},
                "fact_type": mem.fact.fact_type,
                "category": mem.domain,
                "status": state,
                "value": mem.fact.value,
                "subject": mem.fact.subject,
            })

    # -------------------------------------------------------------------
    # Public API: remember()
    # -------------------------------------------------------------------

    def remember(
        self,
        message: str,
        *,
        diagnostics: bool = False,
    ) -> RememberResult:
        """Process a user message through the complete write pipeline.

        Args:
            message: Natural language fact statement from user
            diagnostics: If True, include full pipeline trace

        Returns:
            RememberResult with operation details
        """
        result = RememberResult(message=message)

        if not message or not message.strip():
            result.status = "no_op"
            return result

        try:
            write_result = self._write_pipeline.process(message)
            result.write_result = write_result
            result.status = write_result.status

            if write_result.status in ("question", "greeting", "no_op"):
                return result

            if write_result.memories:
                mem = write_result.memories[0]
                result.memory_id = mem.id
                result.entity = mem.entity.name
                result.fact_summary = mem.fact.content
                result.content = mem.fact.content
                result.lifecycle_state = mem.lifecycle.state.value if hasattr(mem.lifecycle.state, "value") else str(mem.lifecycle.state)

            if write_result.results:
                result.operation = write_result.results[0].action_taken

            # Sync to old store so retriever can find new memories
            self._sync_to_old_store()

        except Exception as e:
            result.status = "error"
            result.diagnostics["error"] = str(e)

        return result

    # -------------------------------------------------------------------
    # Public API: recall()
    # -------------------------------------------------------------------

    def recall(
        self,
        question: str,
        *,
        diagnostics: bool = False,
    ) -> str:
        """Answer a question using the complete read pipeline.

        Args:
            question: Natural language question from user
            diagnostics: If True, could be extended to return AnswerResult

        Returns:
            Natural language answer string
        """
        if not question or not question.strip():
            return "Please type a question."

        # Check intent for general knowledge routing
        try:
            intent = self._intent_detector.detect(question)
            if intent.is_general_knowledge or not intent.is_memory_read:
                # Route to general knowledge via Qwen
                from .. import CHAT_MODEL
                try:
                    messages = [
                        {"role": "system", "content": "You are ALLI, a helpful local personal AI assistant. Always reply in English. Do not invent personal facts about the user."},
                        {"role": "user", "content": question}
                    ]
                    response = self._embedding_store._client.chat(model=CHAT_MODEL, messages=messages)
                    return response["message"]["content"].strip()
                except Exception as e:
                    return f"I couldn't contact the language model: {e}"
        except Exception:
            # If intent detection fails, fall through to memory pipeline
            pass

        try:
            result = self._answer_engine.answer(question, diagnostics=diagnostics)
            return result.answer
        except Exception as e:
            return f"I had trouble processing your question: {e}"

    def recall_detailed(
        self,
        question: str,
    ) -> AnswerResult:
        """Answer a question with full pipeline details.

        Args:
            question: Natural language question from user

        Returns:
            AnswerResult with answer and full pipeline metadata
        """
        if not question or not question.strip():
            return AnswerResult(
                question=question,
                answer="Please type a question.",
                source="empty",
            )

        return self._answer_engine.answer(question, diagnostics=True)

    # -------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources."""
        self._sqlite_adapter.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
