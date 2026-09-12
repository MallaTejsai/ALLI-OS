"""Write pipeline — orchestrates the write path from user message to storage.

This is the main entry point for the write pipeline. It chains:
1. Message understanding
2. Entity resolution
3. Fact extraction
4. Validation
5. Operation application
6. Memory event production
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .understanding import understand_message, MessageUnderstanding
from .entity_resolution import resolve_entity, ResolvedEntity
from .extraction import PatternExtractor, ExtractionResult, MemoryOperation
from .operations import apply_operation, OperationResult
from .validation import validate_memory, ValidationResult
from .models import Memory, MemoryEvent
from .storage import MemoryStore
from .ids import generate_episode_id


# Garbage content patterns — rejects meaningless input
_GARBAGE_PATTERN = re.compile(r"^[^a-zA-Z0-9\s]+$")
_NUMERIC_ONLY = re.compile(r"^\d+$")
_VAGUE_VALUES = frozenset({
    "nothing", "something", "anything", "everything", "none",
    "ok", "okay", "yes", "no", "maybe", "sure",
})


def _is_garbage_content(content: str) -> bool:
    """Check if content is garbage, meaningless, or too short to be a valid fact."""
    if not content or not isinstance(content, str):
        return True
    v = content.strip()
    if len(v) < 2:
        return True
    if _GARBAGE_PATTERN.match(v):
        return True
    if _NUMERIC_ONLY.match(v):
        return True
    if v.lower() in _VAGUE_VALUES:
        return True
    lower = v.lower()
    # Reject content that is clearly a question
    if "?" in v:
        return True
    _QUESTION_START = re.compile(
        r"^(?:what|who|where|when|why|how|which"
        r"|do|did|can|could|would|should"
        r"|is|am|are|was|were"
        r"|tell|give|show)\b"
    )
    if _QUESTION_START.match(lower):
        return True
    # Reject content with embedded question structures
    # e.g., "i phone do i use now" contains "do i use"
    if re.search(r"\b(?:do|did|does|is|are|was|were|can|could|would|should)\s+(?:i|you|he|she|we|they)\b", lower):
        return True
    return False


@dataclass
class WriteResult:
    """Result of processing a user message through the write pipeline."""

    status: str  # "success", "partial", "rejected", "no_op", "question"
    operations: list[MemoryOperation]
    results: list[OperationResult]
    events: list[MemoryEvent]
    memories: list[Memory]
    entity: Optional[ResolvedEntity] = None
    understanding: Optional[MessageUnderstanding] = None
    extraction: Optional[ExtractionResult] = None
    validation_errors: list[str] = field(default_factory=list)
    debug_trace: dict = field(default_factory=dict)


class WritePipeline:
    """Orchestrates the write path from user message to storage.

    Usage:
        pipeline = WritePipeline(store)
        result = pipeline.process("I am learning Python")
    """

    def __init__(
        self,
        store: MemoryStore,
        extractor: Optional[PatternExtractor] = None,
        *,
        old_store: Optional[object] = None,
    ) -> None:
        self._store = store
        self._extractor = extractor or PatternExtractor()
        self._old_store = old_store

    def process(self, message: str) -> WriteResult:
        """Process a user message through the full write pipeline."""
        debug_trace: dict = {"input": message}
        events: list[MemoryEvent] = []
        results: list[OperationResult] = []
        memories: list[Memory] = []
        validation_errors: list[str] = []

        # --- Stage 1: Message Understanding ---
        understanding = understand_message(message)
        debug_trace["understanding"] = {
            "message_type": understanding.message_type.value,
            "is_temporary": understanding.is_temporary,
            "is_uncertain": understanding.is_uncertain,
            "is_third_party": understanding.is_third_party,
            "is_correction": understanding.is_correction,
            "is_deletion": understanding.is_deletion,
        }

        # Questions are not written to memory
        if understanding.message_type.value == "question":
            return WriteResult(
                status="question",
                operations=[],
                results=[],
                events=[],
                memories=[],
                understanding=understanding,
                debug_trace=debug_trace,
            )

        # Greetings are not written to memory
        if understanding.message_type.value == "greeting":
            return WriteResult(
                status="no_op",
                operations=[],
                results=[],
                events=[],
                memories=[],
                understanding=understanding,
                debug_trace=debug_trace,
            )

        # --- Stage 2: Entity Resolution ---
        resolved = resolve_entity(message)
        debug_trace["entity_resolution"] = {
            "entity": resolved.entity.name,
            "entity_type": resolved.entity.entity_type,
            "confidence": resolved.confidence,
            "method": resolved.resolution_method,
        }

        # --- Stage 3: Fact Extraction ---
        extraction = self._extractor.extract(message, understanding)
        debug_trace["extraction"] = {
            "operation_count": len(extraction.operations),
            "operations": [
                {
                    "action": op.action,
                    "domain": op.domain,
                    "fact_type": op.fact_type,
                    "subject": op.subject,
                    "value": op.value,
                    "polarity": op.polarity,
                    "confidence": op.confidence,
                }
                for op in extraction.operations
            ],
        }

        if not extraction.operations:
            return WriteResult(
                status="no_op",
                operations=[],
                results=[],
                events=[],
                memories=[],
                entity=resolved,
                understanding=understanding,
                extraction=extraction,
                debug_trace=debug_trace,
            )

        # --- Stage 4: Apply Operations ---
        for op in extraction.operations:
            # Skip operations with empty or garbage content
            if not op.content or not op.content.strip():
                continue
            if _is_garbage_content(op.content):
                continue

            # Override entity from resolution
            op.entity_name = resolved.entity.name
            op.entity_type = resolved.entity.entity_type

            # Apply to store
            op_result = apply_operation(
                op,
                self._store,
                entity_name=resolved.entity.name,
                entity_type=resolved.entity.entity_type,
            )

            if op_result:
                results.append(op_result)
                events.append(op_result.event)
                if op_result.memory:
                    memories.append(op_result.memory)

                # Validate the created memory
                if op_result.memory:
                    v = validate_memory(op_result.memory)
                    if not v.is_valid:
                        validation_errors.extend(v.errors)

        debug_trace["persistence"] = {
            "results_count": len(results),
            "actions": [r.action_taken for r in results],
            "memory_ids": [r.memory_id for r in results],
        }
        debug_trace["validation"] = {
            "errors": validation_errors,
            "passed": len(validation_errors) == 0,
        }

        # Determine overall status
        if not results:
            status = "no_op"
        elif validation_errors:
            status = "partial"
        elif any(r.action_taken == "unchanged" for r in results):
            status = "no_op" if all(r.action_taken == "unchanged" for r in results) else "success"
        else:
            status = "success"

        # Sync SQLite state to old store so retrieval sees updated lifecycle
        if status == "success" and self._old_store is not None:
            from .adapter import sync_sqlite_to_old_store
            sync_sqlite_to_old_store(self._store, self._old_store)

        return WriteResult(
            status=status,
            operations=extraction.operations,
            results=results,
            events=events,
            memories=memories,
            entity=resolved,
            understanding=understanding,
            extraction=extraction,
            validation_errors=validation_errors,
            debug_trace=debug_trace,
        )
