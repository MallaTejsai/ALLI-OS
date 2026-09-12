"""Semantic + category-aware memory retrieval (ALLI v0.6)."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .embeddings import EmbeddingError, EmbeddingStore, cosine_similarity
from .query_intent import INTENT_FILTERS, QueryIntent, QueryIntentDetector
from .store import MemoryStore


@dataclass
class RetrievalResult:
    """Structured result from memory retrieval."""
    status: str = "ok"  # "ok", "not_found", "ambiguous", "unresolved_entity", "not_personal"
    memories: list[dict] = field(default_factory=list)
    selected: list[dict] = field(default_factory=list)
    intent: Optional[QueryIntent] = None
    grouped: Optional[dict[str, list[dict]]] = None
    debug: Optional[dict] = None

    @property
    def has_results(self) -> bool:
        """True if the result contains memories."""
        if self.selected:
            return True
        return len(self.memories) > 0

    @property
    def is_ambiguous(self) -> bool:
        """True if the result is ambiguous (multiple conflicting candidates)."""
        return self.status == "ambiguous"


class MemoryRetriever:
    """
    Rank active long-term memories using:
      1) query ↔ memory embedding similarity
      2) semantic query-intent category/fact_type filters
      3) preference subject relevance gates
      4) multi-category intents for compound questions

    Inactive/historical memories are never returned as current facts.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedding_store: Optional[EmbeddingStore] = None,
        intent_detector: Optional[QueryIntentDetector] = None,
        *,
        threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ):
        self.store = store
        self.embedding_store = embedding_store or store.embedding_store
        self.intent_detector = intent_detector or QueryIntentDetector(
            self.embedding_store
        )
        self.threshold = (
            threshold
            if threshold is not None
            else config.MEMORY_SIMILARITY_THRESHOLD
        )
        self.max_results = max_results or config.MAX_RETRIEVED_MEMORIES
        self.last_debug: dict = {}

    def detect_intent(self, question: str) -> QueryIntent:
        return self.intent_detector.detect(question)

    def retrieve_relevant_memories(
        self,
        question: str,
        *,
        intent: Optional[QueryIntent] = None,
        threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> list[dict]:
        scored = self.retrieve_with_scores(
            question,
            intent=intent,
            threshold=threshold,
            max_results=max_results,
        )
        return [item["memory"] for item in scored]

    def retrieve_by_requested_intents(
        self,
        question: str,
        *,
        intent: Optional[QueryIntent] = None,
        threshold: Optional[float] = None,
    ) -> dict[str, list[dict]]:
        """
        Retrieve memories grouped by each requested intent.

        Used for compound questions such as "skills and goals".
        """
        intent = intent or self.intent_detector.detect(question)
        requested = intent.requested or ([intent.primary] if intent.is_memory_read else [])
        grouped: dict[str, list[dict]] = {}
        seen_ids: set[str] = set()

        for label in requested:
            filters = dict(INTENT_FILTERS.get(label, {}))
            sub_intent = QueryIntent(
                primary=label,
                score=intent.scores.get(label, intent.score),
                scores=intent.scores,
                is_memory_read=True,
                is_general_knowledge=False,
                preference_polarity=intent.preference_polarity,
                filters=filters,
                requested=[label],
                normalized_query=intent.normalized_query,
            )
            hits = self.retrieve_with_scores(
                question,
                intent=sub_intent,
                threshold=threshold,
            )
            memories = []
            for item in hits:
                memory = item["memory"]
                memory_id = memory.get("id")
                if memory_id in seen_ids and label != intent.primary:
                    # Allow same memory only once across sections unless needed.
                    pass
                memories.append(memory)
                if memory_id:
                    seen_ids.add(str(memory_id))
            grouped[label] = memories

        self.last_debug["requested"] = list(requested)
        self.last_debug["grouped_counts"] = {
            key: len(value) for key, value in grouped.items()
        }
        return grouped

    def retrieve_with_scores(
        self,
        question: str,
        *,
        intent: Optional[QueryIntent] = None,
        threshold: Optional[float] = None,
        max_results: Optional[int] = None,
    ) -> list[dict]:
        threshold = self.threshold if threshold is None else threshold
        max_results = self.max_results if max_results is None else max_results

        # Use cardinality to limit results for singular queries
        if intent and intent.cardinality == "singular":
            max_results = min(max_results, 3)

        intent = intent or self.intent_detector.detect(question)
        entity = getattr(intent, "entity", None)
        self.last_debug = {
            "query": question,
            "normalized_query": intent.normalized_query,
            "intent": intent.primary,
            "requested": list(intent.requested or []),
            "intent_score": intent.score,
            "intent_scores": dict(intent.scores),
            "is_memory_read": intent.is_memory_read,
            "is_general_knowledge": intent.is_general_knowledge,
            "entity": {
                "name": getattr(entity, "name", None) if entity else None,
                "is_resolved": getattr(entity, "is_resolved", False) if entity else False,
                "is_unresolved": getattr(entity, "is_unresolved", False) if entity else False,
            } if entity else None,
            "cardinality": getattr(intent, "cardinality", "unknown"),
            "is_personal": getattr(intent, "is_personal", False),
            "filter_diagnostics": {},
            "candidates": [],
            "selected": [],
        }

        active = self.store.active_memories()
        if not question.strip() or not active:
            return []

        if intent.is_general_knowledge or not intent.is_memory_read:
            return []

        # Entity filtering: apply BEFORE semantic ranking
        active = self._filter_by_entity(active, intent)
        entity = getattr(intent, "entity", None)
        if entity is not None:
            self.last_debug["filter_diagnostics"]["entity_filter"] = {
                "entity_name": getattr(entity, "name", None),
                "memories_before_filter": len(self.store.active_memories()),
                "memories_after_filter": len(active),
            }
        if not active:
            return []

        # An overview is intentionally broad: it summarizes active stored facts
        # rather than only facts semantically close to the query wording.
        if intent.primary == "personal_overview":
            selected = [
                {
                    "memory": memory,
                    "similarity": 1.0,
                    "category_match": True,
                    "final_score": 1.0,
                }
                for memory in active
            ]
            self.last_debug["selected"] = [
                {
                    "id": item["memory"].get("id"),
                    "content": item["memory"].get("content"),
                    "similarity": item["similarity"],
                    "final_score": item["final_score"],
                }
                for item in selected
            ]
            return selected

        embed_text = intent.normalized_query or question
        try:
            question_vec = self.embedding_store.generate_embedding(embed_text)
        except EmbeddingError as error:
            print(f"[Warning] {error}")
            return []

        if not question_vec:
            return []

        allowed_fact_types = self._allowed_fact_types(intent)
        filters = intent.filters or {}
        allowed_categories = filters.get("categories")

        exclude_fact_types = set(filters.get("exclude_fact_types") or [])
        strict = bool(filters.get("strict", False))

        results = []
        for memory in active:
            embedding = memory.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue

            category = str(memory.get("category", "")).lower()
            fact_type = str(memory.get("fact_type", "")).lower()

            category_match = True

            if allowed_categories is not None:
                category_match = category in allowed_categories

            if allowed_fact_types is not None:
                if intent.primary == "preference":
                    category_match = (
                        category_match
                        and category == "preference"
                        and fact_type in allowed_fact_types
                    )
                else:
                    category_match = (
                        category_match
                        and fact_type in allowed_fact_types
                    )

            if fact_type in exclude_fact_types:
                category_match = False

            if strict and not category_match:
                continue

            similarity = cosine_similarity(question_vec, embedding)
            subject_sim = None
            if (
                category == "preference"
                and memory.get("subject")
                and intent.preference_polarity is None
            ):
                try:
                    subject_vec = self.embedding_store.generate_embedding(
                        f"personal preference about {memory['subject']}"
                    )
                    subject_sim = cosine_similarity(question_vec, subject_vec)
                except EmbeddingError:
                    subject_sim = None

                if (
                    similarity < config.PREFERENCE_CONTENT_GATE
                    and (
                        subject_sim is None
                        or subject_sim < config.PREFERENCE_SUBJECT_GATE
                    )
                ):
                    continue

            if strict and category_match:
                if category in {"preference", "skill", "interest", "learning", "possession", "project", "habit", "task"}:
                    effective_threshold = threshold - 0.15
                else:
                    effective_threshold = 0.0
            elif category_match:
                effective_threshold = threshold
            else:
                effective_threshold = threshold + 0.12

            category_score = 1.0 if category_match else 0.0
            final_score = (
                max(0.0, min(1.0, similarity)) * config.SEMANTIC_WEIGHT
                + category_score * config.CATEGORY_WEIGHT
            )

            candidate = {
                "id": memory.get("id"),
                "content": memory.get("content"),
                "category": category,
                "fact_type": fact_type,
                "similarity": similarity,
                "subject_similarity": subject_sim,
                "category_match": category_match,
                "final_score": final_score,
            }
            self.last_debug["candidates"].append(candidate)

            if similarity < effective_threshold:
                continue

            results.append({
                "memory": memory,
                "similarity": similarity,
                "category_match": category_match,
                "final_score": final_score,
            })

        # Fallback: if no results for a strict category, search a related
        # category without merging stored categories.  The primary category
        # stays pure — we only broaden at retrieval time.
        # NOTE: learning→education fallback is intentionally omitted.
        # Education and learning must remain semantically separate per spec.
        if not results and strict and allowed_categories:
            fallback_map = {
                "education": "learning",
                "ambition": "goal",
            }
            for primary_cat in allowed_categories:
                fallback_cat = fallback_map.get(primary_cat)
                if fallback_cat and fallback_cat not in allowed_categories:
                    fallback_results = []
                    for memory in active:
                        embedding = memory.get("embedding")
                        if not isinstance(embedding, list) or not embedding:
                            continue
                        category = str(memory.get("category", "")).lower()
                        if category != fallback_cat:
                            continue
                        similarity = cosine_similarity(question_vec, embedding)
                        # Use 0.0 threshold for fallback — category filter
                        # already constrains the results.
                        fallback_results.append({
                            "memory": memory,
                            "similarity": similarity,
                            "category_match": True,
                            "final_score": similarity * config.SEMANTIC_WEIGHT + config.CATEGORY_WEIGHT,
                        })
                    if fallback_results:
                        results = fallback_results
                        break

        results.sort(
            key=lambda item: (item["final_score"], item["similarity"]),
            reverse=True,
        )
        selected = results[:max_results]
        self.last_debug["selected"] = [
            {
                "id": item["memory"].get("id"),
                "content": item["memory"].get("content"),
                "similarity": item["similarity"],
                "final_score": item["final_score"],
            }
            for item in selected
        ]
        return selected

    def _allowed_fact_types(self, intent: QueryIntent) -> Optional[set[str]]:
        filters = intent.filters or {}
        base = filters.get("fact_types")
        if intent.primary in {"name", "nickname"}:
            secondary = self.intent_detector.secondary_identity_intents(intent)
            return set(secondary)
        if intent.primary == "preference":
            if intent.preference_polarity == "like":
                return {"like"}

            if intent.preference_polarity == "dislike":
                return {"dislike"}

            return {"like", "dislike", "preference"}
        if base is None:
            return None
        return set(base)

    def _filter_by_entity(
        self, memories: list[dict], intent: QueryIntent
    ) -> list[dict]:
        """Filter memories by entity scope BEFORE semantic ranking.

        - If intent has a resolved entity, only return memories matching that entity.
        - If entity is unresolved, return empty (we can't determine scope).
        - If entity is None (no entity detected), return all memories.
        """
        entity = getattr(intent, "entity", None)
        if entity is None:
            return memories

        # Unresolved entity: block all results
        if getattr(entity, "is_unresolved", False):
            return []

        entity_name = getattr(entity, "name", None)
        if not entity_name or entity_name == "self":
            entity_name = "user"

        filtered = []
        for memory in memories:
            mem_entity = memory.get("entity", {})
            if isinstance(mem_entity, dict):
                mem_entity_name = mem_entity.get("name", "user")
            else:
                mem_entity_name = "user"
            # Normalize "self" to "user" for consistent comparison
            if mem_entity_name == "self":
                mem_entity_name = "user"
            if mem_entity_name == entity_name:
                filtered.append(memory)
        return filtered

    def retrieve_structured(
        self, query: str, *, intent: Optional[QueryIntent] = None
    ) -> RetrievalResult:
        """Retrieve memories and return a structured RetrievalResult."""
        if not query or not query.strip():
            return RetrievalResult(status="not_found", memories=[])

        intent = intent or self.detect_intent(query)

        if not intent.is_memory_read:
            return RetrievalResult(status="not_personal", memories=[], intent=intent)

        entity = getattr(intent, "entity", None)
        if entity is not None and getattr(entity, "is_unresolved", False):
            return RetrievalResult(status="unresolved_entity", memories=[], intent=intent)

        memories = self.retrieve_relevant_memories(query, intent=intent)

        if not memories:
            return RetrievalResult(status="not_found", memories=[], intent=intent)

        return RetrievalResult(
            status="found",
            memories=memories,
            selected=[{"memory": m} for m in memories],
            intent=intent,
            debug=self.last_debug,
        )
