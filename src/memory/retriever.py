"""Semantic + category-aware memory retrieval (ALLI v0.6)."""

from __future__ import annotations

from typing import Optional

from . import config
from .embeddings import EmbeddingError, EmbeddingStore, cosine_similarity
from .query_intent import INTENT_FILTERS, QueryIntent, QueryIntentDetector
from .store import MemoryStore


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

        intent = intent or self.intent_detector.detect(question)
        self.last_debug = {
            "query": question,
            "normalized_query": intent.normalized_query,
            "intent": intent.primary,
            "requested": list(intent.requested or []),
            "intent_score": intent.score,
            "intent_scores": dict(intent.scores),
            "is_memory_read": intent.is_memory_read,
            "is_general_knowledge": intent.is_general_knowledge,
            "candidates": [],
            "selected": [],
        }

        active = self.store.active_memories()
        if not question.strip() or not active:
            return []

        if intent.is_general_knowledge or not intent.is_memory_read:
            return []

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
                    category_match = category_match and category == "preference"
                else:
                    category_match = category_match and fact_type in allowed_fact_types
            if fact_type in exclude_fact_types:
                category_match = False

            if strict and not category_match:
                continue

            similarity = cosine_similarity(question_vec, embedding)
            subject_sim = None
            if category == "preference" and memory.get("subject"):
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
                effective_threshold = config.FILTERED_SIMILARITY_THRESHOLD
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
            return {"like", "dislike", "preference"}
        if base is None:
            return None
        return set(base)
