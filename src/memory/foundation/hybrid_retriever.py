"""Hybrid Retrieval Engine for ALLI v0.6.

Consumes StructuredQuery from Session 5 and retrieves ranked memory
candidates using multiple retrieval channels:

  1. Structured retrieval — entity/relation/object matching
  2. Keyword/lexical retrieval — token overlap
  3. Semantic retrieval — embedding cosine similarity
  4. Relationship retrieval — entity relationship matching
  5. Temporal filtering — current/historical/time-specific

Then fuses candidates and ranks them transparently.

Scope:
  - Retrieves ranked memory candidates
  - Does NOT decide evidence sufficiency (Session 7)
  - Does NOT generate answers (Session 8)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Optional

from .query_understanding import StructuredQuery
from .models import Memory, EntityRef


# ---------------------------------------------------------------------------
# Retrieval Result Data Structures
# ---------------------------------------------------------------------------

@dataclass
class RetrievalCandidate:
    """A single retrieval candidate with full scoring metadata."""

    memory: dict  # Old-format memory dict for backward compatibility
    memory_id: str = ""
    content: str = ""

    # Individual channel scores (0.0 – 1.0)
    structured_score: float = 0.0
    keyword_score: float = 0.0
    semantic_score: float = 0.0
    relationship_score: float = 0.0
    temporal_score: float = 0.0

    # Match details
    entity_match: bool = False
    relation_match: bool = False
    object_match: bool = False
    lifecycle_match: bool = True

    # Retrieval sources that contributed
    sources: list[str] = field(default_factory=list)

    # Final ranking score
    final_score: float = 0.0
    rank: int = 0

    def __post_init__(self) -> None:
        if not self.memory_id:
            self.memory_id = self.memory.get("id", "")
        if not self.content:
            self.content = self.memory.get("content", "")


@dataclass
class RetrievalTrace:
    """Debug trace for a retrieval operation."""

    query: str = ""
    structured_query: Optional[dict] = None
    candidate_pool_size: int = 0
    structured_candidates: int = 0
    keyword_candidates: int = 0
    semantic_candidates: int = 0
    relationship_candidates: int = 0
    temporal_filtered: int = 0
    fused_candidates: int = 0
    final_ranked: int = 0
    channel_weights: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class HybridRetrievalResult:
    """Complete result from hybrid retrieval."""

    candidates: list[RetrievalCandidate] = field(default_factory=list)
    trace: RetrievalTrace = field(default_factory=RetrievalTrace)
    status: str = "ok"  # ok, not_found, no_candidates, not_personal

    @property
    def has_results(self) -> bool:
        return len(self.candidates) > 0

    @property
    def top_candidate(self) -> Optional[RetrievalCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def ranked_memories(self) -> list[dict]:
        """Return memories in rank order for backward compatibility."""
        return [c.memory for c in self.candidates]


# ---------------------------------------------------------------------------
# Token Utilities
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "through", "during",
    "before", "after", "above", "below", "between", "out", "off", "over",
    "under", "again", "further", "then", "once", "here", "there", "when",
    "where", "why", "how", "all", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "no", "nor", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "about", "also", "and",
    "but", "or", "if", "while", "that", "this", "these", "those", "what",
    "which", "who", "whom", "my", "your", "his", "her", "its", "our",
    "their", "me", "him", "us", "them", "i", "you", "he", "she", "it",
    "we", "they", "am", "your", "mine", "yours", "his", "hers", "ours",
    "theirs", "myself", "yourself", "himself", "herself", "itself",
    "ourselves", "yourselves", "themselves",
})


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase word tokens."""
    return [
        w for w in re.findall(r"[a-z0-9+#]+", text.lower())
        if w not in _STOP_WORDS and len(w) >= 2
    ]


def token_overlap_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Compute Jaccard-like overlap score between query and document tokens."""
    if not query_tokens or not doc_tokens:
        return 0.0
    query_set = set(query_tokens)
    doc_set = set(doc_tokens)
    intersection = query_set & doc_set
    union = query_set | doc_set
    if not union:
        return 0.0
    return len(intersection) / len(union)


# ---------------------------------------------------------------------------
# Hybrid Retriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Hybrid retrieval engine consuming StructuredQuery.

    Channels:
      1. Structured — entity/relation/object matching
      2. Keyword — token overlap scoring
      3. Semantic — embedding cosine similarity
      4. Relationship — entity relationship matching
      5. Temporal — lifecycle state and time filtering

    Fuses candidates from all channels and ranks by composite score.
    """

    # Channel weights for final ranking
    DEFAULT_WEIGHTS = {
        "structured": 0.30,
        "keyword": 0.15,
        "semantic": 0.35,
        "relationship": 0.10,
        "temporal": 0.10,
    }

    def __init__(
        self,
        store: object,
        embedding_store: object,
        *,
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Args:
            store: MemoryStore with .active_memories() returning list[dict]
            embedding_store: EmbeddingStore with .generate_embedding()
            weights: Optional channel weight overrides
        """
        self.store = store
        self.embedding_store = embedding_store
        self.weights = dict(weights or self.DEFAULT_WEIGHTS)
        self._query_cache: dict[str, list[float]] = {}

    # -------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------

    def retrieve(
        self,
        query: StructuredQuery,
        *,
        max_results: int = 8,
        threshold: float = 0.0,
    ) -> HybridRetrievalResult:
        """Retrieve ranked memory candidates for a StructuredQuery.

        Args:
            query: StructuredQuery from Session 5
            max_results: Maximum candidates to return
            threshold: Minimum final_score threshold

        Returns:
            HybridRetrievalResult with ranked candidates and trace
        """
        trace = RetrievalTrace(
            query=query.raw_query,
            structured_query=query.to_dict(),
            channel_weights=dict(self.weights),
        )

        # Handle empty/whitespace queries
        if not query.raw_query or not query.raw_query.strip():
            return HybridRetrievalResult(
                candidates=[],
                trace=trace,
                status="no_candidates",
            )

        # Get candidate pool
        active = self._get_candidate_pool(query)
        trace.candidate_pool_size = len(active)

        if not active:
            return HybridRetrievalResult(
                candidates=[],
                trace=trace,
                status="no_candidates",
            )

        # Generate query embedding once
        query_vec = self._get_query_embedding(query)

        # Collect candidates from each channel
        structured = self._structured_retrieval(active, query)
        trace.structured_candidates = len(structured)

        keyword = self._keyword_retrieval(active, query)
        trace.keyword_candidates = len(keyword)

        semantic = self._semantic_retrieval(active, query, query_vec)
        trace.semantic_candidates = len(semantic)

        relationship = self._relationship_retrieval(active, query)
        trace.relationship_candidates = len(relationship)

        # Fuse all candidates
        fused = self._fuse_candidates(
            active, structured, keyword, semantic, relationship, query
        )
        trace.fused_candidates = len(fused)

        # Apply temporal filtering/scoring
        temporally_scored = self._apply_temporal_scoring(fused, query)
        trace.temporal_filtered = len(temporally_scored)

        # Rank and select
        ranked = self._rank(temporally_scored, query)
        trace.final_ranked = len(ranked)

        # Apply threshold and limit
        filtered = [c for c in ranked if c.final_score >= threshold]
        final = filtered[:max_results]

        # Assign ranks
        for i, candidate in enumerate(final, 1):
            candidate.rank = i

        return HybridRetrievalResult(
            candidates=final,
            trace=trace,
            status="ok" if final else "not_found",
        )

    # -------------------------------------------------------------------
    # Candidate Pool
    # -------------------------------------------------------------------

    def _get_candidate_pool(self, query: StructuredQuery) -> list[dict]:
        """Get the base candidate pool from the store.

        NEGATED memories (explicitly-contradicted current state, e.g. "I
        don't want to learn Rust anymore.") are always included: they are the
        current state for their subject. For historical queries,
        inactive/superseded memories are included too.
        """
        active = self.store.active_memories()

        all_memories = getattr(self.store, "memories", [])

        # NEGATED memories are current state — include them for every query.
        negated = [
            m
            for m in all_memories
            if isinstance(m, dict)
            and str(m.get("status", "active")).lower() == "negated"
        ]
        if negated:
            active = active + negated

        # For historical queries, we need superseded/inactive memories too
        if query.state == "historical":
            historical = []
            for m in all_memories:
                if not isinstance(m, dict):
                    continue
                status = str(m.get("status", "active")).lower()
                if status in ("inactive", "superseded"):
                    historical.append(m)
            # Combine: active + historical for temporal queries
            return active + historical

        return active

    # -------------------------------------------------------------------
    # Channel 1: Structured Retrieval
    # -------------------------------------------------------------------

    def _structured_retrieval(
        self, memories: list[dict], query: StructuredQuery
    ) -> dict[str, float]:
        """Score memories by structural match to StructuredQuery.

        Returns dict of memory_id → structured_score.
        """
        scores: dict[str, float] = {}

        for memory in memories:
            mem_id = memory.get("id", "")
            score = 0.0

            # Entity match
            mem_entity = self._get_memory_entity(memory)
            if query.entity and query.entity != "user":
                if mem_entity.lower() == query.entity.lower():
                    score += 0.40
                elif mem_entity == "user" and query.subject == "user":
                    score += 0.20
            elif query.subject == "user":
                if mem_entity in ("user", "self", ""):
                    score += 0.30

            # Relation/fact_type match
            mem_fact_type = str(memory.get("fact_type", "")).lower()
            mem_category = str(memory.get("category", "")).lower()
            relation_match = self._match_relation(query.relation, mem_fact_type)
            score += relation_match * 0.30

            # Object/subject match
            mem_subject = str(memory.get("subject", "")).lower()
            mem_content = str(memory.get("content", "")).lower()
            object_match = self._match_object(query.object, mem_subject, mem_content, mem_fact_type, mem_category)
            score += object_match * 0.20

            # Domain match
            if query.domain and query.domain != "unknown":
                if mem_category == query.domain or mem_category.startswith(query.domain):
                    score += 0.10

            if score > 0:
                scores[mem_id] = min(1.0, score)

        return scores

    def _match_relation(self, query_relation: str, fact_type: str) -> float:
        """Match query relation to memory fact_type."""
        if query_relation == "unknown" or not query_relation:
            return 0.0

        # Direct match
        if query_relation == fact_type:
            return 1.0

        # Semantic relation groups
        _RELATION_GROUPS = {
            "possess": {"possession", "possession Ownership", "device", "phone", "laptop"},
            "prefer": {"preference", "like", "dislike", "preference Like", "preference Dislike", "learning"},
            "learn": {"learning", "education", "skill", "course", "study"},
            "identity": {"identity", "name", "nickname"},
            "goal": {"goal", "ambition", "dream", "career_goal", "project_goal"},
            "skill": {"skill", "ability", "proficiency"},
            "disprefer": {"dislike", "preference Dislike"},
            "project": {"project", "project_goal"},
            "not_possess": set(),
            "not_learn": set(),
        }

        group = _RELATION_GROUPS.get(query_relation, set())
        if fact_type in group:
            return 0.80

        # Partial matches
        if query_relation in ("possess", "prefer", "learn") and fact_type:
            if query_relation in fact_type or fact_type in query_relation:
                return 0.60

        return 0.0

    def _match_object(
        self, query_object: str, mem_subject: str, mem_content: str, mem_fact_type: str = "", mem_category: str = ""
    ) -> float:
        """Match query object — category-aware for general queries, specific for exact."""
        if query_object and query_object != "unknown":
            obj_lower = query_object.lower()
            if obj_lower in mem_subject.lower():
                return 1.0
            obj_tokens = set(tokenize(obj_lower))
            content_tokens = set(tokenize(mem_content))
            if obj_tokens and content_tokens:
                overlap = obj_tokens & content_tokens
                if overlap:
                    return min(1.0, len(overlap) / len(obj_tokens))
            # Semantic overlap with category words in content
            for cat_word in ("project","goal","preference","learning","technology","skill","identity","relationship"):
                if cat_word in obj_lower and cat_word in mem_content.lower():
                    return 0.85
            return 0.0
        else:
            # Category-level query: use relation/domain/category to match
            if mem_fact_type in ("project","goal","preference","learning","skill"):
                return 0.75
            if mem_category in ("project","goals","interests","learning","skills"):
                return 0.75
            return 0.5  # neutral for category query

    # -------------------------------------------------------------------
    # Channel 2: Keyword Retrieval
    # -------------------------------------------------------------------

    def _keyword_retrieval(
        self, memories: list[dict], query: StructuredQuery
    ) -> dict[str, float]:
        """Score memories by keyword/token overlap with the normalized query."""
        scores: dict[str, float] = {}

        query_tokens = tokenize(query.normalized_query)
        if not query_tokens:
            return scores

        for memory in memories:
            mem_id = memory.get("id", "")
            content = memory.get("content", "")
            subject = memory.get("subject", "")
            fact_type = memory.get("fact_type", "")

            # Combine subject + fact_type + content for tokenization
            doc_text = f"{subject} {fact_type} {content}"
            doc_tokens = tokenize(doc_text)

            score = token_overlap_score(query_tokens, doc_tokens)

            # Boost for exact phrase match in content
            query_lower = query.normalized_query.lower()
            if query_lower in content.lower():
                score = min(1.0, score + 0.30)

            if score > 0:
                scores[mem_id] = score

        return scores

    # -------------------------------------------------------------------
    # Channel 3: Semantic Retrieval
    # -------------------------------------------------------------------

    def _semantic_retrieval(
        self,
        memories: list[dict],
        query: StructuredQuery,
        query_vec: Optional[list[float]],
    ) -> dict[str, float]:
        """Score memories by embedding cosine similarity."""
        scores: dict[str, float] = {}

        if not query_vec:
            return scores

        for memory in memories:
            mem_id = memory.get("id", "")
            embedding = memory.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                continue

            sim = self._cosine_sim(query_vec, embedding)
            if sim > 0.0:
                scores[mem_id] = sim

        return scores

    def _get_query_embedding(self, query: StructuredQuery) -> Optional[list[float]]:
        """Generate embedding for the query, with caching."""
        cache_key = query.normalized_query or query.raw_query
        if not cache_key:
            return None

        cached = self._query_cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            vec = self.embedding_store.generate_embedding(cache_key)
            if vec:
                self._query_cache[cache_key] = vec
                return vec
        except Exception:
            pass

        # Fallback: try raw query
        try:
            vec = self.embedding_store.generate_embedding(query.raw_query)
            if vec:
                self._query_cache[cache_key] = vec
                return vec
        except Exception:
            pass

        return None

    def _cosine_sim(
        self, a: list[float], b: list[float]
    ) -> float:
        """Compute cosine similarity between two vectors."""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a <= 0 or norm_b <= 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # -------------------------------------------------------------------
    # Channel 4: Relationship Retrieval
    # -------------------------------------------------------------------

    def _relationship_retrieval(
        self, memories: list[dict], query: StructuredQuery
    ) -> dict[str, float]:
        """Score memories based on entity relationship matching."""
        scores: dict[str, float] = {}

        if not query.entity or query.entity == "user":
            return scores

        query_entity_lower = query.entity.lower()

        for memory in memories:
            mem_id = memory.get("id", "")
            mem_entity = self._get_memory_entity(memory).lower()

            # Direct entity match
            if mem_entity == query_entity_lower:
                scores[mem_id] = 1.0
            # Relationship words in content
            elif query_entity_lower in str(memory.get("content", "")).lower():
                scores[mem_id] = 0.50

        return scores

    # -------------------------------------------------------------------
    # Channel 5: Temporal Filtering
    # -------------------------------------------------------------------

    def _apply_temporal_scoring(
        self, candidates: list[RetrievalCandidate], query: StructuredQuery
    ) -> list[RetrievalCandidate]:
        """Apply temporal scoring to candidates."""
        for candidate in candidates:
            mem = candidate.memory
            status = str(mem.get("status", "active")).lower()
            expires_at = mem.get("expires_at")
            superseded_by = mem.get("superseded_by")

            temporal_score = 0.50  # default neutral

            if query.state == "current":
                # Prefer active, non-expired, non-superseded
                if status == "active" and not expires_at and not superseded_by:
                    temporal_score = 1.0
                elif status == "active" and expires_at:
                    temporal_score = 0.70
                elif status in ("inactive", "superseded"):
                    temporal_score = 0.20

            elif query.state == "historical":
                # Prefer inactive/superseded
                if status in ("inactive", "superseded"):
                    temporal_score = 1.0
                elif status == "active" and superseded_by:
                    temporal_score = 0.80
                else:
                    temporal_score = 0.30

            elif query.state == "future":
                temporal_score = 0.50  # no special handling yet

            else:
                # No temporal constraint — prefer active
                if status == "active":
                    temporal_score = 0.80
                else:
                    temporal_score = 0.40

            candidate.temporal_score = temporal_score

        return candidates

    # -------------------------------------------------------------------
    # Candidate Fusion
    # -------------------------------------------------------------------

    def _fuse_candidates(
        self,
        all_memories: list[dict],
        structured: dict[str, float],
        keyword: dict[str, float],
        semantic: dict[str, float],
        relationship: dict[str, float],
        query: StructuredQuery,
    ) -> list[RetrievalCandidate]:
        """Fuse candidates from all channels into a single candidate list."""
        # Collect all memory IDs that appeared in any channel
        all_ids: set[str] = set()
        all_ids.update(structured.keys())
        all_ids.update(keyword.keys())
        all_ids.update(semantic.keys())
        all_ids.update(relationship.keys())

        # Also include all memories (for temporal scoring)
        for m in all_memories:
            mid = m.get("id", "")
            if mid:
                all_ids.add(mid)

        # Build lookup
        mem_lookup: dict[str, dict] = {}
        for m in all_memories:
            mid = m.get("id", "")
            if mid:
                mem_lookup[mid] = m

        candidates = []
        for mem_id in all_ids:
            memory = mem_lookup.get(mem_id)
            if not memory:
                continue

            sources = []
            if mem_id in structured:
                sources.append("structured")
            if mem_id in keyword:
                sources.append("keyword")
            if mem_id in semantic:
                sources.append("semantic")
            if mem_id in relationship:
                sources.append("relationship")

            entity_match = self._check_entity_match(memory, query)
            relation_match = self._check_relation_match(memory, query)
            object_match = self._check_object_match(memory, query)

            candidate = RetrievalCandidate(
                memory=memory,
                memory_id=mem_id,
                structured_score=structured.get(mem_id, 0.0),
                keyword_score=keyword.get(mem_id, 0.0),
                semantic_score=semantic.get(mem_id, 0.0),
                relationship_score=relationship.get(mem_id, 0.0),
                entity_match=entity_match,
                relation_match=relation_match,
                object_match=object_match,
                sources=sources,
            )
            candidates.append(candidate)

        return candidates

    # -------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------

    def _rank(
        self, candidates: list[RetrievalCandidate], query: StructuredQuery
    ) -> list[RetrievalCandidate]:
        """Rank candidates by composite score."""
        for c in candidates:
            # Composite score from channels
            composite = (
                c.structured_score * self.weights.get("structured", 0.30)
                + c.keyword_score * self.weights.get("keyword", 0.15)
                + c.semantic_score * self.weights.get("semantic", 0.35)
                + c.relationship_score * self.weights.get("relationship", 0.10)
                + c.temporal_score * self.weights.get("temporal", 0.10)
            )

            # Bonuses
            if c.entity_match:
                composite += 0.05
            if c.relation_match:
                composite += 0.05
            if c.object_match:
                composite += 0.05

            # Multi-source bonus
            if len(c.sources) >= 3:
                composite += 0.05
            elif len(c.sources) >= 2:
                composite += 0.03

            c.final_score = min(1.0, composite)

        # Sort by final_score descending
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        return candidates

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    def _get_memory_entity(self, memory: dict) -> str:
        """Extract entity name from a memory dict."""
        entity = memory.get("entity", {})
        if isinstance(entity, dict):
            name = entity.get("name", "user")
        elif isinstance(entity, str):
            name = entity
        else:
            name = "user"
        if name in ("self", ""):
            name = "user"
        return name

    def _check_entity_match(self, memory: dict, query: StructuredQuery) -> bool:
        """Check if memory entity matches query entity."""
        if not query.entity or query.entity == "user":
            return True
        mem_entity = self._get_memory_entity(memory)
        return mem_entity.lower() == query.entity.lower()

    def _check_relation_match(self, memory: dict, query: StructuredQuery) -> bool:
        """Check if memory fact_type matches query relation."""
        if not query.relation or query.relation == "unknown":
            return False
        mem_fact = str(memory.get("fact_type", "")).lower()
        return self._match_relation(query.relation, mem_fact) > 0.5

    def _check_object_match(self, memory: dict, query: StructuredQuery) -> bool:
        """Check if memory subject/content matches query object."""
        if not query.object or query.object == "unknown":
            return False
        mem_subject = str(memory.get("subject", "")).lower()
        mem_content = str(memory.get("content", "")).lower()
        return self._match_object(query.object, mem_subject, mem_content) > 0.3
