"""Evidence Grounding Engine for ALLI v0.6.

Consumes StructuredQuery (Session 5) + HybridRetrievalResult (Session 6)
and determines whether retrieved candidates actually provide sufficient
evidence for the query.

Core principle:
  RETRIEVAL ≠ EVIDENCE

A memory can be semantically similar but still be completely irrelevant
to the question. This layer validates evidence correctness.

Scope:
  - Validates evidence correctness
  - Does NOT generate final natural-language answers (Session 8)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .query_understanding import StructuredQuery
from .hybrid_retriever import HybridRetrievalResult, RetrievalCandidate


# ---------------------------------------------------------------------------
# Evidence Assessment Model
# ---------------------------------------------------------------------------

@dataclass
class EvidenceItem:
    """A single piece of evidence with full validation metadata."""

    candidate: RetrievalCandidate
    memory_id: str = ""
    content: str = ""
    entity: str = ""
    fact_type: str = ""
    domain: str = ""
    status: str = "active"  # active, inactive, superseded

    # Validation signals
    entity_valid: bool = False
    relation_valid: bool = False
    object_valid: bool = False
    domain_valid: bool = False
    temporal_valid: bool = True
    lifecycle_valid: bool = True
    is_negated: bool = False
    is_uncertain: bool = False

    # Evidence strength
    strength: float = 0.0  # 0.0 – 1.0
    rejection_reasons: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.memory_id:
            self.memory_id = self.candidate.memory_id
        if not self.content:
            self.content = self.candidate.content
        mem = self.candidate.memory
        if not self.entity:
            entity = mem.get("entity", {})
            self.entity = entity.get("name", "user") if isinstance(entity, dict) else str(entity)
        if not self.fact_type:
            self.fact_type = str(mem.get("fact_type", "")).lower()
        if not self.domain:
            self.domain = str(mem.get("category", "")).lower()
        # Always read status from memory — do NOT default to "active"
            self.status = str(mem.get("status", "active")).lower()


@dataclass
class EvidenceAssessment:
    """Complete evidence grounding assessment."""

    query: Optional[StructuredQuery] = None
    status: str = "NONE"  # SUPPORTED, WEAK, CONFLICTING, NONE
    accepted: list[EvidenceItem] = field(default_factory=list)
    rejected: list[EvidenceItem] = field(default_factory=list)
    conflicts: list[list[EvidenceItem]] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_evidence(self) -> bool:
        return len(self.accepted) > 0

    @property
    def is_supported(self) -> bool:
        return self.status == "SUPPORTED"

    @property
    def is_weak(self) -> bool:
        return self.status == "WEAK"

    @property
    def is_conflicting(self) -> bool:
        return self.status == "CONFLICTING"

    @property
    def is_none(self) -> bool:
        return self.status == "NONE"


# ---------------------------------------------------------------------------
# Evidence Grounder
# ---------------------------------------------------------------------------

class EvidenceGrounder:
    """Validates whether retrieved candidates constitute real evidence.

    Decision rules:
      1. Entity must match (user query → user memory, mom query → mom memory)
      2. Relation must be compatible (phone query → possession memory)
      3. Object must be compatible (phone query → phone memory, not color)
      4. Domain must be compatible (possessions → possessions)
      5. Temporal state must match (current → active, historical → inactive)
      6. Conflicts are detected (two active same-entity same-relation memories)
      7. Negation is preserved (negative memory ≠ positive evidence)
      8. Uncertainty is preserved (uncertain ≠ certain)
      9. No personal facts are invented
    """

    # Relation compatibility groups
    _RELATION_COMPAT: dict[str, set[str]] = {
        "possess": {"possession", "device", "phone", "laptop", "car", "ownership", "owned_item"},
        "prefer": {"preference", "like", "dislike", "interest", "favorite"},
        "learn": {"learning", "education", "skill", "course", "study"},
        "identity": {"identity", "name", "nickname"},
        "goal": {"goal", "ambition", "dream"},
        "skill": {"skill", "ability", "proficiency"},
        "disprefer": {"dislike", "preference_dislike"},
        "not_possess": set(),
        "not_learn": set(),
    }

    # Domain compatibility groups
    _DOMAIN_COMPAT: dict[str, set[str]] = {
        "possessions": {"possession", "device"},
        "interests": {"preference", "interest", "like"},
        "learning": {"learning", "education", "skill"},
        "identity": {"identity"},
        "goals": {"goal", "ambition", "dream"},
        "skills": {"skill", "ability"},
    }

    def ground(
        self,
        query: StructuredQuery,
        retrieval_result: HybridRetrievalResult,
    ) -> EvidenceAssessment:
        """Ground retrieval results against the query.

        Args:
            query: StructuredQuery from Session 5
            retrieval_result: HybridRetrievalResult from Session 6

        Returns:
            EvidenceAssessment with accepted/rejected evidence and status
        """
        assessment = EvidenceAssessment(query=query)

        if not retrieval_result.has_results:
            assessment.status = "NONE"
            assessment.diagnostics["reason"] = "no_candidates"
            return assessment

        # Process each candidate
        for candidate in retrieval_result.candidates:
            item = self._validate_candidate(candidate, query)
            if item.rejection_reasons:
                assessment.rejected.append(item)
            else:
                assessment.accepted.append(item)

        # Detect conflicts
        assessment.conflicts = self._detect_conflicts(assessment.accepted, query)

        # Determine final status
        assessment.status = self._determine_status(assessment)

        # Diagnostics
        assessment.diagnostics = self._build_diagnostics(assessment, retrieval_result)

        return assessment

    # -------------------------------------------------------------------
    # Candidate Validation
    # -------------------------------------------------------------------

    def _validate_candidate(
        self, candidate: RetrievalCandidate, query: StructuredQuery
    ) -> EvidenceItem:
        """Validate a single candidate against the query."""
        item = EvidenceItem(candidate=candidate)
        reasons = []

        # 1. Entity validation
        entity_valid = self._validate_entity(item, query)
        item.entity_valid = entity_valid
        if not entity_valid:
            reasons.append(f"entity_mismatch: memory_entity={item.entity} query_entity={query.entity}")

        # 2. Relation validation
        relation_valid = self._validate_relation(item, query)
        item.relation_valid = relation_valid
        if not relation_valid:
            reasons.append(f"relation_mismatch: memory_fact={item.fact_type} query_relation={query.relation}")

        # 3. Object validation
        object_valid = self._validate_object(item, query)
        item.object_valid = object_valid
        if not object_valid:
            reasons.append(f"object_mismatch: query_object={query.object}")

        # 4. Domain validation
        domain_valid = self._validate_domain(item, query)
        item.domain_valid = domain_valid
        if not domain_valid:
            reasons.append(f"domain_mismatch: memory_domain={item.domain} query_domain={query.domain}")

        # 5. Temporal validation
        temporal_valid = self._validate_temporal(item, query)
        item.temporal_valid = temporal_valid
        if not temporal_valid:
            reasons.append(f"temporal_mismatch: query_state={query.state} memory_status={item.status}")

        # 6. Lifecycle validation
        lifecycle_valid = self._validate_lifecycle(item, query)
        item.lifecycle_valid = lifecycle_valid
        if not lifecycle_valid:
            reasons.append(f"lifecycle_invalid: status={item.status}")

        # 7. Negation detection
        item.is_negated = self._detect_negation(item)

        # 8. Uncertainty detection
        item.is_uncertain = self._detect_uncertainty(item)

        # Calculate strength
        item.strength = self._calculate_strength(item, query)
        item.rejection_reasons = reasons

        return item

    def _validate_entity(self, item: EvidenceItem, query: StructuredQuery) -> bool:
        """Validate that memory entity matches query entity."""
        query_entity = query.entity or query.subject
        if not query_entity or query_entity == "unknown":
            return True  # No entity constraint

        memory_entity = item.entity.lower()
        query_entity_lower = query_entity.lower()

        # Direct match
        if memory_entity == query_entity_lower:
            return True

        # "self" / "user" equivalence
        if memory_entity in ("user", "self") and query_entity_lower in ("user", "self"):
            return True

        return False

    def _validate_relation(self, item: EvidenceItem, query: StructuredQuery) -> bool:
        """Validate that memory fact_type is compatible with query relation."""
        if not query.relation or query.relation == "unknown":
            return True  # No relation constraint

        query_relation = query.relation.lower()
        memory_fact = item.fact_type.lower()

        # Direct match
        if query_relation == memory_fact:
            return True

        # Compatibility groups
        compat_set = self._RELATION_COMPAT.get(query_relation, set())
        if memory_fact in compat_set:
            return True

        # Partial match (query relation appears in fact type or vice versa)
        if query_relation in memory_fact or memory_fact in query_relation:
            return True

        return False

    # Category → instance mappings for generic query objects
    _CATEGORY_INSTANCES: dict[str, set[str]] = {
        "technology": {
            "python", "java", "javascript", "c++", "ruby", "go", "rust", "swift",
            "kotlin", "php", "sql", "html", "css", "react", "angular", "vue",
            "node", "django", "flask", "spring", "rails", "laravel", "docker",
            "kubernetes", "linux", "windows", "macos", "android", "ios",
            "samsung", "iphone", "pixel", "oppo", "oneplus", "xiaomi", "huawei", "motorola", "nokia",
            "samsung s24", "samsung s23", "samsung s22", "iphone 15", "iphone 14", "iphone 13",
            "galaxy", "galaxy s24", "galaxy s23",
        },
        "phone": {"samsung", "iphone", "pixel", "oppo", "oneplus", "xiaomi", "huawei", "motorola", "nokia"},
        "color": {"blue", "red", "green", "orange", "purple", "yellow", "black", "white", "pink", "brown", "gray", "grey"},
        "food": {"pizza", "pasta", "sushi", "tacos", "burger", "steak", "chicken", "rice", "noodles"},
        "music": {"rock", "pop", "jazz", "classical", "hip hop", "electronic", "country", "r&b"},
        "movie": {"action", "comedy", "drama", "horror", "sci-fi", "thriller", "animation"},
        "language": {"python", "java", "javascript", "c++", "ruby", "go", "rust", "swift", "kotlin"},
    }

    def _validate_object(self, item: EvidenceItem, query: StructuredQuery) -> bool:
        """Validate that memory content is compatible with query object."""
        if not query.object or query.object == "unknown":
            return True  # No object constraint

        obj_lower = query.object.lower()
        content_lower = item.content.lower()
        subject_lower = str(item.candidate.memory.get("subject", "")).lower()
        value_lower = str(item.candidate.memory.get("value", "")).lower()

        # Direct match in subject
        if obj_lower in subject_lower:
            return True

        # Direct match in value
        if obj_lower in value_lower:
            return True

        # Token overlap
        obj_tokens = set(re.findall(r"[a-z0-9]+", obj_lower))
        content_tokens = set(re.findall(r"[a-z0-9]+", content_lower))
        if obj_tokens and content_tokens:
            overlap = obj_tokens & content_tokens
            if overlap and len(overlap) / len(obj_tokens) >= 0.5:
                return True

        # Category → instance matching: generic query object matches specific memory subject
        category_instances = self._CATEGORY_INSTANCES.get(obj_lower, set())
        if category_instances:
            # Check if memory subject or value is a known instance of the category
            if subject_lower in category_instances or value_lower in category_instances:
                return True
            # Check if content mentions a category instance
            for instance in category_instances:
                if instance in content_lower:
                    return True

        # Special: "color" matches color-related content
        if obj_lower in ("color", "colour"):
            if any(w in content_lower for w in ("color", "colour", "blue", "red", "green", "orange", "purple", "yellow", "black", "white", "pink")):
                return True

        # Special: "phone" matches phone-related content
        if obj_lower in ("phone", "mobile", "device"):
            if any(w in content_lower for w in ("phone", "mobile", "device", "samsung", "iphone", "pixel", "oppo")):
                return True

        return False

    def _validate_domain(self, item: EvidenceItem, query: StructuredQuery) -> bool:
        """Validate that memory domain is compatible with query domain."""
        if not query.domain or query.domain == "unknown":
            return True  # No domain constraint

        query_domain = query.domain.lower()
        memory_domain = item.domain.lower()

        # Direct match
        if query_domain == memory_domain:
            return True

        # Compatibility groups
        compat_set = self._DOMAIN_COMPAT.get(query_domain, set())
        if memory_domain in compat_set:
            return True

        return False

    def _validate_temporal(self, item: EvidenceItem, query: StructuredQuery) -> bool:
        """Validate temporal compatibility."""
        if not query.state or query.state == "unknown":
            return True  # No temporal constraint

        if query.state == "current":
            # Current queries prefer active, non-superseded memories
            return item.status == "active"

        if query.state == "historical":
            # Historical queries accept inactive/superseded memories
            return item.status in ("inactive", "superseded", "active")

        if query.state == "future":
            return True  # No special handling yet

        return True

    def _validate_lifecycle(self, item: EvidenceItem, query: StructuredQuery) -> bool:
        """Validate lifecycle state."""
        # Expired memories are never valid evidence
        if item.status == "expired":
            return False

        # For current queries, superseded memories are not valid
        if query.state == "current" and item.status == "superseded":
            return False

        return True

    def _detect_negation(self, item: EvidenceItem) -> bool:
        """Detect if the memory contains negation."""
        content = item.content.lower()
        negation_patterns = (
            r"\b(?:don'?t|do\s+not|doesn'?t|did\s+not|didn'?t|never|no\s+longer)\b",
            r"\b(?:cannot|can't|won't|wouldn't|shouldn't)\b",
        )
        for pattern in negation_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _detect_uncertainty(self, item: EvidenceItem) -> bool:
        """Detect if the memory contains uncertainty."""
        content = item.content.lower()
        uncertainty_patterns = (
            r"\b(?:i\s+think|i\s+believe|i\s+feel\s+like|maybe|perhaps|possibly)\b",
            r"\b(?:might|could|not\s+sure|unsure|apparently)\b",
            r"\b(?:i\s+guess|i\s+suppose)\b",
        )
        for pattern in uncertainty_patterns:
            if re.search(pattern, content):
                return True
        return False

    def _calculate_strength(self, item: EvidenceItem, query: StructuredQuery) -> float:
        """Calculate evidence strength (0.0 – 1.0)."""
        score = 0.0

        # Base score from retrieval
        score += item.candidate.final_score * 0.3

        # Entity match bonus
        if item.entity_valid:
            score += 0.25

        # Relation match bonus
        if item.relation_valid:
            score += 0.20

        # Object match bonus
        if item.object_valid:
            score += 0.10

        # Domain match bonus
        if item.domain_valid:
            score += 0.05

        # Temporal bonus
        if item.temporal_valid:
            score += 0.05

        # Lifecycle bonus
        if item.lifecycle_valid:
            score += 0.05

        # Penalties
        if item.is_negated:
            score *= 0.5
        if item.is_uncertain:
            score *= 0.7
        if item.status in ("inactive", "superseded"):
            score *= 0.6

        return min(1.0, max(0.0, score))

    # -------------------------------------------------------------------
    # Conflict Detection
    # -------------------------------------------------------------------

    def _detect_conflicts(
        self, accepted: list[EvidenceItem], query: StructuredQuery
    ) -> list[list[EvidenceItem]]:
        """Detect conflicting evidence among accepted items.

        Only detects conflicts when items are about the SAME specific subject
        (e.g., two different favorite colors), not just the same fact_type
        (e.g., multiple different likes).
        """
        if len(accepted) < 2:
            return []

        # Group by (entity, fact_type, subject) — only same-subject items conflict
        groups: dict[tuple[str, str, str], list[EvidenceItem]] = {}
        for item in accepted:
            subject = str(item.candidate.memory.get("subject", "")).lower()
            key = (item.entity.lower(), item.fact_type.lower(), subject)
            groups.setdefault(key, []).append(item)

        conflicts = []
        for key, items in groups.items():
            if len(items) >= 2:
                # Check if they have different values
                values = set()
                for item in items:
                    value = item.candidate.memory.get("value", "")
                    if value:
                        values.add(value.lower())
                if len(values) > 1:
                    conflicts.append(items)

        return conflicts

    # -------------------------------------------------------------------
    # Status Determination
    # -------------------------------------------------------------------

    def _determine_status(self, assessment: EvidenceAssessment) -> str:
        """Determine the final grounding status."""
        if not assessment.accepted:
            if assessment.rejected:
                return "NONE"
            return "NONE"

        # Check for conflicts
        if assessment.conflicts:
            return "CONFLICTING"

        # Boolean existence queries ("Do I like X?", "Do I have X?") have a
        # definitive YES/NO answer when a single accepted memory is fully
        # validated against the query. Strength penalties (e.g. negation x0.5)
        # exist to weaken *open-ended* answers from weaker evidence — they must
        # not demote a directly-matching "yes"/"no" fact to an "unsure" reply.
        query = assessment.query
        if (
            query is not None
            and query.cardinality == "existence"
            and len(assessment.accepted) == 1
        ):
            item = assessment.accepted[0]
            if (
                item.entity_valid
                and item.relation_valid
                and item.object_valid
                and item.domain_valid
                and item.temporal_valid
                and item.lifecycle_valid
            ):
                return "SUPPORTED"

        # Check strength distribution
        strengths = [item.strength for item in assessment.accepted]
        max_strength = max(strengths) if strengths else 0.0
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.0

        if max_strength >= 0.6 and avg_strength >= 0.4:
            return "SUPPORTED"

        if max_strength >= 0.3:
            return "WEAK"

        return "NONE"

    # -------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------

    def _build_diagnostics(
        self,
        assessment: EvidenceAssessment,
        retrieval_result: HybridRetrievalResult,
    ) -> dict:
        """Build diagnostic information."""
        return {
            "query": assessment.query.raw_query if assessment.query else "",
            "retrieval_status": retrieval_result.status,
            "candidates_evaluated": len(retrieval_result.candidates),
            "accepted_count": len(assessment.accepted),
            "rejected_count": len(assessment.rejected),
            "conflict_count": len(assessment.conflicts),
            "final_status": assessment.status,
            "accepted_entities": [item.entity for item in assessment.accepted],
            "accepted_fact_types": [item.fact_type for item in assessment.accepted],
            "rejection_reasons": [
                {"memory_id": item.memory_id, "reasons": item.rejection_reasons}
                for item in assessment.rejected
            ],
        }
