"""Operation decisions — converts extracted operations into foundation memory records.

Lifecycle engine that decides what happens to existing memories when new
information arrives. Based on structured meaning and memory state, not
sentence-specific rules.

Operations: ADD, UPDATE, MERGE, IGNORE, INVALIDATE, SUPERSEDE

Multi-value fact handling:
  Facts like preferences, interests, skills allow multiple simultaneous
  values. "I like Java" + "I like Python" = both active (not supersede).

  Single-value facts like current phone, name, employer allow only one
  current value. "I use Oppo" + "I use Samsung" = supersede.

The distinction is driven by structured fields (fact_type, domain),
not sentence-level rules.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from .extraction import MemoryOperation, Action
from .models import (
    EntityRef,
    Fact,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    EventType,
    Provenance,
    ProvenanceSource,
    TemporalScope,
)
from .ids import generate_memory_id, generate_event_id
from .storage import MemoryStore


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ---------------------------------------------------------------------------
# Temporal resolution
# ---------------------------------------------------------------------------

def _resolve_temporal(
    is_persistent: bool,
    temporal_hint: Optional[str],
) -> TemporalScope:
    """Resolve temporal scope from hint and persistence flag."""
    now = _utc_now_iso()

    if is_persistent:
        return TemporalScope(valid_from=now, is_persistent=True)

    valid_until = None
    if temporal_hint:
        hint = temporal_hint.lower().strip()
        if "tomorrow" in hint or "tommorrow" in hint:
            valid_until = (datetime.now(timezone.utc) + timedelta(days=1)).replace(
                microsecond=0
            ).isoformat()
        elif "next week" in hint:
            valid_until = (datetime.now(timezone.utc) + timedelta(weeks=1)).replace(
                microsecond=0
            ).isoformat()
        elif "next month" in hint:
            valid_until = (datetime.now(timezone.utc) + timedelta(days=30)).replace(
                microsecond=0
            ).isoformat()
        elif "today" in hint or "tonight" in hint:
            valid_until = datetime.now(timezone.utc).replace(
                hour=23, minute=59, second=59, microsecond=0
            ).isoformat()
        elif "this week" in hint:
            now_dt = datetime.now(timezone.utc)
            days_until_sunday = 6 - now_dt.weekday()
            valid_until = (now_dt + timedelta(days=days_until_sunday)).replace(
                hour=23, minute=59, second=59, microsecond=0
            ).isoformat()
        elif "this month" in hint:
            now_dt = datetime.now(timezone.utc)
            if now_dt.month == 12:
                end = now_dt.replace(day=31, hour=23, minute=59, second=59, microsecond=0)
            else:
                end = now_dt.replace(month=now_dt.month + 1, day=1, hour=23, minute=59, second=59, microsecond=0) - timedelta(days=1)
            valid_until = end.isoformat()

    return TemporalScope(
        valid_from=now,
        valid_until=valid_until,
        is_persistent=False,
        temporal_hint=temporal_hint,
    )


def _determine_importance(operation: MemoryOperation) -> str:
    """Determine importance level from operation content."""
    content = operation.content.lower()

    if operation.domain == "identity":
        return "high"
    if operation.domain == "goals" and operation.confidence >= 0.7:
        return "high"
    if any(kw in content for kw in ("important", "critical", "must", "urgent")):
        return "high"
    if operation.is_uncertain:
        return "low"
    if operation.is_temporary:
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Multi-value vs single-value fact classification
# ---------------------------------------------------------------------------

# Fact types that allow multiple simultaneous active values for the same entity.
# These are set-based, not single-current-value facts.
_MULTI_VALUE_FACT_TYPES = frozenset({
    "preference",    # likes, enjoys, prefers
    "hobby",         # hobbies, pastimes
    "skill",         # skills, proficiencies
    "learning",      # things being learned
    "course",        # courses being taken
    "goal",          # aspirations, targets
    "career_goal",
    "project_goal",
})

# Domains where multiple values are inherently allowed
_MULTI_VALUE_DOMAINS = frozenset({
    "interests",     # preferences, likes, hobbies
    "learning",      # multiple things can be learned simultaneously
    "knowledge",     # multiple skills
    "general",       # general facts about an entity can coexist
})


def _is_multi_value_fact(fact_type: str, domain: str) -> bool:
    """Determine if a fact type allows multiple simultaneous active values.

    Single-value facts (current phone, name, employer) → supersede on contradiction.
    Multi-value facts (likes, interests, skills) → allow multiple active values.

    Classification is based on structured fields, not sentence content.
    """
    if fact_type in _MULTI_VALUE_FACT_TYPES:
        return True
    if domain in _MULTI_VALUE_DOMAINS:
        return True
    return False


# ---------------------------------------------------------------------------
# Decision receipt — structured audit trail
# ---------------------------------------------------------------------------

@dataclass
class DecisionReceipt:
    """Structured record of a lifecycle decision.

    Exposes: candidate, operation, target, reason, resulting state.
    Used for debugging, audit, and explanation.
    """

    candidate_content: str
    candidate_domain: str
    candidate_fact_type: str
    operation: str
    target_memory_id: Optional[str]
    target_content: Optional[str]
    reason: str
    resulting_action: str
    resulting_state: str  # "created", "active", "inactive", "unchanged", "superseded"
    previous_state: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate_content,
            "domain": self.candidate_domain,
            "fact_type": self.candidate_fact_type,
            "operation": self.operation,
            "target_id": self.target_memory_id,
            "target_content": self.target_content,
            "reason": self.reason,
            "action": self.resulting_action,
            "state": self.resulting_state,
            "previous_state": self.previous_state,
        }


# ---------------------------------------------------------------------------
# Subject normalization and similarity
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset({
    "i", "me", "my", "mine", "myself", "am", "is", "are", "was", "were",
    "a", "an", "the", "to", "for", "in", "on", "at", "with", "and", "or",
    "but", "not", "do", "does", "did", "have", "has", "had", "like", "likes",
    "love", "hate", "want", "use", "using", "used", "just", "really", "very",
    "also", "too", "so", "now", "currently", "anymore",
})


def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, remove stop words, collapse whitespace."""
    words = re.findall(r"[a-z0-9+#]+", text.lower())
    return " ".join(w for w in words if w not in _STOP_WORDS and len(w) > 1)


def _subjects_match(subject_a: str, subject_b: str) -> bool:
    """Check if two subjects refer to the same logical thing.

    Uses exact normalized equality after removing stop words and collapsing
    whitespace. This prevents false merges between:

    - Different owners: "friend's Honda Civic" ≠ "brother's Honda Civic"
    - Different attributes: "Honda Civic insurance" ≠ "Honda Civic dealership"
    - Supersets/subsets: "Honda Civic" ≠ "Honda Civic model"

    The normalization strips possessive pronouns (my, your, his, her) so
    "My Honda Civic" == "Honda Civic" (same entity). But it preserves
    relationship tokens (friend, brother) so different owners stay separate.

    For memory safety, false merges are worse than failing to merge an
    ambiguous pair.
    """
    a = _normalize_text(subject_a)
    b = _normalize_text(subject_b)

    if not a or not b:
        return False

    # Exact normalized match
    return a == b


def _values_contradict(value_a: str, value_b: str) -> bool:
    """Check if two values represent contradictory information.

    Two values contradict if they are different and neither is a
    substring/overlap of the other (which would indicate compatible
    additional information).
    """
    a = _normalize_text(value_a)
    b = _normalize_text(value_b)

    if not a or not b:
        return False

    # Same value = not contradictory
    if a == b:
        return False

    # One contains the other = compatible addition (e.g., "Samsung S24" + "Samsung S24 is black")
    if a in b or b in a:
        return False

    # Different values = contradictory (e.g., "Python" vs "Java")
    return True


def _is_compatible_addition(
    existing: Memory,
    new_subject: str,
    new_value: str,
    new_domain: str,
) -> bool:
    """Check if new information is a compatible addition to existing memory.

    Compatible additions are facts about the same entity that provide
    new attributes without contradicting existing information.
    Examples:
      - "I have a Samsung S24" + "My Samsung S24 is black" → compatible
      - "I like Java" + "I use IntelliJ for Java" → compatible
    """
    # Same domain, same entity, different subject = compatible
    if existing.domain == new_domain and existing.entity.name:
        existing_subject = _normalize_text(existing.fact.subject)
        new_subj = _normalize_text(new_subject)

        # If subjects share core tokens but aren't identical → compatible
        if existing_subject and new_subj:
            existing_tokens = set(existing_subject.split())
            new_tokens = set(new_subj.split())
            overlap = existing_tokens & new_tokens

            # If there's significant overlap but subjects differ → compatible
            if overlap and existing_subject != new_subj:
                # Check if the new info adds a different attribute
                diff_existing = existing_tokens - new_tokens
                diff_new = new_tokens - existing_tokens
                if diff_existing or diff_new:
                    return True

    return False


# ---------------------------------------------------------------------------
# Finding existing memories
# ---------------------------------------------------------------------------

# Generic verbs/prepositions carry no entity identity. Matching a negation or
# correction target on these words lets "I don't want to learn Rust anymore."
# (subject "to learn rust") deactivate the first unrelated learning memory,
# because "learn" alone satisfies a 50% content-word overlap. Only distinctive
# words may identify a deletion/invalidation target.
_GENERIC_SUBJECT_WORDS = frozenset({
    "learn", "learning", "study", "studying", "like", "liking", "love",
    "loving", "prefer", "preferring", "want", "wanting", "wish",
    "use", "using", "have", "having", "get", "getting", "make", "making",
    "take", "taking", "build", "building", "develop", "developing",
    "the", "and", "for", "with", "about", "into", "from", "that", "this",
    "any", "more", "anymore", "now", "not", "dont",
})


def _find_existing_active(
    store: MemoryStore,
    entity_name: str,
    fact_type: str,
    subject: str,
    *,
    allow_content_match: bool = False,
) -> Optional[Memory]:
    """Find an existing active memory with the same entity+fact_type+subject.

    Matching strategy (escalating):
    1. Exact normalized subject match (same fact_type required)
    2. Word-set match — same significant words regardless of order (same fact_type)
    3. Content-based match (only when allow_content_match=True) — negated entity
       words appear in original content. Handles cross-fact-type corrections like
       "X is not my Y" where the correction is classified differently from original.
       Generic verbs are excluded — they do not identify the target fact.
    """
    active = store.get_by_entity(entity_name)

    # Pass 1: exact subject match
    for mem in active:
        if (
            mem.lifecycle.is_active
            and mem.fact.fact_type == fact_type
            and _subjects_match(mem.fact.subject, subject)
        ):
            return mem

    # Pass 2: word-set match (handles word-order differences from reversals)
    subj_words = set(_normalize_text(subject).split())
    if len(subj_words) >= 2:
        for mem in active:
            if not mem.lifecycle.is_active:
                continue
            if mem.fact.fact_type != fact_type:
                continue
            mem_words = set(_normalize_text(mem.fact.subject).split())
            if len(mem_words) >= 2 and subj_words == mem_words:
                return mem

    # Pass 3: content-based match (cross-fact-type corrections only)
    # Only used when explicitly allowed (correction/negation paths).
    # Check if DISTINCTIVE subject words appear in the original content.
    if allow_content_match:
        subj_words_list = [
            w
            for w in _normalize_text(subject).split()
            if len(w) > 2 and w not in _GENERIC_SUBJECT_WORDS
        ]
        if subj_words_list:
            for mem in active:
                if not mem.lifecycle.is_active:
                    continue
                content_lower = mem.content.lower()
                matches = sum(1 for w in subj_words_list if w in content_lower)
                if matches >= 1 and matches / len(subj_words_list) >= 0.5:
                    return mem

    return None


def _find_same_fact_type(
    store: MemoryStore,
    entity_name: str,
    fact_type: str,
) -> Optional[Memory]:
    """Find an existing active memory with same entity+fact_type (any subject).

    Used for contradiction detection when subjects don't match exactly
    but the fact type is the same (e.g., two different preferences).

    Excludes historical/temporary memories (is_persistent=False) from
    contradiction matching — a current fact should not supersede a
    historical record merely because they share the same fact type.

    NOTE: For multi-value fact types, this function still finds existing
    memories, but the CALLER must decide whether to supersede or create
    a new parallel active memory.
    """
    active = store.get_by_entity(entity_name)
    for mem in active:
        if (
            mem.lifecycle.is_active
            and mem.fact.fact_type == fact_type
            and mem.temporal.is_persistent
        ):
            return mem
    return None


def _find_all_same_fact_type(
    store: MemoryStore,
    entity_name: str,
    fact_type: str,
) -> list[Memory]:
    """Find ALL existing active memories with same entity+fact_type.

    Unlike _find_same_fact_type which returns only the first match,
    this returns all active memories of the same fact type. Used for
    multi-value fact handling where we need to check if a value already
    exists among multiple active values.
    """
    results = []
    active = store.get_by_entity(entity_name)
    for mem in active:
        if (
            mem.lifecycle.is_active
            and mem.fact.fact_type == fact_type
            and mem.temporal.is_persistent
        ):
            results.append(mem)
    return results


def _find_related_memories(
    store: MemoryStore,
    entity_name: str,
    domain: str,
    subject: str,
) -> list[Memory]:
    """Find active memories in the same entity+domain with related subjects."""
    results = []
    active = store.get_by_entity(entity_name)
    for mem in active:
        if not mem.lifecycle.is_active:
            continue
        if mem.domain != domain:
            continue
        if _subjects_match(mem.fact.subject, subject):
            results.append(mem)
    return results


def _find_deletion_target(
    store: MemoryStore,
    entity_name: str,
    deletion_subject: str,
) -> Optional[Memory]:
    """Find the best active memory to deactivate for a deletion request.

    Searches the specified entity first, then falls back to all entities
    if no match is found. This handles cases like "Forget about Jake's hiking"
    where entity resolution may not resolve the third-party entity.
    """
    # Search in specified entity first
    active = store.get_by_entity(entity_name)

    # Pass 1: exact subject match
    for mem in active:
        if mem.lifecycle.is_active and mem.fact.subject == deletion_subject:
            return mem

    # Pass 2: flexible match
    ds = deletion_subject.lower().strip()
    for mem in active:
        if mem.lifecycle.is_active and (
            ds in mem.fact.subject.lower()
            or ds in mem.content.lower()
            or mem.fact.subject.lower() in ds
        ):
            return mem

    # Pass 3: domain-scoped semantic match
    for mem in active:
        if mem.lifecycle.is_active and _subjects_match(mem.fact.subject, deletion_subject):
            return mem

    # Pass 4: search ALL entities (handles "Forget about X" where entity wasn't resolved)
    for mem in store.get_active():
        if mem.entity.name == entity_name:
            continue  # already checked
        if not mem.lifecycle.is_active:
            continue
        if (
            ds in mem.fact.subject.lower()
            or ds in mem.content.lower()
            or mem.fact.subject.lower() in ds
            or _subjects_match(mem.fact.subject, deletion_subject)
        ):
            return mem

    return None


def _find_contradiction_target(
    store: MemoryStore,
    entity_name: str,
    domain: str,
    subject: str,
    value: str,
) -> Optional[Memory]:
    """Find an active memory that contradicts the new information.

    A contradiction is when the same logical fact has a different value.
    """
    related = _find_related_memories(store, entity_name, domain, subject)
    for mem in related:
        if not mem.lifecycle.is_active:
            continue
        # Check if values contradict
        if _values_contradict(mem.fact.value, value):
            return mem
    return None


# ---------------------------------------------------------------------------
# Lifecycle decision engine
# ---------------------------------------------------------------------------

def _decide_lifecycle_action(
    operation: MemoryOperation,
    store: MemoryStore,
    entity_name: str,
) -> tuple[str, Optional[Memory], str]:
    """Decide the lifecycle action based on existing memory state.

    The decision is driven by:
    - Structured fields (fact_type, domain, subject, value, polarity)
    - Existing memory state (active, inactive, historical)
    - Multi-value vs single-value fact classification
    - Explicit correction/negation signals
    - Temporal context

    Returns (action, existing_memory, reason).
    Actions: "deactivate", "invalidate", "ignore", "supersede", "merge", "create"
    """
    # --- Explicit deactivation requested ---
    if operation.action == Action.DEACTIVATE:
        target = _find_deletion_target(store, entity_name, operation.subject)
        return ("deactivate", target, "user_deletion")

    # --- Explicit INVALIDATE (mark as no longer valid without replacement) ---
    if operation.action == Action.INVALIDATE:
        target = _find_deletion_target(store, entity_name, operation.subject)
        return ("invalidate", target, "user_invalidation")

    # --- Explicit correction/update requested ---
    if operation.action == Action.UPDATE:
        # Historical/temporary statements should not supersede current memories
        if operation.is_temporary:
            return ("create", None, "historical_statement")
        existing = _find_existing_active(store, entity_name, operation.fact_type, operation.subject, allow_content_match=True)
        if existing:
            # Check if content actually differs
            if existing.content.lower().strip() == operation.content.lower().strip():
                return ("ignore", existing, "duplicate_content")
            # Negation-based correction: "I don't like X anymore" → invalidate
            if operation.polarity == "negative":
                return ("invalidate", existing, "negation_correction")
            return ("supersede", existing, "explicit_correction")
        # No exact subject match — try flexible match for negation
        if operation.polarity == "negative":
            neg_target = _find_deletion_target(store, entity_name, operation.subject)
            if neg_target:
                return ("invalidate", neg_target, "negation_correction")
        # No exact subject match — check same-fact-type contradiction
        # But only for single-value fact types
        if not _is_multi_value_fact(operation.fact_type, operation.domain):
            same_type = _find_same_fact_type(store, entity_name, operation.fact_type)
            if same_type and not operation.is_uncertain:
                if _values_contradict(same_type.fact.value, operation.value):
                    return ("supersede", same_type, "fact_type_contradiction")
        # No existing — just create
        return ("create", None, "new_fact")

    # --- Negation detection: "I don't like X anymore" ---
    # This is distinct from a value change — it invalidates the existing fact
    if operation.polarity == "negative" and operation.action == Action.CREATE:
        existing = _find_existing_active(store, entity_name, operation.fact_type, operation.subject, allow_content_match=True)
        if existing:
            # Negation of an existing fact = invalidate it
            return ("invalidate", existing, "negation_of_existing")

    # --- Default: CREATE — need to decide what to do with existing memories ---
    existing = _find_existing_active(store, entity_name, operation.fact_type, operation.subject)

    if existing is not None:
        # Same content = duplicate → ignore
        if existing.content.lower().strip() == operation.content.lower().strip():
            return ("ignore", existing, "exact_duplicate")

        # Same subject, same value = duplicate (different wording) → ignore
        if existing.fact.value.lower().strip() == operation.value.lower().strip():
            return ("ignore", existing, "semantic_duplicate")

        # Same subject, different value
        if _values_contradict(existing.fact.value, operation.value):
            if operation.is_uncertain:
                return ("ignore", existing, "uncertain_contradiction")
            # For single-value facts: supersede
            # For multi-value facts: the values are different subjects/values
            # so this shouldn't happen via _find_existing_active (which matches subject)
            return ("supersede", existing, "value_change")

        # Compatible addition: same entity, related subject, non-contradictory
        if _is_compatible_addition(existing, operation.subject, operation.value, operation.domain):
            return ("merge", existing, "compatible_addition")

    # --- No exact subject match — check for same-fact-type contradictions ---
    # KEY DISTINCTION: Multi-value facts allow multiple active values.
    # Single-value facts require supersession on contradiction.
    if not _is_multi_value_fact(operation.fact_type, operation.domain):
        same_type = _find_same_fact_type(store, entity_name, operation.fact_type)
        if same_type and not operation.is_uncertain:
            if _values_contradict(same_type.fact.value, operation.value):
                return ("supersede", same_type, "fact_type_contradiction")
    else:
        # Multi-value fact: check if this EXACT value already exists
        all_same_type = _find_all_same_fact_type(store, entity_name, operation.fact_type)
        for mem in all_same_type:
            if not operation.is_uncertain:
                # Same value already exists = duplicate
                if mem.fact.value.lower().strip() == operation.value.lower().strip():
                    return ("ignore", mem, "multi_value_duplicate")
                # Same subject, same value = duplicate
                if (
                    _subjects_match(mem.fact.subject, operation.subject)
                    and mem.fact.value.lower().strip() == operation.value.lower().strip()
                ):
                    return ("ignore", mem, "multi_value_duplicate")
        # Multi-value: allow parallel active values (don't supersede)

    # Different subject entirely = new fact
    return ("create", None, "new_fact")


# ---------------------------------------------------------------------------
# Operation application
# ---------------------------------------------------------------------------

@dataclass
class OperationResult:
    """Result of applying a single operation."""

    memory_id: str
    action_taken: str
    event: MemoryEvent
    memory: Optional[Memory] = None
    receipt: Optional[DecisionReceipt] = None


def _create_memory_from_operation(
    operation: MemoryOperation,
    entity_ref: EntityRef,
    provenance_reason: str = "user_statement",
) -> Memory:
    """Create a new Memory from an operation."""
    new_id = generate_memory_id()
    importance = _determine_importance(operation)
    temporal = _resolve_temporal(not operation.is_temporary, operation.temporal_hint)

    # Determine lifecycle state: historical/temporary statements get INACTIVE state
    if operation.is_temporary or (temporal is not None and not temporal.is_persistent):
        lifecycle_state = LifecycleState.INACTIVE.value
    else:
        lifecycle_state = LifecycleState.ACTIVE.value

    return Memory(
        id=new_id,
        entity=entity_ref,
        fact=Fact(
            fact_type=operation.fact_type,
            subject=operation.subject,
            predicate=f"has_{operation.fact_type}",
            value=operation.value,
            content=operation.content,
            polarity=operation.polarity,
        ),
        domain=operation.domain,
        lifecycle=Lifecycle(state=lifecycle_state),
        provenance=Provenance(
            source=ProvenanceSource.USER_STATEMENT.value,
            reason=provenance_reason,
        ),
        temporal=temporal,
        confidence=operation.confidence,
        importance=importance,
    )


def apply_operation(
    operation: MemoryOperation,
    store: MemoryStore,
    entity_name: str = "self",
    entity_type: str = "self",
) -> Optional[OperationResult]:
    """Apply a single memory operation to the store with lifecycle decisions.

    Returns the result including the memory event and decision receipt produced.
    """
    now = _utc_now_iso()

    entity_ref = EntityRef(
        name=entity_name,
        entity_type=entity_type,
    )

    # Decide lifecycle action
    action, existing, reason = _decide_lifecycle_action(operation, store, entity_name)

    # Build decision receipt
    receipt = DecisionReceipt(
        candidate_content=operation.content,
        candidate_domain=operation.domain,
        candidate_fact_type=operation.fact_type,
        operation=action.upper(),
        target_memory_id=existing.id if existing else None,
        target_content=existing.content if existing else None,
        reason=reason,
        resulting_action=action,
        resulting_state="",  # filled below
        previous_state=existing.lifecycle.state if existing else None,
    )

    # --- DEACTIVATE ---
    if action == "deactivate":
        if existing:
            existing.lifecycle.deactivate()
            store.deactivate(existing.id)
            event = MemoryEvent(
                id=generate_event_id(),
                event_type=EventType.DEACTIVATED.value,
                memory_id=existing.id,
                trigger="user_statement",
                details={"content": operation.content, "reason": reason},
            )
            if hasattr(store, 'add_event'):
                store.add_event(event)
            receipt.resulting_state = "inactive"
            return OperationResult(
                memory_id=existing.id,
                action_taken="deactivated",
                event=event,
                memory=existing,
                receipt=receipt,
            )
        return None

    # --- INVALIDATE (mark as no longer valid, preserve provenance) ---
    if action == "invalidate":
        if existing:
            existing.lifecycle.deactivate()
            store.deactivate(existing.id)
            # Negation-based invalidation ("I don't want to learn Rust anymore.")
            # also stores the negative current-state fact so existence queries
            # ("Do I want to learn Rust?") can answer "No" from evidence
            # instead of falling back to unknown. The old positive fact stays
            # inactive for historical queries.
            if "negation" in reason and operation.polarity == "negative":
                negative_memory = _create_memory_from_operation(
                    operation, entity_ref, reason
                )
                negative_memory.lifecycle.negate()
                store.add(negative_memory)
            event = MemoryEvent(
                id=generate_event_id(),
                event_type=EventType.DEACTIVATED.value,
                memory_id=existing.id,
                trigger="user_statement",
                details={
                    "content": operation.content,
                    "reason": reason,
                    "invalidation": True,
                },
            )
            if hasattr(store, 'add_event'):
                store.add_event(event)
            receipt.resulting_state = "inactive"
            return OperationResult(
                memory_id=existing.id,
                action_taken="invalidated",
                event=event,
                memory=existing,
                receipt=receipt,
            )
        return None

    # --- IGNORE (duplicate / unchanged) ---
    if action == "ignore":
        receipt.resulting_state = "unchanged"
        return OperationResult(
            memory_id=existing.id if existing else "",
            action_taken="unchanged",
            event=MemoryEvent(
                id=generate_event_id(),
                event_type=EventType.UPDATED.value,
                memory_id=existing.id if existing else "",
                trigger="user_statement",
                details={"content": operation.content, "reason": reason},
            ),
            memory=existing,
            receipt=receipt,
        )

    # --- SUPERSEDE (value change / correction) ---
    if action == "supersede":
        if existing:
            old_id = existing.id
            existing.lifecycle.supersede("pending")

            new_memory = _create_memory_from_operation(operation, entity_ref, reason)
            new_memory.lifecycle.activate()
            store.add(new_memory)

            # Persist supersession of the old memory
            store.supersede(new_memory.id, old_id)

            # Fix supersession link
            existing.lifecycle.superseded_by = new_memory.id

            event = MemoryEvent(
                id=generate_event_id(),
                event_type=EventType.SUPERSEDED.value,
                memory_id=new_memory.id,
                trigger="user_statement",
                details={
                    "old_id": old_id,
                    "content": operation.content,
                    "reason": reason,
                },
            )
            if hasattr(store, 'add_event'):
                store.add_event(event)
            receipt.resulting_state = "active"
            receipt.target_memory_id = old_id
            return OperationResult(
                memory_id=new_memory.id,
                action_taken="superseded",
                event=event,
                memory=new_memory,
                receipt=receipt,
            )

    # --- MERGE (compatible addition) ---
    if action == "merge":
        new_memory = _create_memory_from_operation(operation, entity_ref, reason)
        new_memory.lifecycle.activate()
        store.add(new_memory)

        event = MemoryEvent(
            id=generate_event_id(),
            event_type=EventType.CREATED.value,
            memory_id=new_memory.id,
            trigger="user_statement",
            details={
                "content": operation.content,
                "reason": reason,
                "related_to": existing.id if existing else None,
            },
        )
        if hasattr(store, 'add_event'):
            store.add_event(event)
        receipt.resulting_state = "active"
        return OperationResult(
            memory_id=new_memory.id,
            action_taken="merged",
            event=event,
            memory=new_memory,
            receipt=receipt,
        )

    # --- CREATE (fresh fact) ---
    new_memory = _create_memory_from_operation(operation, entity_ref, reason)
    # Only activate if not a temporary/historical statement
    if not operation.is_temporary:
        new_memory.lifecycle.activate()
    store.add(new_memory)

    # A positive statement reverses an earlier negation of the same fact:
    # retire the NEGATED memory so both are not treated as current state.
    if new_memory.lifecycle.is_active:
        for mem in store.get_by_entity(entity_name):
            if (
                mem.lifecycle.state == LifecycleState.NEGATED.value
                and mem.id != new_memory.id
                and mem.fact.fact_type == operation.fact_type
                and _subjects_match(mem.fact.subject, operation.subject)
            ):
                mem.lifecycle.deactivate()
                store.deactivate(mem.id)

    event = MemoryEvent(
        id=generate_event_id(),
        event_type=EventType.CREATED.value,
        memory_id=new_memory.id,
        trigger="user_statement",
        details={"content": operation.content, "reason": reason},
    )
    if hasattr(store, 'add_event'):
        store.add_event(event)
    receipt.resulting_state = "active" if new_memory.lifecycle.is_active else "inactive"
    return OperationResult(
        memory_id=new_memory.id,
        action_taken="created",
        event=event,
        memory=new_memory,
        receipt=receipt,
    )
