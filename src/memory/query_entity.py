"""Query-side entity resolution for ALLI v0.6 Phase 5.

Extracts entity information from natural language questions so retrieval
can be entity-scoped.  Resolves self-references (my, I, me) and
possessive references (mom's, my friend's) generically.

No hardcoded people.  Entity names are extracted by syntactic role.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass
class QueryEntity:
    """Entity identified in a query."""

    name: str  # canonical entity name: "user", "mom", "dad", etc.
    entity_type: str  # "self", "person", "organization", "concept"
    relation_to_user: Optional[str] = None
    is_unresolved: bool = False  # True when entity cannot be determined

    @property
    def is_self(self) -> bool:
        return self.entity_type == "self"

    @property
    def is_resolved(self) -> bool:
        return not self.is_unresolved


def resolve_query_entity(query: str) -> QueryEntity:
    """Extract the primary entity referenced in a query.

    Resolution priority:
      1. Possessive third-party: "mom's favourite colour" → entity=mom
      2. "my <person>'s": "my friend's name" → entity=friend
      3. Self-references: "my favourite colour", "what do I like" → entity=user
      4. Pronoun references: "what is their favourite colour" → unresolved
      5. Unresolvable → unresolved

    Returns QueryEntity with name and type.
    """
    text = (query or "").strip()
    if not text:
        return QueryEntity(name="user", entity_type="self")

    lower = text.lower()

    # --- Third-party possessive: "mom's favourite colour" ---
    # Match "<name>'s" at the start or after "what is/are"
    entity = _extract_possessive_entity(lower)
    if entity is not None:
        return entity

    # --- "my <person>'s": "my friend's name" ---
    entity = _extract_my_person_entity(lower)
    if entity is not None:
        return entity

    # --- Self-references: "my ...", "I ...", "me ..." ---
    if _is_self_reference_query(lower):
        return QueryEntity(name="user", entity_type="self")

    # --- Unresolvable pronouns ---
    if _is_unresolved_pronoun(lower):
        return QueryEntity(
            name="unknown",
            entity_type="unknown",
            is_unresolved=True,
        )

    # --- Default: self if the query is clearly personal, else unresolved ---
    if _looks_personal(lower):
        return QueryEntity(name="user", entity_type="self")

    return QueryEntity(
        name="unknown",
        entity_type="unknown",
        is_unresolved=True,
    )


# ---------------------------------------------------------------------------
# Entity extraction patterns (generic, no hardcoded names)
# ---------------------------------------------------------------------------

# Relationship words that can appear after "my" to identify a person
_RELATIONSHIP_WORDS = {
    "mom", "mother", "dad", "father", "brother", "sister",
    "friend", "teacher", "professor", "doctor", "boss", "manager",
    "colleague", "classmate", "roommate", "neighbor",
    "cousin", "uncle", "aunt", "grandma", "grandfather",
    "grandpa", "grandmother", "wife", "husband",
    "girlfriend", "boyfriend", "partner",
}


def _extract_possessive_entity(lower: str) -> Optional[QueryEntity]:
    """Extract entity from '<name>'s ...' patterns.

    Examples:
      "mom's favourite colour" → QueryEntity("mom", "person", "parent")
      "the teacher's opinion" → QueryEntity("teacher", "person", "teacher")
      "my friend's name" is handled by _extract_my_person_entity
    """
    # Match "<word>'s" — the word is the entity name
    match = re.search(r"\b(\w+)'s\b", lower)
    if not match:
        return None

    entity_name = match.group(1).strip()

    # Skip self-reference possessives (handled elsewhere)
    if entity_name in {"my", "i", "me", "your", "you", "our", "we", "who", "what", "where", "when", "why", "how", "it", "that", "this"}:
        return None

    # Derive relationship from common words
    relation = _RELATIONSHIP_MAP.get(entity_name)
    entity_type = "person"

    return QueryEntity(
        name=entity_name,
        entity_type=entity_type,
        relation_to_user=relation,
    )


def _extract_my_person_entity(lower: str) -> Optional[QueryEntity]:
    """Extract entity from 'my <person>'s ...' or 'what does my <person> ...' patterns.

    Examples:
      "my mom's favourite colour" → QueryEntity("mom", "person", "parent")
      "my friend's name" → QueryEntity("friend", "person", "friend")
      "what does my mom like" → QueryEntity("mom", "person", "parent")
    """
    match = re.search(r"\bmy\s+(\w+)(?:'s)?\b", lower)
    if not match:
        return None

    entity_name = match.group(1).strip()

    # Skip non-person words
    if entity_name in {"own", "ownself", "self", "favorite", "favourite", "best", "goal", "goals", "name", "nickname", "skill", "learning", "education", "preference", "interest", "project", "task", "priority", "ambition", "dream", "habit", "phone", "laptop", "device"}:
        return None

    # If it's a known relationship word, assume it's an entity
    if entity_name in _RELATIONSHIP_WORDS:
        relation = _RELATIONSHIP_MAP.get(entity_name)
        entity_type = "person"
        return QueryEntity(
            name=entity_name,
            entity_type=entity_type,
            relation_to_user=relation,
        )

    # Otherwise, require it to be possessive or part of a "does my X" pattern
    # so we don't accidentally match "my car" as a person entity if car is unknown
    if "'s" in match.group(0) or re.search(r"\b(?:does|is|did|has)\s+my\s+" + re.escape(entity_name) + r"\b", lower):
        relation = _RELATIONSHIP_MAP.get(entity_name)
        return QueryEntity(
            name=entity_name,
            entity_type="person",
            relation_to_user=relation,
        )

    return None


def _is_self_reference_query(lower: str) -> bool:
    """Check if the query references the user themselves."""
    # Since third-party and my-person extraction run BEFORE this,
    # if we see self-reference pronouns here, it means the primary
    # entity is likely the user (e.g., "what phone do I have", "what is my favorite color").
    if re.search(r"\b(?:i|me|my|mine|myself)\b", lower):
        return True

    # "who am i", "tell me about me", etc. are covered by the above.
    
    # What if no explicit pronoun but implied? "what is the user's name"
    if re.search(r"\bthe\s+user(?:'s)?\b", lower):
        return True
        
    return False


def _is_unresolved_pronoun(lower: str) -> bool:
    """Check if the query uses an unresolvable pronoun."""
    unresolved_starts = (
        r"^their\s+",
        r"^his\s+",
        r"^her\s+",
        r"^what\s+is\s+their\b",
        r"^what\s+are\s+their\b",
        r"^what\s+does\s+they\b",
        r"^what\s+do\s+they\b",
    )
    for pattern in unresolved_starts:
        if re.search(pattern, lower):
            return True
    return False


def _looks_personal(lower: str) -> bool:
    """Heuristic: does this query look like a personal memory question?"""
    personal_cues = (
        r"\bmy\s+(?:goal|goals?|name|nickname|skill|learning|education|"
        r"preference|interest|project|task|priority|ambition|dream|habit)\b",
        r"\bwhat\s+(?:do|does)\s+(?:i|the user)\s+(?:like|dislike|enjoy|hate|know|want|need|have)\b",
        r"\bwhat\s+am\s+i\s+(?:learning|studying|doing|building|creating)\b",
        r"\bfavou?rite\b",
        r"\babout\s+(?:me|myself)\b",
        r"\bwhat\s+do\s+you\s+know\s+about\s+(?:me|myself)\b",
        r"\bwhat\s+do\s+you\s+remember\s+about\s+(?:me|myself)\b",
    )
    for pattern in personal_cues:
        if re.search(pattern, lower):
            return True
    return False


# Relationship derivation (generic, same as write side)
_RELATIONSHIP_MAP: dict[str, str] = {
    "mom": "parent",
    "mother": "parent",
    "dad": "parent",
    "father": "parent",
    "brother": "sibling",
    "sister": "sibling",
    "friend": "friend",
    "teacher": "teacher",
    "professor": "teacher",
    "doctor": "doctor",
    "boss": "manager",
    "manager": "manager",
    "colleague": "colleague",
    "classmate": "classmate",
    "roommate": "roommate",
    "neighbor": "neighbor",
    "cousin": "relative",
    "uncle": "relative",
    "aunt": "relative",
    "grandma": "relative",
    "grandfather": "relative",
    "grandpa": "relative",
    "grandmother": "relative",
    "wife": "spouse",
    "husband": "spouse",
    "girlfriend": "partner",
    "boyfriend": "partner",
    "partner": "partner",
}
