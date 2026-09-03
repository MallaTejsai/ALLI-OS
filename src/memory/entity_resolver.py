"""Generic entity resolution — extracts entities from user messages.

No hardcoded people. Entities are resolved by syntactic role, not by name.
The system learns entity names from user messages and maps them to entity records.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """A resolved entity (person, organization, concept, or self)."""
    name: str
    entity_type: str  # "self", "person", "organization", "concept"
    id: Optional[str] = None
    relation_to_user: Optional[str] = None  # "self", "parent", "sibling", etc.

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.entity_type,
            "id": self.id,
            "relation_to_user": self.relation_to_user,
        }


# Patterns that indicate the speaker is referring to themselves
_SELF_PATTERNS = [
    re.compile(r"^i\b", re.IGNORECASE),
    re.compile(r"^my\b", re.IGNORECASE),
    re.compile(r"^i'm\b", re.IGNORECASE),
    re.compile(r"^i am\b", re.IGNORECASE),
    re.compile(r"^myself\b", re.IGNORECASE),
]

# Patterns that indicate a possessive entity reference
# "Mom's favourite colour", "My friend's name", "The teacher said"
_POSSESSIVE_PATTERNS = [
    # "Mom's favourite colour" → entity="mom", relation derived from name
    re.compile(r"^(\w+)'s\b", re.IGNORECASE),
    # "My friend's name" → entity="friend", relation="friend"
    re.compile(r"^my\s+(\w+)'s\b", re.IGNORECASE),
    # "The teacher's opinion" → entity="teacher", relation="teacher"
    re.compile(r"^the\s+(\w+)'s\b", re.IGNORECASE),
]

# Patterns that indicate "I am <name>" self-identification
_SELF_IDENTIFICATION_PATTERNS = [
    re.compile(r"^(?:i am|i'm)\s+([A-Za-z][A-Za-z .'-]{0,40})$", re.IGNORECASE),
    re.compile(r"^my\s+name(?:'s)?\s+is\s+([A-Za-z][A-Za-z .'-]{0,40})$", re.IGNORECASE),
]

# Common relationship words that can be derived from entity names
_RELATIONSHIP_KEYWORDS = {
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


def _derive_relation_from_name(name: str) -> Optional[str]:
    """Derive relationship from entity name if it's a common relationship word."""
    lower = name.lower().strip()
    return _RELATIONSHIP_KEYWORDS.get(lower)


def _extract_possessive_entity(text: str) -> Optional[tuple[str, str]]:
    """Extract entity from possessive patterns like 'Mom's favourite colour'.

    Returns (entity_name, rest_of_sentence) or None.
    """
    text = text.strip()
    for pattern in _POSSESSIVE_PATTERNS:
        match = pattern.match(text)
        if match:
            entity_name = match.group(1).strip().lower()
            # Get the rest after the possessive
            rest = text[match.end():].strip()
            return (entity_name, rest)
    return None


def _is_self_reference(text: str) -> bool:
    """Check if text refers to the speaker (user)."""
    text = text.strip()
    return any(p.match(text) for p in _SELF_PATTERNS)


def _extract_self_identification(text: str) -> Optional[str]:
    """Extract name from 'I am <name>' or 'My name is <name>'."""
    text = text.strip()
    for pattern in _SELF_IDENTIFICATION_PATTERNS:
        match = pattern.match(text)
        if match:
            return match.group(1).strip(" .!?")
    return None


@dataclass
class EntityExtraction:
    """Result of entity extraction from a user message."""
    entities: list[Entity] = field(default_factory=list)
    is_self: bool = True
    entity_name: str = "user"
    entity_type: str = "self"
    relation_to_user: Optional[str] = "self"

    def to_dict(self) -> dict:
        return {
            "name": self.entity_name,
            "type": self.entity_type,
            "id": None,
            "relation_to_user": self.relation_to_user,
        }


def extract_entity(text: str) -> EntityExtraction:
    """Extract the primary entity from a user message.

    This is generic — no hardcoded names. It resolves entities by:
    1. Possessive patterns ("Mom's favourite colour")
    2. Self-reference patterns ("I like Python", "My name is X")
    3. Default to self if no entity detected

    Returns EntityExtraction with entity info.
    """
    text = text.strip()
    if not text:
        return EntityExtraction()

    # Check for possessive entity: "Mom's favourite colour", "My friend's name"
    possessive = _extract_possessive_entity(text)
    if possessive:
        entity_name, rest = possessive
        relation = _derive_relation_from_name(entity_name)
        # Possessive entities are usually people (names), not concepts
        entity_type = "person"
        return EntityExtraction(
            entities=[Entity(name=entity_name, entity_type=entity_type, relation_to_user=relation)],
            is_self=False,
            entity_name=entity_name,
            entity_type=entity_type,
            relation_to_user=relation,
        )

    # Check for self-identification: "I am Tejsai", "My name is Tejsai"
    self_name = _extract_self_identification(text)
    if self_name:
        return EntityExtraction(
            entities=[Entity(name="user", entity_type="self", relation_to_user="self")],
            is_self=True,
            entity_name="user",
            entity_type="self",
            relation_to_user="self",
        )

    # Check for self-reference: "I like Python", "My goal is..."
    if _is_self_reference(text):
        return EntityExtraction(
            entities=[Entity(name="user", entity_type="self", relation_to_user="self")],
            is_self=True,
            entity_name="user",
            entity_type="self",
            relation_to_user="self",
        )

    # Default: assume self if no entity detected
    return EntityExtraction(
        entities=[Entity(name="user", entity_type="self", relation_to_user="self")],
        is_self=True,
        entity_name="user",
        entity_type="self",
        relation_to_user="self",
    )


def extract_entities_from_clause(clause: str) -> EntityExtraction:
    """Extract entity from a single clause (may be part of compound statement).

    This is the per-clause entity extraction used by the structured extractor.
    """
    return extract_entity(clause)
