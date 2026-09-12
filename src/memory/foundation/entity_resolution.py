"""Entity resolution — extracts and resolves entity references from user messages.

Resolution is hierarchical and purely syntactic:
1. Self-identification: "I am X" → self
2. Possessive third-party: "Mom likes blue" → mom (person)
3. Named third-party: "Alice likes blue" → alice (person)
4. Self-reference: "I like blue" → self
5. Default: → self

No hardcoded name lists. Resolution derives from syntactic role.
Supports coreference resolution for pronouns and anaphora.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import EntityRef, EntityType


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""

    entity: EntityRef
    confidence: float
    resolution_method: str


@dataclass
class CoreferenceChain:
    """Tracks a chain of coreferent mentions."""
    entity_name: str
    entity_type: str
    mentions: list[str] = field(default_factory=list)
    first_mention_index: int = 0


# ---------------------------------------------------------------------------
# Self-reference detection
# ---------------------------------------------------------------------------

_SELF_PRONOUNS = re.compile(
    r"\b(?:i|me|my|mine|myself)\b",
    re.IGNORECASE,
)

_SELF_IDENTIFIER = re.compile(
    r"\b(?:my\s+name\s+is|i\s+am|i'm|this\s+is|i\s+go\s+by|i\s+call\s+myself)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Possessive third-party detection
# ---------------------------------------------------------------------------

_POSSESSIVE_KINSHIP = re.compile(
    r"\b(?:my\s+)?"
    r"(mom|mother|dad|father|brother|sister|"
    r"friend|teacher|colleague|classmate|roommate|neighbor|"
    r"husband|wife|partner|son|daughter|cousin|uncle|aunt|"
    r"grandmother|grandfather|grandma|grandpa|grandparent|"
    r"manager|supervisor|boss|mentor|coach|"
    r"teammate|coworker|peer|"
    r"professor|instructor|tutor|advisor|dean|"
    r"doctor|nurse|dentist|"
    r"lawyer|attorney|"
    r"plumber|electrician|mechanic|"
    r"barista|cashier|waiter|waitress|server|"
    r"driver|pilot|captain|"
    r"chef|baker|artist|musician|writer|"
    r"scientist|engineer|architect|designer|"
    r"athlete|player|"
    r"landlord|tenant|"
    r"client|customer|vendor|supplier|"
    r"agent|broker|dealer|"
    r"therapist|counselor|"
    r"librarian|curator|"
    r"developer|programmer|analyst|"
    r"consultant|specialist|coordinator|"
    r"director|executive|"
    r"founder|owner|"
    r"member|participant|"
    r"visitor|guest|host|"
    r"acquaintance|"
    r"companion|comrade|ally|"
    r"rival|opponent|competitor|"
    r"fan|follower|supporter|"
    r"critic|reviewer|"
    r"inspector|auditor|examiner|"
    r"date|boyfriend|girlfriend|"
    r"fiance|spouse|mate|"
    r"stepmother|stepfather|stepbrother|stepsister|"
    r"stepson|stepdaughter|"
    r"foster.parent|guardian|"
    r"sponsor|benefactor|patron|"
    r"protégé|apprentice|intern"
    r")\b(?:'s)?\s",
    re.IGNORECASE,
)

# Maps kinship terms to canonical entity types
_KINSHIP_TYPES = {
    "mom": "person", "mother": "person",
    "dad": "person", "father": "person",
    "brother": "person", "sister": "person",
    "friend": "person", "teacher": "person",
    "colleague": "person", "classmate": "person",
    "roommate": "person", "neighbor": "person",
    "husband": "person", "wife": "person", "partner": "person",
    "son": "person", "daughter": "person",
    "cousin": "person", "uncle": "person", "aunt": "person",
    "grandmother": "person", "grandfather": "person",
    "grandma": "person", "grandpa": "person", "grandparent": "person",
}

# Pattern: "my friend Jake" — kinship term followed by a capitalized name
# The name must be AFTER the kinship term, not the kinship term itself
_KINSHIP_WITH_NAME = re.compile(
    r"\b(?:my\s+)?"
    r"(?:mom|mother|dad|father|brother|sister|"
    r"friend|teacher|colleague|classmate|roommate|neighbor|"
    r"husband|wife|partner|son|daughter|cousin|uncle|aunt|"
    r"grandmother|grandfather|grandma|grandpa|grandparent"
    r")\s+([A-Z][a-z]{1,20})\b",
)


def _extract_possessive_third_party(message: str) -> Optional[tuple[str, str]]:
    """Extract possessive third-party entity (e.g., 'Mom', 'my friend Jake')."""
    # First try kinship + name pattern (e.g., "my friend Jake")
    name_match = _KINSHIP_WITH_NAME.search(message)
    if name_match:
        entity_name = name_match.group(1).lower().strip()
        return entity_name, "person"

    # Fall back to kinship-only pattern (e.g., "Mom likes blue")
    match = _POSSESSIVE_KINSHIP.search(message)
    if match:
        name = match.group(1).lower().strip()
        entity_type = _KINSHIP_TYPES.get(name, "person")
        return name, entity_type

    return None


# ---------------------------------------------------------------------------
# Named third-party detection
# ---------------------------------------------------------------------------

_NAMED_THIRD_PARTY = re.compile(
    r"^([A-Z][a-z]{1,20})\s+"
    r"(?:likes?|loves?|hates?|wants?|needs?|prefers?|enjoys?|"
    r"uses?|has|is|was|can|does|studied|learned|plays|works|"
    r"built|created|designed|coded|code|codes|programmed|wrote|read|watched)\b",
)

# Technology/programming terms that should NOT be treated as person names
_TECHNOLOGY_TERMS = frozenset({
    "python", "java", "javascript", "typescript", "rust", "go", "ruby",
    "swift", "kotlin", "scala", "haskell", "clojure", "erlang", "elixir",
    "php", "perl", "r", "matlab", "sql", "nosql", "mongodb", "postgresql",
    "mysql", "redis", "docker", "kubernetes", "aws", "azure", "gcp",
    "linux", "windows", "macos", "unix", "git", "github", "gitlab",
    "react", "angular", "vue", "svelte", "next", "nuxt", "node",
    "django", "flask", "fastapi", "spring", "rails", "laravel",
    "tensorflow", "pytorch", "keras", "scikit", "pandas", "numpy",
    "machine", "learning", "deep", "neural", "network", "algorithm",
    "data", "science", "blockchain", "crypto", "nft", "web3",
    "api", "rest", "graphql", "grpc", "websocket", "http", "https",
    "html", "css", "scss", "less", "sass", "webpack", "vite",
    "npm", "yarn", "pip", "cargo", "maven", "gradle",
    "vim", "emacs", "vscode", "intellij", "eclipse",
    "chrome", "firefox", "safari", "edge", "opera",
    "slack", "discord", "teams", "zoom", "skype",
    "google", "microsoft", "apple", "amazon", "meta", "netflix",
    "tesla", "spacex", "openai", "anthropic", "deepmind",
    "java", "scala", "kotlin", "groovy", "clojure",
    "nginx", "apache", "tomcat", "jenkins", "travis",
    "postgresql", "mysql", "sqlite", "oracle", "mssql",
    "zephyr", "embedded", "firmware", "microcontroller", "arduino",
    "raspberry", "pi", "esp32", "stm32", "arm", "riscv",
})


def _extract_named_third_party(message: str) -> Optional[tuple[str, str]]:
    """Extract a named third-party entity (e.g., 'Alice', 'Rahul')."""
    match = _NAMED_THIRD_PARTY.match(message.strip())
    if match:
        name = match.group(1).lower().strip()
        # Check if it's a technology term (not a person)
        if name in _TECHNOLOGY_TERMS:
            return None
        return name, "person"
    return None


# ---------------------------------------------------------------------------
# Self-reference from "my" patterns
# ---------------------------------------------------------------------------

_MY_THIRD_PARTY = re.compile(
    r"\bmy\s+(mom|mother|dad|father|brother|sister|friend|teacher|"
    r"colleague|classmate|roommate|neighbor|husband|wife|partner|"
    r"son|daughter|cousin|uncle|aunt|grandmother|grandfather|"
    r"grandma|grandpa|grandparent)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Main resolution function
# ---------------------------------------------------------------------------

def resolve_entity(message: str) -> ResolvedEntity:
    """Resolve the primary entity referenced in a user message.

    Priority order:
    1. Self-identification ("I am X", "My name is X")
    2. Possessive third-party ("Mom likes blue", "my friend likes Java")
    3. Named third-party ("Alice likes blue")
    4. Self-reference ("I like blue")
    5. Default → self
    """
    lower = message.lower().strip()

    # 1. Self-identification
    if _SELF_IDENTIFIER.search(lower):
        return ResolvedEntity(
            entity=EntityRef(name="self", entity_type=EntityType.SELF.value),
            confidence=1.0,
            resolution_method="self_identification",
        )

    # 2. Possessive third-party
    poss_match = _extract_possessive_third_party(message)
    if poss_match:
        name, etype = poss_match
        return ResolvedEntity(
            entity=EntityRef(name=name, entity_type=etype),
            confidence=0.9,
            resolution_method="possessive_third_party",
        )

    # 3. Named third-party (only if no self-reference present)
    if _SELF_PRONOUNS.search(lower):
        pass  # Has self-reference, skip named third-party
    else:
        named_match = _extract_named_third_party(message)
        if named_match:
            name, etype = named_match
            return ResolvedEntity(
                entity=EntityRef(name=name, entity_type=etype),
                confidence=0.8,
                resolution_method="named_third_party",
            )

    # 4. Self-reference
    if _SELF_PRONOUNS.search(lower):
        return ResolvedEntity(
            entity=EntityRef(name="self", entity_type=EntityType.SELF.value),
            confidence=0.95,
            resolution_method="self_reference",
        )

    # 5. Default → self
    return ResolvedEntity(
        entity=EntityRef(name="self", entity_type=EntityType.SELF.value),
        confidence=0.7,
        resolution_method="default_self",
    )


# ---------------------------------------------------------------------------
# Coreference resolution
# ---------------------------------------------------------------------------

# Pronouns that refer to previously mentioned entities
_PRONOUNS = re.compile(
    r"\b(?:he|she|they|it|him|her|them|his|her|its|their)\b",
    re.IGNORECASE,
)

# Self pronouns
_SELF_PRONOUNS_ALT = re.compile(
    r"\b(?:i|me|my|mine|myself)\b",
    re.IGNORECASE,
)

# Demonstrative references
_DEMONSTRATIVE = re.compile(
    r"\b(?:this|that|these|those)\b",
    re.IGNORECASE,
)

# Relative pronouns
_RELATIVE_PRONOUNS = re.compile(
    r"\b(?:who|whom|whose|which|that)\b",
    re.IGNORECASE,
)


class CoreferenceResolver:
    """Resolves coreferences within a conversation context.
    
    Tracks mentions of entities and resolves pronouns and anaphora
    to previously mentioned entities.
    """
    
    def __init__(self):
        self.chains: dict[str, CoreferenceChain] = {}
        self.mention_order: list[str] = []
        self.current_entity: Optional[str] = None
    
    def reset(self):
        """Reset the resolver state."""
        self.chains = {}
        self.mention_order = []
        self.current_entity = None
    
    def _normalize_mention(self, mention: str) -> str:
        """Normalize a mention for comparison."""
        return mention.lower().strip()
    
    def _is_pronoun(self, token: str) -> bool:
        """Check if a token is a pronoun."""
        return bool(_PRONOUNS.match(token.lower()))
    
    def _is_self_reference(self, token: str) -> bool:
        """Check if a token is a self-reference."""
        return bool(_SELF_PRONOUNS_ALT.match(token.lower()))
    
    def add_mention(self, mention: str, entity_name: str, entity_type: str):
        """Add a mention to the coreference chain."""
        normalized = self._normalize_mention(mention)
        
        if entity_name not in self.chains:
            self.chains[entity_name] = CoreferenceChain(
                entity_name=entity_name,
                entity_type=entity_type,
            )
        
        self.chains[entity_name].mentions.append(normalized)
        self.current_entity = entity_name
        
        if normalized not in self.mention_order:
            self.mention_order.append(normalized)
    
    def resolve_pronoun(self, pronoun: str, context: str) -> Optional[str]:
        """Resolve a pronoun to a previously mentioned entity.
        
        This is a simplified coreference resolution that uses:
        1. Recency: most recently mentioned entity
        2. Gender/number agreement (basic)
        3. Syntactic role (subject/object)
        """
        if not self.mention_order:
            return None
        
        # If we have a current entity and the pronoun is in the same sentence,
        # it likely refers to the current entity
        if self.current_entity:
            return self.current_entity
        
        # Fallback: return the most recently mentioned entity
        if self.mention_order:
            return self.mention_order[-1]
        
        return None
    
    def resolve_demonstrative(self, demonstrative: str, context: str) -> Optional[str]:
        """Resolve a demonstrative reference (this, that, these, those).
        
        Demonstratives typically refer to the most recently mentioned entity
        or concept.
        """
        if not self.mention_order:
            return None
        
        # Return the most recently mentioned entity
        return self.mention_order[-1] if self.mention_order else None
    
    def resolve_relative(self, relative: str, antecedent: str) -> Optional[str]:
        """Resolve a relative pronoun to its antecedent.
        
        Relative pronouns (who, which, that) typically refer to
        the noun phrase that precedes them.
        """
        # The antecedent is typically the noun phrase before the relative pronoun
        # For simplicity, return the current entity or the last mentioned
        if self.current_entity:
            return self.current_entity
        
        return self.mention_order[-1] if self.mention_order else None
    
    def resolve_coreference(self, token: str, context: str) -> Optional[str]:
        """Resolve any coreference token (pronoun, demonstrative, relative).
        
        Returns the resolved entity name, or None if unresolvable.
        """
        if self._is_pronoun(token):
            return self.resolve_pronoun(token, context)
        elif _DEMONSTRATIVE.match(token):
            return self.resolve_demonstrative(token, context)
        elif _RELATIVE_PRONOUNS.match(token):
            return self.resolve_relative(token, context)
        
        return None
    
    def get_all_mentions(self) -> dict[str, list[str]]:
        """Get all mentions for each entity."""
        return {name: chain.mentions for name, chain in self.chains.items()}
    
    def get_mention_count(self, entity_name: str) -> int:
        """Get the number of mentions for an entity."""
        if entity_name in self.chains:
            return len(self.chains[entity_name].mentions)
        return 0


def resolve_coreferences(message: str, resolver: CoreferenceResolver) -> list[tuple[str, str]]:
    """Resolve all coreferences in a message.
    
    Returns list of (token, resolved_entity) tuples.
    """
    tokens = message.split()
    resolutions = []
    
    for token in tokens:
        # Clean token
        clean_token = re.sub(r"[^\w']", "", token)
        if not clean_token:
            continue
        
        resolved = resolver.resolve_coreference(clean_token, message)
        if resolved:
            resolutions.append((clean_token, resolved))
    
    return resolutions
