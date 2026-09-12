"""Message understanding — classifies user messages for the write pipeline.

Determines message type, detects temporary/uncertain/third-party/correction
signals, extracts temporal hints, and distinguishes personal vs general content.
Purely syntactic — no LLM required.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class MessageType(str, Enum):
    STATEMENT = "statement"
    QUESTION = "question"
    COMMAND = "command"
    GREETING = "greeting"


class ContentScope(str, Enum):
    """Whether content is personal or general knowledge."""
    PERSONAL = "personal"
    GENERAL = "general"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass
class MessageUnderstanding:
    """Structured understanding of a user message."""

    message_type: MessageType = MessageType.STATEMENT
    is_temporary: bool = False
    is_uncertain: bool = False
    is_third_party: bool = False
    is_correction: bool = False
    is_negation: bool = False
    is_deletion: bool = False
    is_personal: bool = True
    is_general: bool = False
    temporal_hint: Optional[str] = None
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Lexical cues — deterministic, fast, no LLM
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(?:hi|hello|hey|sup|what'?s\s+up|good\s+(?:morning|afternoon|evening|night)"
    r"|howdy|greetings|how(?:'s| is) it going|what'?s\s+new)"
    r"(?:\s+there)?\s*[!.]?\s*$",
    re.IGNORECASE,
)

# "yo" only when it's the entire message or followed by punctuation
_GREETING_YO = re.compile(r"^yo\s*[!.]?\s*$", re.IGNORECASE)

_QUESTION_SUFFIX = re.compile(r"\?\s*$")
_QUESTION_STARTERS = re.compile(
    r"^(?:what|who|where|when|why|how|do|did|can|could|would|should|is|are|was|were|tell|give|show)\b",
    re.IGNORECASE,
)

_COMMAND_PATTERNS = re.compile(
    r"^(?:remember|forget|stop|cancel|delete|remove|drop|clear|reset|don'?t)\b",
    re.IGNORECASE,
)

# Historical statements that start with "When" but are not questions
_HISTORICAL_STATEMENT = re.compile(
    r"^when\s+(?:i\s+was|i\s+had|i\s+used\s+to|i\s+studied|i\s+learned|i\s+worked)\b",
    re.IGNORECASE,
)


def classify_message_type(message: str) -> MessageType:
    """Classify message as statement, question, command, or greeting."""
    stripped = message.strip()

    if _GREETING_PATTERNS.match(stripped) or _GREETING_YO.match(stripped):
        return MessageType.GREETING

    if _QUESTION_SUFFIX.search(stripped):
        return MessageType.QUESTION

    # Historical statements starting with "When" are not questions
    if _HISTORICAL_STATEMENT.match(stripped):
        return MessageType.STATEMENT

    if _QUESTION_STARTERS.match(stripped):
        return MessageType.QUESTION

    if _COMMAND_PATTERNS.match(stripped):
        return MessageType.COMMAND

    return MessageType.STATEMENT


# ---------------------------------------------------------------------------
# Temporary statement detection
# ---------------------------------------------------------------------------

_TEMPORAL_ADVERBS = frozenset({
    "temporarily", "for now", "for today", "just today", "for the moment",
    "tomorrow", "tonight", "later", "just now",
    "this week", "this month", "next week", "next month",
})

_TEMPORAL_PATTERNS = (
    re.compile(
        r"\b(?:tomorrow|tommorrow|tonight|today|later|right\s+now|"
        r"at\s+the\s+moment|this\s+(?:week|month|day)|"
        r"next\s+(?:week|month|day|monday|tuesday|wednesday|"
        r"thursday|friday|saturday|sunday))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:until|till|through|for)\s+(?:tomorrow|today|tonight|"
        r"this\s+(?:week|month)|next\s+(?:week|month))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:have|has|had|am|is|are)\b.*?\b(?:exam|test|quiz|deadline|"
        r"appointment|interview|meeting)\b.*?\b(?:tomorrow|today|tonight|"
        r"next|this)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i(?:'m|\s+am)\s+)?(?:preparing|studying|getting\s+ready)\s+"
        r"(?:for|to)\s+\w+.*?\b(?:tomorrow|today|tonight|this\s+week)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bused\s+to\b",
        re.IGNORECASE,
    ),
)


def detect_temporary(message: str) -> tuple[bool, Optional[str]]:
    """Detect if message expresses temporary state and extract temporal hint."""
    lower = message.lower()

    for phrase in _TEMPORAL_ADVERBS:
        if phrase in lower:
            return True, phrase

    for pattern in _TEMPORAL_PATTERNS:
        match = pattern.search(lower)
        if match:
            return True, match.group(0).strip()

    return False, None


# ---------------------------------------------------------------------------
# Uncertainty detection
# ---------------------------------------------------------------------------

_UNCERTAINTY_SIGNALS = re.compile(
    r"\b(?:i\s+think|i\s+believe|i\s+feel\s+like|i\s+feel\s+that|"
    r"maybe\s+i|perhaps\s+i|possibly\s+i|i\s+might|i\s+could|"
    r"i(?:'m|\s+am)\s+considering|i\s+am\s+considering|"
    r"not\s+sure\s+(?:if|i)|i(?:'m|\s+am)\s+not\s+sure\s+(?:if|i)|"
    r"i\s+am\s+not\s+sure\s+(?:if|i)|"
    r"probably|most\s+likely|not\s+certain|not\s+sure)\b",
    re.IGNORECASE,
)

_UNCERTAINTY_PERSONAL = re.compile(
    r"\b(?:i\s+think|i\s+believe|i\s+feel\s+like)\s+"
    r"(?:i\s+)?(?:want|should|would|could|might|plan|am\s+going)\b",
    re.IGNORECASE,
)


def detect_uncertain(message: str) -> bool:
    """Detect if message expresses uncertainty about a personal intention."""
    return bool(_UNCERTAINTY_SIGNALS.search(message))


# ---------------------------------------------------------------------------
# Third-party detection
# ---------------------------------------------------------------------------

_POSSESSIVE_THIRD_PARTY = re.compile(
    r"\b(?:my\s+)?(?:mom|mother|dad|father|brother|sister|"
    r"friend|teacher|colleague|classmate|roommate|neighbor|"
    r"husband|wife|partner|son|daughter|cousin|uncle|aunt|"
    r"grandmother|grandfather|grandma|grandpa|grandparent)\b",
    re.IGNORECASE,
)

_NAMED_THIRD_PARTY = re.compile(
    r"^([A-Z][a-z]{1,20})\s+(?:likes?|loves?|hates?|wants?|needs?|"
    r"prefers?|enjoys?|uses?|has|is|was|can|does)\b",
)


def detect_third_party(message: str) -> bool:
    """Detect if message is about someone other than the user."""
    lower = message.lower().strip()

    if _POSSESSIVE_THIRD_PARTY.search(lower):
        return True

    if _NAMED_THIRD_PARTY.match(message.strip()):
        return True

    third_person = re.match(
        r"^(?:he|she|they|someone|my\s+\w+)\s+"
        r"(?:likes?|loves?|wants?|needs?|prefers?|enjoys?|uses?|has|is|was)\b",
        lower,
    )
    return bool(third_person)


# ---------------------------------------------------------------------------
# Correction / change signal detection
# ---------------------------------------------------------------------------

_CORRECTION_PHRASES = (
    "actually", "in fact", "instead of", "i no longer",
    "i don't want to", "i do not want to", "i don't like",
    "i do not like", "i no longer want", "i no longer use",
    "i switched to", "i switched from",
    "i changed to", "i changed from",
    "i changed my",
    "forget that", "forget my", "remove my", "delete my",
    "no longer", "anymore", "used to",
    "replaced with", "replaced my", "stop remembering",
    "i am no longer", "i'm no longer",
    "is not my", "is no longer my", "is not the", "is no longer the",
    "are not my", "are no longer my",
)


def detect_correction(message: str) -> bool:
    """Detect if message contains an explicit correction or change signal.

    Normalizes apostrophes so both "don't" and "dont" match.
    """
    lower = message.lower().replace("'", "")
    return any(phrase.replace("'", "") in lower for phrase in _CORRECTION_PHRASES)


# ---------------------------------------------------------------------------
# Negation detection
# ---------------------------------------------------------------------------

_NEGATION_PATTERNS = re.compile(
    r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|didn'?t|did\s+not|"
    r"can'?t|cannot|never|no\s+longer|not)\b",
    re.IGNORECASE,
)


def detect_negation(message: str) -> bool:
    """Detect if message contains a negation signal."""
    return bool(_NEGATION_PATTERNS.search(message))


# ---------------------------------------------------------------------------
# Deletion detection
# ---------------------------------------------------------------------------

_DELETION_PATTERNS = re.compile(
    r"\b(?:forget\s+(?:about\s+|that\s+|the\s+)?\S+|remove\s+(?:my|me|that|\w+)|"
    r"delete\s+(?:my|me|that|\w+)|stop\s+remembering|i\s+no\s+longer\s+"
    r"want\s+you\s+to\s+remember)\b",
    re.IGNORECASE,
)


def detect_deletion(message: str) -> bool:
    """Detect if message requests memory deletion."""
    return bool(_DELETION_PATTERNS.search(message))


# ---------------------------------------------------------------------------
# Personal vs General detection
# ---------------------------------------------------------------------------

# Personal pronouns that indicate personal content
# Includes first-person singular (i, me, my) and first-person plural (we, us, our)
_PERSONAL_PRONOUNS = re.compile(
    r"\b(?:i|me|my|mine|myself|i'm|i\s+am|i\s+have|i\s+like|i\s+want|i\s+use|"
    r"we|us|our|ours|ourselves|we're|we've|we'll)\b",
    re.IGNORECASE,
)

# General knowledge question patterns
_GENERAL_KNOWLEDGE_PATTERNS = re.compile(
    r"^(?:what\s+is|what\s+are|who\s+is|who\s+was|where\s+is|when\s+was|"
    r"how\s+does|how\s+do|how\s+to|"
    r"why\s+(?:is|are|was|were|do|does|did|can|could|would|should|has|have|had)|"
    r"who\s+(?:invented|discovered|created|wrote|painted|built|designed|made|founded|started|developed|coined|originated)|"
    r"how\s+much\s+(?:does|do|did|is|are|was|were|has|have|had)|"
    r"can\s+you\s+explain|"
    r"tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\b",
    re.IGNORECASE,
)

# Personal question patterns
_PERSONAL_QUESTION_PATTERNS = re.compile(
    r"\b(?:my|mine|me|myself|i\s+am|i'm|i\s+have|i\s+like|i\s+want|i\s+use|"
    r"we|us|our|ours|ourselves)\b",
    re.IGNORECASE,
)

# General knowledge topics (not exhaustive, but covers common cases)
_GENERAL_TOPICS = re.compile(
    r"\b(?:what\s+is|what\s+are|who\s+is|who\s+was|where\s+is|when\s+was|"
    r"how\s+does|how\s+do|how\s+to|"
    r"python|java|javascript|c\+\+|programming|language|"
    r"math|science|history|geography|physics|chemistry|biology|"
    r"algorithm|data\s+structure|database|network|internet|"
    r"computer|software|hardware|"
    r"president|country|capital|city|river|mountain|"
    r"element|compound|reaction|equation|"
    r"war|revolution|empire|civilization|"
    r"speed|light|gravity|mass|energy|force|power|voltage|"
    r"temperature|pressure|density|velocity|acceleration|"
    r"atom|molecule|electron|proton|neutron|"
    r"planet|star|galaxy|universe|earth|sun|moon|"
    r"language|grammar|vocabulary|syntax|"
    r"algebra|geometry|calculus|statistics|"
    r"music|art|literature|poetry|novel|"
    r"food|nutrition|diet|health|disease|"
    r"economy|market|trade|business|finance)\b",
    re.IGNORECASE,
)


def detect_personal_vs_general(message: str) -> tuple[bool, bool]:
    """Detect if message is personal or general knowledge.
    
    Uses a generalized decision hierarchy:
    1. Personal pronouns → PERSONAL
    2. General knowledge question pattern → GENERAL
    3. Factual world statement (article/noun + verb) → GENERAL
    4. No clear signal → UNKNOWN (neither personal nor general)
    
    Note: General topics alone are NOT sufficient to classify as GENERAL.
    A statement must have structural signals (factual pattern, question pattern)
    OR personal pronouns to be classified. Otherwise it's UNKNOWN.
    
    Returns (is_personal, is_general).
    """
    lower = message.lower().strip()
    
    # Check for personal pronouns (includes we/us/our)
    has_personal = bool(_PERSONAL_PRONOUNS.search(lower))
    
    # Check for general knowledge question patterns
    has_general_pattern = bool(_GENERAL_KNOWLEDGE_PATTERNS.match(lower))
    
    # Check for factual statements about the world (not personal)
    # Pattern 1: Article + noun phrase + verb
    # "The X is/has/was Y" with article is typically factual
    # (The Earth is round, The sky is blue, A triangle has three sides)
    # Excludes casual observations with temporal markers (today, now, currently)
    has_temporal_marker = bool(re.search(
        r"\b(?:today|now|currently|right\s+now|at\s+the\s+moment|these\s+days|lately|recently)\b",
        lower,
    ))
    factual_world_article = re.match(
        r"^(?:the|a|an)\s+(?:\w+\s+){0,5}(?:"
        r"is|are|was|were|has|have|had|"
        r"orbits?|rotates?|contains?|produces?|emits?|absorbs?|"
        r"measures?|weighs?|costs?|equals?|consists?|"
        r"travels?|moves?|flows?|exists?|occurs?|"
        r"boils?|freezes?|melts?|evaporates?|condenses?"
        r")\s+",
        lower,
    ) and not has_temporal_marker
    
    # Pattern 2: Scientific/technical noun + verb (Water boils, Light travels, etc.)
    # This catches factual statements that don't start with articles
    factual_world_noun = re.match(
        r"^(?:water|light|sound|heat|electricity|magnetism|gravity|"
        r"oxygen|carbon|hydrogen|nitrogen|"
        r"photosynthesis|evaporation|condensation|fermentation|"
        r"evolution|gravity|inertia|friction|"
        r"democracy|capitalism|socialism|"
        r"geometry|algebra|calculus|statistics|"
        r"phonetics|syntax|semantics|morphology|"
        r"python|java|javascript|c\+\+|ruby|swift|kotlin|"
        r"html|css|sql|nosql|"
        r"linux|windows|macos|unix|"
        r"docker|kubernetes|aws|azure|gcp|"
        r"tensorflow|pytorch|keras|"
        r"react|angular|vue|svelte)\s+"
        r"(?:is|are|was|were|has|have|had|"
        r"travels?|moves?|flows?|exists?|occurs?|"
        r"boils?|freezes?|melts?|evaporates?|condenses?|"
        r"was\s+(?:created|discovered|invented|built|designed|developed|founded|established)\s+)",
        lower,
    )
    
    # Decision hierarchy (generalized, not sentence-specific):
    
    # 1. Personal pronouns present → PERSONAL
    if has_personal:
        return True, False
    
    # 2. General knowledge question pattern → GENERAL
    if has_general_pattern:
        return False, True
    
    # 3. Factual world statement (starts with article or scientific noun, has verb) → GENERAL
    if factual_world_article or factual_world_noun:
        return False, True
    
    # 4. No clear signal → UNKNOWN (neither personal nor general)
    # General topics alone are NOT sufficient — they need structural signals.
    # This avoids misclassifying casual observations like "The weather is nice today"
    # or opinions like "Music is enjoyable" as general knowledge.
    # The pipeline will validate later.
    return False, False


# ---------------------------------------------------------------------------
# Main understanding function
# ---------------------------------------------------------------------------

def understand_message(message: str) -> MessageUnderstanding:
    """Produce a structured understanding of a user message.

    This is the first stage of the write pipeline. It classifies the
    message and detects semantic signals without extracting facts.
    """
    msg_type = classify_message_type(message)
    is_temp, temp_hint = detect_temporary(message)
    is_uncertain = detect_uncertain(message)
    is_third = detect_third_party(message)
    is_correction = detect_correction(message)
    is_negation = detect_negation(message)
    is_deletion = detect_deletion(message)
    is_personal, is_general = detect_personal_vs_general(message)

    return MessageUnderstanding(
        message_type=msg_type,
        is_temporary=is_temp,
        is_uncertain=is_uncertain,
        is_third_party=is_third,
        is_correction=is_correction,
        is_negation=is_negation,
        is_deletion=is_deletion,
        is_personal=is_personal,
        is_general=is_general,
        temporal_hint=temp_hint,
        confidence=1.0,
    )
