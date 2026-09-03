"""Deterministic validation for memory values — rejects garbage, normalizes fields."""

from __future__ import annotations

import re
from typing import Optional


# Minimum meaningful lengths
MIN_VALUE_LENGTH = 1
MIN_SUBJECT_LENGTH = 1
MIN_CONTENT_LENGTH = 3
MAX_CONTENT_LENGTH = 500

# Characters that indicate garbage / non-meaningful input
GARBAGE_PATTERN = re.compile(r"^[^a-zA-Z0-9\s]+$")

# Temporal-only patterns (no substance)
TEMPORAL_ONLY_PATTERNS = [
    re.compile(r"^(?:again|now|currently|today|tomorrow|tonight|later)\s*$", re.IGNORECASE),
    re.compile(r"^(?:in\s+\d+\s+(?:days?|hours?|minutes?|weeks?))\s*$", re.IGNORECASE),
]

# Universal/vague values
VAGUE_VALUES = frozenset({
    "nothing", "something", "anything", "everything", "none",
    "noting",  # common misspelling
})

# Activity phrases that look like person names but aren't
ACTIVITY_PHRASES = re.compile(
    r"^(?:i(?:'m|\s+am)\s+)?(?:"
    r"learning|studying|trying|exploring|practicing|working|building|"
    r"creating|studying|reading|watching|playing|cooking|running|"
    r"walking|swimming|hiking|traveling|singing|dancing|painting|"
    r"writing|designing|coding|programming|debugging|testing"
    r")\b",
    re.IGNORECASE,
)

# Words that start activity-like patterns
ACTIVITY_STARTERS = frozenset({
    "learn", "learni", "learnin", "learned", "learns",
    "studi", "study", "studied", "studies", "studying",
    "try", "tried", "trying", "tries",
    "explo", "explore", "explored", "explores",
    "pract", "practice", "practiced", "practices",
    "work", "worked", "working", "works",
    "build", "built", "building", "builds",
    "creat", "create", "created", "creating", "creates",
    "read", "reading", "reads",
    "watch", "watching", "watches",
    "play", "playing", "plays",
    "cook", "cooking", "cooks",
    "run", "running", "runs",
    "walk", "walking", "walks",
    "swim", "swimming", "swims",
    "hik", "hike", "hiking", "hikes",
    "trav", "travel", "traveled", "traveling", "travels",
    "sing", "singing", "sings",
    "danc", "dance", "danced", "dancing", "dances",
    "paint", "painting", "paints",
    "writ", "write", "wrote", "writing", "writes",
    "desig", "design", "designed", "designing", "designs",
    "cod", "code", "coded", "coding", "codes",
    "program", "programmed", "programming", "programs",
    "debug", "debugged", "debugging",
    "test", "tested", "testing", "tests",
})

# Negative / negation patterns
NEGATION_PATTERNS = [
    re.compile(r"^(?:i\s+)?(?:do\s+not|don(?:'t|t))\s+(?:know|like|want|have|need|use|understand)\b", re.IGNORECASE),
    re.compile(r"^(?:i\s+)?(?:can(?:'t|t)|cannot)\s+(?:do|use|understand|know)\b", re.IGNORECASE),
]


def is_garbage(value: str) -> bool:
    """Check if a value is meaningless garbage (symbols only, too short, etc.)."""
    if not value or not isinstance(value, str):
        return True
    v = value.strip()
    if len(v) < MIN_VALUE_LENGTH:
        return True
    if GARBAGE_PATTERN.match(v):
        return True
    return False


def is_temporal_only(value: str) -> bool:
    """Check if a value contains only temporal words with no substance."""
    if not value or not isinstance(value, str):
        return True
    v = value.strip()
    return any(p.match(v) for p in TEMPORAL_ONLY_PATTERNS)


def is_vague(value: str) -> bool:
    """Check if a value is a universal/vague word."""
    if not value or not isinstance(value, str):
        return True
    return value.strip().lower() in VAGUE_VALUES


def validate_value(value: str, category: str = "other") -> tuple[bool, str]:
    """Validate a memory value. Returns (is_valid, reason)."""
    if not value or not isinstance(value, str):
        return False, "empty value"

    v = value.strip()

    if len(v) > MAX_CONTENT_LENGTH:
        return False, f"value exceeds {MAX_CONTENT_LENGTH} characters"

    if is_garbage(v):
        return False, "garbage value (symbols only or too short)"

    if is_vague(v):
        return False, "vague universal value"

    if is_temporal_only(v):
        return False, "temporal-only value with no substance"

    return True, ""


def validate_subject(subject: str, category: str = "other") -> tuple[bool, str]:
    """Validate a memory subject. Returns (is_valid, reason)."""
    if not subject or not isinstance(subject, str):
        return False, "empty subject"

    s = subject.strip()

    if len(s) < MIN_SUBJECT_LENGTH:
        return False, "subject too short"

    if is_garbage(s):
        return False, "garbage subject"

    if is_vague(s):
        return False, "vague universal subject"

    return True, ""


def validate_content(content: str) -> tuple[bool, str]:
    """Validate memory content. Returns (is_valid, reason)."""
    if not content or not isinstance(content, str):
        return False, "empty content"

    c = content.strip()

    if len(c) < MIN_CONTENT_LENGTH:
        return False, "content too short"

    if len(c) > MAX_CONTENT_LENGTH:
        return False, "content too long"

    return True, ""


def normalize_value(value: str) -> str:
    """Normalize a memory value by stripping whitespace and normalizing spaces."""
    if not value or not isinstance(value, str):
        return ""
    v = value.strip()
    v = re.sub(r"\s+", " ", v)
    return v


def normalize_subject(subject: str) -> str:
    """Normalize a memory subject by stripping, lowering, and cleaning."""
    if not subject or not isinstance(subject, str):
        return ""
    s = subject.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def is_activity_phrase(text: str) -> bool:
    """Check if text looks like an activity phrase, not a name."""
    if not text or not isinstance(text, str):
        return False
    t = text.strip().lower()
    if ACTIVITY_PHRASES.match(t):
        return True
    first_word = t.split()[0] if t else ""
    if first_word in ACTIVITY_STARTERS:
        return True
    return False


def validate_for_category(value: str, category: str, fact_type: str = "other") -> tuple[bool, str]:
    """Category-specific validation. Returns (is_valid, reason)."""
    if not value or not isinstance(value, str):
        return False, "empty value"

    v = value.strip()

    # Universal validation
    valid, reason = validate_value(v, category)
    if not valid:
        return valid, reason

    # Identity-specific: reject activity phrases
    if category == "identity":
        if is_activity_phrase(v):
            return False, "value looks like an activity phrase, not a name"
        if len(v.split()) > 3:
            return False, "identity value too long for a name"

    # Goal-specific: reject temporal-only
    if category == "goal":
        if is_temporal_only(v):
            return False, "goal value has no substance"

    return True, ""


class ValidationResult:
    """Result of a validation check."""
    
    __slots__ = ("is_valid", "reason", "normalized_value", "normalized_subject")
    
    def __init__(
        self,
        is_valid: bool,
        reason: str = "",
        normalized_value: str = "",
        normalized_subject: str = "",
    ):
        self.is_valid = is_valid
        self.reason = reason
        self.normalized_value = normalized_value
        self.normalized_subject = normalized_subject
    
    def __bool__(self) -> bool:
        return self.is_valid


def validate_memory_input(
    value: str,
    subject: str,
    category: str,
    fact_type: str = "other",
) -> ValidationResult:
    """Full validation of memory input. Returns ValidationResult."""
    # Validate value
    v_valid, v_reason = validate_for_category(value, category, fact_type)
    if not v_valid:
        return ValidationResult(is_valid=False, reason=v_reason)
    
    # Validate subject if provided
    if subject:
        s_valid, s_reason = validate_subject(subject, category)
        if not s_valid:
            return ValidationResult(is_valid=False, reason=s_reason)
    
    # Normalize
    norm_value = normalize_value(value)
    norm_subject = normalize_subject(subject) if subject else ""
    
    return ValidationResult(
        is_valid=True,
        normalized_value=norm_value,
        normalized_subject=norm_subject,
    )
