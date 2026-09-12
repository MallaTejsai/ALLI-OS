"""Generic fact extraction — produces structured operations from user messages.

Extracts facts using pattern-based analysis. Handles multiple domains,
paraphrases, grammar mistakes, slang, and multiple facts per message.

The extraction layer produces structured operations — it never directly
mutates storage. The LLM is used when available; this module provides
the deterministic fallback and schema definitions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Protocol, runtime_checkable

from .understanding import MessageUnderstanding, MessageType


# ---------------------------------------------------------------------------
# Operation representation
# ---------------------------------------------------------------------------

class Action(str):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    MERGE = "MERGE"
    IGNORE = "IGNORE"
    DEACTIVATE = "DEACTIVATE"
    INVALIDATE = "INVALIDATE"
    CANCEL = "CANCEL"


@dataclass
class MemoryOperation:
    """A single memory operation to be applied by the pipeline."""

    action: str
    domain: str
    fact_type: str
    content: str
    subject: str
    value: str
    polarity: Optional[str] = None
    importance: str = "medium"
    confidence: float = 1.0
    entity_name: str = "self"
    entity_type: str = "self"
    temporal_hint: Optional[str] = None
    is_temporary: bool = False
    is_uncertain: bool = False


@dataclass
class ExtractionResult:
    """Result of fact extraction from a user message."""

    operations: list[MemoryOperation]
    message_type: str = "statement"
    is_temporary: bool = False
    is_uncertain: bool = False
    is_third_party: bool = False
    is_correction: bool = False
    is_deletion: bool = False
    temporal_hint: Optional[str] = None
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Structural completeness check
# ---------------------------------------------------------------------------

class CompletenessLevel(str):
    """Completeness level for extracted operations."""
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"
    AMBIGUOUS = "ambiguous"
    GENERAL = "general"


# Vague/placeholder values that indicate incomplete extraction
_VAGUE_SUBJECTS = frozenset({
    "unknown", "it", "that", "this", "something", "anything",
    "nothing", "everything", "none", "stuff", "things",
})

# Minimal subject length (in characters) to be considered meaningful
_MIN_SUBJECT_LENGTH = 3

# Minimal content length (in words) to be considered meaningful
_MIN_CONTENT_WORDS = 2


def check_completeness(operation: MemoryOperation) -> str:
    """Check structural completeness of an extracted memory operation.
    
    Operates on the structured representation, not exact sentences.
    Returns CompletenessLevel indicating whether the operation is ready
    for memory admission.
    
    Levels:
    - COMPLETE: Has meaningful subject and content, ready for admission
    - INCOMPLETE: Missing required semantic information (e.g., "I like" with no object)
    - AMBIGUOUS: Too vague to determine meaning (e.g., "It is nice")
    - GENERAL: General knowledge, not personal memory
    """
    subject = (operation.subject or "").strip().lower()
    content = (operation.content or "").strip()
    value = (operation.value or "").strip()
    
    # Check 1: Content must exist and have meaningful length
    if not content or len(content.strip()) < 2:
        return CompletenessLevel.INCOMPLETE
    
    # Check 2: Content must have at least 2 words (subject + something)
    content_words = content.split()
    if len(content_words) < _MIN_CONTENT_WORDS:
        return CompletenessLevel.INCOMPLETE
    
    # Check 3: Subject must be meaningful (not just "it", "that", "unknown")
    if subject in _VAGUE_SUBJECTS:
        return CompletenessLevel.AMBIGUOUS
    
    # Check 3b: Subject must not be a pronoun + adjective/verb pattern
    # These indicate vague/ambiguous statements: "it nice", "that good", "they coming"
    pronoun_pattern = re.match(
        r"^(?:it|that|this|these|those|they|he|she|we)\s+"
        r"(?:nice|good|bad|great|fine|okay|ok|done|coming|going|here|there|"
        r"right|wrong|true|false|possible|impossible|necessary|important|"
        r"interesting|boring|funny|sad|happy|angry|scared|afraid|worried|"
        r"like|love|hate|enjoy|prefer|want|need|use|have|had|has|is|are|was|were)$",
        subject,
    )
    if pronoun_pattern:
        return CompletenessLevel.AMBIGUOUS
    
    # Check 4: Subject must have minimum length
    if len(subject) < _MIN_SUBJECT_LENGTH:
        return CompletenessLevel.INCOMPLETE
    
    # Check 5: Domain-specific completeness
    if operation.domain == "interests":
        # Preference must have an object (subject should not be just a verb phrase)
        # "I like" → subject="i like" (incomplete)
        # "My favorite is" → subject="favorite" (incomplete)
        # "I love my mom" → subject="mom" (complete)
        verb_only_subjects = frozenset({
            "like", "likes", "loved", "love", "loves", "enjoy", "enjoys",
            "hate", "hates", "dislike", "dislikes", "prefer", "prefers",
            "i like", "i love", "i enjoy", "i hate", "i prefer",
            "favorite", "favourite", "best", "worst",
        })
        if subject in verb_only_subjects:
            return CompletenessLevel.INCOMPLETE
        # Also check if subject is too long (indicates extraction failed)
        if len(subject.split()) > 4:
            return CompletenessLevel.INCOMPLETE
    
    if operation.domain == "goals":
        # Goal must have a target (not just "want to")
        if subject in ("to", "a", "an", "the"):
            return CompletenessLevel.INCOMPLETE
        # Check for incomplete goal phrases
        incomplete_goals = frozenset({
            "want to", "wanna", "going to", "plan to", "hope to",
            "i want to", "i'm going to", "i plan to",
        })
        if subject in incomplete_goals:
            return CompletenessLevel.INCOMPLETE
    
    if operation.domain == "identity":
        # Identity must have a name/role
        if len(subject.split()) < 1:
            return CompletenessLevel.INCOMPLETE
    
    return CompletenessLevel.COMPLETE


def domain_is_general(operation: MemoryOperation) -> bool:
    """Check if operation domain is general (not specific)."""
    return operation.domain == "general"


# ---------------------------------------------------------------------------
# Extractor protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FactExtractor(Protocol):
    """Protocol for fact extraction implementations."""

    def extract(self, message: str, understanding: MessageUnderstanding) -> ExtractionResult:
        ...


# ---------------------------------------------------------------------------
# Domain classification — generic, pattern-based
# ---------------------------------------------------------------------------

_LEARNING_VERBS = re.compile(
    r"\b(?:learn(?:ing|ed|s)?|stud(?:y|ying|ied|ies|ied)|pick(?:ing)?\s+up|"
    r"picked\s+up|studying|practicing|practising|exploring|"
    r"getting\s+into|getting\s+into|"
    r"spending\s+(?:\w+\s+)*time\s+with)\b",
    re.IGNORECASE,
)

_PREFERENCE_VERBS = re.compile(
    r"\b(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed|"
    r"hate|hates|hated|dislike|dislikes|disliked|"
    r"prefer|prefers|preferred|favor|favors|favored|"
    r"favorite|favourite|best|adore|adores|into)\b",
    re.IGNORECASE,
)

_GOAL_VERBS = re.compile(
    r"\b(?:want(?:ing)?|wanna|aim(?:ing)?|plan(?:ning)?|"
    r"hope(?:ful(?:ly)?)?|aspire|aspiring|target|targeting|"
    r"goal|strive|striving|wish(?:ing)?|desire|desiring)\b",
    re.IGNORECASE,
)

_POSSESSION_VERBS = re.compile(
    r"\b(?:have|has|had|own|owns|owned|use|uses|used|"
    r"bought|bought|acquired|got|getting|"
    r"switch(?:ed|ing)?|chang(?:ed|ing)?|upgrad(?:ed|ing)?|"
    r"replaced?)\b",
    re.IGNORECASE,
)

_IDENTITY_VERBS = re.compile(
    r"\b(?:my\s+name\s+is|i\s+am|i'm|this\s+is|i\s+go\s+by|"
    r"call\s+me|i\s+work\s+(?:as|at|for)|i\s+study\s+(?:at|at)|i\s+live\s+(?:in|at))\b",
    re.IGNORECASE,
)

_SKILL_VERBS = re.compile(
    r"\b(?:can|know|knows|skilled|proficient|experienced|"
    r"good\s+at|adept|expert|familiar\s+with)\b",
    re.IGNORECASE,
)


def _classify_domain(message: str) -> str:
    """Classify the domain of a message based on verb patterns."""
    lower = message.lower()

    if _LEARNING_VERBS.search(lower):
        return "learning"
    if _PREFERENCE_VERBS.search(lower):
        return "interests"
    if _GOAL_VERBS.search(lower):
        return "goals"
    if _POSSESSION_VERBS.search(lower):
        return "possessions"
    if _IDENTITY_VERBS.search(lower):
        return "identity"
    if _SKILL_VERBS.search(lower):
        return "knowledge"
    return "general"


# ---------------------------------------------------------------------------
# Subject extraction — generic, from the message
# ---------------------------------------------------------------------------

def _extract_subject_from_message(message: str, domain: str) -> str:
    """Extract the normalized subject/topic from a message."""
    lower = message.lower().strip()

    # Remove common prefixes/fillers
    filler_pattern = re.compile(
        r"^(?:i(?:'m|\s+am)\s+)?"
        r"(?:currently\s+|just\s+|also\s+|really\s+|very\s+|"
        r"so\s+|actually\s+|basically\s+|like\s+)?",
        re.IGNORECASE,
    )
    cleaned = filler_pattern.sub("", lower).strip()

    if domain == "learning":
        match = re.search(
            r"(?:learn(?:ing|ed)?|study(?:ing|ied)?|pick(?:ing)?\s+up|picked\s+up)\s+(?:\w+\s+)?(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return _normalize_subject(match.group(1))

    if domain == "interests":
        # Handle "X is something I like" pattern
        something_match = re.search(
            r"^(\w+)\s+is\s+something\s+(?:i\s+)?(?:like|enjoy|love|want)",
            cleaned,
            re.IGNORECASE,
        )
        if something_match:
            return _normalize_subject(something_match.group(1))

        # Handle "X is my favorite Y" pattern (e.g., "Python is my favorite language")
        fav_match = re.search(
            r"(\w+)\s+is\s+my\s+favorite",
            cleaned,
            re.IGNORECASE,
        )
        if fav_match:
            return _normalize_subject(fav_match.group(1))

        # Handle "my favorite X is Y" pattern (e.g., "my favorite language is Python")
        # Extracts "favorite X" without the value Y, so the subject is stable
        # when the value changes (e.g., Python → Rust).
        # Uses (.+?) to handle multi-word X (e.g., "programming language").
        # Also handles possessive forms via optional (?:\w+(?:'s)?\s+)?.
        my_fav_match = re.search(
            r"my\s+(?:\w+(?:'s)?\s+)?favorite\s+(.+?)\s+is\s+",
            cleaned,
            re.IGNORECASE,
        )
        if my_fav_match:
            return _normalize_subject(f"favorite {my_fav_match.group(1)}")

        # Handle "spending time with X" pattern
        spend_match = re.search(
            r"spending\s+time\s+with\s+(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next|very|really|so|a\s+lot|more|less|lately)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if spend_match:
            return _normalize_subject(spend_match.group(1))

        match = re.search(
            r"(?:like|likes|loved?|enjoy|enjoys|hate|hates|dislike|dislikes|prefer|prefers)\s+(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next|very|really|so|a\s+lot|more|less|again)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return _normalize_subject(match.group(1))

    if domain == "goals":
        match = re.search(
            r"(?:want(?:ing)?|wanna|aim(?:ing)?|plan(?:ning)?|hope(?:ful(?:ly)?)?)\s+(?:to\s+)?(?:be\s+)?(?:a\s+)?(?:become\s+)?(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next|very|really|so)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return _normalize_subject(match.group(1))

    if domain == "possessions":
        # Handle "switched to X" / "changed to X" / "upgraded to X" / "replaced X with Y"
        switch_match = re.search(
            r"(?:switch(?:ed|ing)?|chang(?:ed|ing)?|upgrad(?:ed|ing)?)\s+to\s+(?:a\s+|an\s+|my\s+)?(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if switch_match:
            return _normalize_subject(switch_match.group(1))

        # Handle "replaced X with Y" — extract Y (the new item)
        replace_match = re.search(
            r"replaced?\s+(?:my\s+|a\s+|an\s+)?(.+?)\s+with\s+(?:a\s+|an\s+|my\s+)?(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if replace_match:
            return _normalize_subject(replace_match.group(2))

        match = re.search(
            r"(?:have|has|own|use|used|bought|got)\s+(?:a\s+|an\s+|my\s+)?(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return _normalize_subject(match.group(1))

    if domain == "identity":
        match = re.search(
            r"(?:my\s+name\s+is|i\s+am|i'm|call\s+me|i\s+work\s+as)\s+(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            cleaned,
            re.IGNORECASE,
        )
        if match:
            return _normalize_subject(match.group(1))

    # Fallback: first significant words
    words = re.findall(r"[a-z]+", cleaned)
    stop = {"i", "me", "my", "am", "is", "are", "was", "were", "a", "an",
            "the", "to", "for", "in", "on", "at", "with", "and", "or",
            "but", "not", "do", "does", "did", "have", "has", "had",
            "like", "likes", "love", "hate", "want", "learning", "studying"}
    significant = [w for w in words if w not in stop and len(w) > 1]
    if significant:
        return _normalize_subject(" ".join(significant[:3]))

    return _normalize_subject(cleaned)


def _normalize_subject(text: str) -> str:
    """Normalize a subject string for consistent matching."""
    text = text.lower().strip()
    # Remove articles and common fillers
    text = re.sub(r"\b(?:a|an|the|my|your|his|her|its|our|their)\b", "", text)
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Remove trailing punctuation
    text = re.sub(r"[.,!?;:]+$", "", text)
    return text if text else "unknown"


# ---------------------------------------------------------------------------
# Value extraction
# ---------------------------------------------------------------------------

def _extract_value(message: str, subject: str, domain: str) -> str:
    """Extract the compact value from the message."""
    lower = message.lower().strip()

    # For preference: the subject IS the value
    if domain == "interests":
        # Special case: "my favorite X is Y" — extract Y as the value
        # Uses .+? to handle multi-word X (e.g., "programming language").
        # Also handles possessive forms via optional (?:\w+(?:'s)?\s+)?.
        my_fav_match = re.search(
            r"my\s+(?:\w+(?:'s)?\s+)?favorite\s+.+?\s+is\s+(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            lower,
            re.IGNORECASE,
        )
        if my_fav_match:
            return my_fav_match.group(1).strip().title()
        return subject.title() if subject else message

    # For identity: extract the name/role
    if domain == "identity":
        match = re.search(
            r"(?:my\s+name\s+is|i\s+am|i'm|call\s+me)\s+(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            lower,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().title()
        return subject.title()

    # For possessions: the subject is the value
    if domain == "possessions":
        return subject.title() if subject else message

    # For goals: extract the goal description
    if domain == "goals":
        match = re.search(
            r"(?:want(?:ing)?|wanna|aim(?:ing)?|plan(?:ning)?|hope(?:ful(?:ly)?)?)\s+(?:to\s+)?(?:be\s+)?(?:a\s+)?(?:become\s+)?(.+?)(?:\s+(?:now|currently|today|tomorrow|later|right|this|next)\b|$)",
            lower,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip().title()
        return subject.title()

    # Default: subject as value
    return subject.title() if subject else message.strip()


# ---------------------------------------------------------------------------
# Polarity extraction
# ---------------------------------------------------------------------------

_POSITIVE_SIGNALS = re.compile(
    r"\b(?:like|likes|liked|love|loves|loved|enjoy|enjoys|enjoyed|"
    r"prefer|prefers|preferred|favorite|favourite|best|great|"
    r"awesome|amazing|wonderful|fantastic|excellent|good|nice)\b",
    re.IGNORECASE,
)

_NEGATIVE_SIGNALS = re.compile(
    r"\b(?:hate|hates|hated|dislike|dislikes|disliked|"
    r"don'?t\s+like|do\s+not\s+like|can'?t\s+stand|"
    r"worst|terrible|awful|bad|horrible|ugly|boring|"
    r"bored\s+of|sick\s+of|tired\s+of|done\s+with)\b",
    re.IGNORECASE,
)

_NEGATION_INFIX = re.compile(
    r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|didn'?t|did\s+not|"
    r"never|no\s+longer|not)\b",
    re.IGNORECASE,
)


def _extract_polarity(message: str) -> Optional[str]:
    """Extract positive or negative polarity from a message."""
    lower = message.lower()

    # Negation without any positive signal = standalone negative
    if _NEGATION_INFIX.search(lower) and not _POSITIVE_SIGNALS.search(lower):
        return "negative"

    # Check for negation before positive signals
    if _NEGATION_INFIX.search(lower) and _POSITIVE_SIGNALS.search(lower):
        return "negative"

    if _NEGATIVE_SIGNALS.search(lower):
        return "negative"

    if _POSITIVE_SIGNALS.search(lower):
        return "positive"

    return None


# ---------------------------------------------------------------------------
# Confidence extraction
# ---------------------------------------------------------------------------

def _extract_confidence(message: str, is_uncertain: bool) -> float:
    """Extract confidence level from message."""
    if is_uncertain:
        return 0.3

    lower = message.lower()

    # Low confidence signals
    if re.search(r"\b(?:maybe|perhaps|possibly|probably|not\s+sure|might)\b", lower):
        return 0.5

    # High confidence signals
    if re.search(r"\b(?:definitely|certainly|absolutely|always|never\s+not|"
                 r"100%|for\s+sure|of\s+course)\b", lower):
        return 0.95

    return 0.8


# ---------------------------------------------------------------------------
# Fact type extraction
# ---------------------------------------------------------------------------

def _infer_fact_type(domain: str, message: str) -> str:
    """Infer the fact type from domain and message content."""
    lower = message.lower()

    if domain == "learning":
        if re.search(r"\b(?:course|class|degree|diploma)\b", lower):
            return "course"
        return "learning"

    if domain == "interests":
        if re.search(r"\b(?:hobby|hobbies|pastime)\b", lower):
            return "hobby"
        return "preference"

    if domain == "goals":
        if re.search(r"\b(?:career|job|profession)\b", lower):
            return "career_goal"
        if re.search(r"\b(?:project|app|website|software)\b", lower):
            return "project_goal"
        return "goal"

    if domain == "possessions":
        return "owned_item"

    if domain == "identity":
        if re.search(r"\b(?:name|nickname)\b", lower):
            return "name"
        if re.search(r"\b(?:work|job|role|position)\b", lower):
            return "occupation"
        return "identity"

    if domain == "knowledge":
        return "skill"

    return "general"


# ---------------------------------------------------------------------------
# Multiple facts detection
# ---------------------------------------------------------------------------

_SPLIT_CONJUNCTIONS = re.compile(
    r"\b(?:and|also|plus|as\s+well\s+as|along\s+with|together\s+with)\b",
    re.IGNORECASE,
)


def _extract_deletion_target(message: str) -> Optional[str]:
    """Extract the subject being deleted from a deletion message.

    E.g., 'Forget tea' → 'tea', 'Remove my coffee preference' → 'coffee'
          'Forget about Jake's hiking' → 'jake hiking'
    """
    lower = message.lower().strip()
    # Remove deletion triggers
    cleaned = re.sub(
        r"\b(?:forget|remove|delete|stop\s+remembering|clear)\s+(?:about\s+|that\s+|my\s+|the\s+)?",
        "",
        lower,
    ).strip()
    # Remove trailing temporal/filler words
    cleaned = re.sub(
        r"\s+(?:now|anymore|any\s+more|no\s+more|please|ok|okay|from\s+now|from\s+today)\s*$",
        "",
        cleaned,
    ).strip()
    if cleaned:
        return _normalize_subject(cleaned)
    return None


def _extract_correction_subject(message: str, domain: str) -> Optional[str]:
    """Extract the new subject/value from a correction message.

    E.g., 'Actually I prefer espresso now' → 'espresso'
          'I don't like coffee anymore' → 'coffee'
          'I switched to Samsung S24' → 'samsung s24'
          'Actually my name is Bob' → 'bob'
          'I replaced my ThinkPad with a Macbook' → 'macbook'
          'I changed my phone from iPhone to Samsung' → 'samsung'
          'I switched from Python to Java' → 'java'
    """
    lower = message.lower().strip()

    # For identity domain, extract just the name from "X is my name" / "my name is X"
    if domain == "identity":
        name_match = re.search(
            r"(?:my\s+name\s+is|call\s+me|i(?:'m|\s+am)|this\s+is)\s+(.+?)$",
            lower,
            re.IGNORECASE,
        )
        if name_match:
            return _normalize_subject(name_match.group(1))

    # Handle "replaced X with Y" pattern — extract Y (the new value)
    replaced_match = re.search(
        r"replaced\s+(?:my\s+|our\s+)?(?:the\s+)?(.+?)\s+with\s+(?:a\s+|an\s+|the\s+)?(.+?)$",
        lower,
        re.IGNORECASE,
    )
    if replaced_match:
        return _normalize_subject(replaced_match.group(2))

    # Handle "changed X from A to B" pattern — extract B (the new value)
    changed_from_match = re.search(
        r"changed\s+(?:my\s+|our\s+)?(?:the\s+)?(.+?)\s+from\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(?:a\s+|an\s+|the\s+)?(.+?)$",
        lower,
        re.IGNORECASE,
    )
    if changed_from_match:
        return _normalize_subject(changed_from_match.group(3))

    # Handle "switched from A to B" pattern — extract B (the new value)
    switched_from_match = re.search(
        r"switched\s+from\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+to\s+(?:a\s+|an\s+|the\s+)?(.+?)$",
        lower,
        re.IGNORECASE,
    )
    if switched_from_match:
        return _normalize_subject(switched_from_match.group(2))

    # Handle "changed to Y from X" / "switched to Y from X" pattern — extract Y (the new value)
    switched_to_from_match = re.search(
        r"(?:switched|changed)\s+to\s+(?:a\s+|an\s+|the\s+)?(.+?)\s+from\s+",
        lower,
        re.IGNORECASE,
    )
    if switched_to_from_match:
        return _normalize_subject(switched_to_from_match.group(1))

    # Handle "changed to Y" / "switched to Y" pattern — extract Y (the new value)
    switched_to_match = re.search(
        r"(?:switched|changed)\s+to\s+(?:a\s+|an\s+|the\s+)?(.+?)$",
        lower,
        re.IGNORECASE,
    )
    if switched_to_match:
        return _normalize_subject(switched_to_match.group(1))

    # Handle "X is not my Y" / "X is no longer my Y" reversal pattern
    # Extract "Y X" to match original subject format ("my Y is X" → subject "Y X")
    reversal_match = re.search(
        r"(.+?)\s+(?:is|are)\s+(?:not|no\s+longer)\s+(?:my|the|our)\s+(.+?)(?:\s+(?:anymore|any\s+more|no\s+more|now|currently))?\s*$",
        lower,
        re.IGNORECASE,
    )
    if reversal_match:
        entity_part = reversal_match.group(1).strip()
        attribute_part = reversal_match.group(2).strip()
        return _normalize_subject(f"{attribute_part} {entity_part}")

    # Remove correction/negation prefixes for remaining patterns
    cleaned = re.sub(
        r"\b(?:actually|in\s+fact|instead|rather|but\s+now|no\s+longer|"
        r"i\s+switched\s+to|i\s+changed\s+to|i\s+changed\s+from|"
        r"replaced\s+with|don'?t|doesn'?t|didn'?t|can'?t|never|not|i(?:'m|\s+am)\s+no\s+longer)\s+"
        r"(?:i\s+|we\s+)?",
        "",
        lower,
    ).strip()

    # Remove trailing temporal/filler words
    cleaned = re.sub(
        r"\s+(?:now|anymore|any\s+more|no\s+more|instead|rather|better|more|less|right|currently|today|tomorrow)\s*$",
        "",
        cleaned,
    ).strip()

    if not cleaned:
        return None

    # Extract noun after preference/like verb
    match = re.search(
        r"(?:like|prefer|enjoy|hate|dislike|want|use|bought|got|switched\s+to|changed\s+to)\s+"
        r"(?:a\s+|an\s+|the\s+|my\s+)?(.+?)$",
        cleaned,
        re.IGNORECASE,
    )
    if match:
        return _normalize_subject(match.group(1))

    # Fallback: use cleaned content
    return _normalize_subject(cleaned)


def _split_multiple_facts(message: str) -> list[str]:
    """Split a message into multiple independent fact statements."""
    # Simple split on conjunctions if the message has multiple clauses
    if _SPLIT_CONJUNCTIONS.search(message):
        parts = _SPLIT_CONJUNCTIONS.split(message)
        return [p.strip() for p in parts if p.strip() and len(p.strip()) > 2]
    return [message]


# ---------------------------------------------------------------------------
# Enhanced temporal interpretation
# ---------------------------------------------------------------------------

# Historical signals
_HISTORICAL_SIGNALS = re.compile(
    r"\b(?:used\s+to|was|were|before|previously|formerly|"
    r"in\s+the\s+past|back\s+when|earlier|last\s+year|"
    r"when\s+i\s+was|when\s+i\s+had|when\s+i\s+studied|"
    r"when\s+i\s+learned|when\s+i\s+worked)\b",
    re.IGNORECASE,
)

# Current signals
_CURRENT_SIGNALS = re.compile(
    r"\b(?:currently|now|right\s+now|at\s+the\s+moment|"
    r"these\s+days|lately|recently|presently|"
    r"i\s+am|i'm|i\s+have|i've|"
    r"i\s+do|i\s+use|i\s+like|i\s+want)\b",
    re.IGNORECASE,
)

# Future signals
_FUTURE_SIGNALS = re.compile(
    r"\b(?:will|going\s+to|plan\s+to|intend\s+to|"
    r"want\s+to|hope\s+to|aim\s+to|"
    r"tomorrow|next\s+week|next\s+month|next\s+year|"
    r"later|someday|eventually|in\s+the\s+future)\b",
    re.IGNORECASE,
)

# Stopped/ceased signals
_STOPPED_SIGNALS = re.compile(
    r"\b(?:stopped|ceased|quit|gave\s+up|no\s+longer|"
    r"don'?t\s+anymore|not\s+anymore|"
    r"i\s+don'?t|i\s+can'?t|i\s+won'?t)\b",
    re.IGNORECASE,
)


def _interpret_temporal_context(message: str) -> str:
    """Interpret the temporal context of a message.
    
    Returns: 'historical', 'current', 'future', 'stopped', or 'unknown'
    """
    lower = message.lower()
    
    # Check for stopped/ceased signals first (highest priority)
    if _STOPPED_SIGNALS.search(lower):
        return "stopped"
    
    # Check for historical signals
    if _HISTORICAL_SIGNALS.search(lower):
        return "historical"
    
    # Check for future signals
    if _FUTURE_SIGNALS.search(lower):
        return "future"
    
    # Check for current signals
    if _CURRENT_SIGNALS.search(lower):
        return "current"
    
    # Default: assume current if it's a statement
    return "current"


# ---------------------------------------------------------------------------
# Enhanced negation handling
# ---------------------------------------------------------------------------

# Negation patterns with context
_NEGATION_PATTERNS = re.compile(
    r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|didn'?t|did\s+not|"
    r"can'?t|cannot|never|no\s+longer|not|"
    r"won'?t|will\s+not|wouldn'?t|would\s+not|"
    r"shouldn'?t|should\s+not|"
    r"i\s+don'?t|i\s+do\s+not|"
    r"i\s+can'?t|i\s+cannot|"
    r"i\s+won'?t|i\s+will\s+not)\b",
    re.IGNORECASE,
)

# Negated verbs
_NEGATED_VERBS = re.compile(
    r"\b(?:don'?t|do\s+not|doesn'?t|does\s+not|didn'?t|did\s+not|"
    r"can'?t|cannot|never|no\s+longer|not)\s+"
    r"(?:like|love|enjoy|hate|dislike|prefer|"
    r"use|use|have|own|want|need|"
    r"learn|study|work|build|create)\b",
    re.IGNORECASE,
)


def _extract_negation_context(message: str) -> dict:
    """Extract negation context from a message.
    
    Returns dict with:
    - is_negated: bool
    - negation_scope: what is negated (e.g., "like", "use")
    - negation_type: "explicit" or "implicit"
    """
    lower = message.lower()
    
    is_negated = bool(_NEGATION_PATTERNS.search(lower))
    
    negation_scope = None
    negation_type = None
    
    if is_negated:
        # Find what verb is negated
        match = _NEGATED_VERBS.search(lower)
        if match:
            negation_scope = match.group(0).split()[-1]  # Get the verb after negation
            negation_type = "explicit"
        else:
            # Check for "not" before a verb
            not_match = re.search(
                r"\bnot\s+(?:like|love|enjoy|hate|dislike|prefer|"
                r"use|have|own|want|need|"
                r"learn|study|work|build|create)\b",
                lower,
            )
            if not_match:
                negation_scope = not_match.group(0).split()[-1]
                negation_type = "explicit"
            else:
                negation_type = "implicit"
    
    return {
        "is_negated": is_negated,
        "negation_scope": negation_scope,
        "negation_type": negation_type,
    }


# ---------------------------------------------------------------------------
# Enhanced uncertainty handling
# ---------------------------------------------------------------------------

# Uncertainty signals
_UNCERTAINTY_SIGNALS = re.compile(
    r"\b(?:i\s+think|i\s+believe|i\s+feel\s+like|i\s+feel\s+that|"
    r"maybe\s+i|perhaps\s+i|possibly\s+i|i\s+might|i\s+could|"
    r"i(?:'m|\s+am)\s+considering|i\s+am\s+considering|"
    r"not\s+sure\s+(?:if|i)|i(?:'m|\s+am)\s+not\s+sure\s+(?:if|i)|"
    r"i\s+am\s+not\s+sure\s+(?:if|i)|"
    r"probably|most\s+likely|not\s+certain|not\s+sure|"
    r"i\s+guess|i\s+suppose|i\s+assume|"
    r"it\s+seems|it\s+appears|apparently|"
    r"allegedly|reportedly|supposedly)\b",
    re.IGNORECASE,
)

# Hedged statements
_HEDGED_STATEMENTS = re.compile(
    r"\b(?:i\s+might|i\s+could|i\s+would|i\s+should|"
    r"i\s+may|i\s+might\s+want|"
    r"i\s+am\s+thinking|i\s+am\s+planning|"
    r"i\s+am\s+hoping|i\s+am\s+aiming)\b",
    re.IGNORECASE,
)


def _extract_uncertainty_context(message: str) -> dict:
    """Extract uncertainty context from a message.
    
    Returns dict with:
    - is_uncertain: bool
    - uncertainty_level: "low", "medium", "high"
    - uncertainty_type: "hedged", "speculative", "tentative"
    """
    lower = message.lower()
    
    is_uncertain = bool(_UNCERTAINTY_SIGNALS.search(lower))
    is_hedged = bool(_HEDGED_STATEMENTS.search(lower))
    
    uncertainty_level = "low"
    uncertainty_type = None
    
    if is_uncertain:
        # Determine uncertainty level
        if re.search(r"\b(?:maybe|perhaps|possibly|i\s+think|i\s+believe)\b", lower):
            uncertainty_level = "medium"
        elif re.search(r"\b(?:not\s+sure|not\s+certain|i\s+guess|i\s+suppose)\b", lower):
            uncertainty_level = "high"
        else:
            uncertainty_level = "low"
        
        # Determine uncertainty type
        if is_hedged:
            uncertainty_type = "hedged"
        elif re.search(r"\b(?:maybe|perhaps|possibly|might|could)\b", lower):
            uncertainty_type = "speculative"
        else:
            uncertainty_type = "tentative"
    
    return {
        "is_uncertain": is_uncertain,
        "uncertainty_level": uncertainty_level,
        "uncertainty_type": uncertainty_type,
    }


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

class PatternExtractor:
    """Deterministic pattern-based fact extractor.

    Handles common natural language patterns without an LLM.
    Provides the fallback extraction path.
    """

    def extract(self, message: str, understanding: MessageUnderstanding) -> ExtractionResult:
        """Extract memory operations from a user message."""
        operations: list[MemoryOperation] = []

        # Split into multiple facts if needed
        messages = _split_multiple_facts(message)

        for msg in messages:
            # Strip trailing punctuation so $-anchored filler-word regexes
            # (e.g. r"anymore\s*$") match regardless of punctuation.
            msg_normalized = msg.rstrip(".!?…;:")

            domain = _classify_domain(msg_normalized)
            subject = _extract_subject_from_message(msg_normalized, domain)
            value = _extract_value(msg_normalized, subject, domain)
            polarity = _extract_polarity(msg_normalized)
            confidence = _extract_confidence(msg_normalized, understanding.is_uncertain)
            fact_type = _infer_fact_type(domain, msg_normalized)

            # Determine action
            action = Action.CREATE
            if understanding.is_deletion:
                action = Action.DEACTIVATE
                # For deletions, extract the target subject from the message
                deletion_target = _extract_deletion_target(msg_normalized)
                if deletion_target:
                    subject = deletion_target
                    value = deletion_target.title()
            elif understanding.is_correction:
                action = Action.UPDATE
                # For corrections, try to extract the NEW value being corrected to
                correction_subject = _extract_correction_subject(msg_normalized, domain)
                if correction_subject:
                    subject = correction_subject
                    value = correction_subject.title() if domain == "interests" else correction_subject

            op = MemoryOperation(
                action=action,
                domain=domain,
                fact_type=fact_type,
                content=msg.strip(),
                subject=subject,
                value=value,
                polarity=polarity,
                importance="medium",
                confidence=confidence,
                entity_name="self",
                entity_type="self",
                temporal_hint=understanding.temporal_hint,
                is_temporary=understanding.is_temporary,
                is_uncertain=understanding.is_uncertain,
            )
            operations.append(op)

        return ExtractionResult(
            operations=operations,
            message_type=understanding.message_type.value,
            is_temporary=understanding.is_temporary,
            is_uncertain=understanding.is_uncertain,
            is_third_party=understanding.is_third_party,
            is_correction=understanding.is_correction,
            is_deletion=understanding.is_deletion,
            temporal_hint=understanding.temporal_hint,
            confidence=understanding.confidence,
        )
