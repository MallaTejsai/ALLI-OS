"""Fast Triage — pre-memory classification layer for ALLI v0.6.

General pre-memory triage layer that identifies:
- Greetings
- Acknowledgements
- Low-information conversational statements
- Obvious conversational requests
- Memory-bearing statements
- Questions
- General knowledge questions
- Mixed messages

NOT a phrase cache. Uses general intent/information-content classification.
Mixed messages must still be processed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class TriageCategory(str, Enum):
    """Triage categories per architecture §13.3."""
    GREETING = "greeting"
    ACKNOWLEDGEMENT = "acknowledgement"
    CONVERSATIONAL = "conversational"
    MEMORY_STATEMENT = "memory_statement"
    QUESTION = "question"
    GENERAL_KNOWLEDGE = "general_knowledge"
    MIXED = "mixed"
    MALFORMED = "malformed"


@dataclass
class TriageResult:
    """Result of fast triage classification."""
    category: TriageCategory
    has_memory_content: bool
    has_instruction: bool
    confidence: float


# ---------------------------------------------------------------------------
# Greeting patterns
# ---------------------------------------------------------------------------

_GREETING_PATTERNS = re.compile(
    r"^(?:hi|hello|hey|sup|what'?s\s+up|good\s+(?:morning|afternoon|evening|night)"
    r"|howdy|greetings|how(?:'s| is) it going|what'?s\s+new)"
    r"(?:\s+there)?\s*[!.]?\s*$",
    re.IGNORECASE,
)

_GREETING_YO = re.compile(r"^yo\s*[!.]?\s*$", re.IGNORECASE)

_GREETING_BYE = re.compile(
    r"^(?:bye|goodbye|see\s+ya|later|talk\s+later|gotta\s+go|"
    r"gtg|g2g|night|good\s+night|take\s+care)\s*[!.]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Acknowledgement patterns
# ---------------------------------------------------------------------------

_ACKNOWLEDGEMENT_PATTERNS = re.compile(
    r"^(?:ok|okay|k|sure|alright|alrighty|right|yeah|yep|yup|"
    r"got\s+it|understood|noted|will\s+do|i\s+see|makes?\s+sense|"
    r"cool|nice|great|awesome|thanks|thank\s+you|thx|ty|"
    r"no\s+problem|np|you'?re\s+welcome|anytime)\s*[!.]?\s*$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Question patterns
# ---------------------------------------------------------------------------

_QUESTION_SUFFIX = re.compile(r"\?\s*$")

_QUESTION_STARTERS = re.compile(
    r"^(?:what|who|where|when|why|how|do|did|can|could|would|should|is|are|was|were|"
    r"tell|give|show|explain|define|list)\b",
    re.IGNORECASE,
)

# General knowledge question patterns
# Note: personal questions (containing "my", "I", etc.) are checked FIRST in triage(),
# so these patterns only match non-personal questions.
_GENERAL_KNOWLEDGE_PATTERNS = re.compile(
    r"^(?:what\s+is|what\s+are|who\s+is|who\s+was|where\s+is|when\s+was|"
    r"how\s+does|how\s+do|how\s+to|"
    r"why\s+(?:is|are|was|were|do|does|did|did|can|could|would|should|has|have|had)|"
    r"who\s+(?:invented|discovered|created|wrote|painted|built|designed|made|founded|started|developed|coined|originated)|"
    r"how\s+much\s+(?:does|do|did|is|are|was|were|has|have|had)|"
    r"can\s+you\s+explain|"
    r"tell\s+me\s+about|what\s+do\s+you\s+know\s+about)\b",
    re.IGNORECASE,
)

# Personal question patterns (contain personal pronouns)
_PERSONAL_QUESTION_PATTERNS = re.compile(
    r"\b(?:i|my|mine|me|myself|i'm|i\s+am|i\s+have|i\s+like|i\s+want|i\s+use|"
    r"we|us|our|ours|ourselves)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Conversational request patterns
# ---------------------------------------------------------------------------

_CONVERSATIONAL_PATTERNS = re.compile(
    r"^(?:can\s+you\s+help|help\s+me|could\s+you|would\s+you|"
    r"tell\s+me\s+a\s+joke|what'?s\s+the\s+weather|"
    r"how'?s\s+it\s+going|what'?s\s+up|"
    r"let'?s\s+(?:talk|chat|discuss)|"
    r"i\s+(?:need|want)\s+help|"
    r"what\s+do\s+you\s+think)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Memory instruction patterns
# ---------------------------------------------------------------------------

_INSTRUCTION_PATTERNS = re.compile(
    r"\b(?:remember|forget|store|save|note|don'?t\s+forget|"
    r"stop\s+remembering|remove|delete|clear)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Memory-bearing content detection
# ---------------------------------------------------------------------------

# Personal pronouns that indicate personal content
# Includes first-person singular (i, me, my) and first-person plural (we, us, our)
_PERSONAL_PRONOUNS = re.compile(
    r"\b(?:i|me|my|mine|myself|i'm|i\s+am|i\s+have|i\s+like|i\s+want|i\s+use|"
    r"we|us|our|ours|ourselves|we're|we've|we'll)\b",
    re.IGNORECASE,
)

# Verbs that indicate memory-bearing content
_MEMORY_VERBS = re.compile(
    r"\b(?:learn(?:ing|ed)?|stud(?:y|ying|ied)|like|likes|liked|"
    r"love|loves|loved|enjoy|enjoys|enjoyed|"
    r"hate|hates|hated|dislike|dislikes|"
    r"prefer|prefers|preferred|"
    r"want|wants|wanted|need|needs|needed|"
    r"have|has|had|own|owns|owned|use|uses|used|"
    r"work|works|worked|build|builds|built|"
    r"create|creates|created|design|designs|designed|"
    r"program|programs|coded|code|codes)\b",
    re.IGNORECASE,
)

# Possession indicators
_POSSESSION_INDICATORS = re.compile(
    r"\b(?:my\s+(?:\w+\s+){0,2}(?:phone|laptop|computer|pc|tablet|"
    r"car|bike|watch|headphones|earbuds|camera|"
    r"book|notebook|pen|mouse|keyboard|monitor|screen|"
    r"shirt|pants|shoes|jacket|hat|glasses))\b",
    re.IGNORECASE,
)


def _is_malformed(message: str) -> bool:
    """Check if message is malformed/unintelligible."""
    stripped = message.strip()
    
    # Empty or too short
    if len(stripped) < 2:
        return True
    
    # Only punctuation/special characters
    if re.match(r"^[^a-zA-Z0-9\s]+$", stripped):
        return True
    
    # Only numbers
    if re.match(r"^\d+$", stripped):
        return True
    
    # Gibberish detection: too many consecutive consonants (more than 5)
    if re.search(r"[bcdfghjklmnpqrstvwxyz]{6,}", stripped.lower()):
        return True
    
    return False


def _has_memory_content(message: str) -> bool:
    """Detect if message contains memory-bearing content.
    
    Uses broad detection: any personal pronoun + any verb/content
    indicates potential memory-bearing content. Specific verb lists
    are a bonus for confidence, not a requirement.
    """
    lower = message.lower()
    
    # Check for personal pronouns (primary signal)
    has_personal_pronoun = bool(_PERSONAL_PRONOUNS.search(lower))
    
    if has_personal_pronoun:
        # Personal pronoun present = likely memory content
        # Exception: very short phrases that are just pronouns
        words = lower.split()
        if len(words) >= 3:
            return True
        # Short phrases: check if there's a verb-like word
        if _MEMORY_VERBS.search(lower) or re.search(r"\b(?:is|are|was|were|have|has|had|do|does|did|will|would|can|could|should|may|might|shall)\b", lower):
            return True
    
    # Check for possession indicators
    if _POSSESSION_INDICATORS.search(lower):
        return True
    
    # Check for third-party statements with verbs (expanded)
    third_party = re.match(
        r"^(?:he|she|they|my\s+\w+|someone)\s+"
        r"(?:likes?|loves?|wants?|needs?|prefers?|enjoys?|uses?|has|is|was|"
        r"drives?|rides?|bought|recommend|code|learn|study|play|watch|"
        r"eat|cook|travel|move|visit|meet|write|run|swim|drink|read)\b",
        lower,
    )
    if third_party:
        return True
    
    # Check for "X is Y" identity/property statements
    # This catches things like "My dog prefers chicken" or "My brother drives a Tesla"
    if re.match(r"^.+\s+(?:is|are|was|were|has|have|had|like|likes|love|loves|"
                r"prefer|prefers|enjoy|enjoys|want|wants|need|needs|"
                r"drive|drives|ride|rides|use|uses|drink|drinks|eat|eats)\b", lower):
        # Exclude general knowledge patterns
        if not _GENERAL_KNOWLEDGE_PATTERNS.match(lower):
            # Exclude pure factual statements about the world
            if not re.match(r"^(?:the|a|an|this|that|these|those)\s+", lower):
                return True
    
    return False


def _has_instruction(message: str) -> bool:
    """Detect if message contains a memory instruction."""
    return bool(_INSTRUCTION_PATTERNS.search(message.lower()))


def triage(message: str) -> TriageResult:
    """Perform fast triage on a user message.
    
    This is the first stage of the pipeline. It determines what kind
    of message we're dealing with without deep analysis.
    """
    stripped = message.strip()
    lower = stripped.lower()
    
    # Malformed check
    if _is_malformed(stripped):
        return TriageResult(
            category=TriageCategory.MALFORMED,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.9,
        )
    
    # Greeting check
    if _GREETING_PATTERNS.match(stripped) or _GREETING_YO.match(stripped):
        return TriageResult(
            category=TriageCategory.GREETING,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.95,
        )
    
    # Bye/farewell check (conversational)
    if _GREETING_BYE.match(stripped):
        return TriageResult(
            category=TriageCategory.CONVERSATIONAL,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.9,
        )
    
    # Acknowledgement check
    if _ACKNOWLEDGEMENT_PATTERNS.match(stripped):
        return TriageResult(
            category=TriageCategory.ACKNOWLEDGEMENT,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.9,
        )
    
    # Question check
    is_question = bool(_QUESTION_SUFFIX.search(stripped)) or bool(_QUESTION_STARTERS.match(lower))
    
    if is_question:
        # Check if it's a personal question
        if _PERSONAL_QUESTION_PATTERNS.search(lower):
            return TriageResult(
                category=TriageCategory.QUESTION,
                has_memory_content=True,
                has_instruction=False,
                confidence=0.85,
            )
        # Check if it's a general knowledge question
        if _GENERAL_KNOWLEDGE_PATTERNS.match(lower):
            return TriageResult(
                category=TriageCategory.GENERAL_KNOWLEDGE,
                has_memory_content=False,
                has_instruction=False,
                confidence=0.8,
            )
        # Other questions
        return TriageResult(
            category=TriageCategory.QUESTION,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.7,
        )
    
    # Conversational request check
    if _CONVERSATIONAL_PATTERNS.match(lower):
        # Check if it also has memory content
        if _has_memory_content(stripped):
            return TriageResult(
                category=TriageCategory.MIXED,
                has_memory_content=True,
                has_instruction=_has_instruction(stripped),
                confidence=0.75,
            )
        return TriageResult(
            category=TriageCategory.CONVERSATIONAL,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.8,
        )
    
    # Memory instruction check
    has_instruction = _has_instruction(stripped)
    
    # Memory content check
    has_memory = _has_memory_content(stripped)
    
    if has_memory or has_instruction:
        # Check if it's mixed (has both conversational and memory content)
        # Simple heuristic: if it starts with conversational filler but has memory content
        conversational_fillers = re.match(
            r"^(?:bro|dude|hey|yo|lol|haha|omg|wow|oh|ah|so|like|"
            r"btw|anyway|also|just|finally|actually)\b",
            lower,
        )
        if conversational_fillers and has_memory:
            return TriageResult(
                category=TriageCategory.MIXED,
                has_memory_content=True,
                has_instruction=has_instruction,
                confidence=0.7,
            )
        
        return TriageResult(
            category=TriageCategory.MEMORY_STATEMENT,
            has_memory_content=True,
            has_instruction=has_instruction,
            confidence=0.8,
        )
    
    # Default: if it's a sentence with some content, treat as conversational
    if len(lower.split()) >= 2:
        return TriageResult(
            category=TriageCategory.CONVERSATIONAL,
            has_memory_content=False,
            has_instruction=False,
            confidence=0.6,
        )
    
    # Very short or unclear
    return TriageResult(
        category=TriageCategory.CONVERSATIONAL,
        has_memory_content=False,
        has_instruction=False,
        confidence=0.5,
    )
