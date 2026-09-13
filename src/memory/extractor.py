"""Deterministic durable-fact extraction for ALLI v0.6."""

from __future__ import annotations

import re
from typing import Optional


# Normalization map for common typos that change semantic meaning.
# Keys are normalized (canonical) values; entries are sets of typos.
_CANONICAL_WORDS = {
    "nothing": {"noting", "nothign", "noting"},
    "something": {"somthing", "smething"},
    "anything": {"anythign", "anyting"},
    "everything": {"everythign", "evrything"},
}


def _normalize_goal_value(word: str) -> str:
    """Normalize a single goal word, correcting obvious typos.

    Returns the canonical form if the word is a known typo,
    otherwise returns the original word unchanged.
    """
    lower = word.lower().strip()
    for canonical, typos in _CANONICAL_WORDS.items():
        if lower in typos or lower == canonical:
            return canonical
    return lower


CHANGE_SIGNAL_PHRASES = (
    "actually",
    "in fact",
    "my goal changed",
    "i changed my goal",
    "my career goal changed",
    "instead of",
    "replace ",
    "i no longer want",
    "i don't want to become",
    "i do not want to become",
    "don't want to become",
    "do not want to become",
    "i don't want to learn",
    "i do not want to learn",
    "i don't like",
    "i do not like",
    "i don't want",
    "i do not want",
    "i don't interested",
    "i don't intrested",
    "i do not interested",
    "i do not intrested",
    "not interested",
    "not intrested",
    "no longer interested",
    "no longer intrested",
    "stopped being interested",
    "stopped being intrested",
    "i stopped",
    "i changed from",
    "forget my previous",
    "forget that",
    "remove my",
    "delete my",
    "my name changed",
    "i am no longer studying",
    "changed to",
    "no longer",
    "anymore",
    "used to",
    "but now",
    "i no longer want you to remember",
    "switched to",
    "replaced",
    "replaced with",
    "i now",
    # Centralized state-change / supersession vocabulary (generic, any fact)
    "i changed my mind",
    "i decided against",
    "i decided not to",
    "i no longer prefer",
    "i changed my preference",
    "i used to want",
    "i used to like",
    "i used to prefer",
    "i stopped wanting",
    "i stopped preferring",
    "i've changed my mind",
    "i've decided against",
)

TEMPORARY_PHRASES = (
    "this particular",
    "this specific",
    "this one",
    "right now",
    "at the moment",
    "just today",
    "for today",
    "tomorrow",
    "this week",
    "i am hungry",
    "i'm hungry",
    "i am tired",
    "i'm tired",
    "i have an exam",
    "i've got an exam",
    "i am preparing for",
    "i'm preparing for",
    "preparing for my",
    "studying for my",
    "prepare for my",
    "study for my",
    "i want to prepare for",
    "i want to study for",
    # General temporal signals (independent of subject)
    "currently",
    "at present",
    "for now",
    "just",
    "right now working",
    "currently working",
    "at the moment working",
    "today i am",
    "today i'm",
    "this morning",
    "this afternoon",
    "tonight",
)

DELETE_PATTERNS = (
    r"\bforget that i\b",
    r"\bforget my\b",
    r"\bremove my\b",
    r"\bdelete my\b",
    r"\bi no longer want you to remember\b",
    r"\bstop remembering\b",
    r"\bi\s+stopped\s+learning\b",
)

# Negated / cancellation patterns for temporary statements.
# Must be checked BEFORE is_temporary_message to avoid silent dropping.
TASK_CANCEL_PATTERNS = (
    r"\bi\s+(?:don'?t|do\s+not|no\s+longer)\s+have\b",
    r"\b(?:cancel|forget|drop)\s+(?:my|the|an?)\b",
    r"\bno\s+(?:longer\s+)?(?:have|need)\s+(?:a|an|the|my|any)\b",
    r"\bthere\s+is\s+no\b",
)


def has_explicit_change_signal(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in CHANGE_SIGNAL_PHRASES)


def is_temporary_message(message: str) -> bool:
    lower = message.lower()
    if any(phrase in lower for phrase in TEMPORARY_PHRASES):
        return True

    # Time-bounded statements describe a temporary state, not a durable fact.
    # Matches: "until tomorrow", "have X tomorrow", "have X next week", etc.
    # Also handles common misspellings like "tommorrow".
    return bool(
        re.search(
            r"(?:\b(?:until|till|through)\b"
            r"|\bhave\b.*?\b(?:tomorrow|tommorrow|tonight|today)\b"
            r"|\bhave\b.*?\b(?:next|this)\s+(?:week|month|monday|tuesday|"
            r"wednesday|thursday|friday|saturday|sunday)\b"
            r"|\bhave\b.*?\b(?:exam|test|quiz|deadline|appointment|interview)\b"
            r".*?\b(?:tomorrow|tommorrow|tonight|today|next|this)\b"
            r")",
            lower,
            flags=re.IGNORECASE,
        )
    )


def has_third_party_quoted_speech(message: str) -> bool:
    """Return True when a quoted first-person claim is attributed to someone else."""
    return bool(
        re.search(
            r"\b(?:my\s+(?:friend|brother|sister|parent|teacher|classmate)|"
            r"he|she|they|someone|[A-Z][a-z]+)\s+"
            r"(?:said|says|told\s+me|wrote)\s*[,;:]?\s*[\"'""'']",
            message,
            flags=re.IGNORECASE,
        )
    )


def _is_uncertain_personal_intention(text: str) -> bool:
    """Detect if a statement expresses uncertain personal intention.

    Distinguishes uncertain personal intentions from general opinions:
    - "I think Python is useful" → general opinion (not uncertain intention)
    - "I think I want to learn Java" → uncertain personal intention
    - "I believe I should become a doctor" → uncertain personal intention
    - "I feel like I should learn Flutter" → uncertain personal intention

    Returns True if the statement is an uncertain personal intention
    that should not be stored as a definite fact.
    """
    lower = text.lower()

    # Uncertainty signals that prefix personal intentions
    uncertain_personal = re.search(
        r"\b(?:i\s+think|i\s+believe|i\s+feel\s+like|i\s+feel\s+that|"
        r"maybe\s+i|perhaps\s+i|possibly\s+i|i\s+might|i\s+could|"
        r"not\s+sure\s+(?:if|i)|i'm\s+not\s+sure\s+(?:if|i)|"
        r"i\s+am\s+not\s+sure\s+(?:if|i))\s+"
        r"(?:want\s+to|wanna|should|would\s+like\s+to|plan\s+to|"
        r"am\s+going\s+to|will\s+try\s+to|could\s+become|"
        r"might\s+become|want\s+to\s+become|want\s+to\s+be|"
        r"should\s+become|should\s+be)\b",
        lower,
    )
    if uncertain_personal:
        return True

    # "I think I want to..." pattern (broader)
    if re.search(r"\bi\s+think\s+i\s+(?:want|should|would|could|might|plan)\b", lower):
        return True

    # "I believe I want to..." pattern
    if re.search(r"\bi\s+believe\s+i\s+(?:want|should|would|could|might|plan)\b", lower):
        return True

    return False


# ---------------------------------------------------------------------------
# Possession subject classification (generic, no hardcoded brands)
# ---------------------------------------------------------------------------

_DEVICE_KEYWORDS = {
    "phone", "mobile", "laptop", "computer", "tablet", "device",
    "watch", "earbuds", "headphones", "speaker", "keyboard", "mouse",
    "monitor", "tv", "camera", "car", "bike", "motorcycle", "vehicle",
}

_EDUCATION_KEYWORDS = {
    "diploma", "degree", "course", "college", "university", "school",
    "bachelor", "master", "phd", "b.tech", "m.tech", "engineering",
    "computer science", "cs", "certificate",
}

_SUBSTANCE_KEYWORDS = {
    "coffee", "tea", "water", "juice", "beer", "wine", "drink",
}


def _classify_possession_subject(item: str) -> str:
    """Classify a possession item into a generic subject category.

    Returns a generic subject string for storage. No hardcoded brands/models.
    """
    lower = item.lower()

    # Check device keywords
    for kw in _DEVICE_KEYWORDS:
        if kw in lower:
            return kw

    # Check education keywords
    for kw in _EDUCATION_KEYWORDS:
        if kw in lower:
            return "education"

    # Check substance keywords
    for kw in _SUBSTANCE_KEYWORDS:
        if kw in lower:
            return kw

    # Generic: use the first significant word as subject
    words = re.findall(r"[a-z]+", lower)
    # Filter out articles and common fillers
    stop_words = {"a", "an", "the", "is", "was", "my", "your", "his", "her", "its", "our", "their"}
    significant = [w for w in words if w not in stop_words and len(w) > 2]
    if significant:
        return significant[0]

    return "item"


def looks_like_question(message: str) -> bool:
    text = message.strip().lower()
    # Normalize common typos so "wat" → "what", "phne" → "phone", etc.
    for canonical, typos in _QUERY_TYPO_MAP.items():
        text = re.sub(r"\b(" + "|".join(typos) + r")\b", canonical, text)
    if text.endswith("?"):
        return True
    return bool(
        re.match(
            r"^(?:what|who|where|when|why|how|which"
            r"|do|did|can|could|would|should"
            r"|is|am|are|was|were"
            r"|tell|give|show)\b",
            text,
        )
    )


# General typo map shared with query_intent.normalize_memory_query.
_QUERY_TYPO_MAP: dict[str, list[str]] = {
    "phone": ["phne", "phon", "phnoe", "fone", "pfone"],
    "what": ["wat", "wht", "whaat", "hwat", "wha"],
    "which": ["wich", "whch", "whihc"],
    "learning": ["learing", "lerning", "leanring"],
    "studying": ["studing", "studying", "stuyding"],
    "interested": ["intrested", "intresting", "intersted"],
    "python": ["pyhton", "pytohn", "pyhon"],
    "favorite": ["favrite", "favourite", "favrouite"],
    "goals": ["goalz", "gaols"],
    "skills": ["skils", "skillz", "sklls"],
    "nothing": ["noting", "nothign"],
    "something": ["somthing", "smething"],
    "anything": ["anythign", "anyting"],
    "everything": ["everythign", "evrything"],
    "tomorrow": ["tommorrow", "tommorow", "tomorow"],
    "computer": ["compuer", "compter", "computor"],
}


def preference_polarity(content: str) -> Optional[str]:
    text = content.lower()
    if any(
        phrase in text
        for phrase in (
            "dislikes",
            "does not like",
            "don't like",
            "do not like",
            "hates",
            "dislike",
            "hate",
            "prefers not",
        )
    ):
        return "dislike"
    if any(phrase in text for phrase in ("likes", "prefers", "like", "love", "enjoy")):
        # Avoid matching "dislike" already handled above.
        if "dislike" in text or "don't like" in text:
            return "dislike"
        return "like"
    return None


def looks_like_person_name(value: str) -> bool:
    """
    Accept only short person-name values for identity extraction.

    Rejects activity phrases such as:
      interested in web development
      intrested in editing
      learning python
    """
    text = (value or "").strip()
    if not text or len(text) > 40:
        return False

    lower = text.lower()
    # Activity / state phrases are never personal names.
    if re.search(
        r"\b(in|and|or|with|at|from|into|about|learning|studying|"
        r"interested|intrested|interest|working|trying|using|doing|"
        r"going|want|wanna|hope|plan|building|creating|making|"
        r"coding|debugging|writing|developing|reading)\b",
        lower,
    ):
        return False

    tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
    if not tokens or len(tokens) > 3:
        return False

    first = tokens[0].lower()
    blocked_prefixes = (
        "learn", "stud", "interest", "intrest", "work", "try", "use",
        "tired", "hungr", "busy", "happy", "sad", "edit", "develop",
    )
    if any(first.startswith(prefix) for prefix in blocked_prefixes):
        return False

    return True


def split_list_items(text: str) -> list[str]:
    """Split 'A and B' / 'A, B, and C' / 'A or B' into separate items."""
    raw = (text or "").strip().rstrip(".!?")
    if not raw:
        return []
    # Normalize Oxford-comma "and" / "or"
    cleaned = re.sub(r"\s*,\s*and\s+", ", ", raw, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*,\s*or\s+", ", ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+and\s+", ", ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+or\s+", ", ", cleaned, flags=re.IGNORECASE)
    parts = [part.strip(" .!?") for part in cleaned.split(",")]
    return [part for part in parts if len(part) >= 2]


def strip_goal_filler(phrase: str) -> str:
    """Normalize a goal role phrase to a comparable subject."""
    text = (phrase or "").strip().rstrip(".!?")
    # Peel leading fillers repeatedly: "becoming a software engineer"
    for _ in range(4):
        cleaned = re.sub(
            r"^(?:becoming|become|to become|to be|being|be|a|an|the)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()
        if cleaned == text:
            break
        text = cleaned
    return text


def strip_temporal_suffix(value: str) -> str:
    """Remove trailing state-change words from a stored fact's topic."""
    value = (value or "").strip()
    # Strip trailing temporal words and preceding prepositions (in/the)
    value = re.sub(
        r"\s*(?:in|the)?\s*(?:again|anymore|now|currently|today|still|future)\s*$",
        "",
        value,
        flags=re.IGNORECASE,
    )
    # Strip if the entire value is a temporal word
    if value.lower() in {"again", "anymore", "now", "currently", "today", "still", "future"}:
        return ""
    return value


def goal_match_score(hint: str, memory: dict) -> float:
    """
    Lexical confidence that `hint` refers to this goal memory.

    Does not authorize writes by itself — callers still require an explicit
    change/delete signal from the extractor.
    """
    hint_norm = normalize_goal_text(strip_goal_filler(hint or ""))
    if not hint_norm:
        return 0.0

    content_norm = normalize_goal_text(
        strip_temporal_suffix(str(memory.get("content", "")))
    )
    value_norm = normalize_goal_text(
        strip_temporal_suffix(strip_goal_filler(str(memory.get("value", "") or "")))
    )
    subject_norm = normalize_goal_text(
        strip_temporal_suffix(str(memory.get("subject", "") or ""))
    )

    for cand in (content_norm, value_norm, subject_norm):
        if not cand:
            continue
        if hint_norm == cand:
            return 1.0
        if hint_norm in cand or cand in hint_norm:
            return 0.95

    hint_tokens = set(hint_norm.split())
    for cand in (content_norm, value_norm, subject_norm):
        if not cand:
            continue
        cand_tokens = set(cand.split())
        if not cand_tokens:
            continue
        if hint_tokens <= cand_tokens:
            return 0.9
        if cand_tokens <= hint_tokens and len(cand_tokens) >= 1:
            return 0.85
        overlap = len(hint_tokens & cand_tokens) / len(hint_tokens | cand_tokens)
        if overlap >= 0.7:
            return 0.8
        if overlap >= 0.5 and len(hint_tokens & cand_tokens) >= 2:
            return 0.75
    return 0.0


def extract_goal_operations(message: str) -> list[dict]:
    """
    Detect explicit goal replacement or deletion ops.

    May return multiple DELETE_GOAL operations when several goals are rejected.
    Ambiguous "goal changed" without an identifiable old goal is flagged so the
    manager ADDs safely instead of overwriting a random memory.
    """
    text = message.strip()
    if not text:
        return []

    # --- Unambiguous from → to replacement ---
    replace_patterns = [
        r"(?:my\s+(?:career\s+)?goal\s+changed|i\s+changed\s+my\s+(?:career\s+)?goal)"
        r"\s+from\s+(.+?)\s+to\s+(.+?)[.!?]?$",
        r"instead of(?:\s+becoming|\s+being)?\s+(.+?),\s*"
        r"i want to(?:\s+become|\s+be)?\s+(.+?)[.!?]?$",
    ]
    for pattern in replace_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            old = strip_goal_filler(match.group(1))
            new = strip_goal_filler(match.group(2))
            if len(old) < 2 or len(new) < 2:
                continue
            return [{
                "action": "REPLACE_GOAL",
                "old_content_hint": old,
                "category": "goal",
                "fact_type": "career_goal",
                "content": f"User wants to become {new}",
                "value": new,
                "importance": "high",
                "ambiguous": False,
            }]

    # --- Explicit goal rejection (supports multiple targets) ---
    delete_match = re.search(
        r"\bi\s+(?:do not|don't)\s+want\s+to(?:\s+become|\s+be)?\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not delete_match:
        delete_match = re.search(
            r"\bi\s+no longer\s+want\s+to(?:\s+become|\s+be)?\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
    if delete_match:
        blob = delete_match.group(1).strip().rstrip(".!?")
        blob = re.sub(r"\s+anymore\s*$", "", blob, flags=re.IGNORECASE)
        items = split_list_items(blob) or [blob]
        ops = []
        for item in items:
            role = strip_goal_filler(item)
            if len(role) < 2:
                continue
            ops.append({
                "action": "DELETE_GOAL",
                "old_content_hint": role,
                "category": "goal",
                "importance": "high",
            })
        if ops:
            return ops

    # --- Ambiguous goal change (no identifiable old goal) ---
    if re.search(
        r"\b(?:my\s+(?:career\s+)?goal\s+changed|i\s+changed\s+my\s+(?:career\s+)?goal)\b",
        text,
        flags=re.IGNORECASE,
    ):
        new_match = re.search(
            r"(?:now[,.]?\s*)?i\s+want\s+to(?:\s+become|\s+be)?\s+(.+?)[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if not new_match:
            new_match = re.search(
                r"\b(?:now[,.]?\s*)?(?:to\s+become|become)\s+(.+?)[.!?]?$",
                text,
                flags=re.IGNORECASE,
            )
        if new_match:
            new = strip_goal_filler(new_match.group(1))
            if len(new) >= 2:
                return [{
                    "action": "REPLACE_GOAL",
                    "old_content_hint": "",
                    "category": "goal",
                    "fact_type": "career_goal",
                    "content": f"User wants to become {new}",
                    "value": new,
                    "importance": "high",
                    "ambiguous": True,
                }]
        # Change language present but unparseable — block fall-through ADD.
        return []

    return []


def extract_goal_replacement(message: str) -> Optional[dict]:
    """Backward-compatible single-op wrapper around extract_goal_operations."""
    ops = extract_goal_operations(message)
    return ops[0] if ops else None


def normalize_topic_text(text: str) -> str:
    """
    Normalize interest/learning topic strings for matching.

    Applies light spelling repairs and filler stripping. Does not authorize writes.
    """
    value = (text or "").lower().strip()
    value = value.replace("&", " and ")
    value = value.replace("’", "'")
    # Common typos seen in real testing.
    value = re.sub(r"\bdevelopement\b", "development", value)
    value = re.sub(r"\bintrested\b", "interested", value)
    value = re.sub(r"\bintrest\b", "interest", value)
    value = re.sub(r"\bweb\s*dev\b", "web development", value)
    value = re.sub(
        r"\b(?:interest in|interested in|interest|learning|knows|know)\b",
        " ",
        value,
    )
    value = re.sub(
        r"^user\s+(?:is interested in|wants to learn|is learning|knows)\s+",
        "",
        value,
    )
    value = re.sub(r"[^a-z0-9+#\s]", " ", value)
    stop = {
        "a", "an", "the", "to", "my", "user", "i", "also", "in", "of", "for",
    }
    words = [w for w in value.split() if w not in stop]
    return " ".join(words)


def topic_match_score(hint: str, memory: dict) -> float:
    """Lexical confidence that hint refers to this topic memory."""
    hint_norm = normalize_topic_text(hint or "")
    if not hint_norm:
        return 0.0

    candidates = [
        normalize_topic_text(str(memory.get("content", ""))),
        normalize_topic_text(str(memory.get("value", "") or "")),
        normalize_topic_text(str(memory.get("subject", "") or "")),
    ]
    for cand in candidates:
        if not cand:
            continue
        if hint_norm == cand:
            return 1.0
        if hint_norm in cand or cand in hint_norm:
            return 0.95

    hint_tokens = set(hint_norm.split())
    for cand in candidates:
        if not cand:
            continue
        cand_tokens = set(cand.split())
        if not cand_tokens:
            continue
        if hint_tokens <= cand_tokens:
            return 0.9
        if cand_tokens <= hint_tokens and len(cand_tokens) >= 1:
            return 0.85
        overlap = len(hint_tokens & cand_tokens) / len(hint_tokens | cand_tokens)
        if overlap >= 0.7:
            return 0.8
        if overlap >= 0.5 and len(hint_tokens & cand_tokens) >= 2:
            return 0.75
    return 0.0


def extract_interest_and_learning_deletes(message: str) -> list[dict]:
    """
    Explicit interest/learning rejection ops.

    Does not touch goals. Learning deletes never target skills.
    """
    text = message.strip()
    if not text:
        return []

    # --- Learning rejection (category=learning only) ---
    learn_match = re.search(
        r"\bi\s+(?:do not|don't)\s+want\s+to\s+learn\s+(.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not learn_match:
        learn_match = re.search(
            r"\bi\s+no longer\s+want\s+to\s+learn\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
    if not learn_match:
        learn_match = re.search(
            r"\bi\s+stopped\s+learning\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
    if learn_match:
        blob = learn_match.group(1).strip().rstrip(".!?")
        blob = re.sub(r"\s+anymore\s*$", "", blob, flags=re.IGNORECASE)
        items = split_list_items(blob) or [blob]
        ops = []
        for item in items:
            topic = item.strip()
            if len(topic) < 1:
                continue
            ops.append({
                "action": "DELETE_MATCH",
                "category": "learning",
                "content_hint": topic,
                "importance": "medium",
                "explicit_change": True,
            })
        if ops:
            return ops

    # --- Interest rejection (including broken English / typos) ---
    interest_patterns = [
        r"\bi\s+am\s+no longer\s+(?:interested|intrested|intrest)\s+(?:in\s+)?(.+)$",
        r"\bi(?:'m)?\s+no longer\s+(?:interested|intrested|intrest)\s+(?:in\s+)?(.+)$",
        r"\bi\s+am\s+not\s+(?:interested|intrested|intrest)\s+(?:in\s+)?(.+)$",
        r"\bi(?:'m)?\s+not\s+(?:interested|intrested|intrest)\s+(?:in\s+)?(.+)$",
        r"\bi\s+(?:do not|don't)\s+(?:want to be\s+)?(?:interested|intrested|intrest)\s+(?:in\s+)?(.+)$",
        r"\bi\s+stopped\s+being\s+(?:interested|intrested|intrest)\s+(?:in\s+)?(.+)$",
        r"\bno longer\s+(?:interested|intrested)\s+in\s+(.+)$",
      
        r"\bi\s+(?:do not|don't)\s+want\s+(?!to\s+become\b|to\s+learn\b|to\s+be\b)(.+)$",
        r"\bi\s+no longer\s+like\s+(.+)$",
    ]
    for pattern in interest_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        blob = match.group(1).strip().rstrip(".!?")
        blob = re.sub(r"\s+anymore\s*$", "", blob, flags=re.IGNORECASE)
        if len(blob) < 2:
            continue
        items = split_list_items(blob)
        # Prefer per-item deletes when the user lists multiple interests.
        # Also keep the full blob so a single combined interest memory can match.
        hints: list[str] = []
        if items and len(items) > 1:
            hints.append(blob)
            hints.extend(items)
        else:
            hints.append(blob)
        unique_hints = []
        seen = set()
        for hint in hints:
            key = normalize_topic_text(hint)
            if not key or key in seen:
                continue
            seen.add(key)
            unique_hints.append(hint.strip())
            
        if unique_hints:
            return [{
                "action": "DELETE_MATCH",
                "category": "interest",
                "content_hints": unique_hints,
                "importance": "medium",
                "explicit_change": True,
            }]

    return []


def normalize_topic_subject(subject: str) -> str:
    """Normalize skill/interest/learning subjects for duplicate matching."""
    return normalize_preference_subject(subject or "")


def normalize_preference_subject(subject: str) -> str:
    """
    Normalize preference subjects so equivalent phrasings match.

    Examples that collapse to the same key:
      C / C program / C programming / C language / programming in C
      Java / Java programming / the Java language
    """
    text = subject.lower().strip()
    text = text.replace("’", "'")
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)

    # "programming in C" / "program in Java" → "C" / "Java"
    text = re.sub(
        r"\b(?:programming|program|coding|code)\s+in\s+",
        " ",
        text,
    )
    text = re.sub(r"\bin\s+(?=[a-z0-9+#]+)", " ", text)

    for old in (
        "programming language",
        "programming",
        "program",
        "language",
        "coding",
        "code",
    ):
        text = re.sub(rf"\b{re.escape(old)}\b", " ", text)

    stop = {
        "a", "an", "the", "my", "this", "that", "in", "of", "for",
        "with", "using", "to", "again",
    }
    words = [word for word in text.split() if word not in stop]
    return " ".join(words)


def preference_subjects_match(left: str, right: str) -> bool:
    """Return True when two preference subjects are equivalent."""
    a = normalize_preference_subject(left or "")
    b = normalize_preference_subject(right or "")

    if not a or not b:
        return False

    return a == b


def preference_subject(content: str) -> str:
    text = content.lower()
    match = re.search(
        r"\b(?:dislikes|likes|hates|prefers|dislike|like|love|enjoy)\s+(.+)",
        text,
    )
    if not match:
        return ""
    return normalize_preference_subject(match.group(1).strip())


def normalize_goal_text(content: str) -> str:
    text = content.lower()
    text = re.sub(r"^user\s+(wants|wants to|aims|plans|dreams)\s+", "", text)
    text = re.sub(
        r"\b(want to become|want to be|aim to become|aim to|plan to become|"
        r"plan to|dream of becoming|dream of|dream is to|goal is to|"
        r"become|be)\b",
        " ",
        text,
    )
    text = re.sub(r"[^a-z0-9+#\s]", " ", text)
    words = [
        w
        for w in text.split()
        if w not in {"a", "an", "the", "to", "my", "user", "i"}
    ]
    return " ".join(words)


class MemoryExtractor:
    """
    Detect durable personal facts from a user message.

    Returns proposed operations. The SafeMemoryManager validates and applies them.
    Qwen is never given write access to memory files.
    """

    def extract(self, user_message: str) -> list[dict]:
        text = user_message.strip()
        if not text or looks_like_question(text):
            return []

        # Extract entity information from the statement
        from .entity_resolver import extract_entity
        entity_extraction = extract_entity(text)
        entity_name = "user"
        entity_type = "self"
        if entity_extraction.entities:
            primary_entity = entity_extraction.entities[0]
            entity_name = primary_entity.name
            entity_type = primary_entity.entity_type

        # Task cancellation must be checked BEFORE is_temporary_message
        # to avoid silently dropping negated forms like
        # "I don't have any task tomorrow".
        if self._is_task_cancellation(text):
            cancel_op = self._extract_task_cancellation(text)
            if cancel_op:
                cancel_op["entity_name"] = entity_name
                cancel_op["entity_type"] = entity_type
            return [cancel_op] if cancel_op else []

        if is_temporary_message(text):
            task_op = self._extract_task(text)
            if task_op:
                task_op["entity_name"] = entity_name
                task_op["entity_type"] = entity_type
            return [task_op] if task_op else []

        # Do not treat another person's quoted first-person statement as the
        # user's own durable fact.
        if has_third_party_quoted_speech(text):
            return []

        # Uncertain personal intentions should not be stored as definite facts.
        # "I think I want to learn Java" → reject (uncertain)
        # "I think Python is useful" → allow (general opinion)
        if _is_uncertain_personal_intention(text):
            return []

        # Explicit interest / learning rejection (before goal ops; never touches goals)
        topic_deletes = extract_interest_and_learning_deletes(text)
        if topic_deletes:
            return topic_deletes

        # Explicit deletion requests
        delete_ops = self._extract_delete(text)
        if delete_ops:
            return delete_ops

        goal_ops = extract_goal_operations(text)
        if goal_ops:
            return goal_ops

        # Goal-change / rejection language must never fall through to ADD.
        if re.search(
            r"\b(?:goal\s+changed|changed\s+my\s+(?:career\s+)?goal|"
            r"don'?t want to become|do not want to become|"
            r"no longer want to become)\b",
            text,
            flags=re.IGNORECASE,
        ):
            return []

        # Interest / learning rejection language must never fall through to ADD.
        if re.search(
            r"\b(?:don'?t|do not)\s+(?:interested|intrested|intrest)\b|"
            r"\b(?:not|no longer)\s+(?:interested|intrested)\b|"
            r"\bstopped being\s+(?:interested|intrested)\b|"
            r"\b(?:don'?t|do not|no longer)\s+want to learn\b|"
            r"\bstopped learning\b",
            text,
            flags=re.IGNORECASE,
        ):
            return []

        # Interests / skills / learning before name so "I am interested/intrested…"
        # never falls through to the identity "I am <name>" path.
        # Process every clause in a compound statement (comma-separated, with optional
        # trailing "and X" on the last clause), collecting one operation per extractor
        # per clause so that one user message can produce multiple memory operations.
        clauses: list[str] = []
        parts = text.split(",")
        for i, part in enumerate(parts):
            p = part.strip()
            if i < len(parts) - 1:
                clauses.append(p)
            else:
                # Last clause: strip leading "and " if present
                if p.lower().startswith("and "):
                    p = p[4:].strip()
                clauses.append(p)

        # Expand clauses that contain "and" or "or" lists for preferences/skills/interests
        # e.g., "I like Python and Java" → ["I like Python", "I like Java"]
        expanded_clauses: list[str] = []
        for clause in clauses:
            lower_clause = clause.lower()
            # Check if this is a preference/skill/interest statement with multiple items
            if re.search(r"\b(?:i\s+)?(?:like|love|enjoy|dislike|hate)\s+", lower_clause):
                match = re.search(
                    r"\b(?:i\s+)?(?:like|love|enjoy|dislike|hate)\s+(.+)",
                    clause,
                    flags=re.IGNORECASE,
                )
                if match:
                    verb_match = re.search(
                        r"\b(?:i\s+)?(like|love|enjoy|dislike|hate)\s+",
                        clause,
                        flags=re.IGNORECASE,
                    )
                    if verb_match:
                        verb = verb_match.group(1)
                        subject_part = match.group(1).strip().rstrip(".!?")
                        items = split_list_items(subject_part)
                        if len(items) > 1:
                            for item in items:
                                expanded_clauses.append(f"I {verb} {item}")
                            continue
            expanded_clauses.append(clause)
        clauses = expanded_clauses

        operations: list[dict] = []
        for clause in clauses:
            if not clause:
                continue
            for extractor in (
                self._extract_nickname,
                self._extract_preference,
                self._extract_skill_or_interest,
                self._extract_name,
                self._extract_goal,
                self._extract_education,
                self._extract_ambition,
                self._extract_dream,
                self._extract_priority,
                self._extract_possession,
            ):
                operation = extractor(clause)
                if not operation:
                    continue
                if isinstance(operation, list):
                    for op in operation:
                        op["entity_name"] = entity_name
                        op["entity_type"] = entity_type
                    operations.extend(operation)
                else:
                    operation["entity_name"] = entity_name
                    operation["entity_type"] = entity_type
                    operations.append(operation)

        # Filter out garbage extractions from malformed input
        validated = []
        for op in operations:
            if self._is_valid_operation(op):
                validated.append(op)
        return validated

    @staticmethod
    def _is_valid_operation(op: dict) -> bool:
        """Reject extractions that are clearly nonsensical."""
        value = (op.get("value") or "").strip().lower()
        content = (op.get("content") or "").strip().lower()
        _INVALID_VALUES = {
            "now", "today", "tomorrow", "yesterday", "here", "there",
            "yes", "no", "ok", "okay", "maybe", "something", "nothing",
            "anything", "everything", "stuff", "thing", "things", "it",
            "that", "this", "what", "which", "when", "where", "how", "why",
            "do", "did", "does", "is", "are", "was", "were", "am",
            "i", "you", "he", "she", "we", "they", "me", "him", "her",
            "us", "my", "your", "his", "its", "our", "their",
        }
        if value in _INVALID_VALUES:
            return False
        if content and len(content.split()) < 3:
            return False
        return True

    def _extract_delete(self, text: str) -> list[dict]:
        lower = text.lower()
        if not any(re.search(pattern, lower) for pattern in DELETE_PATTERNS):
            return []

        # Forget preference
        match = re.search(
            r"forget that i\s+(?:like|love|enjoy|dislike|hate)\s+(.+?)[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            subject = match.group(1).strip()
            return [{
                "action": "DELETE_MATCH",
                "category": "preference",
                "content_hint": subject,
                "importance": "medium",
            }]

        match = re.search(
            r"(?:forget|remove|delete)\s+my\s+(.+?)\s+goal[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return [{
                "action": "DELETE_MATCH",
                "category": "goal",
                "content_hint": match.group(1).strip(),
                "importance": "high",
            }]

        if re.search(
            r"(?:forget|remove|delete).*(?:nickname|nick name)|"
            r"no longer want you to remember my nickname",
            lower,
        ):
            return [{
                "action": "DELETE_MATCH",
                "category": "identity",
                "fact_type": "nickname",
                "content_hint": "nickname",
                "importance": "high",
            }]

        match = re.search(
            r"\bi\s+stopped\s+learning\s+(.+?)[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            topic = match.group(1).strip()
            items = split_list_items(topic) or [topic]
            ops = []
            for item in items:
                cleaned = item.strip()
                if len(cleaned) < 1:
                    continue
                ops.append({
                    "action": "DELETE_MATCH",
                    "category": "learning",
                    "content_hint": cleaned,
                    "importance": "medium",
                })
            return ops

        return []

    def _extract_nickname(self, text: str) -> Optional[dict]:
        match = re.search(
            r"\bmy\s+(?:nick\s*name|nickname)\s+is\s+(.+?)[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if not match:
            return None
        value = match.group(1).strip(" .!?")
        if not value:
            return None
        content = f"User's nickname is {value}"
        return {
            "action": "UPSERT",
            "category": "identity",
            "fact_type": "nickname",
            "content": content,
            "value": value,
            "importance": "high",
        }

    def _extract_name(self, text: str) -> Optional[dict]:
        # Explicit rename: "My name changed to Rahul"
        match = re.search(
            r"\bmy\s+name\s+changed(?:\s+from\s+.+?)?\s+to\s+(.+?)[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip(" .!?")
            if looks_like_person_name(value):
                return {
                    "action": "UPSERT",
                    "category": "identity",
                    "fact_type": "name",
                    "content": f"User's name is {value}",
                    "value": value,
                    "importance": "high",
                    "explicit_change": True,
                }

        match = re.search(
            r"\bmy\s+name(?:'s)?\s+is\s+(.+?)[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            value = match.group(1).strip(" .!?")
            if looks_like_person_name(value):
                return {
                    "action": "UPSERT",
                    "category": "identity",
                    "fact_type": "name",
                    "content": f"User's name is {value}",
                    "value": value,
                    "importance": "high",
                }
            return None

        match = re.search(
            r"\b(?:i am|i'm)\s+([A-Za-z][A-Za-z .'-]{0,40})[.!?]?$",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            name = match.group(1).strip(" .!?")
            if not looks_like_person_name(name):
                return None
            return {
                "action": "UPSERT",
                "category": "identity",
                "fact_type": "name",
                "content": f"User's name is {name}",
                "value": name,
                "importance": "high",
            }
        return None

    def _extract_preference(self, text: str) -> Optional[dict]:
        negative_patterns = [
            r"\bi don't like\s+(.+)",
            r"\bi do not like\s+(.+)",
            r"\bi dislike\s+(.+)",
            r"\bi hate\s+(.+)",
            r"\bi don't enjoy\s+(.+)",
            r"\bi do not enjoy\s+(.+)",
        ]
        for pattern in negative_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                subject = match.group(1).strip().rstrip(".!?")
                subject = strip_temporal_suffix(subject)
                if len(subject) < 1:
                    return None
                # Allow single-token subjects such as "C".
                if len(subject) == 1 and not subject.isalnum():
                    return None
                # Reject universal/vague expressions that do not name a real
                # preference subject (e.g. "I don't like anything").
                if subject.lower() in {"anything", "nothing", "something", "everything"}:
                    return None
                # Handle multiple values: "I like Python and Java"
                items = split_list_items(subject)
                if len(items) > 1:
                    # Return first item; the caller (extract) will process remaining via clauses
                    content = f"User dislikes {items[0]}"
                    return {
                        "action": "UPSERT",
                        "category": "preference",
                        "fact_type": "dislike",
                        "content": content,
                        "value": "dislike",
                        "subject": normalize_preference_subject(items[0]),
                        "importance": "medium",
                        "preference_polarity": "dislike",
                        "preference_subject": normalize_preference_subject(items[0]),
                    }
                content = f"User dislikes {subject}"
                return {
                    "action": "UPSERT",
                    "category": "preference",
                    "fact_type": "dislike",
                    "content": content,
                    "value": "dislike",
                    "subject": normalize_preference_subject(subject),
                    "importance": "medium",
                    "preference_polarity": "dislike",
                    "preference_subject": normalize_preference_subject(subject),
                }

        positive_patterns = [
            r"\bi like\s+(.+)",
            r"\bi love\s+(.+)",
            r"\bi enjoy\s+(.+)",
            r"\bi prefer\s+(.+)",
            r"\bmy\s+favou?rite\s+(\w+)\s+is\s+(.+)",
            r"\bfavou?rite\s+(\w+)\s+is\s+(.+)",
        ]
        for pattern in positive_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                # Handle "My favourite X is Y" / "Favourite X is Y" patterns
                if "favou?rite" in pattern:
                    # Pattern has two groups: (category_word, value)
                    if match.lastindex and match.lastindex >= 2:
                        category_word = match.group(1).strip().rstrip(".!?")
                        value = match.group(2).strip().rstrip(".!?")
                        subject = normalize_preference_subject(f"{category_word} {value}")
                        content = f"User likes {value}"
                        return {
                            "action": "UPSERT",
                            "category": "preference",
                            "fact_type": "like",
                            "content": content,
                            "value": "like",
                            "subject": subject,
                            "importance": "medium",
                            "preference_polarity": "like",
                            "preference_subject": subject,
                        }
                    continue
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                subject = match.group(1).strip().rstrip(".!?")
                subject = strip_temporal_suffix(subject)
                if len(subject) < 1:
                    return None
                # Allow single-token subjects such as "C".
                if len(subject) == 1 and not subject.isalnum():
                    return None
                # Reject universal/vague expressions that do not name a real
                # preference subject (e.g. "I like nothing").
                if subject.lower() in {"anything", "nothing", "something", "everything"}:
                    return None
                # Handle multiple values: "I like Python and Java"
                items = split_list_items(subject)
                if len(items) > 1:
                    # Return first item; the caller (extract) will process remaining via clauses
                    content = f"User likes {items[0]}"
                    return {
                        "action": "UPSERT",
                        "category": "preference",
                        "fact_type": "like",
                        "content": content,
                        "value": "like",
                        "subject": normalize_preference_subject(items[0]),
                        "importance": "medium",
                        "preference_polarity": "like",
                        "preference_subject": normalize_preference_subject(items[0]),
                    }
                content = f"User likes {subject}"
                return {
                    "action": "UPSERT",
                    "category": "preference",
                    "fact_type": "like",
                    "content": content,
                    "value": "like",
                    "subject": normalize_preference_subject(subject),
                    "importance": "medium",
                    "preference_polarity": "like",
                    "preference_subject": normalize_preference_subject(subject),
                }
        return None

    def _extract_goal(self, text: str) -> Optional[dict]:
        # Normalize common action-verb typos so build_match can catch them.
        normalized = text
        normalized = re.sub(r"\bbulid\b", "build", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bdevelope\b", "develop", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\bcreaet\b", "create", normalized, flags=re.IGNORECASE)
        # "I want to become build X" is malformed; normalize to "I want to build X".
        normalized = re.sub(
            r"\b(?:want to|wanna)\s+become\s+(build|create|develop|make|design|write|code)\b",
            r"want to \1",
            normalized,
            flags=re.IGNORECASE,
        )

        # "I want to build..." / "I am building..." / "I am creating..." may be project.
        build_match = re.search(
            r"\bi(?:\s+also)?\s+want to build\s+(.+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if not build_match:
            build_match = re.search(
                r"\bi\s+(?:am|'m)\s+(?:building|creating|developing)\s+(.+)",
                normalized,
                flags=re.IGNORECASE,
            )
        if build_match:
            project = build_match.group(1).strip().rstrip(".!?")
            if len(project) >= 3:
                content = f"User wants to build {project}"
                return {
                    "action": "ADD",
                    "category": "project",
                    "fact_type": "project",
                    "content": content,
                    "value": project,
                    "importance": "medium",
                }

        # NOTE: ambition and dream are handled by _extract_ambition and
        # _extract_dream respectively, called before this method in extract().

        goal_patterns = [
            r"\bmy (?:main |primary )?goal is to\s+(.+)",
            r"\bi aim to\s+(.+)",
            r"\bi plan to\s+(.+)",
            r"\bi hope to(?:\s+become|\s+be)?\s+(.+)",
            r"\bi wanna be\s+(.+)",
            r"\bi(?:\s+also)?\s+want to become\s+(.+)",
            r"\bi(?:\s+also)?\s+want to be\s+(.+)",
            r"\bi(?:\s+also)?\s+want to\s+(.+)",
        ]
        for pattern in goal_patterns:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if match:
                goal = match.group(1).strip().rstrip(".!?")
                if len(goal) < 3:
                    continue
                # Normalize wording toward a stable content form.
                if not goal.lower().startswith("become") and re.search(
                    r"become|be a|be an",
                    " " + goal,
                    flags=re.IGNORECASE,
                ):
                    content = f"User wants to {goal}"
                elif goal.lower().startswith("become") or goal.lower().startswith("be "):
                    content = f"User wants to {goal}"
                else:
                    content = f"User wants to {goal}"

                # Prefer "become X" form when user said become/be.
                become = re.search(
                    r"(?:become|be)\s+(.+)$",
                    goal,
                    flags=re.IGNORECASE,
                )
                if become or re.search(
                    r"want to become|want to be|goal is to become|dream is to",
                    normalized,
                    flags=re.IGNORECASE,
                ):
                    target = become.group(1) if become else goal
                    target = re.sub(
                        r"^(?:become|be)\s+",
                        "",
                        target,
                        flags=re.IGNORECASE,
                    )
                    content = f"User wants to become {target.strip()}"

                # "want to learn …" is handled as learning, not career goal.
                if re.search(r"\blearn\b", goal, flags=re.IGNORECASE):
                    continue

                # Goals whose extracted value starts with an action verb
                # (build, create, develop, etc.) are projects, not goals.
                # Also reject "become build/create/..." which is malformed.
                if re.match(
                    r"^(?:build|create|develop|make|design|write|code)\b",
                    goal,
                    flags=re.IGNORECASE,
                ):
                    continue
                if re.match(
                    r"^become\s+(?:build|create|develop|make|design|write|code)\b",
                    goal,
                    flags=re.IGNORECASE,
                ):
                    continue

                value = re.sub(
                    r"^User wants to (?:become |build )?",
                    "",
                    content,
                    flags=re.IGNORECASE,
                ).strip()

                # Reject goals with empty or non-substantive content.
                # A goal must have a substantive object after "become"/"be";
                # temporal words or the bare verb do not qualify.
                value_lower = value.lower().strip()
                if not value_lower:
                    return None
                if value_lower == "become":
                    return None
                if value_lower in {"again", "anymore", "now", "currently", "today", "still", "future"}:
                    return None
                # Reject universal/vague pronouns that do not express a real goal.
                # Use normalization to catch common typos like "noting" → "nothing".
                normalized = _normalize_goal_value(value_lower)
                if normalized in {"nothing", "something", "anything", "everything"}:
                    return None
                # Also reject negation words and vague pronouns that cannot be goals.
                if value_lower in {"no", "none", "no one", "nobody", "neither", "nor"}:
                    return None
                # Also reject temporal phrases like "in future" where the
                # extracted value consists of a preposition + temporal noun.
                words = value_lower.split()
                if len(words) == 2 and words[1] in {"again", "anymore", "now", "currently", "today", "still", "future"}:
                    return None

                return {
                    "action": "ADD",
                    "category": "goal",
                    "fact_type": "career_goal",
                    "content": content,
                    "value": value,
                    "subject": normalize_goal_text(content),
                    "importance": "high",
                }
        return None

    def _extract_education(self, text: str):
        """Extract education/background facts.

        Handles formal education statements like:
        - "I am in diploma second year"
        - "I am pursuing B.Tech in computer science"
        - "I study at university"
        """
        # Explicit education context patterns
        education_patterns = [
            # "I am in diploma/degree X year"
            r"\bi\s+(?:am\s+in|study\s+in|pursuing)\s+(.+?)[.!?]?$",
            # "I study at ..."
            r"\bi\s+study\s+(?:at|in)\s+(.+?)[.!?]?$",
            # "My education is ..."
            r"\bmy\s+(?:education|course|degree|diploma)\s+is\s+(.+?)[.!?]?$",
        ]
        for pattern in education_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                detail = match.group(1).strip().rstrip(".!?")
                if len(detail) >= 2:
                    content = f"User is studying {detail}"
                    return {
                        "action": "ADD",
                        "category": "education",
                        "fact_type": "education",
                        "content": content,
                        "value": detail,
                        "subject": normalize_topic_subject(detail),
                        "importance": "high",
                    }
        return None

    def _extract_ambition(self, text: str) -> Optional[dict]:
        """Extract ambition statements — professional aspirations."""
        ambition_patterns = [
            r"\bmy ambition is to\s+(.+)",
            r"\bi ambition to\s+(.+)",
            r"\bmy ambition is\s+(.+)",
            r"\bi aspire to\s+(.+)",
        ]
        for pattern in ambition_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                goal = match.group(1).strip().rstrip(".!?")
                if len(goal) < 3:
                    continue
                content = f"User's ambition is to {goal}"
                value = strip_goal_filler(goal)
                return {
                    "action": "ADD",
                    "category": "ambition",
                    "fact_type": "ambition",
                    "content": content,
                    "value": value,
                    "subject": normalize_goal_text(content),
                    "importance": "high",
                }
        return None

    def _extract_dream(self, text: str) -> Optional[dict]:
        """Extract dream/aspiration statements."""
        dream_patterns = [
            r"\bmy dream is to\s+(.+)",
            r"\bi dream of\s+(.+)",
            r"\bi dream to\s+(.+)",
            r"\bmy dream is\s+(.+)",
        ]
        for pattern in dream_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                goal = match.group(1).strip().rstrip(".!?")
                if len(goal) < 3:
                    continue
                content = f"User's dream is to {goal}"
                value = strip_goal_filler(goal)
                return {
                    "action": "ADD",
                    "category": "dream",
                    "fact_type": "dream",
                    "content": content,
                    "value": value,
                    "subject": normalize_goal_text(content),
                    "importance": "high",
                }
        return None

    def _extract_priority(self, text: str) -> Optional[dict]:
        """Extract explicit priority statements.

        Matches patterns like:
        - "My main goal is to get a good rank in ECET" → priority + goal
        - "Actually, my current main priority is ECET" → priority update
        - "My top priority is ECET" → priority
        """
        # "my main goal is X" → store as goal + set priority
        main_goal_match = re.search(
            r"\bmy (?:main|primary|top)\s+goal is to\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if main_goal_match:
            goal_text = main_goal_match.group(1).strip().rstrip(".!?")
            if len(goal_text) >= 3:
                # This is handled by _extract_goal; we just flag it as priority
                # The goal extraction will create the goal, and we add priority here
                return {
                    "action": "ADD",
                    "category": "priority",
                    "fact_type": "priority",
                    "content": f"User's current main priority is {goal_text}",
                    "value": goal_text,
                    "importance": "high",
                    "linked_category": "goal",
                }

        # "my current main priority is X" → priority update
        priority_match = re.search(
            r"\bmy\s+(?:current\s+)?(?:main|primary|top)\s+priority is\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if priority_match:
            topic = priority_match.group(1).strip().rstrip(".!?")
            if len(topic) >= 2:
                return {
                    "action": "ADD",
                    "category": "priority",
                    "fact_type": "priority",
                    "content": f"User's current main priority is {topic}",
                    "value": topic,
                    "importance": "high",
                }

        return None

    def _extract_possession(self, text: str) -> Optional[dict]:
        """Extract generic possession/device statements.

        Matches patterns like:
        - "I have an Oppo A53" → possession, device
        - "I own a MacBook" → possession, device
        - "I possess a laptop" → possession, device
        - "I use an iPhone" → possession, device
        - "My phone is an iPhone" → possession, device

        Uses generic semantic structure, no hardcoded brands/models.
        """
        lower = text.lower()

        # "I have X" — but NOT "I have X tomorrow" (task) or "I have to X" (goal)
        have_match = re.search(
            r"\bi (?:have|have got|ve got)\s+(?:an?\s+)?(.+)",
            lower,
        )
        if have_match:
            item = have_match.group(1).strip().rstrip(".!?")
            # Reject if followed by temporal markers (task) or "to" (obligation)
            if re.search(r"\b(?:tomorrow|today|tonight|next|this|due|deadline)\b", item):
                return None
            if re.search(r"\bto\s+(?:do|go|finish|complete|submit|pay|buy|get)\b", item):
                return None
            # Reject abstract concepts (feelings, ideas, thoughts, etc.)
            abstract_concepts = {
                "feeling", "feelings", "idea", "ideas", "thought", "thoughts",
                "opinion", "opinions", "belief", "beliefs", "sense", "impression",
                "feeling that", "idea that", "thought that", "feeling like",
            }
            if any(item.startswith(concept) or item == concept for concept in abstract_concepts):
                return None
            if len(item) >= 2:
                return {
                    "action": "ADD",
                    "category": "possession",
                    "fact_type": "owned_item",
                    "content": f"User has {item}",
                    "value": item,
                    "subject": _classify_possession_subject(item),
                    "importance": "medium",
                }

        # "I own X"
        own_match = re.search(
            r"\bi (?:own|possess)\s+(?:an?\s+)?(.+)",
            lower,
        )
        if own_match:
            item = own_match.group(1).strip().rstrip(".!?")
            if len(item) >= 2:
                return {
                    "action": "ADD",
                    "category": "possession",
                    "fact_type": "owned_item",
                    "content": f"User owns {item}",
                    "value": item,
                    "subject": _classify_possession_subject(item),
                    "importance": "medium",
                }

        # "I use X"
        use_match = re.search(
            r"\bi (?:use|am using)\s+(?:an?\s+)?(.+)",
            lower,
        )
        if use_match:
            item = use_match.group(1).strip().rstrip(".!?")
            if len(item) >= 2:
                return {
                    "action": "ADD",
                    "category": "possession",
                    "fact_type": "owned_item",
                    "content": f"User uses {item}",
                    "value": item,
                    "subject": _classify_possession_subject(item),
                    "importance": "medium",
                }

        # "My X is Y" — e.g., "My phone is an iPhone"
        my_is_match = re.search(
            r"\bmy\s+(\w+)\s+is\s+(?:an?\s+)?(.+)",
            lower,
        )
        if my_is_match:
            subject_word = my_is_match.group(1).strip()
            item = my_is_match.group(2).strip().rstrip(".!?")
            # Only treat as possession if the subject is a device/item category
            device_subjects = {
                "phone", "mobile", "laptop", "computer", "tablet",
                "device", "watch", "earbuds", "headphones", "speaker",
                "keyboard", "mouse", "monitor", "tv", "camera",
                "car", "bike", "motorcycle", "vehicle",
            }
            if subject_word in device_subjects and len(item) >= 2:
                return {
                    "action": "ADD",
                    "category": "possession",
                    "fact_type": "owned_item",
                    "content": f"User's {subject_word} is {item}",
                    "value": item,
                    "subject": subject_word,
                    "importance": "medium",
                }

        return None

    def _extract_task(self, text: str) -> Optional[dict]:
        """Extract temporary event/task with time reference.

        Matches patterns like:
        - "i have ct exam tomorrow"
        - "i have a test next week"
        - "i have an appointment today"
        - "i have deadline friday"
        - "i am studying for my mid exam"
        - "i am preparing for my exam"
        - "i have a meeting at 3 PM"
        - "i have a class at 10 AM"
        """
        lower = text.lower()

        # Match "have X [time reference]" patterns
        task_match = re.search(
            r"\b(?:i\s+)?(?:have|got|hav?e)\s+(?:an?\s+)?(.+?)"
            r"\s+(?:tomorrow|tommorrow|tonight|today|"
            r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"this\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
            lower,
            flags=re.IGNORECASE,
        )
        if not task_match:
            # Match "have X at TIME" patterns (e.g., "have a meeting at 3 PM")
            time_match = re.search(
                r"\b(?:i\s+)?(?:have|got|hav?e)\s+(?:an?\s+)?(.+?)"
                r"\s+(?:at\s+\d{1,2}(?:\s*(?:am|pm|a\.m\.|p\.m\.))?|"
                r"at\s+(?:noon|midnight|morning|afternoon|evening))\b",
                lower,
                flags=re.IGNORECASE,
            )
            if time_match:
                event_text = time_match.group(1).strip().rstrip(".!?")
                if len(event_text) >= 2:
                    # Extract the full time reference
                    full_time = re.search(
                        r"\b(?:at\s+\d{1,2}(?:\s*(?:am|pm|a\.m\.|p\.m\.))?|"
                        r"at\s+(?:noon|midnight|morning|afternoon|evening))\b",
                        lower,
                        flags=re.IGNORECASE,
                    )
                    temporal_hint = full_time.group(0).strip() if full_time else None
                    content = f"User has {event_text} {temporal_hint or ''}".strip()
                    return {
                        "action": "ADD",
                        "category": "task",
                        "fact_type": "task",
                        "content": content,
                        "value": event_text,
                        "subject": event_text,
                        "importance": "medium",
                        "temporal_hint": temporal_hint,
                    }
            # Match "studying for X" / "preparing for X" patterns (temporary tasks)
            prep_match = re.search(
                r"\b(?:i(?:'m| am)\s+)?(?:studying|preparing)\s+for\s+(?:my\s+)?(.+)|"
                r"\b(?:i\s+want\s+to\s+)?(?:prepare|studying)\s+for\s+(?:my\s+)?(.+)",
                lower,
                flags=re.IGNORECASE,
            )
            if prep_match:
                event_text = (prep_match.group(1) or prep_match.group(2) or "").strip().rstrip(".!?")
                if len(event_text) >= 2:
                    content = f"User is preparing for {event_text}"
                    return {
                        "action": "ADD",
                        "category": "task",
                        "fact_type": "task",
                        "content": content,
                        "value": event_text,
                        "subject": event_text,
                        "importance": "medium",
                    }
            return None

        event_text = task_match.group(1).strip().rstrip(".!?")
        if len(event_text) < 2:
            return None

        # Extract time reference for expiration
        time_ref = re.search(
            r"\b(tomorrow|tommorrow|tonight|today|"
            r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"this\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
            lower,
            flags=re.IGNORECASE,
        )
        temporal_hint = time_ref.group(1) if time_ref else None

        content = f"User has {event_text} {temporal_hint or ''}".strip()
        return {
            "action": "ADD",
            "category": "task",
            "fact_type": "task",
            "content": content,
            "value": event_text,
            "subject": event_text,
            "importance": "medium",
            "temporal_hint": temporal_hint,
        }

    @staticmethod
    def _is_task_cancellation(text: str) -> bool:
        """Return True if the message is a negated task or cancellation request.

        Must be checked BEFORE is_temporary_message to avoid dropping
        cancellations like "I don't have any task tomorrow".
        """
        lower = text.lower()
        return any(re.search(pat, lower) for pat in TASK_CANCEL_PATTERNS)

    def _extract_task_cancellation(self, text: str) -> Optional[dict]:
        """Extract a task-cancellation operation.

        Matches negated task statements and returns a CANCEL action that
        the manager can use to deactivate the matching active task.
        """
        lower = text.lower()

        # Extract the event noun phrase (e.g. "exam", "ct exam", "task")
        cancel_match = re.search(
            r"(?:don'?t|do\s+not|no\s+longer)\s+have\s+(?:an?\s+)?(.+?)"
            r"\s+(?:tomorrow|tommorrow|tonight|today|"
            r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"this\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
            lower,
            flags=re.IGNORECASE,
        )
        if cancel_match:
            event_text = cancel_match.group(1).strip().rstrip(".!?")
            time_ref = re.search(
                r"(tomorrow|tommorrow|tonight|today|"
                r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
                r"friday|saturday|sunday)|"
                r"this\s+(?:week|month|monday|tuesday|wednesday|thursday|"
                r"friday|saturday|sunday)|"
                r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
                lower,
                flags=re.IGNORECASE,
            )
            temporal_hint = time_ref.group(1) if time_ref else None
            if len(event_text) >= 2:
                return {
                    "action": "CANCEL",
                    "category": "task",
                    "fact_type": "task",
                    "value": event_text,
                    "temporal_hint": temporal_hint,
                }

        # Generic cancellation: "cancel my exam", "forget my task", "drop the quiz"
        generic_cancel = re.search(
            r"(?:cancel|forget|drop)\s+(?:my|the|an?)\s+(.+?)(?:\.|$)",
            lower,
            flags=re.IGNORECASE,
        )
        if generic_cancel:
            event_text = generic_cancel.group(1).strip().rstrip(".!?")
            if len(event_text) >= 2:
                return {
                    "action": "CANCEL",
                    "category": "task",
                    "fact_type": "task",
                    "value": event_text,
                    "temporal_hint": None,
                }

        # "there is no X tomorrow"
        there_no_match = re.search(
            r"there\s+is\s+no\s+(.+?)"
            r"\s+(?:tomorrow|tommorrow|tonight|today|"
            r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"this\s+(?:week|month|monday|tuesday|wednesday|thursday|"
            r"friday|saturday|sunday)|"
            r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
            lower,
            flags=re.IGNORECASE,
        )
        if there_no_match:
            event_text = there_no_match.group(1).strip().rstrip(".!?")
            time_ref = re.search(
                r"(tomorrow|tommorrow|tonight|today|"
                r"next\s+(?:week|month|monday|tuesday|wednesday|thursday|"
                r"friday|saturday|sunday)|"
                r"this\s+(?:week|month|monday|tuesday|wednesday|thursday|"
                r"friday|saturday|sunday)|"
                r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b",
                lower,
                flags=re.IGNORECASE,
            )
            temporal_hint = time_ref.group(1) if time_ref else None
            if len(event_text) >= 2:
                return {
                    "action": "CANCEL",
                    "category": "task",
                    "fact_type": "task",
                    "value": event_text,
                    "temporal_hint": temporal_hint,
                }

        # Fallback: if we detected a cancellation pattern but couldn't parse
        # the event, still return a CANCEL with the full text as value so the
        # manager can do fuzzy matching against active tasks.
        if any(re.search(pat, lower) for pat in TASK_CANCEL_PATTERNS):
            return {
                "action": "CANCEL",
                "category": "task",
                "fact_type": "task",
                "value": text.strip().rstrip(".!?"),
                "temporal_hint": None,
            }

        return None

    def _extract_skill_or_interest(self, text: str):
        """Return one op dict, a list of ops, or None."""
        # Learning intent — distinct from skills and career goals.
        learn_match = re.search(
            r"\bi(?:\s+want to|'d like to|would like to)\s+learn\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if learn_match:
            topic = learn_match.group(1).strip().rstrip(".!?")
            if len(topic) >= 1:
                return {
                    "action": "ADD",
                    "category": "learning",
                    "fact_type": "learning",
                    "content": f"User wants to learn {topic}",
                    "value": topic,
                    "subject": normalize_topic_subject(topic),
                    "importance": "medium",
                }

        # Existing skills / knowledge.
        know_match = re.search(
            r"\bi (?:know|already know|can use|am skilled in|am good at)\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if know_match:
            skill = know_match.group(1).strip().rstrip(".!?")
            if len(skill) >= 1:
                return {
                    "action": "ADD",
                    "category": "skill",
                    "fact_type": "skill",
                    "content": f"User knows {skill}",
                    "value": skill,
                    "subject": normalize_topic_subject(skill),
                    "importance": "medium",
                }

        # Active learning status — "I am learning X" maps to learning category.
        learn_status_match = re.search(
            r"\bi(?:'m| am)\s+(?:learning|studing|studying)\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if learn_status_match:
            topic = learn_status_match.group(1).strip().rstrip(".!?")
            topic = strip_temporal_suffix(topic)
            if len(topic) >= 1:
                # Check for formal education context signals in the topic
                formal_signals = re.search(
                    r"\b(diploma|degree|course|college|university|school|"
                    r"bachelor|master|phd|b\.?tech|m\.?tech|engineering|"
                    r"computer science|cs|cse|ecet|first year|second year|"
                    r"third year|fourth year|semester|year)\b",
                    topic,
                    flags=re.IGNORECASE,
                )
                if formal_signals:
                    return {
                        "action": "ADD",
                        "category": "education",
                        "fact_type": "studying",
                        "content": f"User is studying {topic}",
                        "value": topic,
                        "subject": normalize_topic_subject(topic),
                        "importance": "high",
                    }
                return {
                    "action": "ADD",
                    "category": "learning",
                    "fact_type": "learning",
                    "content": f"User is learning {topic}",
                    "value": topic,
                    "subject": normalize_topic_subject(topic),
                    "importance": "medium",
                }

        # Interests — including common typos of "interested".
        interest_match = re.search(
            r"\bi am (?:interested|intrested|intrest|into)\s+(?:in\s+)?(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if not interest_match:
            interest_match = re.search(
                r"\bi(?:'m)?\s+(?:interested|intrested)\s+in\s+(.+)",
                text,
                flags=re.IGNORECASE,
            )
        if interest_match:
            interest_blob = interest_match.group(1).strip().rstrip(".!?")
            if len(interest_blob) < 2:
                return None
            items = split_list_items(interest_blob)
            if len(items) <= 1:
                items = [interest_blob]
            ops = []
            for interest in items:
                ops.append({
                    "action": "ADD",
                    "category": "interest",
                    "fact_type": "interest",
                    "content": f"User is interested in {interest}",
                    "value": interest,
                    "subject": normalize_topic_subject(interest),
                    "importance": "medium",
                })
            return ops if len(ops) > 1 else ops[0]
        return None
