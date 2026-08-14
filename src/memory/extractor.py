"""Deterministic durable-fact extraction for ALLI v0.6."""

from __future__ import annotations

import re
from typing import Optional


CHANGE_SIGNAL_PHRASES = (
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


def has_explicit_change_signal(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in CHANGE_SIGNAL_PHRASES)


def is_temporary_message(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in TEMPORARY_PHRASES)


def looks_like_question(message: str) -> bool:
    text = message.strip().lower()
    if text.endswith("?"):
        return True
    return bool(
        re.match(
            r"^(what|who|where|when|why|how|do you|did i|can you|tell me)\b",
            text,
        )
    )


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
        r"going|want|wanna|hope|plan)\b",
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


def goal_match_score(hint: str, memory: dict) -> float:
    """
    Lexical confidence that `hint` refers to this goal memory.

    Does not authorize writes by itself — callers still require an explicit
    change/delete signal from the extractor.
    """
    hint_norm = normalize_goal_text(strip_goal_filler(hint or ""))
    if not hint_norm:
        return 0.0

    content_norm = normalize_goal_text(str(memory.get("content", "")))
    value_norm = normalize_goal_text(
        strip_goal_filler(str(memory.get("value", "") or ""))
    )
    subject_norm = normalize_goal_text(str(memory.get("subject", "") or ""))

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
        "with", "using", "to",
    }
    words = [word for word in text.split() if word not in stop]
    return " ".join(words)


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

        if is_temporary_message(text):
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
        for extractor in (
            self._extract_nickname,
            self._extract_preference,
            self._extract_skill_or_interest,
            self._extract_name,
            self._extract_goal,
        ):
            operation = extractor(text)
            if not operation:
                continue
            if isinstance(operation, list):
                return operation
            return [operation]

        return []

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
                if len(subject) < 1:
                    return None
                # Allow single-token subjects such as "C".
                if len(subject) == 1 and not subject.isalnum():
                    return None
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
        ]
        for pattern in positive_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                subject = match.group(1).strip().rstrip(".!?")
                if len(subject) < 1:
                    return None
                # Allow single-token subjects such as "C".
                if len(subject) == 1 and not subject.isalnum():
                    return None
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
        # "I also want to build..." may be project rather than career goal.
        build_match = re.search(
            r"\bi(?:\s+also)?\s+want to build\s+(.+)",
            text,
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

        goal_patterns = [
            r"\bmy goal is to\s+(.+)",
            r"\bmy dream is to\s+(.+)",
            r"\bi dream of\s+(.+)",
            r"\bi aim to\s+(.+)",
            r"\bi plan to\s+(.+)",
            r"\bi hope to(?:\s+become|\s+be)?\s+(.+)",
            r"\bi wanna be\s+(.+)",
            r"\bi(?:\s+also)?\s+want to become\s+(.+)",
            r"\bi(?:\s+also)?\s+want to be\s+(.+)",
            r"\bi(?:\s+also)?\s+want to\s+(.+)",
        ]
        for pattern in goal_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
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
                    text,
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

                value = re.sub(
                    r"^User wants to (?:become |build )?",
                    "",
                    content,
                    flags=re.IGNORECASE,
                ).strip()

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
            if len(topic) >= 2:
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
            if len(skill) >= 2:
                return {
                    "action": "ADD",
                    "category": "skill",
                    "fact_type": "skill",
                    "content": f"User knows {skill}",
                    "value": skill,
                    "subject": normalize_topic_subject(skill),
                    "importance": "medium",
                }

        # Learning status (not a completed skill).
        learn_status_match = re.search(
            r"\bi(?:'m| am)\s+learning\s+(.+)",
            text,
            flags=re.IGNORECASE,
        )
        if learn_status_match:
            topic = learn_status_match.group(1).strip().rstrip(".!?")
            if len(topic) >= 2:
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
