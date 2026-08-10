"""
Grounded answer generation from retrieved structured memories.

Formats stored facts into concise answers. Does not invent personal information.
Does not map question strings to hard-coded replies.
"""

from __future__ import annotations

from typing import Optional

from .query_intent import QueryIntent


MISSING_TEMPLATES = {
    "name": "I don't currently have any stored information about your name.",
    "nickname": "I don't currently have any stored information about your nickname.",
    "goal": "I don't currently have any stored information about your goals.",
    "skill": "I don't currently have any stored information about your skills.",
    "learning": "I don't currently have any stored information about your learning goals.",
    "preference": "I don't currently have any stored information about your preferences.",
    "interest": "I don't currently have any stored information about your interests.",
    "project": "I don't currently have any stored information about your projects.",
    "habit": "I don't currently have any stored information about your habits.",
    "background": "I don't currently have any stored information about your background.",
    "personal_overview": "I don't currently have any long-term information stored about you.",
}


def missing_answer(intent: QueryIntent, question: str = "") -> str:
    """Build a clear missing-info response without guessing."""
    import re

    text = (question or "").strip().lower()
    noun = re.search(r"\bfavou?rite\s+([a-z0-9]+)", text)
    if noun:
        return f"I don't have your favorite {noun.group(1)} stored yet."

    return MISSING_TEMPLATES.get(
        intent.primary,
        "I don't currently have that information stored.",
    )


SECTION_TITLES = {
    "name": "Name",
    "nickname": "Nickname",
    "skill": "Skills",
    "goal": "Goals",
    "learning": "Learning",
    "preference": "Preferences",
    "interest": "Interests",
    "project": "Projects",
    "habit": "Habits",
    "background": "Education",
}


def grounded_answer(
    intent: QueryIntent,
    memories: list[dict],
    question: str = "",
    grouped: Optional[dict[str, list[dict]]] = None,
) -> str:
    """
    Build a concise answer strictly from retrieved memories.

    If no memories were retrieved for a memory-read intent, report missing info.
    For multi-category questions, pass `grouped` intent→memories.
    """
    if grouped and len(grouped) > 1:
        return _multi_category_answer(intent, grouped, question=question)

    if intent.is_memory_read and not memories:
        return missing_answer(intent, question)

    if not memories:
        return missing_answer(intent, question)

    return _single_intent_answer(intent.primary, memories, question=question)


def _multi_category_answer(
    intent: QueryIntent,
    grouped: dict[str, list[dict]],
    question: str = "",
) -> str:
    sections = []
    order = intent.requested or list(grouped.keys())
    for label in order:
        memories = grouped.get(label) or []
        title = SECTION_TITLES.get(label, label.title())
        if not memories:
            sections.append(
                f"{title}:\n- I don't currently have any stored information "
                f"about your {title.lower()}."
            )
            continue

        if label in {"skill", "goal", "interest", "learning", "project", "habit", "background"}:
            bullets = []
            for memory in memories:
                value = _display_value(memory)
                if label == "goal" and not value.lower().startswith(("become ", "be ")):
                    if memory.get("fact_type") == "learning_goal":
                        if not value.lower().startswith("learn"):
                            value = f"learn {value}"
                    else:
                        value = f"become {value}"
                bullets.append(f"- {value}")
            sections.append(f"{title}:\n" + "\n".join(bullets))
        elif label == "preference":
            body = _preference_answer(memories)
            sections.append(f"{title}:\n- {body}")
        elif label in {"name", "nickname"}:
            body = _single_intent_answer(label, memories, question=question)
            sections.append(f"{title}:\n- {body}")
        else:
            bullets = [f"- {m.get('content', '')}" for m in memories]
            sections.append(f"{title}:\n" + "\n".join(bullets))

    if not sections:
        return missing_answer(intent, question)
    return "\n\n".join(sections)


def _single_intent_answer(
    primary: str,
    memories: list[dict],
    question: str = "",
) -> str:
    if primary == "name":
        return _identity_answer(memories, preferred="name", allow_both=False)

    if primary == "nickname":
        return _identity_answer(memories, preferred="nickname", allow_both=False)

    if primary == "goal":
        return _goal_answer(memories)

    if primary == "skill":
        return _list_answer(
            memories,
            singular="Your skill is",
            plural="Your skills are",
            use_values=True,
            fallback_prefix="Your skills:",
        )

    if primary == "learning":
        return _list_answer(
            memories,
            singular="You are learning",
            plural="You are learning",
            use_values=True,
            fallback_prefix="Your learning:",
        )

    if primary == "preference":
        return _preference_answer(memories)

    if primary == "interest":
        return _list_answer(
            memories,
            singular="You are interested in",
            plural="You are interested in",
            use_values=True,
            fallback_prefix="Your interests:",
        )

    if primary == "project":
        return _list_answer(
            memories,
            singular="Your project is",
            plural="Your projects are",
            use_values=True,
            fallback_prefix="Your projects:",
        )

    if primary == "habit":
        return _list_answer(
            memories,
            singular="Your habit is",
            plural="Your habits are",
            use_values=True,
            fallback_prefix="Your habits:",
        )

    if primary == "background":
        return _list_answer(
            memories,
            singular="About your background:",
            plural="About your background:",
            use_values=False,
            fallback_prefix="About your background:",
        )

    if primary == "personal_overview":
        lines = ["Here is what I currently have stored about you:"]
        for memory in memories:
            lines.append(f"- {memory.get('content', '')}")
        return "\n".join(lines)

    if len(memories) == 1:
        return memories[0].get(
            "content",
            "I don't currently have that information stored.",
        )
    lines = ["Here is the relevant stored information:"]
    for memory in memories:
        lines.append(f"- {memory.get('content', '')}")
    return "\n".join(lines)


def _identity_answer(
    memories: list[dict],
    *,
    preferred: str,
    allow_both: bool = False,
) -> str:
    preferred_memories = [
        m for m in memories if m.get("fact_type") == preferred
    ]
    if not preferred_memories and preferred == "name":
        return MISSING_TEMPLATES["name"]
    if not preferred_memories and preferred == "nickname":
        return MISSING_TEMPLATES["nickname"]

    names = [m for m in memories if m.get("fact_type") == "name"]
    nicks = [m for m in memories if m.get("fact_type") == "nickname"]
    if allow_both and names and nicks:
        name_val = _display_value(names[0])
        nick_val = _display_value(nicks[0])
        return (
            f"I can call you {nick_val}, your nickname, "
            f"or {name_val}, your name."
        )

    target = preferred_memories[0] if preferred_memories else memories[0]
    value = _display_value(target)
    if preferred == "name":
        return f"Your name is {value}."
    return f"Your nickname is {value}."


def _goal_answer(memories: list[dict]) -> str:
    import re

    parts = []
    for memory in memories:
        value = _display_value(memory)
        if not re.match(r"^(become|be)\b", value, flags=re.IGNORECASE):
            if memory.get("fact_type") == "learning_goal":
                if not value.lower().startswith("learn"):
                    value = f"learn {value}"
            else:
                value = f"become {value}"
        parts.append(value)

    if not parts:
        return MISSING_TEMPLATES["goal"]
    if len(parts) == 1:
        return f"Your goal is to {parts[0]}."
    return f"Your goals are to {_join_natural(parts)}."


def _display_value(memory: dict) -> str:
    if memory.get("category") == "preference":
        subject = memory.get("subject")
        if isinstance(subject, str) and subject.strip():
            return subject.strip()

    value = memory.get("value")
    if (
        isinstance(value, str)
        and value.strip()
        and value.strip().lower() not in {"like", "dislike"}
    ):
        return value.strip()

    content = str(memory.get("content", "")).strip()
    for prefix in (
        "User's name is ",
        "User's nickname is ",
        "User wants to become ",
        "User wants to learn ",
        "User wants to ",
        "User likes ",
        "User dislikes ",
        "User is interested in ",
        "User knows ",
        "User is learning ",
    ):
        if content.lower().startswith(prefix.lower()):
            return content[len(prefix):].strip()
    return content


def _list_answer(
    memories: list[dict],
    *,
    singular: str,
    plural: str,
    use_values: bool,
    fallback_prefix: str,
) -> str:
    values = []
    for memory in memories:
        if use_values:
            values.append(_display_value(memory))
        else:
            values.append(str(memory.get("content", "")).strip())

    values = [v for v in values if v]
    if not values:
        return "I don't currently have that information stored."

    if len(values) == 1:
        # Avoid double "to to" if value already starts with become/to
        value = values[0]
        if singular.endswith(" to") and value.lower().startswith("become "):
            return f"{singular} {value}."
        if singular.endswith(" to") and not value.lower().startswith(("become ", "be ")):
            return f"{singular} {value}."
        return f"{singular} {value}."

    joined = _join_natural(values)
    if plural.endswith(" to"):
        return f"{plural} {joined}."
    return f"{plural} {joined}."


def _join_natural(parts: list[str]) -> str:
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _preference_answer(memories: list[dict]) -> str:
    likes = []
    dislikes = []
    for memory in memories:
        subject = _display_value(memory)
        polarity = memory.get("fact_type") or memory.get("value")
        if polarity == "dislike":
            dislikes.append(subject)
        else:
            likes.append(subject)

    parts = []
    if likes:
        if len(likes) == 1:
            parts.append(f"You like {likes[0]}.")
        else:
            parts.append(f"You like {_join_natural(likes)}.")
    if dislikes:
        if len(dislikes) == 1:
            parts.append(f"You dislike {dislikes[0]}.")
        else:
            parts.append(f"You dislike {_join_natural(dislikes)}.")
    if not parts:
        return MISSING_TEMPLATES["preference"]
    return " ".join(parts)


def llm_memory_system_prompt() -> str:
    return (
        "You are ALLI, a personal AI assistant.\n"
        "Answer only from the supplied memories.\n"
        "Treat memories as stored user facts.\n"
        "Do not infer unsupported personal facts.\n"
        "Do not invent skills from goals or learning plans.\n"
        "Do not invent dreams or preferences.\n"
        "If the answer is not contained in the memories, say that the "
        "information is not stored.\n"
        "Keep simple memory answers concise.\n"
        "Do not ask unnecessary follow-up questions.\n"
        "Always reply in English."
    )
