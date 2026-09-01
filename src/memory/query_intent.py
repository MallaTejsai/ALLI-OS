"""
Semantic query-intent detection for ALLI memory reads.

Intent is inferred by embedding similarity to concept prototypes,
with light query normalization and multi-category support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from . import config
from .embeddings import EmbeddingError, EmbeddingStore, cosine_similarity


def normalize_memory_query(query: str) -> str:
    """
    Normalize phrasing before intent embedding.

    Collapses equivalent nickname spellings so semantic matching is stable.
    """
    text = (query or "").strip().lower()
    text = text.replace("\u2019", "'")
    # nick-name / nick name → nickname (generic, not a one-off answer rule)
    text = re.sub(r"\bnick[\s\-]+name\b", "nickname", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# Normalization map for common typos that change semantic meaning.
# Used to normalize vague/invalid goal checks so typos like "noting"
# are treated identically to "nothing".
_CANONICAL_WORDS = {
    "nothing": {"noting", "nothign", "noting"},
    "something": {"somthing", "smething"},
    "anything": {"anythign", "anyting"},
    "everything": {"everythign", "evrything"},
}


def _normalize_goal_word(word: str) -> str:
    """Normalize a single word, correcting obvious typos."""
    # Strip trailing punctuation before checking
    stripped = word.strip(".,!?;:'\"")
    lower = stripped.lower()
    for canonical, typos in _CANONICAL_WORDS.items():
        if lower in typos or lower == canonical:
            return canonical
    return word  # Return original with punctuation intact


def _normalize_vague_goals(text: str) -> str:
    """Replace common typos of vague goal words with canonical forms.

    This ensures that "noting" is treated the same as "nothing" in
    vague/incomplete statement detection.
    """
    lower = text.lower()
    words = lower.split()
    normalized = [_normalize_goal_word(w) for w in words]
    return " ".join(normalized)


# Prototype phrases describe the *meaning* of each memory concept.
CONCEPT_PROTOTYPES: dict[str, list[str]] = {
    "name": [
        "What is the user's real given name?",
        "Who is the user by legal or personal name?",
        "Tell me the person's actual name.",
        "What name did the user say they have?",
    ],
    "nickname": [
        "What is the user's nickname or alias?",
        "What informal name do people use for the user?",
        "What friendly name can I use when addressing the user?",
        "What do people call the user for short?",
    ],
    "goal": [
        "What are the user's life goals and aims?",
        "What does the user want to become in the future?",
        "What is the user trying to achieve long term?",
        "What career outcomes is the user aiming for?",
    ],
    "ambition": [
        "What are the user's professional ambitions?",
        "What career drive and aspiration does the user have?",
        "What does the user aim to achieve professionally?",
        "What is the user's ambition in life?",
    ],
    "dream": [
        "What are the user's dreams and aspirations?",
        "What does the user dream of becoming?",
        "What is the user's lifelong vision?",
        "What does the user dream of achieving?",
    ],
    "skill": [
        "What skills abilities and competencies does the user already have?",
        "What does the user already know how to do well?",
        "Which subjects or tools has the user mastered?",
        "Does the user know a particular skill already?",
        "Do I already know how to use this technology?",
        "Am I skilled at this subject?",
    ],
    "learning": [
        "What does the user want to learn next?",
        "What topics is the user currently learning?",
        "What learning goals does the user have?",
        "What am I learning?",
        "What topics is the user currently studying?",
    ],
    "preference": [
        "What does the user like or dislike?",
        "What are the user's preferences and tastes?",
        "What does the user enjoy or hate?",
        "What do I like?",
    ],
    "interest": [
        "What are the user's interests and hobbies?",
        "What topics is the user interested in?",
    ],
    "education": [
        "What is the user's education background?",
        "What is the user studying formally?",
        "Where does the user study or what degree are they pursuing?",
        "What course or diploma is the user enrolled in?",
        "What am I studying in school or college?",
    ],
    "project": [
        "What projects is the user building or working on?",
        "What is the user creating or developing?",
    ],
    "habit": [
        "What habits or daily routines does the user have?",
        "What does the user usually do regularly?",
    ],
    "task": [
        "What temporary tasks does the user have?",
        "What does the user need to do soon?",
    ],
    "priority": [
        "What is the user's current main priority?",
        "What is the user focusing on right now?",
        "What is the user's top priority?",
    ],
    "personal_overview": [
        "What do you remember about the user overall?",
        "Summarize the stored personal facts about the user.",
        "What personal information is known about me?",
    ],
    "general_knowledge": [
        "Explain a technical or academic concept in general.",
        "How does a programming language feature work?",
        "Teach me about science math or technology unrelated to my personal life.",
        "What is the definition of a computer science topic?",
    ],
}


# Lexical cues used only to detect *multiple* requested categories in one question.
# They do not replace semantic retrieval of memories.
CATEGORY_QUERY_CUES: dict[str, tuple[str, ...]] = {
    "name": (r"\bmy name\b", r"\bwho am i\b"),
    "nickname": (r"\bnickname\b", r"\bnick[\s\-]+name\b", r"\bcall me\b"),
    "goal": (
        r"\bgoals?\b",
        r"\b(?:my\s+)?goals?\b",
        r"\bwant to become\b",
    ),
    "ambition": (r"\bambitions?\b",),
    "dream": (r"\bdreams?\b",),
    "skill": (r"\bskills?\b",),
    "learning": (r"\blearning\b", r"\bam i learning\b", r"\bwhat i(?:'m| am) learning\b", r"\bwant to learn\b"),
    "preference": (
    r"\bpreferences?\b",
    r"\bwhat\s+(?:do\s+)?i\s+(?:like|dislike|enjoy|hate)\b",
    r"\bwhat\s+don'?t\s+i\s+(?:like|enjoy)\b",
    r"\bwhich\b.*\b(?:i\s+)?(?:like|dislike|enjoy|hate)\b",
),
    "interest": (r"\binterests?\b", r"\bhobb(?:y|ies)\b"),
    "education": (r"\bstud(?:ying|ing)\b", r"\bstud(?:y|ies)\b", r"\beducation\b", r"\bcollege\b", r"\bbackground\b", r"\bcourse\b", r"\bdiploma\b"),
    "project": (r"\bprojects?\b", r"\bbuild(?:ing)?\b", r"\bcreating\b", r"\bdevelop(?:ing)?\b"),
    "habit": (r"\bhabits?\b", r"\broutines?\b"),
    "task": (r"\btasks?\b", r"\bto[\s-]dos?\b", r"\bneed to do\b", r"\bhave\b.*?\b(?:tomorrow|tommorrow|today|tonight)\b"),
    "priority": (r"\bpriority\b", r"\bpriorities\b", r"\bmain focus\b", r"\btop focus\b"),
}


INTENT_FILTERS: dict[str, dict] = {
    "name": {
        "categories": {"identity"},
        "fact_types": {"name"},
        "strict": True,
    },
    "nickname": {
        "categories": {"identity"},
        "fact_types": {"nickname"},
        "strict": True,
    },
    "goal": {
        "categories": {"goal"},
        "fact_types": None,
        "exclude_fact_types": {"learning_goal"},
        "strict": True,
    },
    "ambition": {
        "categories": {"ambition"},
        "fact_types": None,
        "strict": True,
    },
    "dream": {
        "categories": {"dream"},
        "fact_types": None,
        "strict": True,
    },
    "skill": {
        "categories": {"skill"},
        "fact_types": {"skill"},
        "strict": True,
    },
    "learning": {
        "categories": {"learning"},
        "fact_types": {"learning", "learning_goal"},
        "strict": True,
    },
    "preference": {
        "categories": {"preference"},
        "fact_types": None,
        "strict": True,
    },
    "interest": {
        "categories": {"interest"},
        "fact_types": None,
        "strict": True,
    },
    "education": {
        "categories": {"education"},
        "fact_types": None,
        "strict": True,
    },
    "project": {
        "categories": {"project"},
        "fact_types": None,
        "strict": True,
    },
    "habit": {
        "categories": {"routine"},
        "fact_types": None,
        "strict": True,
    },
    "task": {
        "categories": {"task"},
        "fact_types": None,
        "strict": True,
    },
    "priority": {
        "categories": {"priority"},
        "fact_types": {"priority"},
        "strict": True,
    },
    "personal_overview": {
        "categories": None,
        "fact_types": None,
        "strict": False,
    },
}


MEMORY_READ_INTENTS = frozenset(INTENT_FILTERS.keys())

CONFUSABLE_INTENTS = {
    "skill": frozenset({"learning"}),
    "learning": frozenset({"skill"}),
    "goal": frozenset({"learning", "ambition", "dream"}),
    "ambition": frozenset({"goal", "dream"}),
    "dream": frozenset({"goal", "ambition"}),
    "name": frozenset({"nickname"}),
    "nickname": frozenset({"name"}),
    "education": frozenset({"learning"}),
}


@dataclass
class QueryIntent:
    """Result of semantic intent detection."""

    primary: str
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    is_memory_read: bool = False
    preference_polarity: Optional[str] = None
    is_general_knowledge: bool = False
    filters: dict = field(default_factory=dict)
    requested: list[str] = field(default_factory=list)
    normalized_query: str = ""
    needs_clarification: bool = False
    clarification_message: str = ""

    @property
    def label(self) -> str:
        return self.primary

    @property
    def is_multi(self) -> bool:
        return len(self.requested) > 1


class QueryIntentDetector:
    """Detect memory-read intent via prototype embeddings + normalization."""

    def __init__(
        self,
        embedding_store: Optional[EmbeddingStore] = None,
        *,
        intent_threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ):
        self.embedding_store = embedding_store or EmbeddingStore()
        self.intent_threshold = (
            intent_threshold
            if intent_threshold is not None
            else config.INTENT_SIMILARITY_THRESHOLD
        )
        self.margin = (
            margin if margin is not None else config.INTENT_SCORE_MARGIN
        )
        self._prototype_vectors: dict[str, list[list[float]]] = {}
        self._ready = False

    def _ensure_prototypes(self) -> None:
        if self._ready:
            return
        for concept, phrases in CONCEPT_PROTOTYPES.items():
            self._prototype_vectors[concept] = [
                self.embedding_store.generate_embedding(phrase)
                for phrase in phrases
            ]
        self._ready = True

    def detect(self, query: str) -> QueryIntent:
        raw = (query or "").strip()
        normalized = normalize_memory_query(raw)
        preference_polarity = self._detect_preference_polarity(normalized)
        if not normalized:
            return QueryIntent(
                primary="unknown",
                score=0.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

        # Forecasts are not reads of today's stored profile.  In particular,
        # a goal is an aspiration, not evidence of a guaranteed future fact.
        if self._is_future_forecast(normalized):
            return QueryIntent(
                primary="unknown",
                score=0.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

        # These ask for advice or a relationship to an unnamed object, not a
        # stored profile list.  They may be answered by the chat layer, with
        # memory supplied as optional context, but must not retrieve a category
        # merely because it is semantically adjacent.
        if self._is_advice_or_ambiguous_memory_query(normalized):
            # Vague preferences and incomplete statements need clarification,
            # not an LLM response.
            clarification = self._get_clarification_message(normalized)
            if clarification:
                return QueryIntent(
                    primary="unknown",
                    score=0.0,
                    is_memory_read=False,
                    is_general_knowledge=False,
                    normalized_query=normalized,
                    needs_clarification=True,
                    clarification_message=clarification,
                )
            return QueryIntent(
                primary="unknown",
                score=0.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

        # Clearly general-knowledge questions (math, definitions, facts)
        # must never enter the memory-read pipeline.
        if self._is_clearly_general_knowledge(normalized):
            return QueryIntent(
                primary="general_knowledge",
                score=1.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

        try:
            self._ensure_prototypes()
            query_vec = self.embedding_store.generate_embedding(normalized)
        except EmbeddingError:
            return self._cue_only_fallback(normalized)

        scores: dict[str, float] = {}
        for concept, vectors in self._prototype_vectors.items():
            best = 0.0
            for vector in vectors:
                best = max(best, cosine_similarity(query_vec, vector))
            scores[concept] = best

        # Keep name and nickname queries from collapsing into each other.
        if re.search(r"\bnickname\b", normalized):
            scores["nickname"] = max(
                scores.get("nickname", 0.0),
                scores.get("name", 0.0) + 0.05,
            )
            if scores.get("name", 0.0) >= scores.get("nickname", 0.0):
                scores["name"] = scores["nickname"] - 0.08
        elif re.search(r"\bmy name\b|\bwho am i\b", normalized):
            scores["name"] = max(
                scores.get("name", 0.0),
                scores.get("nickname", 0.0) + 0.05,
            )
            if scores.get("nickname", 0.0) >= scores.get("name", 0.0):
                scores["nickname"] = scores["name"] - 0.08

        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        primary, best_score = ranked[0]
        cued = self._cued_intents(normalized)

        # Explicit category cues win over near-miss semantic neighbors
        # (e.g. "dreams" must stay goal, not interest).
        if cued:
            best_cued = max(cued, key=lambda concept: scores.get(concept, 0.0))
            cued_score = scores.get(best_cued, 0.0)
            if cued_score >= self.intent_threshold * 0.85:
                primary = best_cued
                best_score = cued_score

        is_general = (
            primary == "general_knowledge"
            and best_score >= self.intent_threshold
        )
        memory_floor = (
            self.intent_threshold * 0.85 if cued else self.intent_threshold
        )
        is_memory = (
            primary in MEMORY_READ_INTENTS
            and best_score >= memory_floor
            and not is_general
        )

        # Strong temporal cue: "have X tomorrow/today" → always task.
        # The semantic similarity for terse temporal phrases can fall below
        # the intent threshold, but the pattern itself is unambiguous.
        if "task" in cued and re.search(
            r"\bhave\b.*?\b(?:tomorrow|tommorrow|today|tonight)\b", normalized
        ):
            primary = "task"
            best_score = max(scores.get("task", 0.0), 0.5)
            is_memory = True

        if is_general:
            for concept, score in ranked[1:]:
                if (
                    concept in MEMORY_READ_INTENTS
                    and score >= self.intent_threshold
                    and (best_score - score) < self.margin
                ):
                    primary = concept
                    best_score = score
                    is_general = False
                    is_memory = True
                    break

        requested: list[str] = []
        if is_memory:
            requested = self._select_requested_intents(
                normalized, scores, primary, best_score, cued=cued
            )
            primary = requested[0]
            best_score = scores.get(primary, best_score)

        filters = dict(INTENT_FILTERS.get(primary, {})) if is_memory else {}

        return QueryIntent(
            primary=primary if (is_memory or is_general) else "unknown",
            score=best_score,
            scores=scores,
            is_memory_read=is_memory,
    preference_polarity=(
        preference_polarity
        if (is_memory or primary == "preference")
        else None
    ),
            is_general_knowledge=is_general,
            filters=filters,
            requested=requested if is_memory else [],
            normalized_query=normalized,
        )
    def _detect_preference_polarity(
        self,
        normalized_query: str,
    ) -> Optional[str]:
        """Detect whether a preference question asks for likes or dislikes."""
        text = normalized_query.lower()

        if re.search(
            r"\bwhat\s+(?:(?:do|does)\s+)?(?:i|the user)\s+dislike\b",
            text,
        ):
            return "dislike"

        if re.search(
            r"\bwhat\s+(?:(?:do|does)\s+)?(?:i|the user)\s+"
            r"(?:do\s+not|don't|does\s+not|doesn't|not)\s+like\b"
            r"|\bwhat\s+(?:don't|do\s+not|doesn't|does\s+not)\s+"
            r"(?:i|the user)\s+like\b",
            text,
        ):
            return "dislike"

        if re.search(
            r"\bwhat\s+(?:(?:do|does)\s+)?(?:i|the user)\s+like\b",
            text,
        ):
            return "like"

        if re.search(
            r"\b(?:what|which)\b.*\b(?:don'?t|do\s+not)\s+"
            r"i\s+enjoy\b",
            text,
        ):
            return "dislike"

        if re.search(
            r"\b(?:what|which)\b.*\b(?:do\s+)?i\s+enjoy\b",
            text,
        ):
            return "like"

        return None

    def _cue_only_fallback(self, normalized: str) -> QueryIntent:
        """Deterministic fallback when embeddings are unavailable.

        Uses lexical cues to route queries to the correct memory category
        without requiring Ollama. Returns 'unknown' only when no cues match.
        """
        cued = self._cued_intents(normalized)
        if not cued:
            return QueryIntent(
                primary="unknown",
                score=0.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

        # Prefer the first cued intent (order in CATEGORY_QUERY_CUES
        # reflects priority for overlapping matches).
        primary = cued[0]
        filters = dict(INTENT_FILTERS.get(primary, {}))
        requested = list(dict.fromkeys(cued))

        return QueryIntent(
            primary=primary,
            score=0.8,
            scores={concept: 0.8 for concept in cued},
            is_memory_read=primary in MEMORY_READ_INTENTS,
            is_general_knowledge=False,
            filters=filters,
            requested=requested,
            normalized_query=normalized,
        )

    @staticmethod
    def _is_clearly_general_knowledge(normalized_query: str) -> bool:
        """Identify questions that are clearly about general knowledge, not personal memory."""
        return bool(
            re.search(
                r"\bwhat\s+is\s+\d+\s*[+\-*/]\s*\d+\b|"
                r"\bwhat\s+is\s+the\s+(?:capital|population|currency|president|"
                r"founder|ceo|meaning|definition|formula|speed)\b|"
                r"\bwho\s+(?:invented|discovered|created|wrote|composed)\b|"
                r"\bhow\s+(?:do|does|did|can|could|would|should)\s+"
                r"(?:you|we|one|i|people|humans|computers|animals)\b|"
                r"\bwhen\s+(?:was|did|were|is)\s+(?:the|a|an)\b|"
                r"\bwhere\s+(?:is|are|was|were)\s+(?:the|a|an)\b",
                normalized_query,
            )
        )

    @staticmethod
    def _is_future_forecast(normalized_query: str) -> bool:
        """Identify speculative future questions rather than memory reads."""
        return bool(
            re.search(
                r"\b(?:will\s+i|will\s+my|my\s+future|future\s+career|"
                r"in\s+\d+\s+years?|definitely\s+be)\b",
                normalized_query,
            )
        )

    @staticmethod
    def _is_advice_or_ambiguous_memory_query(normalized_query: str) -> bool:
        """Keep recommendations, vague statements, and unnamed-object questions
        out of memory reads."""
        # Normalize typos (e.g., "noting" → "nothing") so vague/invalid
        # goal statements are caught regardless of spelling.
        query = _normalize_vague_goals(normalized_query)
        return bool(
            re.search(
                r"\bwhat\s+should\s+i\s+learn\b|"
                r"\bwhich\s+technology\s+is\s+best\s+for\s+me\b|"
                r"\bwhat\s+do\s+i\s+like\s+about\s+"
                r"(?:something|it|that|this)\b|"
                r"\bwhat\s+should\s+i\s+study\b|"
                # Vague preference statements — not specific enough to store.
                r"\bi\s+(?:'m\s+)?(?:like|love|enjoy|hate|dislike)s?\s+"
                r"(?:anything|everything|something|nothing)\b|"
                r"\bi\s+(?:'m\s+)?(?:like|love|enjoy|hate|dislike)s?\s+all\b|"
                r"\bi\s+(?:'m\s+)?(?:like|love|enjoy|hate|dislike)s?\s+both\b|"
                r"\bi\s+don'?t\s+(?:like|love|enjoy|hate|dislike)\s+"
                r"(?:anything|everything|something|nothing)\b|"
                r"\bi\s+do\s+not\s+(?:like|love|enjoy|hate|dislike)\s+"
                r"(?:anything|everything|something|nothing)\b|"
                # Incomplete statements — no object, cannot be a memory query.
                r"\bi\s+(?:want\s+to|wanna|aim\s+to|plan\s+to)\s*$|"
                r"\bi\s+(?:want\s+to|wanna)\s+(?:become|be)\s*$|"
                # Vague/invalid goal statements — pronoun objects are meaningless.
                r"\bi\s+(?:want\s+to|wanna)\s+(?:become|be)\s+"
                r"(?:nothing|something|anything|everything)\b",
                query,
            )
        )

    @staticmethod
    def _get_clarification_message(normalized_query: str) -> str:
        """Return a clarification message for vague/incomplete statements,
        or empty string if the query should be routed to the LLM instead."""
        # Normalize typos (e.g., "noting" → "nothing") so vague/invalid
        # goal statements are caught regardless of spelling.
        query = _normalize_vague_goals(normalized_query)
        # Vague preference statements
        if re.search(
            r"\bi\s+(?:'m\s+)?(?:like|love|enjoy|hate|dislike)s?\s+"
            r"(?:anything|everything|something|nothing)\b|"
            r"\bi\s+(?:'m\s+)?(?:like|love|enjoy|hate|dislike)s?\s+all\b|"
            r"\bi\s+(?:'m\s+)?(?:like|love|enjoy|hate|dislike)s?\s+both\b|"
            r"\bi\s+don'?t\s+(?:like|love|enjoy|hate|dislike)\s+"
            r"(?:anything|everything|something|nothing)\b|"
            r"\bi\s+do\s+not\s+(?:like|love|enjoy|hate|dislike)\s+"
            r"(?:anything|everything|something|nothing)\b",
            query,
        ):
            return (
                "Could you tell me something specific you like or dislike? "
                "For example, 'I like Python' or 'I dislike cold weather'."
            )
        # Incomplete statements
        if re.search(
            r"\bi\s+(?:want\s+to|wanna|aim\s+to|plan\s+to)\s*$|"
            r"\bi\s+(?:want\s+to|wanna)\s+(?:become|be)\s*$",
            query,
        ):
            return (
                "What would you like to do or become? "
                "Please complete your thought so I can remember it."
            )
        # Vague/invalid goal statements
        if re.search(
            r"\bi\s+(?:want\s+to|wanna)\s+(?:become|be)\s+"
            r"(?:nothing|something|anything|everything)\b",
            query,
        ):
            return (
                "That's too vague for me to remember. "
                "Could you tell me a specific goal, like "
                "'I want to become a software engineer'?"
            )
        return ""

    def _cued_intents(self, normalized_query: str) -> list[str]:
        found = []
        for intent, patterns in CATEGORY_QUERY_CUES.items():
            for pattern in patterns:
                if re.search(pattern, normalized_query, flags=re.IGNORECASE):
                    found.append(intent)
                    break
        return found

    def _select_requested_intents(
        self,
        normalized_query: str,
        scores: dict[str, float],
        primary: str,
        primary_score: float,
        cued: Optional[list[str]] = None,
    ) -> list[str]:
        cued = list(cued or self._cued_intents(normalized_query))
        looks_compound = bool(
            re.search(r"\band\b|,|/|\bas well as\b", normalized_query)
        )
        order = [
            "name", "nickname", "skill", "goal", "ambition", "dream",
            "learning", "education", "preference",
            "interest", "project", "habit", "task", "priority",
            "personal_overview",
        ]

        # Compound questions with multiple explicit cues: preserve question order.
        if looks_compound and len(cued) >= 2:
            return list(dict.fromkeys(cued))

        requested = [primary]
        for intent in cued:
            if intent not in requested and intent in MEMORY_READ_INTENTS:
                if intent == "name" and "nickname" in normalized_query:
                    continue
                requested.append(intent)

        # Only add semantic companions for compound wording when cues are sparse.
        if looks_compound and len(cued) < 2:
            multi_floor = max(self.intent_threshold, primary_score * 0.70)
            for concept, score in sorted(
                scores.items(), key=lambda item: item[1], reverse=True
            ):
                if concept not in MEMORY_READ_INTENTS or concept == "personal_overview":
                    continue
                if concept in requested:
                    continue
                if score < multi_floor:
                    continue
                conflict = False
                for selected in requested:
                    if concept in CONFUSABLE_INTENTS.get(selected, frozenset()):
                        if concept not in cued:
                            conflict = True
                            break
                if conflict:
                    continue
                requested.append(concept)

        # Keep primary first for single-intent answers; for compounds, keep cue order.
        if looks_compound and len(requested) > 1 and cued:
            ordered = [item for item in cued if item in requested]
            for item in requested:
                if item not in ordered:
                    ordered.append(item)
            return ordered or [primary]

        ordered = [item for item in order if item in requested]
        if primary in ordered:
            return [primary] + [item for item in ordered if item != primary]
        return ordered or [primary]

    def secondary_identity_intents(
        self,
        intent: QueryIntent,
        *,
        closeness: float = 0.05,
    ) -> list[str]:
        """When name/nickname scores are close, allow both identity types."""
        if not intent.is_memory_read:
            return []
        if intent.primary not in {"name", "nickname"}:
            return [intent.primary]

        normalized = intent.normalized_query or ""
        if "nickname" in normalized:
            return ["nickname"]
        if intent.primary == "name" and re.search(
            r"\bmy name\b|\bwho am i\b", normalized
        ):
            return ["name"]

        name_score = intent.scores.get("name", 0.0)
        nick_score = intent.scores.get("nickname", 0.0)
        if abs(name_score - nick_score) <= closeness and min(
            name_score, nick_score
        ) >= (self.intent_threshold - 0.02):
            return ["name", "nickname"]
        return [intent.primary]
