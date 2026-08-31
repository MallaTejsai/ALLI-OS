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
    text = text.replace("’", "'")
    # nick-name / nick name → nickname (generic, not a one-off answer rule)
    text = re.sub(r"\bnick[\s\-]+name\b", "nickname", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
        "What are the user's life goals dreams ambitions and aims?",
        "What does the user want to become in the future?",
        "What is the user trying to achieve long term?",
        "What career or life outcomes is the user aiming for?",
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
    "project": [
        "What projects is the user building or working on?",
        "What is the user creating or developing?",
    ],
    "habit": [
        "What habits or daily routines does the user have?",
        "What does the user usually do regularly?",
    ],
    "background": [
        "What is the user's education college course or background?",
        "Where does the user study or what did they study?",
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
        r"\bdreams?\b",
        r"\bambitions?\b",
        r"\baims?\b",
        r"\b(?:my\s+)?goals?\b",
        r"\bwant to become\b",
    ),
    "skill": (r"\bskills?\b",),
    "learning": (r"\blearning\b", r"\bam i learning\b", r"\bwhat i(?:'m| am) learning\b", r"\bwant to learn\b"),
    "preference": (
    r"\bpreferences?\b",
    r"\bwhat\s+(?:do\s+)?i\s+(?:like|dislike|enjoy|hate)\b",
    r"\bwhat\s+don'?t\s+i\s+(?:like|enjoy)\b",
    r"\bwhich\b.*\b(?:i\s+)?(?:like|dislike|enjoy|hate)\b",
),
    "interest": (r"\binterests?\b", r"\bhobb(?:y|ies)\b"),
    "project": (r"\bprojects?\b",),
    "habit": (r"\bhabits?\b", r"\broutines?\b"),
    "background": (r"\beducation\b", r"\bcollege\b", r"\bbackground\b"),
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
    "background": {
        "categories": {"education"},
        "fact_types": None,
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
    "goal": frozenset({"learning"}),
    "name": frozenset({"nickname"}),
    "nickname": frozenset({"name"}),
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
            return QueryIntent(
                primary="unknown",
                score=0.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

        try:
            self._ensure_prototypes()
            query_vec = self.embedding_store.generate_embedding(normalized)
        except EmbeddingError:
            return QueryIntent(
                primary="unknown",
                score=0.0,
                is_memory_read=False,
                is_general_knowledge=True,
                normalized_query=normalized,
            )

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
        """Keep recommendations and unnamed-object questions out of memory reads."""
        return bool(
            re.search(
                r"\bwhat\s+should\s+i\s+learn\b|"
                r"\bwhich\s+technology\s+is\s+best\s+for\s+me\b|"
                r"\bwhat\s+do\s+i\s+like\s+about\s+"
                r"(?:something|it|that|this)\b",
                normalized_query,
            )
        )

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
            "name", "nickname", "skill", "goal", "learning", "preference",
            "interest", "project", "habit", "background", "personal_overview",
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
