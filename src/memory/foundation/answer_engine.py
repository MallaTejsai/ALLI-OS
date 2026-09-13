"""Answer Engine for ALLI v0.6.

Orchestrates the full reasoning pipeline:

  question
  → StructuredQuery          (Session 5)
  → HybridRetrievalResult    (Session 6)
  → EvidenceAssessment       (Session 7)
  → validated evidence
  → Qwen
  → final natural-language answer

Qwen is called ONLY when grounded evidence exists (SUPPORTED status).
Qwen receives ONLY accepted evidence — never the full memory database,
never rejected candidates, never fabricated facts.

Scope:
  - Answer generation from validated evidence
  - Does NOT rewrite evidence grounding (Session 7)
  - Does NOT rewrite hybrid retrieval (Session 6)
  - Does NOT rewrite query understanding (Session 5)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from .query_understanding import StructuredQuery, understand_query
from .hybrid_retriever import HybridRetriever, HybridRetrievalResult
from .evidence_grounding import EvidenceGrounder, EvidenceAssessment, EvidenceItem


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "qwen2.5:7b"
_MEMORY_DEBUG = os.environ.get("ALLI_MEMORY_DEBUG", "0") == "1"


# ---------------------------------------------------------------------------
# Answer Result
# ---------------------------------------------------------------------------

@dataclass
class AnswerResult:
    """Complete answer result with full pipeline trace."""

    question: str = ""
    answer: str = ""

    # Pipeline stages
    structured_query: Optional[StructuredQuery] = None
    retrieval_result: Optional[HybridRetrievalResult] = None
    evidence_assessment: Optional[EvidenceAssessment] = None

    # Status
    source: str = ""  # "evidence", "unknown", "weak", "conflict", "error"
    called_qwen: bool = False

    # Diagnostics
    diagnostics: dict = field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        return bool(self.answer.strip())


# ---------------------------------------------------------------------------
# Evidence Formatting for Qwen
# ---------------------------------------------------------------------------

def _format_evidence_for_qwen(
    question: str,
    assessment: EvidenceAssessment,
) -> str:
    """Format accepted evidence as a concise evidence block for Qwen.

    Qwen receives ONLY this text plus the question.
    No rejected candidates. No full database. No invented context.
    """
    lines = []
    for item in assessment.accepted:
        content = item.content.strip()
        if not content:
            continue

        # Add metadata hints
        meta_parts = []
        if item.entity:
            meta_parts.append(f"entity={item.entity}")
        if item.fact_type:
            meta_parts.append(f"fact={item.fact_type}")
        if item.status:
            meta_parts.append(f"status={item.status}")
        if item.is_negated:
            meta_parts.append("NEGATED")
        if item.is_uncertain:
            meta_parts.append("UNCERTAIN")

        meta = ", ".join(meta_parts)
        lines.append(f"- {content} [{meta}]")

    return "\n".join(lines)


def _build_qwen_prompt(
    question: str,
    assessment: EvidenceAssessment,
) -> list[dict]:
    """Build the system + user messages for Qwen.

    Contract:
      1. System message: strict grounding instructions
      2. User message: question + ONLY accepted evidence
    """
    evidence_text = _format_evidence_for_qwen(question, assessment)

    system_msg = (
        "You are ALLI, a personal AI assistant.\n"
        "Answer the user's question using ONLY the evidence provided below.\n"
        "Rules:\n"
        "- Answer ONLY from the supplied evidence.\n"
        "- Do NOT invent, infer, or fabricate personal facts.\n"
        "- Do NOT use information not present in the evidence.\n"
        "- If the evidence does not contain the answer, say so clearly.\n"
        "- Keep answers concise and natural.\n"
        "- Do not mention evidence scores, metadata, or internal details.\n"
        "- If a fact is NEGATED, respect the negation.\n"
        "- If a fact is UNCERTAIN, preserve the uncertainty.\n"
        "- Always reply in English."
    )

    user_msg = (
        f"Question: {question}\n\n"
        f"Evidence:\n{evidence_text}\n\n"
        f"Answer using ONLY the evidence above. "
        f"If the evidence does not support an answer, say so."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


# ---------------------------------------------------------------------------
# Unknown / Weak / Conflict Responses
# ---------------------------------------------------------------------------

def _unknown_response(question: str) -> str:
    """Return a natural unknown response when no evidence exists."""
    q = question.lower().strip()

    # Try to extract the topic from the question
    topic = _extract_topic(q)
    if topic:
        return f"I don't have any stored information about your {topic}."

    return "I don't currently have that information stored."


def _weak_response(question: str) -> str:
    """Return a conservative response for weak evidence."""
    q = question.lower().strip()
    topic = _extract_topic(q)
    if topic:
        return f"I'm not fully sure about your {topic}. The information I have is uncertain."
    return "I'm not fully sure about that. The information I have is uncertain."


def _conflict_response(question: str, assessment: EvidenceAssessment) -> str:
    """Return a response explaining conflicting evidence."""
    q = question.lower().strip()
    topic = _extract_topic(q)

    # Try to show the conflict
    conflict_values = []
    for group in assessment.conflicts:
        for item in group:
            value = item.candidate.memory.get("value", item.content)
            if value not in conflict_values:
                conflict_values.append(value)

    if len(conflict_values) >= 2:
        values_text = " and ".join(conflict_values[:3])
        if topic:
            return (
                f"I have conflicting information about your {topic}: "
                f"I have {values_text}. "
                f"Could you clarify which is current?"
            )
        return (
            f"I have conflicting information: {values_text}. "
            f"Could you clarify which is current?"
        )

    return "I have conflicting information stored. Could you clarify what is currently correct?"


def _extract_topic(question: str) -> Optional[str]:
    """Try to extract the topic noun from a question."""
    import re
    q = question.lower().strip().rstrip("?")

    # Common question patterns — ordered by specificity
    patterns = [
        (r"(?:what|which)\s+is\s+my\s+(.+?)(?:\s*\?|$)", 1),
        (r"(?:what|which)\s+is\s+(?:the\s+)?my\s+(.+?)(?:\s*\?|$)", 1),
        (r"(?:what|which)\s+(.+?)(?:\s+do\s+(?:i|you)\s+have|\s+did\s+(?:i|you)\s+have|\s*$)", 1),
        (r"(?:do|does)\s+(?:i|you)\s+like\s+(.+?)(?:\s*\?|$)", 1),
        (r"(?:what|which)\s+(?:is|are)\s+(?:the|my)\s+(.+?)(?:\s*\?|$)", 1),
        (r"(?:what)\s+(.+?)(?:\s+do\s+(?:i|you)\s+(?:like|use|have)|\s*$)", 1),
    ]

    for pattern, group in patterns:
        match = re.search(pattern, q)
        if match:
            topic = match.group(group).strip()
            # Clean up common artifacts
            topic = topic.rstrip(".")
            topic = re.sub(r"\b(?:do|does|is|are|did)\b", "", topic).strip()
            if topic and len(topic) > 1:
                # Reject verb-phrase fragments (e.g. "was i debugging") — they
                # are not grammatical topics and only make the unknown response
                # awkward. A topic must be a noun-ish phrase: it may not start
                # with an auxiliary verb nor embed a subject pronoun.
                if re.match(r"^(?:was|were|did|do|does|done|is|are|am|was|will|would|could|should|can)\b", topic):
                    continue
                if re.search(r"\b(?:i|you|he|she|we|they)\b", topic):
                    continue
                return topic

    return None


# ---------------------------------------------------------------------------
# Answer Engine
# ---------------------------------------------------------------------------

class AnswerEngine:
    """Complete reasoning pipeline from question to grounded answer.

    Usage:
        engine = AnswerEngine(hybrid_retriever, embed_store)
        result = engine.answer("What is my favorite color?")
        print(result.answer)
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        embedding_store: object,
        *,
        model: str = _DEFAULT_MODEL,
        allow_qwen: bool = True,
        ollama_client=None,
    ):
        """
        Args:
            hybrid_retriever: Session 6 HybridRetriever instance
            embedding_store: EmbeddingStore for query embedding
            model: Ollama model name for answer generation
            allow_qwen: Whether to call Qwen (False = dry run, no LLM)
            ollama_client: Pre-configured ollama.Client instance (optional)
        """
        self.hybrid_retriever = hybrid_retriever
        self.embedding_store = embedding_store
        self.model = model
        self.allow_qwen = allow_qwen
        self.grounder = EvidenceGrounder()
        self._ollama_client = ollama_client

    def answer(
        self,
        question: str,
        *,
        diagnostics: bool = False,
    ) -> AnswerResult:
        """Run the full pipeline and return an answer.

        Args:
            question: Natural language user question
            diagnostics: If True, include full pipeline trace

        Returns:
            AnswerResult with answer text and pipeline metadata
        """
        result = AnswerResult(question=question)

        # Step 1: Query Understanding (Session 5)
        try:
            sq = understand_query(question)
            result.structured_query = sq
        except Exception as e:
            result.source = "error"
            result.answer = "I couldn't understand your question."
            result.diagnostics["understanding_error"] = str(e)
            return result

        # Step 2: Hybrid Retrieval (Session 6)
        try:
            retrieval = self.hybrid_retriever.retrieve(sq, max_results=8)
            result.retrieval_result = retrieval
        except Exception as e:
            result.source = "error"
            result.answer = "I had trouble searching my memory."
            result.diagnostics["retrieval_error"] = str(e)
            return result

        # Step 3: Evidence Grounding (Session 7)
        try:
            assessment = self.grounder.ground(sq, retrieval)
            result.evidence_assessment = assessment
        except Exception as e:
            result.source = "error"
            result.answer = "I had trouble analyzing the evidence."
            result.diagnostics["grounding_error"] = str(e)
            return result

        # Step 4: Answer based on evidence status
        if assessment.is_none:
            result.answer = _unknown_response(question)
            result.source = "unknown"
            return result

        if assessment.is_weak:
            result.answer = _weak_response(question)
            result.source = "weak"
            return result

        if assessment.is_conflicting:
            result.answer = _conflict_response(question, assessment)
            result.source = "conflict"
            return result

        # SUPPORTED — call Qwen with validated evidence only
        if not self.allow_qwen:
            result.answer = self._offline_answer(question, assessment)
            result.source = "evidence_offline"
            return result

        try:
            answer_text = self._call_qwen(question, assessment)
            result.answer = answer_text
            result.source = "evidence"
            result.called_qwen = True
        except Exception as e:
            # Qwen unreachable — fall back to evidence summary
            result.answer = self._offline_answer(question, assessment)
            result.source = "evidence_offline"
            result.diagnostics["qwen_error"] = str(e)

        return result

    def _call_qwen(
        self,
        question: str,
        assessment: EvidenceAssessment,
    ) -> str:
        """Call Qwen with the grounding contract."""
        messages = _build_qwen_prompt(question, assessment)

        if self._ollama_client is not None:
            response = self._ollama_client.chat(model=self.model, messages=messages)
        else:
            import ollama
            response = ollama.Client().chat(model=self.model, messages=messages)
        return response["message"]["content"].strip()

    def _offline_answer(
        self,
        question: str,
        assessment: EvidenceAssessment,
    ) -> str:
        """Generate a simple evidence-based answer without Qwen.

        Used when Qwen is unreachable or allow_qwen=False.
        """
        if not assessment.accepted:
            return _unknown_response(question)

        # Build a simple list of evidence facts
        facts = []
        for item in assessment.accepted:
            content = item.content.strip()
            if content:
                facts.append(content)

        if not facts:
            return _unknown_response(question)

        if len(facts) == 1:
            return facts[0]

        return "Here is what I have stored:\n" + "\n".join(f"- {f}" for f in facts)
