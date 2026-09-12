"""LLM-constrained structured extraction for ALLI v0.6.

Replaces regex-based extraction with deterministic LLM calls.
The LLM produces structured JSON; Python validates and applies.

No direct database writes from the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .entity_resolver import extract_entity, EntityExtraction
from .validation import validate_memory_input, is_garbage, is_vague, is_temporal_only


# Structured extraction schema — what the LLM must output
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["statement", "question", "greeting", "command", "clarification"],
        },
        "is_temporary": {
            "type": "boolean",
            "description": "True if the statement describes a temporary state, not a durable fact",
        },
        "temporal_hint": {
            "type": ["string", "null"],
            "description": "Temporal reference like 'tomorrow', 'next week', 'in 2 days'",
        },
        "is_negation": {
            "type": "boolean",
            "description": "True if the statement negates a previous fact",
        },
        "negation_target": {
            "type": ["string", "null"],
            "description": "What is being negated (e.g., 'python', 'goal')",
        },
        "is_third_party": {
            "type": "boolean",
            "description": "True if describing someone else's fact, not the user's",
        },
        "operations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["ADD", "UPSERT", "DELETE", "REPLACE", "CANCEL"],
                    },
                    "category": {
                        "type": "string",
                        "enum": [
                            "identity", "preference", "skill", "learning",
                            "education", "goal", "ambition", "dream",
                            "project", "routine", "task", "priority", "other",
                        ],
                    },
                    "fact_type": {
                        "type": "string",
                        "description": "Specific fact type within category",
                    },
                    "content": {
                        "type": "string",
                        "description": "Natural language fact statement",
                    },
                    "value": {
                        "type": "string",
                        "description": "Compact value of the fact",
                    },
                    "subject": {
                        "type": "string",
                        "description": "Normalized topic key for matching",
                    },
                    "polarity": {
                        "type": ["string", "null"],
                        "enum": ["like", "dislike", None],
                    },
                    "importance": {
                        "type": "string",
                        "enum": ["high", "medium", "low", "critical"],
                    },
                    "confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                    "entity_name": {
                        "type": "string",
                        "description": "Entity this fact belongs to (e.g., 'user', 'mom')",
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["self", "person", "organization", "concept"],
                    },
                    "relation_to_user": {
                        "type": ["string", "null"],
                        "description": "Relationship to user (e.g., 'self', 'parent')",
                    },
                    "replaces_id": {
                        "type": ["string", "null"],
                        "description": "Memory ID to replace (for REPLACE action)",
                    },
                    "priority": {
                        "type": ["string", "null"],
                        "description": "Priority level if explicitly stated",
                    },
                },
                "required": [
                    "action", "category", "fact_type", "content",
                    "value", "subject", "importance", "confidence",
                    "entity_name", "entity_type",
                ],
            },
        },
    },
    "required": ["intent", "is_temporary", "is_negation", "is_third_party", "operations"],
}

# Prompt template for the LLM
EXTRACTION_PROMPT = """You are a memory extraction system. Extract durable personal facts from user messages.

OUTPUT FORMAT: Strict JSON matching this schema:
{schema}

RULES:
1. Only extract facts the user is telling you about themselves or others.
2. For questions, greetings, or commands, return empty operations array.
3. For temporary statements (exams, appointments), set is_temporary=true.
4. For negations ("I don't like X", "I no longer want to become Y"), set is_negation=true.
5. For third-party statements ("Mom likes Java"), set is_third_party=true and set entity_name to the person.
6. Entity resolution is GENERIC — no hardcoded names. Use entity_name from the text.
7. Categories: identity, preference, skill, learning, education, goal, ambition, dream, project, routine, task, priority, other.
8. Confidence: 0.0-1.0 based on how certain the extraction is.
9. NEVER invent facts. Only extract what the user explicitly states.

EXAMPLES:

User: "I like Python"
→ {{"intent":"statement","is_temporary":false,"is_negation":false,"is_third_party":false,"operations":[{{"action":"ADD","category":"preference","fact_type":"like","content":"User likes Python","value":"like","subject":"python","polarity":"like","importance":"medium","confidence":0.95,"entity_name":"user","entity_type":"self"}}]}}

User: "Mom's favourite colour is red"
→ {{"intent":"statement","is_temporary":false,"is_negation":false,"is_third_party":true,"operations":[{{"action":"ADD","category":"preference","fact_type":"like","content":"Mom's favourite colour is red","value":"like","subject":"favourite colour","polarity":"like","importance":"medium","confidence":0.9,"entity_name":"mom","entity_type":"person","relation_to_user":"parent"}}]}}

User: "I want to become an AI engineer"
→ {{"intent":"statement","is_temporary":false,"is_negation":false,"is_third_party":false,"operations":[{{"action":"ADD","category":"goal","fact_type":"career_goal","content":"User wants to become an AI engineer","value":"AI engineer","subject":"ai engineer","importance":"high","confidence":0.95,"entity_name":"user","entity_type":"self"}}]}}

User: "I don't like Python anymore"
→ {{"intent":"statement","is_temporary":false,"is_negation":true,"negation_target":"python","is_third_party":false,"operations":[{{"action":"DELETE","category":"preference","fact_type":"like","content":"User dislikes Python","value":"dislike","subject":"python","polarity":"dislike","importance":"medium","confidence":0.9,"entity_name":"user","entity_type":"self"}}]}}

User: "I am learning electronics in my diploma"
→ {{"intent":"statement","is_temporary":false,"is_negation":false,"is_third_party":false,"operations":[{{"action":"ADD","category":"education","fact_type":"studying","content":"User is studying electronics in diploma","value":"electronics","subject":"electronics","importance":"high","confidence":0.9,"entity_name":"user","entity_type":"self"}}]}}

User: "My dream is to become a doctor"
→ {{"intent":"statement","is_temporary":false,"is_negation":false,"is_third_party":false,"operations":[{{"action":"ADD","category":"dream","fact_type":"dream","content":"User's dream is to become a doctor","value":"doctor","subject":"doctor","importance":"high","confidence":0.95,"entity_name":"user","entity_type":"self"}}]}}

User: "I have exam tomorrow"
→ {{"intent":"statement","is_temporary":true,"temporal_hint":"tomorrow","is_negation":false,"is_third_party":false,"operations":[{{"action":"ADD","category":"task","fact_type":"task","content":"User has exam tomorrow","value":"exam","subject":"exam","importance":"medium","confidence":0.95,"entity_name":"user","entity_type":"self"}}]}}

User: "What is my goal?"
→ {{"intent":"question","is_temporary":false,"is_negation":false,"is_third_party":false,"operations":[]}}

User: "My name is Tejsai"
→ {{"intent":"statement","is_temporary":false,"is_negation":false,"is_third_party":false,"operations":[{{"action":"UPSERT","category":"identity","fact_type":"name","content":"User's name is Tejsai","value":"Tejsai","subject":"name","importance":"high","confidence":0.98,"entity_name":"user","entity_type":"self"}}]}}

User message: "{message}"
"""


@dataclass
class StructuredOperation:
    """A single extracted memory operation."""
    action: str  # ADD, UPSERT, DELETE, REPLACE, CANCEL
    category: str
    fact_type: str
    content: str
    value: str
    subject: str
    polarity: Optional[str] = None
    importance: str = "medium"
    confidence: float = 0.8
    entity_name: str = "user"
    entity_type: str = "self"
    relation_to_user: Optional[str] = None
    replaces_id: Optional[str] = None
    priority: Optional[str] = None

    def to_dict(self) -> dict:
        d = {
            "action": self.action,
            "category": self.category,
            "fact_type": self.fact_type,
            "content": self.content,
            "value": self.value,
            "subject": self.subject,
            "importance": self.importance,
            "confidence": self.confidence,
        }
        if self.polarity:
            d["preference_polarity"] = self.polarity
            d["preference_subject"] = self.subject
        if self.relation_to_user:
            d["relation_to_user"] = self.relation_to_user
        if self.replaces_id:
            d["replaces_id"] = self.replaces_id
        if self.priority:
            d["priority"] = self.priority
        if self.entity_name != "user":
            d["entity_name"] = self.entity_name
        if self.entity_type != "self":
            d["entity_type"] = self.entity_type
        return d


@dataclass
class StructuredExtraction:
    """Result of structured extraction from a user message."""
    intent: str  # statement, question, greeting, command, clarification
    is_temporary: bool = False
    temporal_hint: Optional[str] = None
    is_negation: bool = False
    negation_target: Optional[str] = None
    is_third_party: bool = False
    operations: list[StructuredOperation] = field(default_factory=list)
    raw_response: Optional[str] = None
    parse_error: Optional[str] = None

    @property
    def is_question(self) -> bool:
        return self.intent == "question"

    @property
    def has_operations(self) -> bool:
        return len(self.operations) > 0


class StructuredExtractor:
    """LLM-constrained extraction of structured memory operations.

    Uses Ollama with structured output to extract facts from user messages.
    Falls back to the old regex extractor if LLM is unavailable.
    """

    def __init__(
        self,
        model: str = "qwen2.5:7b",
        max_retries: int = 2,
        fallback_extractor: Optional[Any] = None,
        ollama_client=None,
    ):
        self.model = model
        self.max_retries = max_retries
        self._fallback = fallback_extractor
        self._client = ollama_client

    def _get_client(self):
        """Return the Ollama client (pre-configured or module-level fallback)."""
        if self._client is not None:
            return self._client
        try:
            import ollama
            return ollama
        except ImportError:
            return None

    def extract(self, user_message: str) -> list[dict]:
        """Extract memory operations from a user message.

        Returns a list of operation dicts compatible with SafeMemoryManager.
        Falls back to regex extraction if LLM is unavailable or fails.
        """
        if not user_message or not user_message.strip():
            return []

        # Try LLM extraction
        try:
            extraction = self._extract_with_llm(user_message)
            if extraction is not None and extraction.has_operations:
                return [op.to_dict() for op in extraction.operations]
            if extraction is not None and extraction.is_question:
                return []  # Questions don't produce memory operations
        except Exception:
            pass

        # Fallback to old regex extractor
        if self._fallback is not None:
            return self._fallback.extract(user_message)
        return []

    def _extract_with_llm(self, user_message: str) -> Optional[StructuredExtraction]:
        """Call LLM for structured extraction with bounded retries."""
        client = self._get_client()
        if client is None:
            return None

        prompt = EXTRACTION_PROMPT.format(
            message=user_message,
            schema=json.dumps(EXTRACTION_SCHEMA, indent=2),
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0},
                    format="json",
                )
                content = response.get("message", {}).get("content", "")
                extraction = self._parse_response(content, user_message)
                if extraction is not None:
                    return extraction
            except Exception:
                if attempt < self.max_retries:
                    continue
                break

        return None

    def _parse_response(
        self, response_text: str, original_message: str
    ) -> Optional[StructuredExtraction]:
        """Parse LLM JSON response into StructuredExtraction."""
        if not response_text:
            return None

        # Try to extract JSON from response (may be wrapped in markdown)
        json_match = re.search(r"```(?:json)?\s*(.*?)```", response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            json_str = response_text.strip()

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            return StructuredExtraction(
                intent="statement",
                raw_response=response_text,
                parse_error="Invalid JSON",
            )

        # Validate required fields
        if "intent" not in data or "operations" not in data:
            return StructuredExtraction(
                intent="statement",
                raw_response=response_text,
                parse_error="Missing required fields",
            )

        # Parse operations
        operations = []
        for op_data in data.get("operations", []):
            op = self._parse_operation(op_data, original_message)
            if op is not None:
                operations.append(op)

        return StructuredExtraction(
            intent=data.get("intent", "statement"),
            is_temporary=data.get("is_temporary", False),
            temporal_hint=data.get("temporal_hint"),
            is_negation=data.get("is_negation", False),
            negation_target=data.get("negation_target"),
            is_third_party=data.get("is_third_party", False),
            operations=operations,
            raw_response=response_text,
        )

    def _parse_operation(
        self, op_data: dict, original_message: str
    ) -> Optional[StructuredOperation]:
        """Parse a single operation from LLM output with validation."""
        if not isinstance(op_data, dict):
            return None

        action = op_data.get("action", "").upper()
        if action not in {"ADD", "UPSERT", "DELETE", "REPLACE", "CANCEL"}:
            return None

        category = op_data.get("category", "other")
        if category not in {
            "identity", "preference", "skill", "learning", "education",
            "goal", "ambition", "dream", "project", "routine", "task",
            "priority", "other",
        }:
            category = "other"

        content = str(op_data.get("content", "")).strip()
        value = str(op_data.get("value", "")).strip()
        subject = str(op_data.get("subject", "")).strip()

        if not content or not value:
            return None

        # Validate via deterministic validation
        validation = validate_memory_input(value, subject, category, op_data.get("fact_type", "other"))
        if not validation.is_valid:
            return None

        # Use normalized values
        if validation.normalized_value:
            value = validation.normalized_value
        if validation.normalized_subject:
            subject = validation.normalized_subject

        # Validate confidence
        confidence = op_data.get("confidence", 0.8)
        if not isinstance(confidence, (int, float)):
            confidence = 0.8
        confidence = max(0.0, min(1.0, float(confidence)))

        # Validate importance
        importance = op_data.get("importance", "medium")
        if importance not in {"high", "medium", "low", "critical"}:
            importance = "medium"

        return StructuredOperation(
            action=action,
            category=category,
            fact_type=op_data.get("fact_type", "other"),
            content=content,
            value=value,
            subject=subject,
            polarity=op_data.get("polarity"),
            importance=importance,
            confidence=confidence,
            entity_name=op_data.get("entity_name", "user"),
            entity_type=op_data.get("entity_type", "self"),
            relation_to_user=op_data.get("relation_to_user"),
            replaces_id=op_data.get("replaces_id"),
            priority=op_data.get("priority"),
        )


def create_structured_extractor(
    fallback_extractor: Optional[Any] = None,
    model: str = "qwen2.5:7b",
    ollama_client=None,
) -> StructuredExtractor:
    """Factory function to create a StructuredExtractor with fallback."""
    return StructuredExtractor(
        model=model,
        fallback_extractor=fallback_extractor,
        ollama_client=ollama_client,
    )
