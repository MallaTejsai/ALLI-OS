"""
STEP 6 — Golden Scenario Framework.

Defines structured golden benchmark scenarios for architecture validation.
Architecture-neutral; references dataset/step 5 and result/step 4.
Does NOT execute candidates. Defines expected behavior only.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, List


class ScenarioFamily(Enum):
    DIRECT_RETRIEVAL = "direct_retrieval"
    PARAPHRASE = "paraphrase"
    IMPERFECT_ENGLISH = "imperfect_english"
    ATTRIBUTE_QUERY = "attribute_query"
    RELATIONSHIP_QUERY = "relationship_query"
    REVERSE_RELATIONSHIP = "reverse_relationship"
    MULTI_HOP = "multi_hop"
    PRONOUN = "pronoun"
    PERSONAL_VS_OTHER = "personal_vs_other"
    PURPOSE_VS_FACT = "purpose_vs_fact"
    CURRENT_TEMPORAL = "current_temporal"
    HISTORICAL_STATE = "historical_state"
    POINT_IN_TIME = "point_in_time"
    CONTRADICTION = "contradiction"
    CORRECTION = "correction"
    DUPLICATE = "duplicate"
    COMPATIBLE = "compatible"
    NEGATIVE_KNOWLEDGE = "negative_knowledge"
    PLAN_VS_FACT = "plan_vs_fact"
    POSSIBILITY = "possibility"
    BROAD_PROFILE = "broad_profile"
    MULTI_PART = "multi_part"
    EVIDENCE_CONFLICT = "evidence_conflict"
    PROVENANCE = "provenance"
    DELETION_SAFETY = "deletion_safety"
    RESTART = "restart"
    INDEX_RECOVERY = "index_recovery"
    MODEL_FAILURE = "model_failure"
    PROMPT_INJECTION = "prompt_injection"
    LARGE_SCALE = "large_scale"
    CROSS_SESSION = "cross_session"
    MEMORY_VS_ANSWER = "memory_vs_answer"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class Severity(Enum):
    S0_CRITICAL = "S0"
    S1_MAJOR = "S1"
    S2_MODERATE = "S2"
    S3_MINOR = "S3"


@dataclass(frozen=True)
class ScenarioIdentity:
    scenario_id: str
    version: str = "0.1.0"


@dataclass(frozen=True)
class ExpectedEvidence:
    memory_ids: List[str] = field(default_factory=list)
    relationship_ids: List[str] = field(default_factory=list)
    evidence_notes: str = ""


@dataclass(frozen=True)
class ExpectedAnswer:
    answer_class: str  # DIRECT, DERIVED, UNKNOWN, AMBIGUOUS, CONFLICTED
    expected_text_pattern: Optional[str] = None  # Not mandatory; semantic match preferred
    clarification_required: bool = False
    conflict_must_surface: bool = False


@dataclass(frozen=True)
class GoldenScenario:
    scenario_id: str
    family: ScenarioFamily
    difficulty: Difficulty
    severity: Severity
    dataset_ref: str  # Reference to dataset version/config
    initial_state_ref: Optional[str] = None  # Episode/dataset reference
    query_text: str
    expected_interpretation: str  # Structured description of intended meaning
    expected_memory_refs: List[str] = field(default_factory=list)
    expected_evidence: Optional[ExpectedEvidence] = None
    expected_answer: ExpectedAnswer
    expected_behavior: List[str] = field(default_factory=list)
    failure_expectation: Optional[str] = None
    scope: Optional[str] = None
    provenance_ref: Optional[str] = None
    version: str = "0.1.0"
    metadata: dict = field(default_factory=dict)
