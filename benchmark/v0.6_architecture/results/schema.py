"""
STEP 4 — Structured Result Schema for ALLI v0.6 Architecture Benchmark.

Defines how benchmark executions record outcomes in structured,
reproducible, architecture-neutral form.

NOT the metrics engine.
NOT the benchmark runner.
NOT the dataset system.
NOT Candidate A.

Only defines the result data contract that later metrics/runner/components
will consume.

Uses existing identity/version concepts from STEP 2 (CandidateContract) and
STEP 3 (ExperimentConfiguration / VersionConfiguration).
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, Union
from uuid import UUID, uuid4

# Reuse existing identity types from contract
from benchmark.v0_6_architecture.candidates.contract import (
    EpisodeId, EntityId, MemoryId, RelationshipId, StatusCode
)


class ExecutionStatus(Enum):
    """Overall benchmark execution status."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    PARTIAL = "PARTIAL"
    SKIPPED = "SKIPPED"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED = "UNSUPPORTED"


class OperationStatus(Enum):
    """Per-operation execution status."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    UNSUPPORTED = "UNSUPPORTED"
    INVALID_REQUEST = "INVALID_REQUEST"
    RECOVERABLE_ERROR = "RECOVERABLE_ERROR"
    FATAL_ERROR = "FATAL_ERROR"
    TIMEOUT = "TIMEOUT"


class AnswerClass(Enum):
    """Answer classification from Phase 3.3."""
    DIRECT = "DIRECT"
    DERIVED = "DERIVED"
    UNKNOWN = "UNKNOWN"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTED = "CONFLICTED"


class SeverityLevel(Enum):
    """Failure severity from Phase 3.3."""
    S0_CRITICAL = "S0_CRITICAL"
    S1_MAJOR = "S1_MAJOR"
    S2_MODERATE = "S2_MODERATE"
    S3_MINOR = "S3_MINOR"


class FailureCategory(Enum):
    """Failure taxonomy from Phase 3.4."""
    UQ = "UQ"  # Query understanding
    ER = "ER"  # Entity resolution
    RF = "RF"  # Retrieval
    RK = "RK"  # Ranking
    TM = "TM"  # Temporal handling
    RL = "RL"  # Relationship handling
    EV = "EV"  # Evidence
    AN = "AN"  # Answer/answer reasoning
    AB = "AB"  # Ambiguity
    ST = "ST"  # Storage/state
    PS = "PS"  # Persistence
    SC = "SC"  # Security
    RC = "RC"  # Recovery
    MD = "MD"  # Memory/deletion


class MetadataType(Enum):
    """Type of extension metadata."""
    DESCRIPTIVE = "DESCRIPTIVE"
    EXTENSION = "EXTENSION"


@dataclass(frozen=True)
class ScenarioIdentity:
    """Stable scenario identifier."""
    scenario_id: str
    category: str  # From Phase 3.3 family list
    difficulty: str = "MEDIUM"


@dataclass(frozen=True)
class DatasetIdentity:
    """Stable dataset identifier."""
    dataset_version: str
    dataset_size: int = 0  # Number of records used
    dataset_seed: int = 0


@dataclass(frozen=True)
class RetrievalResultItem:
    """Single retrieved item — no metric calculation."""
    memory_id: str
    rank: int
    retrieval_channel: str  # "lexical", "vector", "relationship", "combined"
    score: Optional[float] = None
    score_type: Optional[str] = None  # "similarity", "relevance", etc.
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence_relationship_id: Optional[str] = None


@dataclass(frozen=True)
class EvidenceReference:
    """Evidence reference separate from answer."""
    evidence_id: str
    source_memory_id: Optional[str] = None
    evidence_type: str = "MEMORY"  # MEMORY, RELATIONSHIP, OBSERVATION
    support_relationship: str = "SUPPORTS"  # SUPPORTS, CONTRADICTS, NEUTRAL
    provenance_reference: Optional[str] = None
    relevance_observation: Optional[str] = None  # Human-readable observation only


@dataclass(frozen=True)
class AnswerResult:
    """Candidate answer representation — not correctness judgment."""
    answer_text: str
    answer_class: AnswerClass
    confidence: Optional[float] = None
    claims: list[str] = field(default_factory=list)  # Structured claims if available
    evidence_references: list[str] = field(default_factory=list)  # Evidence IDs
    unsupported_claims: list[str] = field(default_factory=list)  # Observed unsupported claims
    ambiguity_notes: list[str] = field(default_factory=list)
    conflict_notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ExpectedResult:
    """Ground truth / expected interpretation — separate from candidate output."""
    scenario_id: str
    expected_memory_ids: list[str] = field(default_factory=list)
    expected_entity_ids: list[str] = field(default_factory=list)
    expected_relationship_ids: list[str] = field(default_factory=list)
    expected_answer_class: Optional[AnswerClass] = None
    expected_evidence_ids: list[str] = field(default_factory=list)
    expected_structure: Optional[str] = None  # Structured interpretation


@dataclass(frozen=True)
class TimingObservation:
    """Timing observations — metrics engine computes aggregates."""
    query_understanding_ms: Optional[float] = None
    planning_ms: Optional[float] = None
    entity_resolution_ms: Optional[float] = None
    retrieval_ms: Optional[float] = None
    ranking_ms: Optional[float] = None
    evidence_selection_ms: Optional[float] = None
    answer_generation_ms: Optional[float] = None
    verification_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None


@dataclass(frozen=True)
class ResourceObservation:
    """Optional resource measurements — observations only."""
    cpu_time_s: Optional[float] = None
    wall_time_s: Optional[float] = None
    peak_ram_mb: Optional[float] = None
    peak_vram_mb: Optional[float] = None
    disk_usage_bytes: Optional[int] = None
    db_size_bytes: Optional[int] = None
    index_size_bytes: Optional[int] = None
    model_startup_time_ms: Optional[float] = None
    recovery_time_ms: Optional[float] = None


@dataclass(frozen=True)
class FailureRecord:
    """Structured failure — not raw traceback unless appropriate."""
    category: FailureCategory
    code: str  # Short machine-readable code
    message: str  # Human-readable
    operation: Optional[str] = None  # Which contract operation
    scenario_id: Optional[str] = None
    recoverable: bool = True
    severity: Optional[SeverityLevel] = None
    stage: Optional[str] = None  # "query_understanding", "retrieval", etc.
    timestamp: Optional[datetime] = None
    duration_ms: Optional[float] = None
    exception_type: Optional[str] = None  # Exception class name (not full traceback)
    diagnostic_details: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkResult:
    """
    Structured benchmark result — architecture-neutral, candidate-neutral.

    Records what happened during a benchmark execution without computing
    metrics or selecting winners.

    All identity fields preserved from STEP 3 experiment configuration.
    """

    # Identity — must match experiment configuration
    experiment_id: str
    configuration_fingerprint: str
    result_id: str = field(default_factory=lambda: f"res_{uuid4().hex[:12]}")
    scenario_id: str

    # Candidate / dataset / session identity
    candidate_name: str
    candidate_version: str
    dataset_version: str
    benchmark_version: str
    result_schema_version: str = "0.1.0"

    # Execution
    execution_profile: str  # "ISOLATED" or "SEQUENTIAL_STATEFUL"
    execution_status: ExecutionStatus

    # Per-operation results (not aggregated scores)
    operation_statuses: list[dict[str, Any]] = field(default_factory=list)
    retrieval_results: list[RetrievalResultItem] = field(default_factory=list)
    evidence_references: list[EvidenceReference] = field(default_factory=list)
    answer_result: Optional[AnswerResult] = None
    expected_result: Optional[ExpectedResult] = None

    # Observations (raw measurements — metrics engine aggregates)
    timing_observations: Optional[TimingObservation] = None
    resource_observations: Optional[ResourceObservation] = None

    # Failures (structured — not hidden)
    failures: list[FailureRecord] = field(default_factory=list)
    error_details: Optional[dict[str, Any]] = None

    # Reproducibility
    random_seed: Optional[int] = None
    timestamp: Optional[datetime] = None
    source_revision: Optional[dict[str, Any]] = None

    # Extensible metadata — clearly scoped
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.experiment_id:
            raise ValueError("experiment_id is required")
        if not self.result_id:
            raise ValueError("result_id is required")
        if not self.scenario_id:
            raise ValueError("scenario_id is required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "result_id": self.result_id,
            "scenario_id": self.scenario_id,
            "candidate_name": self.candidate_name,
            "candidate_version": self.candidate_version,
            "dataset_version": self.dataset_version,
            "benchmark_version": self.benchmark_version,
            "result_schema_version": self.result_schema_version,
            "execution_profile": self.execution_profile,
            "execution_status": self.execution_status.value,
            "operation_statuses": self.operation_statuses,
            "retrieval_results": [
                {
                    "memory_id": r.memory_id,
                    "rank": r.rank,
                    "retrieval_channel": r.retrieval_channel,
                    "score": r.score,
                    "score_type": r.score_type,
                    "metadata": r.metadata,
                    "evidence_relationship_id": r.evidence_relationship_id,
                }
                for r in self.retrieval_results
            ],
            "evidence_references": [
                {
                    "evidence_id": e.evidence_id,
                    "source_memory_id": e.source_memory_id,
                    "evidence_type": e.evidence_type,
                    "support_relationship": e.support_relationship,
                    "provenance_reference": e.provenance_reference,
                    "relevance_observation": e.relevance_observation,
                }
                for e in self.evidence_references
            ],
            "answer_result": self.answer_result.to_dict() if self.answer_result else None,
            "expected_result": self.expected_result.to_dict() if self.expected_result else None,
            "timing_observations": self.timing_observations.to_dict() if self.timing_observations else None,
            "resource_observations": self.resource_observations.to_dict() if self.resource_observations else None,
            "failures": [
                {
                    "category": f.category.value,
                    "code": f.code,
                    "message": f.message,
                    "operation": f.operation,
                    "scenario_id": f.scenario_id,
                    "recoverable": f.recoverable,
                    "severity": f.severity.value if f.severity else None,
                    "stage": f.stage,
                    "timestamp": f.timestamp.isoformat() if f.timestamp else None,
                    "duration_ms": f.duration_ms,
                    "exception_type": f.exception_type,
                    "diagnostic_details": f.diagnostic_details,
                }
                for f in self.failures
            ],
            "error_details": self.error_details,
            "random_seed": self.random_seed,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "source_revision": self.source_revision,
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "BenchmarkResult":
        return BenchmarkResult(
            experiment_id=data["experiment_id"],
            configuration_fingerprint=data["configuration_fingerprint"],
            result_id=data.get("result_id", f"res_{data.get('experiment_id', 'unknown')[:8]}"),
            scenario_id=data["scenario_id"],
            candidate_name=data["candidate_name"],
            candidate_version=data["candidate_version"],
            dataset_version=data["dataset_version"],
            benchmark_version=data["benchmark_version"],
            result_schema_version=data.get("result_schema_version", "0.1.0"),
            execution_profile=data.get("execution_profile", "ISOLATED"),
            execution_status=ExecutionStatus(data.get("execution_status", "SUCCESS")),
            operation_statuses=data.get("operation_statuses", []),
            retrieval_results=[RetrievalResultItem(**r) for r in data.get("retrieval_results", [])],
            evidence_references=[EvidenceReference(**e) for e in data.get("evidence_references", [])],
            answer_result=AnswerResult(**data["answer_result"]) if data.get("answer_result") else None,
            expected_result=ExpectedResult(**data["expected_result"]) if data.get("expected_result") else None,
            timing_observations=TimingObservation(**data["timing_observations"]) if data.get("timing_observations") else None,
            resource_observations=ResourceObservation(**data["resource_observations"]) if data.get("resource_observations") else None,
            failures=[FailureRecord(**f) if isinstance(f, dict) else f for f in data.get("failures", [])],
            error_details=data.get("error_details"),
            random_seed=data.get("random_seed"),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            source_revision=data.get("source_revision"),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def from_json(json_str: str) -> "BenchmarkResult":
        import json
        return BenchmarkResult.from_dict(json.loads(json_str))
