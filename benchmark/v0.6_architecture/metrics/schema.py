"""STEP 8 — Metric Schema."""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Any, List


class MetricType(Enum):
    RETRIEVAL = "retrieval"
    SEMANTIC_CORRECTNESS = "semantic_correctness"
    EVIDENCE_SUPPORT = "evidence_support"
    PERFORMANCE = "performance"


@dataclass(frozen=True)
class RetrievalMetric:
    """Retrieval metric result — no arbitrary composite score."""
    recall_at_k: Optional[float] = None  # None if unavailable/unsupported
    precision_at_k: Optional[float] = None
    mrr: Optional[float] = None
    ndcg: Optional[float] = None
    k: int = 10
    relevant_items: int = 0
    retrieved_items: int = 0
    denominator: int = 0  # Explicit, never hidden


@dataclass(frozen=True)
class SemanticCorrectness:
    """Semantic correctness — structured, not arbitrary score."""
    answer_class_correct: bool = False
    evidence_supported: bool = False
    unsupported_claims_detected: bool = False
    conflict_detected: bool = False
    ambiguity_surface: bool = False
    unknown_abstention: bool = False
    evaluation_status: str = "VALID"  # VALID / UNSUPPORTED / UNEVALUATED / INVALID


@dataclass(frozen=True)
class EvidenceSupport:
    """Evidence support observation — distinction preserved."""
    evidence_refs_present: bool = False
    expected_evidence_found: bool = False
    conflicting_evidence_present: bool = False
    unsupported_evidence_claimed: bool = False
    evidence_sufficiency: Optional[str] = None  # sufficient / insufficient / unknown


@dataclass(frozen=True)
class PerformanceMetric:
    """Performance observations — percentiles only when data exists."""
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    total_observations: int = 0
    population: str = "all"  # scope identifier


@dataclass(frozen=True)
class MetricConfiguration:
    """Metric evaluator configuration."""
    benchmark_version: str = "0.6.0"
    metric_version: str = "0.1.0"
    k_values: List[int] = field(default_factory=lambda: [1, 5, 10])
    percentile_method: str = "linear"  # linear / nearest / midpoint
    include_semantic: bool = True
    include_performance: bool = True
    include_retrieval: bool = True


@dataclass(frozen=True)
class MetricResult:
    """Metric result — no composite score, no winner selection."""
    experiment_id: str
    configuration_fingerprint: str
    scenario_id: str
    metric_type: MetricType
    metric_name: str  # e.g., "recall_at_10"
    value: Optional[float] = None
    scope: str = "per_scenario"
    population_count: int = 0
    denominator: int = 0
    status: str = "VALID"  # VALID / UNSUPPORTED / UNEVALUATED / INVALID
    timestamp: Optional[str] = None
    metric_version: str = "0.1.0"
    config_version: str = "0.1.0"

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "scenario_id": self.scenario_id,
            "metric_type": self.metric_type.value,
            "metric_name": self.metric_name,
            "value": self.value,
            "scope": self.scope,
            "population_count": self.population_count,
            "denominator": self.denominator,
            "status": self.status,
            "timestamp": self.timestamp,
            "metric_version": self.metric_version,
            "config_version": self.config_version,
        }

    def to_json(self) -> str:
        import json
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)
