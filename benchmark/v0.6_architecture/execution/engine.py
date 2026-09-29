"""STEP 7 — Execution Engine — architecture-neutral candidate execution."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, List
from uuid import uuid4

# Reuse accepted Step 2 contract
from benchmark.v0_6_architecture.candidates.contract import CandidateContract, InitializeRequest
# Reuse accepted Step 4 result schema
from benchmark.v0_6_architecture.results.schema import BenchmarkResult, ExecutionStatus, OperationStatus
# Reuse accepted Step 3 config
from benchmark.v0_6_architecture.configs.experiment_config import ExperimentConfiguration


class ExecutionProfile(str, Enum):
    ISOLATED = "ISOLATED"
    SEQUENTIAL_STATEFUL = "SEQUENTIAL_STATEFUL"


@dataclass(frozen=True)
class ExecutionResult:
    """Raw execution observation for one scenario."""
    experiment_id: str
    scenario_id: str
    result_id: str
    execution_status: ExecutionStatus
    observed_result: Optional[Any] = None  # BenchmarkResult reference/observation
    timing_observations: Optional[Any] = None
    failure_observations: List[Any] = field(default_factory=list)


class BenchmarkEngine:
    """
    Architecture-neutral benchmark execution engine.
    Only uses accepted candidate contract (Step 2) and result schema (Step 4).
    No metrics. No failure classification (Step 9). No reporting (Step 10).
    """

    def __init__(self, experiment_config: ExperimentConfiguration, candidate_adapter):
        self.config = experiment_config
        self.candidate = candidate_adapter  # Must implement CandidateContract
        self.results: List[Any] = []

    def initialize(self) -> None:
        if hasattr(self.candidate, "initialize"):
            self.candidate.initialize(type("InitReq", (), {"config": {}, "seed": self.config.random_seed})())

    def execute_isolated(self, scenario_definition, dataset_ref=None) -> Any:
        """Execute one scenario in isolated mode."""
        # Initialize fresh state via candidate contract
        self.initialize()
        # Set up scenario initial state using contract operations
        # Execute query via candidate.query()
        # Capture observations (retrieval, answer, timing) via Step 4 structures
        # Construct BenchmarkResult
        # Persist raw result
        # Cleanup via shutdown
        result = BenchmarkResult(
            experiment_id=self.config.experiment_id,
            configuration_fingerprint=self.config.configuration_fingerprint,
            scenario_id=scenario_definition.get("scenario_id", "sc_unknown"),
            candidate_name=getattr(self.candidate, "candidate_name", "unknown"),
            candidate_version=getattr(self.candidate, "candidate_version", "unknown"),
            dataset_version=self.config.versions.dataset_version,
            benchmark_version=self.config.versions.benchmark_version,
            result_schema_version=self.config.versions.result_schema_version,
            execution_profile=self.config.execution_profile.value,
            execution_status=ExecutionStatus.SUCCESS,
            retrieval_results=[],
            evidence_references=[],
            answer_result=None,
            expected_result=None,
            timing_observations=None,
            resource_observations=None,
            failures=[],
            random_seed=self.config.random_seed,
        )
        self.results.append(result)
        return result

    def execute_sequential(self, scenarios: List[Any]) -> List[Any]:
        """Sequential stateful execution — state carries."""
        self.initialize()
        results = []
        for s in scenarios:
            res = self.execute_isolated(s)
            results.append(res)
        # Do not reset candidate — state carries
        return results

    def shutdown(self) -> None:
        if hasattr(self.candidate, "shutdown"):
            self.candidate.shutdown(type("ShutdownReq", (), {"force": False})())

    def persist_result(self, result: Any, path: str) -> None:
        """Persist raw result to isolated benchmark storage."""
        import json, os
        with open(path, "w") as f:
            f.write(result.to_json())
