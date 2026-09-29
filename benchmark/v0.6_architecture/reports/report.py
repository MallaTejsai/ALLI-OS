"""STEP 10 — Reporting framework (structured report from results/metrics/failures)."""
from benchmark.v0_6_architecture.results.schema import BenchmarkResult
from benchmark.v0_6_architecture.metrics.schema import MetricResult
from benchmark.v0_6_architecture.failures.schema import FailureClassification
from typing import Optional, List, Any
import json

class BenchmarkReport:
    """Structured report referencing observed results, metrics, failures — not computing them."""
    def __init__(self,
                 experiment_id: str,
                 configuration_fingerprint: str,
                 scenario_id: str,
                 benchmark_version: str = "0.6.0",
                 result: Optional[BenchmarkResult] = None,
                 metric_results: Optional[List[Any]] = None,
                 failure_classifications: Optional[List[Any]] = None,
                 metadata: Optional[dict] = None):
        self.experiment_id = experiment_id
        self.configuration_fingerprint = configuration_fingerprint
        self.scenario_id = scenario_id
        self.benchmark_version = benchmark_version
        self.result = result
        self.metric_results = metric_results or []
        self.failure_classifications = failure_classifications or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "configuration_fingerprint": self.configuration_fingerprint,
            "scenario_id": self.scenario_id,
            "benchmark_version": self.benchmark_version,
            "result_reference": self.result.to_dict() if self.result else None,
            "metric_references": [
                m.to_dict() if hasattr(m, "to_dict") else str(m)
                for m in self.metric_results
            ],
            "failure_references": [
                f.to_dict() if hasattr(f, "to_dict") else str(f)
                for f in self.failure_classifications
            ],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)
