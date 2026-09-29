"""STEP 11 - Minimal Smoke Pipeline - actual execution."""
import pytest
from benchmark.v0_6_architecture.candidates.adapter import StubCandidateAdapter
from benchmark.v0_6_architecture.configs.experiment_config import ExperimentConfigurationBuilder
from benchmark.v0_6_architecture.execution.engine import BenchmarkEngine
from benchmark.v0_6_architecture.results.schema import BenchmarkResult
from benchmark.v0_6_architecture.metrics.schema import MetricResult, MetricConfiguration
from benchmark.v0_6_architecture.failures.schema import FailureClassification

def test_smoke_full_pipeline():
    # Step 5 dataset (minimal stub)
    from benchmark.v0_6_architecture.dataset.schema import DatasetConfig, MemoryClaim
    # Step 6 scenario (minimal stub)
    from benchmark.v0_6_architecture.scenarios.schema import GoldenScenario, ScenarioFamily, Difficulty, Severity, ExpectedAnswer
    # Step 2 adapter
    adapter = StubCandidateAdapter()
    # Step 3 config
    config = ExperimentConfigurationBuilder(testing=True).build()
    # Step 7 execution
    engine = BenchmarkEngine(config, adapter)
    result = engine.execute_isolated({"scenario_id": "sc_smoke_001"})
    # Step 4 result
    assert isinstance(result, BenchmarkResult)
    # Step 8 metrics exist
    from benchmark.v0_6_architecture.metrics.retrieval import RetrievalMetrics
    assert RetrievalMetrics.compute_recall_at_k(set(), ["m1"], 1) == 0.0
    # Step 9 failure exists
    from benchmark.v0_6_architecture.failures.schema import SeverityLevel, FailureCategory
    assert FailureCategory.UQ is not None
    # Verify pipeline completed
    assert result.experiment_id == config.experiment_id
    engine.shutdown()
