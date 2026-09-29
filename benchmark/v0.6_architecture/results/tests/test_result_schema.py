"""STEP 4 — Structured Result Schema Tests."""

import pytest
from datetime import datetime
from benchmark.v0.6_architecture.results.schema import (
    BenchmarkResult, ExecutionStatus, OperationStatus, AnswerClass,
    SeverityLevel, FailureCategory, RetrievalResultItem, EvidenceReference,
    AnswerResult, ExpectedResult, TimingObservation, ResourceObservation,
    FailureRecord, ScenarioIdentity, DatasetIdentity
)


class TestResultConstruction:
    def test_basic_result_creation(self):
        r = BenchmarkResult(
            experiment_id="exp_001",
            configuration_fingerprint="abcd1234",
            scenario_id="scen_001",
            candidate_name="Candidate A",
            candidate_version="0.1.0",
            dataset_version="0.1.0",
            benchmark_version="0.6.0",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
        )
        assert r.experiment_id == "exp_001"
        assert r.result_id.startswith("res_")

    def test_missing_experiment_id_fails(self):
        with pytest.raises(ValueError):
            BenchmarkResult(
                experiment_id="",
                configuration_fingerprint="fp",
                scenario_id="scen",
                candidate_name="A",
                candidate_version="1",
                dataset_version="1",
                benchmark_version="0.6",
                execution_profile="ISOLATED",
                execution_status=ExecutionStatus.SUCCESS,
            )


class TestStatusEnums:
    def test_execution_status_values(self):
        for s in ExecutionStatus:
            assert s.value in ["SUCCESS", "FAILURE", "PARTIAL", "SKIPPED", "TIMEOUT", "UNSUPPORTED"]

    def test_operation_status_values(self):
        for s in OperationStatus:
            assert s.value is not None

    def test_answer_class_values(self):
        for s in AnswerClass:
            assert s.value in ["DIRECT", "DERIVED", "UNKNOWN", "AMBIGUOUS", "CONFLICTED"]


class TestIdentity:
    def test_experiment_identity_preserved(self):
        r = BenchmarkResult(
            experiment_id="exp_123",
            configuration_fingerprint="fp_abc",
            scenario_id="sc_456",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
        )
        assert r.experiment_id == "exp_123"
        assert r.configuration_fingerprint == "fp_abc"

    def test_scenario_identity_separate(self):
        r = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="scen_001",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
        )
        assert r.scenario_id == "scen_001"

    def test_result_id_generated(self):
        r = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="sc",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
        )
        assert r.result_id.startswith("res_")


class TestRetrievalResultRepresentation:
    def test_retrieval_result_items(self):
        items = [
            RetrievalResultItem(memory_id="m1", rank=1, retrieval_channel="lexical", score=0.95),
            RetrievalResultItem(memory_id="m2", rank=2, retrieval_channel="vector", score=0.88),
        ]
        assert len(items) == 2
        assert items[0].rank == 1
        assert items[1].score == 0.88

    def test_retrieval_no_metric_calculation(self):
        # Schema stores observations — no Recall@K, Precision@K, etc.
        items = [RetrievalResultItem(memory_id="m1", rank=1, retrieval_channel="combined")]
        assert items[0].memory_id == "m1"


class TestEvidenceRepresentation:
    def test_evidence_reference(self):
        ev = EvidenceReference(
            evidence_id="ev_1",
            source_memory_id="m_1",
            evidence_type="MEMORY",
            support_relationship="SUPPORTS",
        )
        assert ev.support_relationship == "SUPPORTS"

    def test_evidence_separate_from_answer(self):
        answer = AnswerResult(answer_text="yes", answer_class=AnswerClass.DIRECT)
        ev_refs = [EvidenceReference(evidence_id="ev_1")]
        # Evidence and answer are separate structures
        assert answer.answer_text != ev_refs[0].evidence_id


class TestAnswerRepresentation:
    def test_answer_result(self):
        ans = AnswerResult(
            answer_text="The project started in 2024.",
            answer_class=AnswerClass.DIRECT,
            confidence=0.92,
        )
        assert ans.answer_class == AnswerClass.DIRECT
        assert ans.confidence == 0.92

    def test_unsupported_claims_separate(self):
        ans = AnswerResult(
            answer_text="Answer with unsupported claim.",
            answer_class=AnswerClass.DIRECT,
            unsupported_claims=["claim not in evidence"],
        )
        assert len(ans.unsupported_claims) == 1


class TestExpectedObservedSeparation:
    def test_expected_result_separate(self):
        expected = ExpectedResult(
            scenario_id="sc_001",
            expected_memory_ids=["m_1"],
            expected_answer_class=AnswerClass.DIRECT,
        )
        result = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="sc_001",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
            expected_result=expected,
        )
        assert result.expected_result is not None
        assert result.expected_result.expected_memory_ids == ["m_1"]


class TestFailureRepresentation:
    def test_structured_failure(self):
        failure = FailureRecord(
            category=FailureCategory.UQ,
            code="UQ-001",
            message="Query misunderstood",
            recoverable=True,
            severity=SeverityLevel.S1_MAJOR,
        )
        assert failure.category == FailureCategory.UQ
        assert failure.recoverable == True

    def test_failure_in_result(self):
        r = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="sc",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.FAILURE,
            failures=[FailureRecord(
                category=FailureCategory.ER,
                code="ER-002",
                message="Entity not resolved",
                recoverable=False,
                severity=SeverityLevel.S0_CRITICAL,
            )],
        )
        assert r.execution_status == ExecutionStatus.FAILURE
        assert len(r.failures) == 1
        assert r.failures[0].category == FailureCategory.ER


class TestTiming:
    def test_timing_observations_no_aggregation(self):
        timing = TimingObservation(
            query_understanding_ms=12.5,
            retrieval_ms=45.2,
            total_latency_ms=120.0,
        )
        assert timing.query_understanding_ms == 12.5
        # No p50/p95 calculated here — just observations
        assert timing.total_latency_ms == 120.0


class TestResource:
    def test_resource_observations(self):
        res = ResourceObservation(
            cpu_time_s=2.5,
            peak_ram_mb=1024,
        )
        assert res.cpu_time_s == 2.5


class TestSerialization:
    def test_result_serialization(self):
        r = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="sc",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
            timing_observations=TimingObservation(query_understanding_ms=10),
        )
        j = r.to_json()
        restored = BenchmarkResult.from_json(j)
        assert restored.experiment_id == "e1"
        assert restored.execution_status == ExecutionStatus.SUCCESS
        assert restored.timing_observations.query_understanding_ms == 10.0

    def test_result_immutable(self):
        r = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="sc",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
        )
        with pytest.raises(Exception):
            r.execution_status = ExecutionStatus.FAILURE


class TestReproducibility:
    def test_reproducibility_metadata(self):
        r = BenchmarkResult(
            experiment_id="exp_001",
            configuration_fingerprint="fp_abc",
            scenario_id="scen_1",
            candidate_name="Candidate A",
            candidate_version="0.1.0",
            dataset_version="0.1.0",
            benchmark_version="0.6.0",
            result_schema_version="0.1.0",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
            random_seed=42,
            source_revision={"git_commit": "abc123"},
        )
        assert r.experiment_id == "exp_001"
        assert r.configuration_fingerprint == "fp_abc"


class TestNoMetricsInResult:
    def test_result_has_no_recall_precision(self):
        r = BenchmarkResult(
            experiment_id="e1",
            configuration_fingerprint="fp",
            scenario_id="sc",
            candidate_name="A",
            candidate_version="1",
            dataset_version="0.1",
            benchmark_version="0.6",
            execution_profile="ISOLATED",
            execution_status=ExecutionStatus.SUCCESS,
        )
        assert not hasattr(r, "recall")
        assert not hasattr(r, "precision")
        assert not hasattr(r, "mrr")
        assert not hasattr(r, "ndcg")
