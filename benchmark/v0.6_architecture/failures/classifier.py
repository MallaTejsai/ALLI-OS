"""STEP 9 — Failure Classification."""
from benchmark.v0_6_architecture.failures.schema import (
    SeverityLevel, FailureCategory, ClassificationStatus, FailureClassification
)

class FailureClassifier:
    """Deterministic failure classifier — uses evidence, not LLM."""

    @staticmethod
    def classify(
        experiment_id: str,
        scenario_id: str,
        result_id: Optional[str],
        execution_status: Any,
        retrieval_observations: Optional[List[Any]] = None,
        answer_result: Optional[Any] = None,
        expected_result: Optional[Any] = None,
    ) -> FailureClassification:
        # Basic deterministic classification based on observed status and available evidence
        # Primary category derived from execution status + evidence presence
        # Not a full inference engine — structured framework for evidence-based classification

        category = FailureCategory.AN  # Default if unclassifiable; overridden by evidence
        status = ClassificationStatus.UNCLASSIFIED
        severity = SeverityLevel.S2_MODERATE

        if execution_status is not None and hasattr(execution_status, "value"):
            status_val = execution_status.value
            if status_val == "FAILURE":
                status = ClassificationStatus.CONFIRMED
                category = FailureCategory.AN  # Default — requires evidence for specificity
                severity = SeverityLevel.S1_MAJOR
            elif status_val == "SUCCESS":
                status = ClassificationStatus.CONFIRMED
                category = FailureCategory.UQ  # No failure — but taxonomy must be representable
                severity = SeverityLevel.S3_MINOR

        if retrieval_observations is not None and len(retrieval_observations) == 0 and expected_result is not None:
            # Required evidence missing — could indicate retrieval failure
            category = FailureCategory.RF
            status = ClassificationStatus.PROBABLE
            severity = SeverityLevel.S1_MAJOR

        return FailureClassification(
            classification_id=f"fc_{scenario_id}_{experiment_id}",
            experiment_id=experiment_id,
            scenario_id=scenario_id,
            result_id=result_id or "",
            primary_category=category,
            status=status,
            severity=severity,
            evidence_refs=[],
            observed_symptoms=[str(execution_status) if execution_status else "unknown"],
        )
