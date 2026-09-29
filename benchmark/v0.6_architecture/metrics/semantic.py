"""STEP 8 — Semantic correctness — structured, no LLM, no cloud."""
from benchmark.v0.6_architecture.metrics.schema import SemanticCorrectness

class SemanticEvaluator:
    """Structured semantic evaluation — observation only."""

    @staticmethod
    def evaluate_answer_class(observed_class: str, expected_class: str) -> bool:
        return observed_class == expected_class

    @staticmethod
    def evaluate_evidence_support(evidence_refs: list, expected_refs: list) -> bool:
        # Structured reference comparison — not truth judgment
        return len(set(evidence_refs) & set(expected_refs)) > 0

    @staticmethod
    def evaluate_unsupported_claims(observed_claims: list, evidence_refs: list) -> int:
        # Count unsupported if structured
        return len(observed_claims) if observed_claims else 0
