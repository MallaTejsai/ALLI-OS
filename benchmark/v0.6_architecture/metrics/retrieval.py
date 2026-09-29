"""STEP 8 — Retrieval metrics (no composite score, no winner)."""
from benchmark.v0_6_architecture.metrics.schema import RetrievalMetric, MetricResult, MetricType

class RetrievalMetrics:
    """Standard retrieval metric definitions — observation-based."""

    @staticmethod
    def compute_recall_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
        if not relevant_ids:
            return 0.0  # No false positive; no relevant items = 0
        retrieved_at_k = retrieved_ids[:k]
        relevant_retrieved = len(set(retrieved_at_k) & relevant_ids)
        return relevant_retrieved / len(relevant_ids)

    @staticmethod
    def compute_precision_at_k(relevant_ids: set, retrieved_ids: list, k: int) -> float:
        retrieved_at_k = retrieved_ids[:k]
        if not retrieved_at_k:
            return 0.0
        relevant_retrieved = len(set(retrieved_at_k) & relevant_ids)
        return relevant_retrieved / len(retrieved_at_k)

    @staticmethod
    def compute_mrr(relevant_ids: set, retrieved_ids: list) -> float:
        for rank, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant_ids:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def compute_ndcg(relevant_ids: set, retrieved_ids: list, k: int, grades: dict = None) -> float:
        # Simplified graded NDCG — binary relevance if grades not provided
        if not relevant_ids:
            return 0.0
        dcg = 0.0
        for i, item_id in enumerate(retrieved_ids[:k], start=1):
            rel = 1 if item_id in relevant_ids else 0
            dcg += rel / (2.0 ** (i - 1))  # Simplified DCG formulation
        # Ideal DCG = all relevant in top positions
        ideal_relevant = min(len(relevant_ids), k)
        idcg = sum(1.0 / (2.0 ** (i - 1)) for i in range(1, ideal_relevant + 1))
        if idcg == 0:
            return 0.0
        return dcg / idcg
