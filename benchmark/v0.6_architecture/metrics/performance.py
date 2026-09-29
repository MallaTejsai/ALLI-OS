"""STEP 8 — Performance metrics — raw percentiles only."""
from benchmark.v0.6_architecture.metrics.schema import PerformanceMetric

class PerformanceMetrics:
    @staticmethod
    def compute_percentile(values: list[float], p: float) -> float:
        if not values:
            return None  # Not zero — unavailable
        sorted_vals = sorted(values)
        k = (len(sorted_vals) - 1) * (p / 100.0)
        f = int(k)
        c = min(f + 1, len(sorted_vals) - 1)
        if f == c:
            return sorted_vals[f]
        return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

    @staticmethod
    def compute_p50(values: list[float]) -> float:
        return PerformanceMetrics.compute_percentile(values, 50)

    @staticmethod
    def compute_p95(values: list[float]) -> float:
        return PerformanceMetrics.compute_percentile(values, 95)

    @staticmethod
    def compute_p99(values: list[float]) -> float:
        return PerformanceMetrics.compute_percentile(values, 99)
