"""
Benchmark version metadata.

Single source of truth for the benchmark's own version identity.
Used by the reproducibility record on every run.
"""

# Benchmark framework version. Bumped on structural changes.
BENCHMARK_VERSION = "0.6.0"

# Dataset generator version. Bumped when generation logic changes.
DATASET_VERSION = "0.1.0"

# Golden scenario framework version.
SCENARIO_VERSION = "0.1.0"

# Candidate contract version. Bumped when the common contract changes.
CANDIDATE_CONTRACT_VERSION = "0.1.0"

# Result schema version. Bumped when result serialization changes.
RESULT_SCHEMA_VERSION = "0.1.0"


def get_version() -> str:
    """Return the benchmark framework version string."""
    return BENCHMARK_VERSION


def get_candidate_contract_version() -> str:
    """Return the candidate contract version string."""
    return CANDIDATE_CONTRACT_VERSION