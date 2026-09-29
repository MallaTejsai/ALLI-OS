"""
ALLI v0.6 Architecture Benchmark — Package Root.

This package is the benchmark workspace for objectively evaluating
competing memory architecture candidates for the ALLI v0.6 Memory
Engine.

Scope:
    This benchmark is a development-time validation instrument. It is
    isolated from production ALLI (src/, memory/, production databases,
    conversation archives). Nothing in this package is imported by or
    imported into production code.

Purpose:
    Evaluate architecture candidates (A, B, C) against a common contract
    using deterministic synthetic datasets and golden scenarios, before
    the final v0.6 architecture is locked.

The benchmark does NOT:
    - modify production ALLI
    - select an architecture prematurely
    - hardcode individual questions
    - optimize for passing tests
    - hide failures behind aggregate scores

Note on imports:
    Due to the dot in the directory name 'v0.6_architecture', this
    package cannot be imported with standard Python imports.

    To use the benchmark:
    1. Add the benchmark directory to sys.path
    2. Import using the internal package name 'alli_bench'

    Example:
        import sys
        sys.path.insert(0, 'benchmark/v0.6_architecture')
        from alli_bench.candidates import CandidateContract
"""

__all__ = [
    "candidates",
    "dataset",
    "scenarios",
    "runners",
    "metrics",
    "reports",
    "configs",
    "results",
]

# Package metadata
__package_name__ = "alli_bench"
__benchmark_workspace__ = "benchmark/v0.6_architecture/"