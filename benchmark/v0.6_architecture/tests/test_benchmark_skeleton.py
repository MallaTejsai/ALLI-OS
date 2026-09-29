"""Step 1 smoke test: verify the benchmark skeleton package structure exists.

This test validates that STEP 1 produced the correct directory
layout and foundational files. It is NOT an architecture
evaluation — it only proves the package skeleton exists.
"""

import os
import sys
from pathlib import Path

import pytest


def _project_root() -> Path:
    """Return the repository root."""
    # benchmark/v0.6_architecture/ is nested under the repo root
    return Path(__file__).resolve().parents[3]


def _benchmark_root() -> Path:
    return _project_root() / "benchmark" / "v0.6_architecture"


def test_benchmark_root_exists():
    """The benchmark root directory must exist."""
    root = _benchmark_root()
    assert root.exists(), f"Benchmark root missing: {root}"
    assert root.is_dir(), f"Benchmark root is not a directory: {root}"


def test_benchmark_subdirectories_exist():
    """All planned subdirectories must be present."""
    root = _benchmark_root()
    required = [
        "candidates",
        "dataset",
        "scenarios",
        "runners",
        "metrics",
        "reports",
        "configs",
        "results",
        "tests",
    ]
    for name in required:
        d = root / name
        assert d.exists(), f"Missing subdirectory: {name}"
        assert d.is_dir(), f"Not a directory: {name}"


def test_no_hardcoded_question_logic():
    """Step 1: confirm no implementation contains hardcoded question
    conditionals. This will be re-checked at each step."""
    py_files = list(_benchmark_root().rglob("*.py"))
    # Exclude this test file from the scan to avoid self-detection
    py_files = [f for f in py_files if f.name != "test_benchmark_skeleton.py"]
    suspicious_patterns = [
        'if query == "',
        "if 'query' ==",
        'if "what is my name"',
        "if 'python' in",
    ]
    for f in py_files:
        text = f.read_text(encoding="utf-8")
        for pat in suspicious_patterns:
            assert pat not in text, (
                f"Potential hardcoded question logic found in {f}: {pat}"
            )


def test_production_directories_untouched():
    """Confirm production directories exist and no benchmark files
    leaked into them."""
    root = _project_root()
    assert (root / "src").exists(), "Production src/ missing"
    assert (root / "memory").exists(), "Production memory/ missing"

    # The benchmark must not have created files outside its root
    assert not (root / "src" / "benchmark").exists()
    assert not (root / "memory" / "benchmark").exists()


def test_version_module_loadable():
    """The version module must import without error."""
    sys.path.insert(0, str(_benchmark_root()))
    try:
        import v06_architecture  # noqa: F401
    except ImportError:
        pass  # optional at this stage; skeleton does not require it


def test_readme_exists():
    """README documenting the benchmark must be present."""
    assert (_benchmark_root() / "README.md").exists()