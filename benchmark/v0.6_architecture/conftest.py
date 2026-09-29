"""Benchmark infrastructure execution fix (permanent, reproducible)."""
# The directory 'v0.6_architecture' contains a dot that Python treats as
# package separator. All benchmark source files reference the import path
# 'benchmark.v0_6_architecture...'. We make that path resolve permanently
# by injecting the exact package namespace the source expects.
import sys
from pathlib import Path

# Repository root
REPO = Path(__file__).resolve().parents[2]  # benchmark/v0.6_architecture/conftest.py -> repo root
BENCH_DIR = REPO / "benchmark" / "v0.6_architecture"

sys.path.insert(0, str(BENCH_DIR))

# Create the package namespace that source files import
import types
if "benchmark" not in sys.modules:
    benchmark_pkg = types.ModuleType("benchmark")
    benchmark_pkg.__path__ = [str(REPO / "benchmark")]
    sys.modules["benchmark"] = benchmark_pkg

# Alias v0_6_architecture -> actual dir contents
if "benchmark.v0_6_architecture" not in sys.modules:
    v06 = types.ModuleType("benchmark.v0_6_architecture")
    v06.__path__ = [str(BENCH_DIR)]
    sys.modules["benchmark.v0_6_architecture"] = v06
    # Subpackages
    for sub in ("candidates", "configs", "dataset", "execution",
                "failures", "metrics", "reports", "results",
                "runners", "scenarios", "tests"):
        submod = types.ModuleType(f"benchmark.v0_6_architecture.{sub}")
        submod.__path__ = [str(BENCH_DIR / sub)]
        sys.modules[f"benchmark.v0_6_architecture.{sub}"] = submod
