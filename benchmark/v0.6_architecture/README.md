# ALLI v0.6 Architecture Benchmark

`benchmark/v0.6_architecture/`

This is the **architecture-validation benchmark** for the ALLI v0.6 Memory
Engine. It is a development-time instrument, not part of production ALLI.

## Purpose

Objectively evaluate competing memory architecture candidates **before**
the final v0.6 architecture is locked. The benchmark tests correctness
first and performance second. It is designed to **expose architectural
weaknesses**, not to hide them behind aggregate scores.

## Scope

- **In scope:** everything under `benchmark/v0.6_architecture/`
- **Out of scope (do not touch):** `src/`, `memory/`, production v0.6
  code, `conversation_archive.json`, `v06_dev.sqlite`, production
  databases and configuration, existing memory implementation

The benchmark must never read from or write to production ALLI state.

## Structure

```
benchmark/v0.6_architecture/
├── candidates/     # architecture candidate adapters (A, B, C)
├── dataset/        # synthetic dataset generator
├── scenarios/      # golden scenario framework
├── runners/        # benchmark execution engine
├── metrics/        # metric collection
├── reports/        # report generation
├── configs/        # experiment configuration
├── results/        # raw results (immutable)
└── tests/          # benchmark infrastructure tests
```

## Implementation Order (locked)

1. Benchmark package/skeleton
2. Common candidate contract and adapter layer
3. Experiment/configuration schema
4. Structured result schema
5. Synthetic dataset generator foundation
6. Golden scenario framework
7. Benchmark execution engine
8. Metrics collection
9. Failure classification
10. Reporting
11. Minimal smoke test (stub candidate)
12. Candidate A implementation

Candidate B and Candidate C are **not** to be implemented until Candidate
A runs through the complete pipeline.

## Hard Rules

- No hardcoded individual questions. Scenarios live in data, not in
  implementation conditionals.
- No production changes.
- No premature architecture selection.
- No premature optimization.
- No model fine-tuning.
- No arbitrary weighted winner score.
- No benchmark modifications designed to make a candidate pass.
- Architecture-neutral: the same scenario runs against A, B, and C
  through one common contract.

## Reproducibility

Every run records experiment ID, benchmark version, dataset version,
scenario version, candidate, candidate version, configuration, random
seed, model, embedding model, reranker, database version, OS, hardware,
timestamp, and git commit. Raw results are preserved; reports are
derived from raw results and never silently overwrite them.