"""
Candidates subpackage for the ALLI v0.6 Architecture Benchmark.

This package contains:

- The common candidate contract (interface definition).
- The adapter layer that lets the benchmark runner invoke any
  architecture candidate through one stable interface.
- Minimal stubs for Candidate A, B, and C to prove the adapter
  architecture works.

This is NOT the implementation of any memory architecture.
The actual SQLite-centric (A), SQLite+Graph (B), and Graph-native
(C) implementations are deferred until after the benchmark
infrastructure is proven on a stub.
"""

from benchmark.v0_6_architecture.candidates.contract import (
    CandidateContract,
    OperationResult,
    StatusCode,
    EpisodeId,
    EntityId,
    MemoryId,
    RelationshipId,
)
from benchmark.v0_6_architecture.candidates.adapter import CandidateAdapter, StubCandidateAdapter
from benchmark.v0_6_architecture.candidates.adapter_b import CandidateBAdapter

__all__ = [
    "CandidateContract",
    "OperationResult",
    "StatusCode",
    "EpisodeId",
    "EntityId",
    "MemoryId",
    "RelationshipId",
    "CandidateAdapter",
    "StubCandidateAdapter",
    "CandidateBAdapter",
]