"""
Common Candidate Contract for ALLI v0.6 Architecture Benchmark.

This module defines the architecture-neutral interface that all
candidates (A: SQLite-centric, B: SQLite+Graph, C: Graph-native)
must implement to participate in the benchmark.

The contract is an interface boundary, NOT an implementation of any
memory architecture.

Operations (16 total, locked by specification):

1. initialize()          - Bring candidate to operational state
2. create_episode()      - Create a new episode container
3. create_entity()       - Create a new entity
4. create_memory()       - Create/store a memory
5. update_memory()       - Update an existing memory
6. supersede_memory()    - Historical replacement (not physical deletion)
7. create_relationship() - Create a relationship between entities
8. query()               - Higher-level query execution
9. retrieve()            - Retrieve memories/evidence
10. get_history()        - Expose historical state
11. delete()             - Deletion (distinct from supersession)
12. backup()             - Persistence capture for recovery testing
13. restore()            - Restore from backup
14. rebuild_indexes()    - Derived index recovery
15. health_check()       - Operational and integrity status
16. shutdown()           - Clean lifecycle termination

This contract uses benchmark-specific types (e.g., EpisodeId,
EntityId) rather than premature production domain models.
These can be mapped to canonical v0.6 types when they exist.

Error/Result Semantics:
Every operation returns an OperationResult with an explicit
StatusCode. The benchmark runner can distinguish:
- SUCCESS
- FAILURE
- INVALID_REQUEST
- UNSUPPORTED_OPERATION
- RECOVERABLE_ERROR
- FATAL_ERROR

No exceptions are silently converted to PASS.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Generic, Optional, TypeVar, Union
from uuid import UUID, uuid4


# ──────────────────────────────────────────────────────────────
# Stable Identifiers (benchmark contract types)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class EpisodeId:
    """Stable identifier for an episode container."""
    value: str

    @staticmethod
    def generate() -> "EpisodeId":
        return EpisodeId(f"ep_{uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class EntityId:
    """Stable identifier for an entity."""
    value: str

    @staticmethod
    def generate() -> "EntityId":
        return EntityId(f"ent_{uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class MemoryId:
    """Stable identifier for a memory."""
    value: str

    @staticmethod
    def generate() -> "MemoryId":
        return MemoryId(f"mem_{uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class RelationshipId:
    """Stable identifier for a relationship."""
    value: str

    @staticmethod
    def generate() -> "RelationshipId":
        return RelationshipId(f"rel_{uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class BackupId:
    """Stable identifier for a backup."""
    value: str

    @staticmethod
    def generate() -> "BackupId":
        return BackupId(f"bak_{uuid4().hex[:12]}")

    def __str__(self) -> str:
        return self.value


# ──────────────────────────────────────────────────────────────
# Request Types (benchmark contract types)
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InitializeRequest:
    """Request to initialize the candidate."""
    config: dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None


@dataclass(frozen=True)
class CreateEpisodeRequest:
    """Request to create a new episode."""
    episode_id: Optional[EpisodeId] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CreateEntityRequest:
    """Request to create a new entity."""
    entity_type: str = "OTHER"  # PERSON, PROJECT, TECHNOLOGY, ORGANIZATION, EDUCATION, PLACE, DEVICE, PRODUCT, CONCEPT, OTHER
    name: str = ""
    entity_id: Optional[EntityId] = None
    attributes: dict[str, Any] = field(default_factory=dict)
    episode_id: Optional[EpisodeId] = None


@dataclass(frozen=True)
class CreateMemoryRequest:
    """Request to create/store a memory."""
    episode_id: EpisodeId
    content: str = ""
    memory_id: Optional[MemoryId] = None
    entity_ids: list[EntityId] = field(default_factory=list)
    memory_kind: str = "FACT"  # FACT, PREFERENCE, GOAL, PLAN, STATE, EVENT, EXPERIENCE, RELATIONSHIP, OBSERVATION
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    provenance: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class UpdateMemoryRequest:
    """Request to update an existing memory."""
    memory_id: MemoryId
    content: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[dict[str, Any]] = None


@dataclass(frozen=True)
class SupersedeMemoryRequest:
    """Request to supersede a memory (historical replacement)."""
    old_memory_id: MemoryId
    new_memory: CreateMemoryRequest
    supersession_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CreateRelationshipRequest:
    """Request to create a relationship between entities."""
    source_entity_id: EntityId
    target_entity_id: EntityId
    relationship_id: Optional[RelationshipId] = None
    relationship_type: str = ""
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryRequest:
    """Request for higher-level query execution."""
    query_text: str
    episode_id: Optional[EpisodeId] = None
    max_results: int = 10
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrieveRequest:
    """Request to retrieve memories/evidence."""
    query_text: str
    episode_id: Optional[EpisodeId] = None
    entity_ids: Optional[list[EntityId]] = None
    memory_kinds: Optional[list[str]] = None
    max_results: int = 10
    temporal_scope: Optional[dict[str, Any]] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GetHistoryRequest:
    """Request for historical state."""
    entity_id: Optional[EntityId] = None
    memory_id: Optional[MemoryId] = None
    episode_id: Optional[EpisodeId] = None
    point_in_time: Optional[datetime] = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeleteRequest:
    """Request to delete (distinct from supersession)."""
    target_type: str  # "memory" | "entity" | "episode" | "relationship"
    target_id: Union[MemoryId, EntityId, EpisodeId, RelationshipId]
    reason: str = ""
    hard_delete: bool = False


@dataclass(frozen=True)
class BackupRequest:
    """Request to create a backup."""
    backup_id: Optional[BackupId] = None
    description: str = ""
    include_indexes: bool = True


@dataclass(frozen=True)
class RestoreRequest:
    """Request to restore from backup."""
    backup_id: BackupId
    target_episode_id: Optional[EpisodeId] = None


@dataclass(frozen=True)
class RebuildIndexesRequest:
    """Request to rebuild derived indexes."""
    index_types: Optional[list[str]] = None
    episode_id: Optional[EpisodeId] = None


@dataclass(frozen=True)
class HealthCheckRequest:
    """Request for health/integrity check."""
    deep_check: bool = False


@dataclass(frozen=True)
class ShutdownRequest:
    """Request for clean shutdown."""
    force: bool = False


# ──────────────────────────────────────────────────────────────
# Result Types
# ──────────────────────────────────────────────────────────────

T = TypeVar("T")


@dataclass
class OperationResult(Generic[T]):
    """Structured result for every contract operation."""
    status: "StatusCode"
    data: Optional[T] = None
    message: str = ""
    error_details: Optional[dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def is_success(self) -> bool:
        return self.status == StatusCode.SUCCESS

    @property
    def is_failure(self) -> bool:
        return self.status in (
            StatusCode.FAILURE,
            StatusCode.INVALID_REQUEST,
            StatusCode.UNSUPPORTED_OPERATION,
            StatusCode.RECOVERABLE_ERROR,
            StatusCode.FATAL_ERROR,
        )

    @staticmethod
    def success(data: T, message: str = "") -> "OperationResult[T]":
        return OperationResult(status=StatusCode.SUCCESS, data=data, message=message)

    @staticmethod
    def failure(message: str, error_details: Optional[dict[str, Any]] = None) -> "OperationResult[T]":
        return OperationResult(
            status=StatusCode.FAILURE,
            message=message,
            error_details=error_details,
        )

    @staticmethod
    def invalid_request(message: str, error_details: Optional[dict[str, Any]] = None) -> "OperationResult[T]":
        return OperationResult(
            status=StatusCode.INVALID_REQUEST,
            message=message,
            error_details=error_details,
        )

    @staticmethod
    def unsupported(message: str = "Operation not supported by this candidate") -> "OperationResult[T]":
        return OperationResult(
            status=StatusCode.UNSUPPORTED_OPERATION,
            message=message,
        )

    @staticmethod
    def recoverable_error(message: str, error_details: Optional[dict[str, Any]] = None) -> "OperationResult[T]":
        return OperationResult(
            status=StatusCode.RECOVERABLE_ERROR,
            message=message,
            error_details=error_details,
        )

    @staticmethod
    def fatal_error(message: str, error_details: Optional[dict[str, Any]] = None) -> "OperationResult[T]":
        return OperationResult(
            status=StatusCode.FATAL_ERROR,
            message=message,
            error_details=error_details,
        )


class StatusCode(Enum):
    """Explicit status codes for operation results."""
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    INVALID_REQUEST = "INVALID_REQUEST"
    UNSUPPORTED_OPERATION = "UNSUPPORTED_OPERATION"
    RECOVERABLE_ERROR = "RECOVERABLE_ERROR"
    FATAL_ERROR = "FATAL_ERROR"


# ──────────────────────────────────────────────────────────────
# Operation-Specific Result Payload Types
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class InitializeResult:
    candidate_name: str
    candidate_version: str
    capabilities: list[str] = field(default_factory=list)
    initialization_time_ms: float = 0.0


@dataclass(frozen=True)
class CreateEpisodeResult:
    episode_id: EpisodeId


@dataclass(frozen=True)
class CreateEntityResult:
    entity_id: EntityId


@dataclass(frozen=True)
class CreateMemoryResult:
    memory_id: MemoryId


@dataclass(frozen=True)
class UpdateMemoryResult:
    memory_id: MemoryId
    updated_fields: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SupersedeMemoryResult:
    old_memory_id: MemoryId
    new_memory_id: MemoryId


@dataclass(frozen=True)
class CreateRelationshipResult:
    relationship_id: RelationshipId


@dataclass(frozen=True)
class QueryResult:
    answer: str
    answer_class: str  # DIRECT, DERIVED, UNKNOWN, AMBIGUOUS, CONFLICTED
    evidence: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrieveResult:
    memories: list[dict[str, Any]] = field(default_factory=list)
    total_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GetHistoryResult:
    history: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeleteResult:
    deleted_id: str
    deleted_type: str
    hard_deleted: bool


@dataclass(frozen=True)
class BackupResult:
    backup_id: BackupId
    size_bytes: int
    checksum: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class RestoreResult:
    backup_id: BackupId
    restored_count: int
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True)
class RebuildIndexesResult:
    indexes_rebuilt: list[str]
    duration_ms: float


@dataclass(frozen=True)
class HealthCheckResult:
    healthy: bool
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ShutdownResult:
    clean: bool
    message: str = ""


# ──────────────────────────────────────────────────────────────
# Candidate Contract (Interface)
# ──────────────────────────────────────────────────────────────

class CandidateContract(ABC):
    """
    The common interface that all architecture candidates must implement.

    This is the boundary between the benchmark runner and the candidate.
    The runner MUST NOT depend on candidate internals.

    All operations return OperationResult with explicit StatusCode.
    Unsupported operations MUST return UNSUPPORTED_OPERATION, not raise.
    """

    @property
    @abstractmethod
    def candidate_name(self) -> str:
        """Human-readable name of this candidate (e.g., 'Candidate A - SQLite')."""
        pass

    @property
    @abstractmethod
    def candidate_version(self) -> str:
        """Version of this candidate implementation."""
        pass

    @abstractmethod
    def initialize(self, request: InitializeRequest) -> OperationResult[InitializeResult]:
        """Bring candidate to operational state. Must be called first."""
        pass

    @abstractmethod
    def create_episode(self, request: CreateEpisodeRequest) -> OperationResult[CreateEpisodeResult]:
        """Create a new episode container."""
        pass

    @abstractmethod
    def create_entity(self, request: CreateEntityRequest) -> OperationResult[CreateEntityResult]:
        """Create a new entity."""
        pass

    @abstractmethod
    def create_memory(self, request: CreateMemoryRequest) -> OperationResult[CreateMemoryResult]:
        """Create/store a memory."""
        pass

    @abstractmethod
    def update_memory(self, request: UpdateMemoryRequest) -> OperationResult[UpdateMemoryResult]:
        """Update an existing memory."""
        pass

    @abstractmethod
    def supersede_memory(self, request: SupersedeMemoryRequest) -> OperationResult[SupersedeMemoryResult]:
        """
        Historically replace/supersede a memory.

        CRITICAL: This is NOT physical deletion. The benchmark tests
        that superseded memories remain accessible via get_history().
        """
        pass

    @abstractmethod
    def create_relationship(self, request: CreateRelationshipRequest) -> OperationResult[CreateRelationshipResult]:
        """Create a relationship between entities."""
        pass

    @abstractmethod
    def query(self, request: QueryRequest) -> OperationResult[QueryResult]:
        """Higher-level query execution (returns answer + evidence)."""
        pass

    @abstractmethod
    def retrieve(self, request: RetrieveRequest) -> OperationResult[RetrieveResult]:
        """Retrieve memories/evidence (raw retrieval, no answer generation)."""
        pass

    @abstractmethod
    def get_history(self, request: GetHistoryRequest) -> OperationResult[GetHistoryResult]:
        """
        Expose historical state.

        CRITICAL: Must return superseded/deleted memories for scenarios
        testing historical state, point-in-time queries, and supersession.
        """
        pass

    @abstractmethod
    def delete(self, request: DeleteRequest) -> OperationResult[DeleteResult]:
        """
        Delete a target.

        CRITICAL: Deletion is semantically distinct from supersession.
        Benchmark tests verify this distinction is preserved.
        """
        pass

    @abstractmethod
    def backup(self, request: BackupRequest) -> OperationResult[BackupResult]:
        """Create a backup for persistence/recovery testing."""
        pass

    @abstractmethod
    def restore(self, request: RestoreRequest) -> OperationResult[RestoreResult]:
        """Restore from a backup."""
        pass

    @abstractmethod
    def rebuild_indexes(self, request: RebuildIndexesRequest) -> OperationResult[RebuildIndexesResult]:
        """Rebuild derived indexes (for index recovery testing)."""
        pass

    @abstractmethod
    def health_check(self, request: HealthCheckRequest) -> OperationResult[HealthCheckResult]:
        """Return operational status and integrity checks."""
        pass

    @abstractmethod
    def shutdown(self, request: ShutdownRequest) -> OperationResult[ShutdownResult]:
        """Clean lifecycle termination."""
        pass