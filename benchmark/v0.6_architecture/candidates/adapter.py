"""
Adapter Layer for ALLI v0.6 Architecture Benchmark.

This module provides the adapter layer that allows the benchmark runner
to interact with different architecture candidates through a common interface.

The adapter layer is architecture-neutral and does not contain
implementation details of any specific candidate.

Key components:

1. CandidateAdapter - Base class that defines the adapter interface
2. StubCandidateAdapter - Minimal stub implementation for testing
3. Adapter pattern that isolates candidate-specific logic
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

# Import from contract module
from .contract import (
    CandidateContract,
    OperationResult,
    StatusCode,
    InitializeRequest,
    CreateEpisodeRequest,
    CreateEntityRequest,
    CreateMemoryRequest,
    UpdateMemoryRequest,
    SupersedeMemoryRequest,
    CreateRelationshipRequest,
    QueryRequest,
    RetrieveRequest,
    GetHistoryRequest,
    DeleteRequest,
    BackupRequest,
    RestoreRequest,
    RebuildIndexesRequest,
    HealthCheckRequest,
    ShutdownRequest,
    OperationResult,
    StatusCode,
    InitializeResult,
    CreateEpisodeResult,
    CreateEntityResult,
    CreateMemoryResult,
    UpdateMemoryResult,
    SupersedeMemoryResult,
    CreateRelationshipResult,
    QueryResult,
    RetrieveResult,
    GetHistoryResult,
    DeleteResult,
    BackupResult,
    RestoreResult,
    RebuildIndexesResult,
    HealthCheckResult,
    ShutdownResult,
    EpisodeId,
    EntityId,
    MemoryId,
    RelationshipId,
    BackupId,
)


class CandidateAdapter(CandidateContract, ABC):
    """
    Base class for candidate adapters.

    This is the architecture-neutral interface that the benchmark runner
    uses to communicate with any candidate. The adapter layer handles
    translating the common contract into candidate-specific calls.
    """

    @property
    @abstractmethod
    def candidate_name(self) -> str:
        """Human-readable name of this candidate."""
        pass

    @property
    @abstractmethod
    def candidate_version(self) -> str:
        """Version of this candidate implementation."""
        pass

    # All 16 operations must be implemented by concrete adapters
    # The stub implementation will show how this works

    # The methods below are abstract and must be implemented by concrete adapters


class StubCandidateAdapter(CandidateAdapter):
    """
    Minimal stub implementation to demonstrate the adapter pattern works.

    This stub is NOT an actual memory architecture. It's a test double
    that proves the adapter pattern is functional.

    The stub implements all 16 operations with basic functionality:
    - Creates mock data structures
    - Returns structured results with proper status codes
    - Demonstrates lifecycle management
    - Shows how unsupported operations would be handled
    """

    def __init__(self):
        """Initialize the stub candidate."""
        self.initialized = False
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.entities: Dict[str, Dict[str, Any]] = {}
        self.memories: Dict[str, Dict[str, Any]] = {}
        self.relationships: Dict[str, Dict[str, Any]] = {}
        self.history: List[Dict[str, Any]] = []

    @property
    def candidate_name(self) -> str:
        return "Stub Candidate"

    @property
    def candidate_version(self) -> str:
        return "0.1.0-stub"

    def initialize(self, request: InitializeRequest) -> OperationResult[InitializeResult]:
        """Initialize the candidate."""
        self.initialized = True
        return OperationResult.success(
            InitializeResult(
                candidate_name=self.candidate_name,
                candidate_version=self.candidate_version,
                capabilities=["initialize", "create_episode", "create_entity", "create_memory"],
                initialization_time_ms=0.0
            ),
            "Candidate initialized successfully"
        )

    def create_episode(self, request: CreateEpisodeRequest) -> OperationResult[CreateEpisodeResult]:
        """Create a new episode container."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        episode_id = request.episode_id or EpisodeId.generate()
        self.episodes[episode_id.value] = {
            "metadata": request.metadata,
            "created_at": datetime.utcnow().isoformat(),
            "content": "stub_episode_content"
        }
        return OperationResult.success(
            CreateEpisodeResult(episode_id=episode_id),
            "Episode created successfully"
        )

    def create_entity(self, request: CreateEntityRequest) -> OperationResult[CreateEntityResult]:
        """Create a new entity."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        entity_id = request.entity_id or EntityId.generate()
        self.entities[entity_id.value] = {
            "entity_type": request.entity_type,
            "name": request.name,
            "attributes": request.attributes,
            "episode_id": request.episode_id.value if request.episode_id else None,
            "created_at": datetime.utcnow().isoformat()
        }
        return OperationResult.success(
            CreateEntityResult(entity_id=entity_id),
            "Entity created successfully"
        )

    def create_memory(self, request: CreateMemoryRequest) -> OperationResult[CreateMemoryResult]:
        """Create/store a memory."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        memory_id = request.memory_id or MemoryId.generate()
        self.memories[memory_id.value] = {
            "episode_id": request.episode_id.value,
            "entity_ids": [e.value for e in request.entity_ids],
            "content": request.content,
            "memory_kind": request.memory_kind,
            "confidence": request.confidence,
            "timestamp": request.timestamp.isoformat() if request.timestamp else datetime.utcnow().isoformat(),
            "provenance": request.provenance,
            "metadata": request.metadata,
            "created_at": datetime.utcnow().isoformat()
        }

        # Also add to history for get_history() functionality
        history_entry = {
            "operation": "create_memory",
            "memory_id": memory_id.value,
            "timestamp": datetime.utcnow().isoformat(),
            "content": request.content[:100]  # truncate for brevity
        }
        self.history.append(history_entry)

        return OperationResult.success(
            CreateMemoryResult(memory_id=memory_id),
            "Memory created successfully"
        )

    def update_memory(self, request: UpdateMemoryRequest) -> OperationResult[UpdateMemoryResult]:
        """Update an existing memory."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        if request.memory_id.value not in self.memories:
            return OperationResult.failure("Memory not found", {"error": "memory_not_found"})

        # Update the memory
        self.memories[request.memory_id.value]["content"] = request.content or self.memories[request.memory_id.value]["content"]
        if request.confidence is not None:
            self.memories[request.memory_id.value]["confidence"] = request.confidence
        if request.metadata is not None:
            self.memories[request.memory_id.value]["metadata"] = request.metadata

        # Add to history
        history_entry = {
            "operation": "update_memory",
            "memory_id": request.memory_id.value,
            "timestamp": datetime.utcnow().isoformat(),
            "action": "update",
            "content": request.content or "no content change"
        }
        self.history.append(history_entry)

        return OperationResult.success(
            UpdateMemoryResult(
                memory_id=request.memory_id,
                updated_fields=["content"] if request.content else ["metadata"]
            ),
            "Memory updated successfully"
        )

    def supersede_memory(self, request: SupersedeMemoryRequest) -> OperationResult[SupersedeMemoryResult]:
        """
        Historically replace/supersede a memory.

        CRITICAL: This is NOT physical deletion. The benchmark tests
        that superseded memories remain accessible via get_history().
        """
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        if request.old_memory_id.value not in self.memories:
            return OperationResult.failure("Old memory not found", {"error": "memory_not_found"})

        # Create the new memory
        new_memory_result = self.create_memory(request.new_memory)
        if not new_memory_result.is_success:
            return new_memory_result

        # Add to history
        history_entry = {
            "operation": "supersede_memory",
            "old_memory_id": request.old_memory_id.value,
            "new_memory_id": new_memory_result.data.memory_id.value,
            "timestamp": datetime.utcnow().isoformat(),
            "supersession_reason": request.supersession_reason,
            "metadata": request.metadata
        }
        self.history.append(history_entry)

        return OperationResult.success(
            SupersedeMemoryResult(
                old_memory_id=request.old_memory_id,
                new_memory_id=new_memory_result.data.memory_id
            ),
            "Memory superseded successfully"
        )

    def create_relationship(self, request: CreateRelationshipRequest) -> OperationResult[CreateRelationshipResult]:
        """Create a relationship between entities."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        relationship_id = request.relationship_id or RelationshipId.generate()
        self.relationships[relationship_id.value] = {
            "source_entity_id": request.source_entity_id.value,
            "target_entity_id": request.target_entity_id.value,
            "relationship_type": request.relationship_type,
            "confidence": request.confidence,
            "timestamp": request.timestamp.isoformat() if request.timestamp else datetime.utcnow().isoformat(),
            "metadata": request.metadata
        }

        # Add to history
        history_entry = {
            "operation": "create_relationship",
            "relationship_id": relationship_id.value,
            "timestamp": datetime.utcnow().isoformat(),
            "source": request.source_entity_id.value,
            "target": request.target_entity_id.value,
            "type": request.relationship_type
        }
        self.history.append(history_entry)

        return OperationResult.success(
            CreateRelationshipResult(relationship_id=relationship_id),
            "Relationship created successfully"
        )

    def query(self, request: QueryRequest) -> OperationResult[QueryResult]:
        """Higher-level query execution (returns answer + evidence)."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        # Simple stub implementation that returns a basic answer based on query text
        # This is just to prove the interface works - real implementations would be much more sophisticated
        answer = "stub answer based on query"
        answer_class = "DIRECT"  # Could be DERIVED, UNKNOWN, AMBIGUOUS, CONFLICTED based on query complexity

        # Generate mock evidence
        evidence = [
            {
                "memory_id": "mock_mem_1",
                "content": "stub evidence content",
                "memory_kind": "FACT",
                "relevance_score": 0.95
            },
            {
                "memory_id": "mock_mem_2",
                "content": "additional stub evidence",
                "memory_kind": "FACT",
                "relevance_score": 0.85
            }
        ]

        return OperationResult.success(
            QueryResult(
                answer=answer,
                answer_class=answer_class,
                evidence=evidence,
                confidence=0.9,
                metadata={
                    "query_length": len(request.query_text),
                    "options": request.options
                }
            ),
            "Query executed successfully"
        )

    def retrieve(self, request: RetrieveRequest) -> OperationResult[RetrieveResult]:
        """Retrieve memories/evidence (raw retrieval, no answer generation)."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        # Simple stub that returns memories matching basic criteria
        memories = []

        # Filter by episode if specified
        if request.episode_id:
            for mem in self.memories.values():
                if mem["episode_id"] == request.episode_id.value:
                    memories.append(mem)

        # Filter by entity if specified
        if request.entity_ids:
            entity_ids = {e.value for e in request.entity_ids}
            filtered_memories = []
            for mem in self.memories.values():
                if any(e.value in entity_ids for e in mem["entity_ids"]):
                    filtered_memories.append(mem)
            memories = filtered_memories

        # Filter by memory kind if specified
        if request.memory_kinds:
            filtered_memories = [
                mem for mem in memories if mem["memory_kind"] in request.memory_kinds
            ]

        return OperationResult.success(
            RetrieveResult(
                memories=memories,
                total_count=len(memories),
                metadata={
                    "filter_by_episode": request.episode_id is not None,
                    "filter_by_entity": request.entity_ids is not None,
                    "filter_by_memory_kind": request.memory_kinds is not None,
                    "query_length": len(request.query_text)
                }
            ),
            "Memories retrieved successfully"
        )

    def get_history(self, request: GetHistoryRequest) -> OperationResult[GetHistoryResult]:
        """Expose historical state."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        history = []

        # Filter by entity
        if request.entity_id:
            entity_id = request.entity_id.value
            filtered_history = [
                entry for entry in self.history
                if (entry.get("source_entity_id") == entity_id or
                    entry.get("target_entity_id") == entity_id or
                    entry.get("memory_id") == entity_id)
            ]
            history = filtered_history

        # Filter by memory
        if request.memory_id:
            memory_id = request.memory_id.value
            if memory_id in self.memories:
                # Find all history entries related to this memory
                for entry in self.history:
                    if entry.get("memory_id") == memory_id:
                        history.append(entry)

        # Filter by episode
        if request.episode_id:
            episode_id = request.episode_id.value
            filtered_history = [
                entry for entry in self.history
                if entry.get("episode_id") == episode_id
            ]
            history = filtered_history

        # Filter by point in time
        if request.point_in_time:
            # Simple temporal filter - only entries before the specified time
            history = [
                entry for entry in self.history
                if datetime.fromisoformat(entry["timestamp"]) <= request.point_in_time
            ]

        return OperationResult.success(
            GetHistoryResult(
                history=history,
                metadata={
                    "total_entries": len(history),
                    "filtered_by_entity": request.entity_id is not None,
                    "filtered_by_memory": request.memory_id is not None,
                    "filtered_by_episode": request.episode_id is not None,
                    "point_in_time_filtered": request.point_in_time is not None
                }
            ),
            "Historical state retrieved successfully"
        )

    def delete(self, request: DeleteRequest) -> OperationResult[DeleteResult]:
        """Delete a target."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        if request.target_type == "memory" and request.target_id.value in self.memories:
            # Mark as deleted in history
            history_entry = {
                "operation": "delete_memory",
                "memory_id": request.target_id.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": request.reason
            }
            self.history.append(history_entry)
            return OperationResult.success(
                DeleteResult(
                    deleted_id=request.target_id.value,
                    deleted_type=request.target_type,
                    hard_deleted=request.hard_delete
                ),
                "Memory marked for deletion"
            )

        if request.target_type == "entity" and request.target_id.value in self.entities:
            # Mark as deleted in history
            history_entry = {
                "operation": "delete_entity",
                "entity_id": request.target_id.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": request.reason
            }
            self.history.append(history_entry)
            return OperationResult.success(
                DeleteResult(
                    deleted_id=request.target_id.value,
                    deleted_type=request.target_type,
                    hard_deleted=request.hard_delete
                ),
                "Entity marked for deletion"
            )

        if request.target_type == "episode" and request.target_id.value in self.episodes:
            # Mark as deleted in history
            history_entry = {
                "operation": "delete_episode",
                "episode_id": request.target_id.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": request.reason
            }
            self.history.append(history_entry)
            return OperationResult.success(
                DeleteResult(
                    deleted_id=request.target_id.value,
                    deleted_type=request.target_type,
                    hard_deleted=request.hard_delete
                ),
                "Episode marked for deletion"
            )

        if request.target_type == "relationship" and request.target_id.value in self.relationships:
            # Mark as deleted in history
            history_entry = {
                "operation": "delete_relationship",
                "relationship_id": request.target_id.value,
                "timestamp": datetime.utcnow().isoformat(),
                "reason": request.reason
            }
            self.history.append(history_entry)
            return OperationResult.success(
                DeleteResult(
                    deleted_id=request.target_id.value,
                    deleted_type=request.target_type,
                    hard_deleted=request.hard_delete
                ),
                "Relationship marked for deletion"
            )

        return OperationResult.failure(
            "Target not found or unsupported type",
            {"error": "target_not_found"}
        )

    def backup(self, request: BackupRequest) -> OperationResult[BackupResult]:
        """Create a backup for persistence/recovery testing."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        backup_id = request.backup_id or BackupId.generate()
        # In a real implementation, this would serialize the candidate state
        # For the stub, we'll just record metadata
        backup_result = BackupResult(
            backup_id=backup_id,
            size_bytes=0,  # 0 for stub
            checksum="stub_checksum",
            timestamp=datetime.utcnow()
        )
        return OperationResult.success(
            backup_result,
            "Backup created successfully"
        )

    def restore(self, request: RestoreRequest) -> OperationResult[RestoreResult]:
        """Restore from a backup."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        # In a real implementation, this would restore state from backup
        # For the stub, we'll just mark as successful
        return OperationResult.success(
            RestoreResult(
                backup_id=request.backup_id,
                restored_count=0  # 0 in stub
            ),
            "Restore operation completed (stub)"
        )

    def rebuild_indexes(self, request: RebuildIndexesRequest) -> OperationResult[RebuildIndexesResult]:
        """Rebuild derived indexes."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        # In a real implementation, this would rebuild derived indexes
        # For the stub, we'll just return a success result
        return OperationResult.success(
            RebuildIndexesResult(
                indexes_rebuilt=["stub_index_1", "stub_index_2"] if request.index_types else [],
                duration_ms=0.0
            ),
            "Indexes rebuilt successfully"
        )

    def health_check(self, request: HealthCheckRequest) -> OperationResult[HealthCheckResult]:
        """Return operational status and integrity checks."""
        if not self.initialized:
            return OperationResult.failure("Candidate not initialized", {"error": "candidate_not_initialized"})

        # Simple health check - check if basic state is consistent
        healthy = self.initialized and len(self.memories) >= 0
        checks = {
            "initialized": self.initialized,
            "memory_count": len(self.memories),
            "history_length": len(self.history)
        }

        return OperationResult.success(
            HealthCheckResult(
                healthy=healthy,
                checks=checks,
                details={"status": "operational"},
                issues=[]
            ),
            "Health check passed"
        )

    def shutdown(self, request: ShutdownRequest) -> OperationResult[ShutdownResult]:
        """Clean lifecycle termination."""
        # In a real implementation, this would clean up resources
        # For the stub, we'll just mark as clean
        return OperationResult.success(
            ShutdownResult(
                clean=True,
                message="Candidate shut down successfully"
            ),
            "Candidate shut down successfully"
        )


# Export the public API
__all__ = [
    "CandidateAdapter",
    "CandidateContract",
    "OperationResult",
    "StatusCode",
    "InitializeRequest",
    "CreateEpisodeRequest",
    "CreateEntityRequest",
    "CreateMemoryRequest",
    "UpdateMemoryRequest",
    "SupersedeMemoryRequest",
    "CreateRelationshipRequest",
    "QueryRequest",
    "RetrieveRequest",
    "GetHistoryRequest",
    "DeleteRequest",
    "BackupRequest",
    "RestoreRequest",
    "RebuildIndexesRequest",
    "HealthCheckRequest",
    "ShutdownRequest"
]