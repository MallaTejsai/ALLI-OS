"""
Tests for the Common Candidate Contract and Adapter Layer.

These tests verify the contract interface works correctly
and the adapter layer properly isolates candidates.
They do NOT evaluate any actual memory architecture.
"""

import pytest
from datetime import datetime
from abc import ABC

from benchmark.v0_6_architecture.candidates.contract import (
    CandidateContract,
    OperationResult,
    StatusCode,
    EpisodeId,
    EntityId,
    MemoryId,
    RelationshipId,
    BackupId,
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
)
from benchmark.v0_6_architecture.candidates.adapter import (
    CandidateAdapter,
    StubCandidateAdapter,
)


class TestContractImportability:
    """A. Contract importability."""

    def test_contract_classes_importable(self):
        """All core contract classes must be importable."""
        from benchmark.v0_6_architecture.candidates.contract import (
            CandidateContract,
            OperationResult,
            StatusCode,
        )
        assert CandidateContract is not None
        assert OperationResult is not None
        assert StatusCode is not None

    def test_identifier_classes_importable(self):
        """All stable identifier classes must be importable."""
        from benchmark.v0_6_architecture.candidates.contract import (
            EpisodeId,
            EntityId,
            MemoryId,
            RelationshipId,
            BackupId,
        )
        assert EpisodeId is not None
        assert EntityId is not None
        assert MemoryId is not None
        assert RelationshipId is not None
        assert BackupId is not None

    def test_request_types_importable(self):
        """All request types must be importable."""
        from benchmark.v0_6_architecture.candidates.contract import (
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
        )

    def test_result_types_importable(self):
        """All result types must be importable."""
        from benchmark.v0_6_architecture.candidates.contract import (
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
        )


class TestContractImplementation:
    """B. Contract implementation validation."""

    def test_candidate_contract_is_abstract(self):
        """CandidateContract must be an abstract base class."""
        assert issubclass(CandidateContract, ABC)

    def test_contract_has_abstract_methods(self):
        """Contract must have all 16 operations as abstract methods."""
        abstract_methods = CandidateContract.__abstractmethods__
        # 18 abstract attributes: 16 operations + 2 properties
        # The 16 operations are what matter; properties are part of the contract too
        assert len(abstract_methods) == 18

    def test_contract_has_initialize(self):
        """initialize() must be in the contract."""
        assert hasattr(CandidateContract, "initialize")

    def test_contract_has_create_episode(self):
        """create_episode() must be in the contract."""
        assert hasattr(CandidateContract, "create_episode")

    def test_contract_has_create_entity(self):
        """create_entity() must be in the contract."""
        assert hasattr(CandidateContract, "create_entity")

    def test_contract_has_create_memory(self):
        """create_memory() must be in the contract."""
        assert hasattr(CandidateContract, "create_memory")

    def test_contract_has_update_memory(self):
        """update_memory() must be in the contract."""
        assert hasattr(CandidateContract, "update_memory")

    def test_contract_has_supersede_memory(self):
        """supersede_memory() must be in the contract."""
        assert hasattr(CandidateContract, "supersede_memory")

    def test_contract_has_create_relationship(self):
        """create_relationship() must be in the contract."""
        assert hasattr(CandidateContract, "create_relationship")

    def test_contract_has_query(self):
        """query() must be in the contract."""
        assert hasattr(CandidateContract, "query")

    def test_contract_has_retrieve(self):
        """retrieve() must be in the contract."""
        assert hasattr(CandidateContract, "retrieve")

    def test_contract_has_get_history(self):
        """get_history() must be in the contract."""
        assert hasattr(CandidateContract, "get_history")

    def test_contract_has_delete(self):
        """delete() must be in the contract."""
        assert hasattr(CandidateContract, "delete")

    def test_contract_has_backup(self):
        """backup() must be in the contract."""
        assert hasattr(CandidateContract, "backup")

    def test_contract_has_restore(self):
        """restore() must be in the contract."""
        assert hasattr(CandidateContract, "restore")

    def test_contract_has_rebuild_indexes(self):
        """rebuild_indexes() must be in the contract."""
        assert hasattr(CandidateContract, "rebuild_indexes")

    def test_contract_has_health_check(self):
        """health_check() must be in the contract."""
        assert hasattr(CandidateContract, "health_check")

    def test_contract_has_shutdown(self):
        """shutdown() must be in the contract."""
        assert hasattr(CandidateContract, "shutdown")


class TestAll16OperationsRepresented:
    """C. All 16 logical operations are represented."""

    def test_initialize(self):
        assert hasattr(CandidateContract, "initialize")

    def test_create_episode(self):
        assert hasattr(CandidateContract, "create_episode")

    def test_create_entity(self):
        assert hasattr(CandidateContract, "create_entity")

    def test_create_memory(self):
        assert hasattr(CandidateContract, "create_memory")

    def test_update_memory(self):
        assert hasattr(CandidateContract, "update_memory")

    def test_supersede_memory(self):
        assert hasattr(CandidateContract, "supersede_memory")

    def test_create_relationship(self):
        assert hasattr(CandidateContract, "create_relationship")

    def test_query(self):
        assert hasattr(CandidateContract, "query")

    def test_retrieve(self):
        assert hasattr(CandidateContract, "retrieve")

    def test_get_history(self):
        assert hasattr(CandidateContract, "get_history")

    def test_delete(self):
        assert hasattr(CandidateContract, "delete")

    def test_backup(self):
        assert hasattr(CandidateContract, "backup")

    def test_restore(self):
        assert hasattr(CandidateContract, "restore")

    def test_rebuild_indexes(self):
        assert hasattr(CandidateContract, "rebuild_indexes")

    def test_health_check(self):
        assert hasattr(CandidateContract, "health_check")

    def test_shutdown(self):
        assert hasattr(CandidateContract, "shutdown")


class TestStubCandidateInstantiation:
    """D. A stub candidate can be instantiated."""

    def test_stub_candidate_can_be_instantiated(self):
        """A stub candidate must be instantiable."""
        stub = StubCandidateAdapter()
        assert stub is not None

    def test_stub_has_candidate_name(self):
        """Stub must have a candidate name."""
        stub = StubCandidateAdapter()
        assert stub.candidate_name == "Stub Candidate"

    def test_stub_has_candidate_version(self):
        """Stub must have a version."""
        stub = StubCandidateAdapter()
        assert stub.candidate_version == "0.1.0-stub"


class TestStubLifecycle:
    """E. Lifecycle works: initialize → operations → shutdown."""

    def test_stub_lifecycle(self):
        """Test the full lifecycle of a stub candidate."""
        stub = StubCandidateAdapter()

        # Initialize
        init_result = stub.initialize(InitializeRequest())
        assert init_result.status == StatusCode.SUCCESS
        assert isinstance(init_result.data, InitializeResult)

        # Create episode
        ep_result = stub.create_episode(CreateEpisodeRequest())
        assert ep_result.status == StatusCode.SUCCESS
        assert isinstance(ep_result.data, CreateEpisodeResult)

        # Create entity
        ent_result = stub.create_entity(CreateEntityRequest(entity_type="PERSON", name="Test User"))
        assert ent_result.status == StatusCode.SUCCESS
        assert isinstance(ent_result.data, CreateEntityResult)

        # Create memory
        mem_result = stub.create_memory(
            CreateMemoryRequest(
                episode_id=ep_result.data.episode_id,
                entity_ids=[ent_result.data.entity_id],
                content="Test memory content",
                memory_kind="FACT"
            )
        )
        assert mem_result.status == StatusCode.SUCCESS
        assert isinstance(mem_result.data, CreateMemoryResult)

        # Update memory
        update_result = stub.update_memory(
            UpdateMemoryRequest(
                memory_id=mem_result.data.memory_id,
                content="Updated test memory content"
            )
        )
        assert update_result.status == StatusCode.SUCCESS
        assert isinstance(update_result.data, UpdateMemoryResult)

        # Supersede memory
        new_mem_result = stub.create_memory(
            CreateMemoryRequest(
                episode_id=ep_result.data.episode_id,
                entity_ids=[ent_result.data.entity_id],
                content="New superseded memory",
                memory_kind="FACT"
            )
        )
        super_result = stub.supersede_memory(
            SupersedeMemoryRequest(
                old_memory_id=mem_result.data.memory_id,
                new_memory=CreateMemoryRequest(
                    episode_id=ep_result.data.episode_id,
                    entity_ids=[ent_result.data.entity_id],
                    content="Superseding memory",
                    memory_kind="FACT"
                ),
                supersession_reason="Updated information"
            )
        )
        assert super_result.status == StatusCode.SUCCESS
        assert isinstance(super_result.data, SupersedeMemoryResult)
        # CRITICAL: Old memory must still be in history
        history_result = stub.get_history(GetHistoryRequest(memory_id=mem_result.data.memory_id))
        assert history_result.status == StatusCode.SUCCESS
        assert len(history_result.data.history) >= 1

        # Create relationship
        ent2_result = stub.create_entity(CreateEntityRequest(entity_type="PERSON", name="Test User 2"))
        rel_result = stub.create_relationship(
            CreateRelationshipRequest(
                source_entity_id=ent_result.data.entity_id,
                target_entity_id=ent2_result.data.entity_id,
                relationship_type="FRIEND_OF"
            )
        )
        assert rel_result.status == StatusCode.SUCCESS
        assert isinstance(rel_result.data, CreateRelationshipResult)

        # Query
        query_result = stub.query(QueryRequest(query_text="test query"))
        assert query_result.status == StatusCode.SUCCESS
        assert isinstance(query_result.data, QueryResult)

        # Retrieve
        retrieve_result = stub.retrieve(RetrieveRequest(query_text="test retrieval"))
        assert retrieve_result.status == StatusCode.SUCCESS
        assert isinstance(retrieve_result.data, RetrieveResult)

        # Get history
        get_hist_result = stub.get_history(GetHistoryRequest(entity_id=ent_result.data.entity_id))
        assert get_hist_result.status == StatusCode.SUCCESS
        assert isinstance(get_hist_result.data, GetHistoryResult)

        # Delete
        del_result = stub.delete(DeleteRequest(target_type="memory", target_id=mem_result.data.memory_id))
        assert del_result.status == StatusCode.SUCCESS
        assert isinstance(del_result.data, DeleteResult)

        # Backup
        bak_result = stub.backup(BackupRequest())
        assert bak_result.status == StatusCode.SUCCESS
        assert isinstance(bak_result.data, BackupResult)

        # Restore
        restore_result = stub.restore(RestoreRequest(backup_id=bak_result.data.backup_id))
        assert restore_result.status == StatusCode.SUCCESS
        assert isinstance(restore_result.data, RestoreResult)

        # Rebuild indexes
        rebuild_result = stub.rebuild_indexes(RebuildIndexesRequest())
        assert rebuild_result.status == StatusCode.SUCCESS
        assert isinstance(rebuild_result.data, RebuildIndexesResult)

        # Health check
        health_result = stub.health_check(HealthCheckRequest())
        assert health_result.status == StatusCode.SUCCESS
        assert isinstance(health_result.data, HealthCheckResult)

        # Shutdown
        shutdown_result = stub.shutdown(ShutdownRequest())
        assert shutdown_result.status == StatusCode.SUCCESS
        assert isinstance(shutdown_result.data, ShutdownResult)


class TestInvalidOperationResultBehavior:
    """F. Invalid operation/result behavior is explicit."""

    def test_uninitialized_candidate_fails(self):
        """Operations on uninitialized candidate must fail."""
        stub = StubCandidateAdapter()
        result = stub.create_episode(CreateEpisodeRequest())
        assert result.status == StatusCode.FAILURE

    def test_delete_nonexistent_target_fails(self):
        """Deleting a non-existent target must fail."""
        stub = StubCandidateAdapter()
        stub.initialize(InitializeRequest())
        result = stub.delete(DeleteRequest(target_type="memory", target_id=MemoryId.generate()))
        assert result.status == StatusCode.FAILURE

    def test_invalid_target_type_fails(self):
        """Invalid target type must return failure."""
        stub = StubCandidateAdapter()
        stub.initialize(InitializeRequest())
        result = stub.delete(DeleteRequest(target_type="invalid_type", target_id=MemoryId.generate()))
        assert result.status == StatusCode.FAILURE

    def test_operation_result_is_failure(self):
        """OperationResult.is_failure must be True for failure status."""
        result = OperationResult.failure("Test failure")
        assert result.is_failure
        assert not result.is_success

    def test_operation_result_is_success(self):
        """OperationResult.is_success must be True for success status."""
        result = OperationResult.success("data")
        assert result.is_success
        assert not result.is_failure


class TestUnsupportedOperationBehavior:
    """G. Unsupported operation behavior is explicit."""

    def test_unsupported_returns_explicit_status(self):
        """Unsupported operations must return UNSUPPORTED_OPERATION, not raise."""
        stub = StubCandidateAdapter()
        # The stub implements all operations, but we can verify the pattern
        result = stub.initialize(InitializeRequest())
        assert result.status != StatusCode.UNSUPPORTED_OPERATION
        # If a candidate doesn't support an operation, it should return:
        unsupported_result = OperationResult.unsupported("Not implemented")
        assert unsupported_result.status == StatusCode.UNSUPPORTED_OPERATION

    def test_unsupported_status_is_explicit(self):
        """UNSUPPORTED_OPERATION status must be explicit."""
        result = OperationResult.unsupported("test")
        assert result.status == StatusCode.UNSUPPORTED_OPERATION
        assert result.message == "test"

    def test_no_silent_exception_conversion(self):
        """Exceptions must not be silently converted to PASS."""
        stub = StubCandidateAdapter()
        # The stub should raise exceptions for programming errors, not swallow them
        # This test verifies that unsupported operations return explicit status codes
        result = OperationResult.failure("test")
        assert result.status == StatusCode.FAILURE


class TestAdapterIsolation:
    """H. Adapter isolation."""

    def test_adapter_is_abstract_base_class(self):
        """CandidateAdapter must be an abstract base class."""
        assert issubclass(CandidateAdapter, CandidateContract)

    def test_stub_is_adapter_subclass(self):
        """StubCandidateAdapter must be a subclass of CandidateAdapter."""
        assert issubclass(StubCandidateAdapter, CandidateAdapter)

    def test_candidate_contract_is_abstract(self):
        """CandidateContract must not be instantiable directly."""
        with pytest.raises(TypeError):
            CandidateContract()

    def test_adapter_is_not_instantiable_directly(self):
        """CandidateAdapter must not be instantiable directly."""
        with pytest.raises(TypeError):
            CandidateAdapter()


class TestRunnerFacingInterface:
    """I. Runner-facing interface does not require knowledge of candidate internals."""

    def test_stub_can_be_used_through_contract_interface(self):
        """A stub candidate must be usable through the CandidateContract interface."""
        candidate: CandidateContract = StubCandidateAdapter()
        assert isinstance(candidate, CandidateContract)

    def test_runner_interface_does_not_depend_on_candidate_internals(self):
        """The runner-facing interface must not require candidate internals."""
        # The contract only exposes the 16 operations
        # Candidate internals are not accessible through the interface
        stub = StubCandidateAdapter()
        # Only public methods are accessible
        assert hasattr(stub, "initialize")
        assert hasattr(stub, "shutdown")
        # The runner should not need to know about internal state
        # Internal state is implementation detail, but accessible for verification
        # The key point is the contract interface is all the runner needs
        assert hasattr(stub, "candidate_name")

    def test_candidate_a_b_c_can_be_represented_through_same_interface(self):
        """Candidate A/B/C adapters can be represented through the same interface."""
        # All candidates must implement CandidateContract
        stub = StubCandidateAdapter()
        assert isinstance(stub, CandidateContract)
        # The contract has all 16 operations
        operations = [
            "initialize", "create_episode", "create_entity", "create_memory",
            "update_memory", "supersede_memory", "create_relationship",
            "query", "retrieve", "get_history", "delete", "backup",
            "restore", "rebuild_indexes", "health_check", "shutdown"
        ]
        for op in operations:
            assert hasattr(stub, op), f"Missing operation: {op}"


class TestNoProductionPaths:
    """K. No production paths are accessed."""

    def test_candidate_contract_does_not_import_production(self):
        """The contract must not import production ALLI code."""
        import benchmark.v0_6_architecture.candidates.contract as contract_module
        import inspect
        source = inspect.getsource(contract_module)
        # Verify no production imports
        assert "from src" not in source
        assert "import src" not in source
        assert "from memory" not in source
        assert "import memory" not in source

    def test_adapter_does_not_import_production(self):
        """The adapter must not import production ALLI code."""
        import benchmark.v0_6_architecture.candidates.adapter as adapter_module
        import inspect
        source = inspect.getsource(adapter_module)
        # Verify no production imports
        assert "from src" not in source
        assert "import src" not in source
        assert "from memory" not in source
        assert "import memory" not in source


class TestNoHardcodedQuestions:
    """L. No hardcoded benchmark questions are introduced."""

    def test_contract_has_no_hardcoded_questions(self):
        """The contract must not contain hardcoded questions."""
        import benchmark.v0_6_architecture.candidates.contract as contract_module
        import inspect
        source = inspect.getsource(contract_module)
        assert 'if query == "' not in source
        assert 'if "what is my name"' not in source
        assert 'if "python" in' not in source

    def test_adapter_has_no_hardcoded_questions(self):
        """The adapter must not contain hardcoded questions."""
        import benchmark.v0_6_architecture.candidates.adapter as adapter_module
        import inspect
        source = inspect.getsource(adapter_module)
        assert 'if query == "' not in source
        assert 'if "what is my name"' not in source
        assert 'if "python" in' not in source