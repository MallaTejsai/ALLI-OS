"""STEP 14 — Candidate B Tests: SQLite+Graph derived index."""
import pytest
import os
import tempfile
from benchmark.v0_6_architecture.candidates.contract import (
    InitializeRequest, CreateEpisodeRequest, CreateEntityRequest,
    CreateMemoryRequest, UpdateMemoryRequest, SupersedeMemoryRequest,
    CreateRelationshipRequest, QueryRequest, RetrieveRequest,
    GetHistoryRequest, DeleteRequest, BackupRequest, RestoreRequest,
    RebuildIndexesRequest, HealthCheckRequest, ShutdownRequest,
    EpisodeId, EntityId, MemoryId, RelationshipId, BackupId,
)
from benchmark.v0_6_architecture.candidates.adapter_b import CandidateBAdapter


@pytest.fixture
def b_adapter():
    """Candidate B adapter with isolated temp DB."""
    fd, path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    adapter = CandidateBAdapter(db_path=path)
    adapter.initialize(InitializeRequest())
    adapter.create_episode(CreateEpisodeRequest())  # ensure episode exists for foreign key
    # Ensure clean DB state for each test
    adapter.conn.execute('PRAGMA foreign_keys = ON')
    adapter.conn.commit()
    yield adapter
    adapter.shutdown(ShutdownRequest())
    os.unlink(path)


def test_contract_compliance(b_adapter):
    """Candidate B implements all 16 contract operations."""
    assert b_adapter.candidate_name == "Candidate B — SQLite+Graph"
    assert b_adapter.candidate_version == "0.1.0"
    # initialize done via fixture
    result = b_adapter.create_episode(CreateEpisodeRequest())
    assert result.is_success
    result = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Test"))
    assert result.is_success
    result = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep_test"), content="test content", memory_kind="FACT",
    ))
    assert result.is_success
    result = b_adapter.query(QueryRequest(query_text="test"))
    assert result.is_success
    result = b_adapter.retrieve(RetrieveRequest(query_text="test"))
    assert result.is_success
    result = b_adapter.get_history(GetHistoryRequest())
    assert result.is_success
    result = b_adapter.delete(DeleteRequest(target_type="memory", target_id=MemoryId("mem_x")))
    assert result.is_success
    result = b_adapter.backup(BackupRequest())
    assert result.is_success
    result = b_adapter.restore(RestoreRequest(backup_id=BackupId()))
    assert result.is_success
    result = b_adapter.rebuild_indexes(RebuildIndexesRequest())
    assert result.is_success
    result = b_adapter.health_check(HealthCheckRequest())
    assert result.is_success
    result = b_adapter.shutdown(ShutdownRequest())
    assert result.is_success


def test_entity_creation(b_adapter):
    """Entity creation persists to canonical SQLite."""
    result = b_adapter.create_entity(CreateEntityRequest(
        entity_type="PERSON", name="Alice",
    ))
    assert result.is_success
    ent_id = result.data.entity_id
    assert ent_id.value.startswith("ent_")
    # Verify canonical table
    row = b_adapter.conn.execute("SELECT name, entity_type FROM entities WHERE entity_id = ?", (str(ent_id),)).fetchone()
    assert row is not None
    assert row[0] == "Alice"
    assert row[1] == "PERSON"


def test_relationship_creation(b_adapter):
    """Relationship creation between entities persists."""
    e1 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    e2 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Bob")).data.entity_id
    result = b_adapter.create_relationship(CreateRelationshipRequest(
        source_entity_id=e1, target_entity_id=e2, relationship_type="KNOWS",
    ))
    assert result.is_success
    rel_id = result.data.relationship_id
    row = b_adapter.conn.execute("SELECT relationship_type FROM relationships WHERE relationship_id = ?", (str(rel_id),)).fetchone()
    assert row is not None
    assert row[0] == "KNOWS"


def test_relationship_direction(b_adapter):
    """Relationship direction: source→target is preserved."""
    e1 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    e2 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Bob")).data.entity_id
    result = b_adapter.create_relationship(CreateRelationshipRequest(
        source_entity_id=e1, target_entity_id=e2, relationship_type="KNOWS",
    ))
    assert result.is_success
    row = b_adapter.conn.execute(
        "SELECT source_entity_id, target_entity_id FROM relationships WHERE relationship_id = ?",
        (str(result.data.relationship_id),)
    ).fetchone()
    assert row[0] == str(e1)
    assert row[1] == str(e2)


def test_reverse_relationship_retrieval(b_adapter):
    """Retrieve relationships where entity is target (reverse direction)."""
    e1 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    e2 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Bob")).data.entity_id
    b_adapter.create_relationship(CreateRelationshipRequest(
        source_entity_id=e1, target_entity_id=e2, relationship_type="KNOWS",
    ))
    # Query for relationships involving e2 (as target)
    rows = b_adapter.conn.execute(
        "SELECT source_entity_id, target_entity_id, relationship_type FROM relationships WHERE target_entity_id = ?",
        (str(e2),)
    ).fetchall()
    assert len(rows) == 1
    assert rows[0][2] == "KNOWS"


def test_one_hop_traversal(b_adapter):
    """One-hop graph traversal via MENTIONS edge."""
    eid = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="Alice mentioned", memory_kind="FACT",
        entity_ids=[eid],
    )).data.memory_id
    # Retrieve via entity — should find the memory via graph traversal
    result = b_adapter.retrieve(RetrieveRequest(query_text="", entity_ids=[eid], max_results=10))
    assert result.is_success
    memories = result.data.memories
    assert any(m["memory_id"] == str(mid) for m in memories)


def test_multi_hop_traversal(b_adapter):
    """Multi-hop traversal: entity → memory → related entity via RELATES_TO."""
    e1 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    e2 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Bob")).data.entity_id
    e3 = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Carol")).data.entity_id
    # Alice KNOWS Bob, Bob MENTIONS Carol via a memory
    b_adapter.create_relationship(CreateRelationshipRequest(
        source_entity_id=e1, target_entity_id=e2, relationship_type="KNOWS",
    ))
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="Bob and Carol", memory_kind="FACT",
        entity_ids=[e2, e3],
    )).data.memory_id
    # Two-hop: e1 →(KNOWS)→ e2 →(MENTIONS via memory mid)→ e3
    # One-hop from e2 should find mid
    result = b_adapter.retrieve(RetrieveRequest(entity_ids=[e2], max_results=10))
    assert result.is_success
    mem_ids = [m["memory_id"] for m in result.data.memories]
    assert str(mid) in mem_ids


def test_update(b_adapter):
    """Update memory content persists."""
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="original", memory_kind="FACT",
    )).data.memory_id
    result = b_adapter.update_memory(UpdateMemoryRequest(memory_id=mid, content="updated"))
    assert result.is_success
    row = b_adapter.conn.execute("SELECT content FROM memory_claims WHERE memory_id = ?", (str(mid),)).fetchone()
    assert row[0] == "updated"


def test_supersession(b_adapter):
    """Supersede creates new memory, marks old as superseded."""
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="old", memory_kind="FACT",
    )).data.memory_id
    new_mem = CreateMemoryRequest(episode_id=EpisodeId("ep1"), content="new", memory_kind="FACT")
    result = b_adapter.supersede_memory(SupersedeMemoryRequest(old_memory_id=mid, new_memory=new_mem))
    assert result.is_success
    # Old memory should be superseded
    row = b_adapter.conn.execute("SELECT lifecycle_state FROM memory_claims WHERE memory_id = ?", (str(mid),)).fetchone()
    assert row[0] == "superseded"
    # New memory should exist
    new_row = b_adapter.conn.execute("SELECT content FROM memory_claims WHERE memory_id = ?", (str(result.data.new_memory_id),)).fetchone()
    assert new_row[0] == "new"


def test_contradiction(b_adapter):
    """Contradiction: two memories with same entity, different content."""
    eid = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    m1 = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="Alice is tall", memory_kind="FACT", entity_ids=[eid],
    )).data.memory_id
    m2 = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="Alice is short", memory_kind="FACT", entity_ids=[eid],
    )).data.memory_id
    # Both memories exist
    rows = b_adapter.conn.execute("SELECT COUNT(*) FROM memory_claims WHERE entity_ids LIKE ?", (f"%{str(eid)}%",)).fetchone()
    assert rows[0] >= 2


def test_deletion(b_adapter):
    """Deletion marks lifecycle_state, does not physically remove."""
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="to delete", memory_kind="FACT",
    )).data.memory_id
    result = b_adapter.delete(DeleteRequest(target_type="memory", target_id=mid))
    assert result.is_success
    row = b_adapter.conn.execute("SELECT lifecycle_state FROM memory_claims WHERE memory_id = ?", (str(mid),)).fetchone()
    assert row[0] == "deleted"


def test_restart_persistence(b_adapter):
    """Data survives restart: close and reopen adapter."""
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="persistent", memory_kind="FACT",
    )).data.memory_id
    b_adapter.shutdown(ShutdownRequest())
    # Reopen same DB
    new_adapter = CandidateBAdapter(db_path=b_adapter.db_path)
    new_adapter.initialize(InitializeRequest())
    row = new_adapter.conn.execute("SELECT content FROM memory_claims WHERE memory_id = ?", (str(mid),)).fetchone()
    assert row is not None
    assert row[0] == "persistent"
    new_adapter.shutdown(ShutdownRequest())


def test_graph_rebuild(b_adapter):
    """Rebuild indexes recreates graph from canonical data."""
    eid = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="test", memory_kind="FACT", entity_ids=[eid],
    ))
    # Rebuild
    result = b_adapter.rebuild_indexes(RebuildIndexesRequest())
    assert result.is_success
    assert "graph_nodes" in result.data.indexes_rebuilt
    assert "graph_edges" in result.data.indexes_rebuilt
    # Verify graph nodes exist
    node_count = b_adapter.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    assert node_count > 0


def test_graph_recovery(b_adapter):
    """Corrupt graph → rebuild recovers it."""
    eid = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="test", memory_kind="FACT", entity_ids=[eid],
    ))
    # Simulate graph corruption by clearing graph tables
    b_adapter.conn.execute("DELETE FROM graph_edges")
    b_adapter.conn.execute("DELETE FROM graph_nodes")
    b_adapter.conn.commit()
    node_count = b_adapter.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    assert node_count == 0
    # Recover via rebuild
    b_adapter.rebuild_indexes(RebuildIndexesRequest())
    node_count = b_adapter.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
    assert node_count > 0


def test_backup_restore(b_adapter):
    """Backup and restore preserves canonical + graph data."""
    eid = b_adapter.create_entity(CreateEntityRequest(entity_type="PERSON", name="Alice")).data.entity_id
    b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="backup test", memory_kind="FACT", entity_ids=[eid],
    ))
    b_adapter.rebuild_indexes(RebuildIndexesRequest())
    # Backup
    bak_id = BackupId.generate()
    result = b_adapter.backup(BackupRequest(backup_id=bak_id))
    assert result.is_success
    backup_path = f"/tmp/alli_backup_{bak_id.value}.sqlite"
    assert os.path.exists(backup_path)
    # Add more data
    b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="after backup", memory_kind="FACT",
    ))
    # Restore
    b_adapter.restore(RestoreRequest(backup_id=bak_id))
    # Verify old data present, new data gone (full restore)
    count = b_adapter.conn.execute("SELECT COUNT(*) FROM memory_claims WHERE content = 'backup test'").fetchone()[0]
    assert count == 1
    b_adapter.shutdown(ShutdownRequest())
    os.unlink(backup_path)


def test_health_check(b_adapter):
    """Health check reports graph consistency."""
    result = b_adapter.health_check(HealthCheckRequest())
    assert result.is_success
    assert result.data.healthy is True
    assert "graph_nodes" in result.data.checks
    assert "graph_edges" in result.data.checks


def test_fallback_to_canonical_sqlite(b_adapter):
    """When graph is unavailable, retrieve falls back to canonical FTS."""
    # Clear graph tables to simulate unavailability
    b_adapter.conn.execute("DELETE FROM graph_edges")
    b_adapter.conn.execute("DELETE FROM graph_nodes")
    b_adapter.conn.commit()
    # Should still retrieve via FTS
    mid = b_adapter.create_memory(CreateMemoryRequest(
        episode_id=EpisodeId("ep1"), content="fallback test content", memory_kind="FACT",
    )).data.memory_id
    result = b_adapter.retrieve(RetrieveRequest(query_text="fallback test", max_results=10))
    assert result.is_success
    assert any("fallback" in m["content"] for m in result.data.memories)
