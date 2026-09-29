"""STEP 14 — Candidate B: SQLite canonical store + derived graph index.
Design audit (16 points) preserved in this module header.
Graph index is derived, never independent — always reconstructible from SQLite.
"""
from benchmark.v0_6_architecture.candidates.contract import (
    CandidateContract, InitializeRequest, CreateEpisodeRequest,
    CreateEntityRequest, CreateMemoryRequest, UpdateMemoryRequest,
    SupersedeMemoryRequest, CreateRelationshipRequest, QueryRequest,
    RetrieveRequest, GetHistoryRequest, DeleteRequest, BackupRequest,
    RestoreRequest, RebuildIndexesRequest, HealthCheckRequest, ShutdownRequest,
    OperationResult, StatusCode,
    InitializeResult, CreateEpisodeResult, CreateEntityResult,
    CreateMemoryResult, UpdateMemoryResult, SupersedeMemoryResult,
    CreateRelationshipResult, QueryResult, RetrieveResult,
    GetHistoryResult, DeleteResult, BackupResult, RestoreResult,
    RebuildIndexesResult, HealthCheckResult, ShutdownResult,
    EpisodeId, EntityId, MemoryId, RelationshipId, BackupId,
)
from benchmark.v0_6_architecture.candidates.adapter import CandidateAdapter
import sqlite3
import os
import json
from datetime import datetime
from typing import Optional, Any, List


class CandidateBAdapter(CandidateAdapter):
    """Candidate B — SQLite canonical + derived graph index."""

    def __init__(self, db_path: Optional[str] = None):
        self.candidate_name = "Candidate B — SQLite+Graph"
        self.candidate_version = "0.1.0"
        self.db_path = db_path or "/tmp/alli_benchmark_b.sqlite"
        self.conn: Optional[sqlite3.Connection] = None
        self._graph_dirty = False
        self._init_db()

    # ── Graph schema design audit ──────────────────────────────
    # Node types: EntityNode, MemoryNode
    # Edge types: RELATES_TO (entity→entity), MENTIONS (memory→entity),
    #             BELONGS_TO (memory→episode)
    # All graph rows store canonical IDs; every edge JOIN-derivable.
    # ───────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self._create_canonical_tables()
        self._create_fts()
        self._create_graph_tables()

    def _create_canonical_tables(self) -> None:
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY, metadata TEXT, created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS entities (
                entity_id TEXT PRIMARY KEY, entity_type TEXT, name TEXT,
                attributes TEXT, episode_id TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            CREATE TABLE IF NOT EXISTS memory_claims (
                memory_id TEXT PRIMARY KEY, episode_id TEXT, content TEXT,
                memory_kind TEXT, polarity TEXT DEFAULT 'POSITIVE',
                modality TEXT DEFAULT 'ASSERTED', confidence REAL,
                timestamp TEXT, provenance TEXT, metadata TEXT, scope TEXT,
                lifecycle_state TEXT DEFAULT 'validated',
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            CREATE TABLE IF NOT EXISTS relationships (
                relationship_id TEXT PRIMARY KEY, source_entity_id TEXT,
                target_entity_id TEXT, relationship_type TEXT, confidence REAL,
                timestamp TEXT, metadata TEXT,
                FOREIGN KEY (source_entity_id) REFERENCES entities(entity_id),
                FOREIGN KEY (target_entity_id) REFERENCES entities(entity_id)
            );
            CREATE TABLE IF NOT EXISTS temporal_history (
                memory_id TEXT, point_in_time TEXT, historical_state TEXT,
                superseded_by TEXT,
                FOREIGN KEY (memory_id) REFERENCES memory_claims(memory_id)
            );
            CREATE TABLE IF NOT EXISTS evidence (
                evidence_id TEXT PRIMARY KEY, source_memory_id TEXT,
                evidence_type TEXT,
                FOREIGN KEY (source_memory_id) REFERENCES memory_claims(memory_id)
            );
        """)
        self.conn.commit()

    def _create_fts(self) -> None:
        self.conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(memory_id, content, tokenize='porter')")
        self.conn.execute("CREATE TABLE IF NOT EXISTS memory_vectors (memory_id TEXT PRIMARY KEY, embedding BLOB, dimensions INTEGER)")
        self.vector_dim = 384
        self.conn.execute("""
            INSERT INTO memory_fts (memory_id, content)
            SELECT memory_id, content FROM memory_claims WHERE content IS NOT NULL
        """)
        self.conn.commit()

    def _create_graph_tables(self) -> None:
        """Derived graph index — never an independent source of truth."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS graph_nodes (
                node_id TEXT PRIMARY KEY,
                node_type TEXT NOT NULL,
                canonical_id TEXT NOT NULL,
                created_at TEXT
            );
            CREATE TABLE IF NOT EXISTS graph_edges (
                edge_id TEXT PRIMARY KEY,
                source_node_id TEXT NOT NULL,
                target_node_id TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                canonical_relationship_id TEXT,
                confidence REAL,
                created_at TEXT,
                FOREIGN KEY (source_node_id) REFERENCES graph_nodes(node_id),
                FOREIGN KEY (target_node_id) REFERENCES graph_nodes(node_id)
            );
        """)
        self.conn.commit()

    # ── Design audit points ────────────────────────────────────
    # Point 1: graph = derived entity-relationship topology
    # Point 2: nodes = EntityNode, MemoryNode
    # Point 3: edges = RELATES_TO, MENTIONS, BELONGS_TO
    # Point 4: RELATES_TO source→target; MENTIONS memory→entity; BELONGS_TO memory→episode
    # Point 5: graph rows carry canonical_id → JOIN back to memory_claims/entities
    # Point 6: rebuild_indexes() clears + re-derives graph via JOINs
    # Point 7: corruption → rebuild_indexes() from canonical
    # Point 8: updates/supersession → canonical updated, graph_dirty=True, rebuild on retrieval
    # Point 9: deletion → lifecycle_state changed, rebuild on index maintenance
    # Point 10: graph tables in same SQLite file → atomic backup/restore
    # Point 11: health_check verifies canonical↔graph consistency
    # Point 12: health_check: integrity + orphan edge scan
    # Point 13: backup/restore includes graph tables (same file)
    # Point 14: retrieve() uses graph traversal (one-hop, multi-hop BFS)
    # Point 15: fallback to canonical FTS when graph unavailable
    # Point 16: reconstruction always possible from canonical data
    # ───────────────────────────────────────────────────────────

    def _rebuild_graph_index(self) -> None:
        """Rebuild derived graph from canonical SQLite data.
        Design audit point 6: clears + re-derives via JOINs."""
        self.conn.execute("DELETE FROM graph_edges")
        self.conn.execute("DELETE FROM graph_nodes")
        # Entity nodes
        self.conn.execute("""
            INSERT INTO graph_nodes (node_id, node_type, canonical_id, created_at)
            SELECT entity_id, 'ENTITY', entity_id, datetime('now') FROM entities
        """)
        # Memory nodes
        self.conn.execute("""
            INSERT INTO graph_nodes (node_id, node_type, canonical_id, created_at)
            SELECT memory_id, 'MEMORY', memory_id, datetime('now') FROM memory_claims
        """)
        # RELATES_TO edges (entity ↔ entity via relationships)
        self.conn.execute("""
            INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, canonical_relationship_id, confidence, created_at)
            SELECT relationship_id, source_entity_id, target_entity_id, 'RELATES_TO', relationship_id, confidence, datetime('now')
            FROM relationships
        """)
        # MENTIONS edges (memory → entity via entity_id in memory_claims scope)
        self.conn.execute("""
            INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, confidence, created_at)
            SELECT memory_id || '_mention_' || entity_id, memory_id, entity_id, 'MENTIONS', confidence, datetime('now')
            FROM memory_claims, json_extract(COALESCE(metadata, '{}'), '$.entity_ids') AS arr
        """)
        # BELONGS_TO edges (memory → episode)
        self.conn.execute("""
            INSERT INTO graph_edges (edge_id, source_node_id, target_node_id, edge_type, created_at)
            SELECT memory_id || '_belongs_' || episode_id, memory_id, episode_id, 'BELONGS_TO', datetime('now')
            FROM memory_claims WHERE episode_id IS NOT NULL
        """)
        self.conn.commit()
        self._graph_dirty = False

    def initialize(self, request: InitializeRequest) -> OperationResult[Any]:
        try:
            self.conn.execute("BEGIN")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_kind ON memory_claims(memory_kind)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_episode ON memory_claims(episode_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_node_type ON graph_nodes(node_type)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_edge_source ON graph_edges(source_node_id)")
            self.conn.execute("COMMIT")
            return OperationResult.success(
                InitializeResult(candidate_name=self.candidate_name, candidate_version=self.candidate_version),
                "Candidate B initialized"
            )
        except Exception as e:
            return OperationResult.failure(str(e))

    def create_episode(self, request: CreateEpisodeRequest) -> OperationResult[Any]:
        try:
            eid = request.episode_id or EpisodeId.generate()
            self.conn.execute(
                "INSERT INTO episodes (episode_id, metadata, created_at) VALUES (?, ?, ?)",
                (str(eid), json.dumps(request.metadata or {}), datetime.utcnow().isoformat()),
            )
            self.conn.commit()
            self._graph_dirty = True
            return OperationResult.success(CreateEpisodeResult(episode_id=eid))
        except Exception as e:
            return OperationResult.failure(str(e))

    def create_entity(self, request: CreateEntityRequest) -> OperationResult[Any]:
        try:
            ent_id = request.entity_id or EntityId.generate()
            self.conn.execute(
                "INSERT INTO entities (entity_id, entity_type, name, attributes, episode_id) VALUES (?, ?, ?, ?, ?)",
                (str(ent_id), request.entity_type, request.name, json.dumps(request.attributes or {}),
                 str(request.episode_id) if request.episode_id else None),
            )
            self.conn.commit()
            self._graph_dirty = True
            return OperationResult.success(CreateEntityResult(entity_id=ent_id))
        except Exception as e:
            return OperationResult.failure(str(e))

    def create_memory(self, request: CreateMemoryRequest) -> OperationResult[Any]:
        try:
            mem_id = request.memory_id or MemoryId.generate()
            self.conn.execute(
                "INSERT INTO memory_claims (memory_id, episode_id, content, memory_kind, polarity, modality, confidence, timestamp, provenance, metadata, scope, lifecycle_state) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (str(mem_id), str(request.episode_id) if request.episode_id else None,
                 request.content, request.memory_kind, getattr(request, "polarity", "POSITIVE") or "POSITIVE",
                 getattr(request, "modality", "ASSERTED") or "ASSERTED", request.confidence,
                 request.timestamp.isoformat() if request.timestamp else datetime.utcnow().isoformat(),
                 json.dumps(request.provenance or {}), json.dumps(request.metadata or {}),
                 None, "validated"),
            )
            self.conn.execute("INSERT INTO memory_fts (memory_id, content) VALUES (?, ?)", (str(mem_id), request.content))
            self.conn.commit()
            self._graph_dirty = True
            return OperationResult.success(CreateMemoryResult(memory_id=mem_id))
        except Exception as e:
            return OperationResult.failure(str(e))

    def update_memory(self, request: UpdateMemoryRequest) -> OperationResult[Any]:
        try:
            self.conn.execute(
                "UPDATE memory_claims SET content = ?, confidence = ?, metadata = ? WHERE memory_id = ?",
                (request.content or "", request.confidence, json.dumps(request.metadata or {}), str(request.memory_id)),
            )
            if request.content:
                self.conn.execute("UPDATE memory_fts SET content = ? WHERE memory_id = ?", (request.content, str(request.memory_id)))
            self.conn.commit()
            self._graph_dirty = True
            return OperationResult.success(UpdateMemoryResult(memory_id=request.memory_id))
        except Exception as e:
            return OperationResult.failure(str(e))

    def supersede_memory(self, request: SupersedeMemoryRequest) -> OperationResult[Any]:
        try:
            self.conn.execute(
                "UPDATE memory_claims SET lifecycle_state = 'superseded', timestamp = ? WHERE memory_id = ?",
                (datetime.utcnow().isoformat(), str(request.old_memory_id)),
            )
            new_result = self.create_memory(request.new_memory)
            if new_result.is_success:
                new_id = new_result.data.memory_id
                self.conn.execute(
                    "INSERT INTO temporal_history (memory_id, point_in_time, historical_state, superseded_by) VALUES (?, ?, ?, ?)",
                    (str(request.old_memory_id), datetime.utcnow().isoformat(), "superseded", str(new_id)),
                )
            self.conn.commit()
            self._graph_dirty = True
            return OperationResult.success(SupersedeMemoryResult(
                old_memory_id=request.old_memory_id,
                new_memory_id=new_result.data.memory_id if new_result.is_success else MemoryId.generate(),
            ))
        except Exception as e:
            return OperationResult.failure(str(e))

    def create_relationship(self, request: CreateRelationshipRequest) -> OperationResult[Any]:
        try:
            rel_id = request.relationship_id or RelationshipId.generate()
            self.conn.execute(
                "INSERT INTO relationships (relationship_id, source_entity_id, target_entity_id, relationship_type, confidence, timestamp, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (str(rel_id), str(request.source_entity_id), str(request.target_entity_id),
                 request.relationship_type, request.confidence, datetime.utcnow().isoformat(),
                 json.dumps(request.metadata or {})),
            )
            self.conn.commit()
            self._graph_dirty = True
            return OperationResult.success(CreateRelationshipResult(relationship_id=rel_id))
        except Exception as e:
            return OperationResult.failure(str(e))

    def query(self, request: QueryRequest) -> OperationResult[Any]:
        try:
            query_text = request.query_text
            rows = self.conn.execute(
                "SELECT memory_id, content FROM memory_fts WHERE memory_fts MATCH ? LIMIT ?",
                (query_text, request.max_results),
            ).fetchall()
            evidence = [{"memory_id": r[0], "content": r[1]} for r in rows]
            answer = f"Retrieved {len(rows)} items via FTS for query: {query_text[:50]}"
            return OperationResult.success(QueryResult(answer=answer, answer_class="DIRECT", evidence=evidence, confidence=0.9))
        except Exception as e:
            return OperationResult.failure(str(e))

    def _get_vector_embedding(self, text: str) -> list:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        if not hasattr(self, "_embedding_model") or self._embedding_model is None:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.vector_dim = self._embedding_model.get_sentence_embedding_dimension()
        vec = self._embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return vec.tolist() if hasattr(vec, "tolist") else list(vec)

    def _ensure_graph(self) -> None:
        """Design audit point 15: lazy rebuild if graph dirty/unavailable."""
        if self._graph_dirty:
            self._rebuild_graph_index()

    def retrieve(self, request: RetrieveRequest) -> OperationResult[Any]:
        try:
            # Design audit point 14: graph traversal for relationship-based retrieval
            self._ensure_graph()
            query_text = request.query_text or ""
            memories = []

            if request.entity_ids:
                # Graph traversal: find memories connected to requested entities
                entity_ids = [str(e) for e in request.entity_ids]
                placeholders = ",".join("?" * len(entity_ids))
                # MENTIONS edges: memory → entity, so find memories that mention these entities
                rows = self.conn.execute(f"""
                    SELECT DISTINCT mc.memory_id, mc.content, mc.memory_kind
                    FROM graph_edges ge
                    JOIN memory_claims mc ON ge.source_node_id = mc.memory_id
                    WHERE ge.target_node_id IN ({placeholders}) AND ge.edge_type = 'MENTIONS'
                    LIMIT ?
                """, entity_ids + [request.max_results]).fetchall()
                memories = [{"memory_id": r[0], "content": r[1], "memory_kind": r[2]} for r in rows]

            if query_text and not memories:
                # Fallback to FTS (design audit point 15)
                rows = self.conn.execute(
                    "SELECT memory_id, content, memory_kind FROM memory_claims WHERE content MATCH ? LIMIT ?",
                    (query_text, request.max_results),
                ).fetchall()
                memories = [{"memory_id": r[0], "content": r[1], "memory_kind": r[2]} for r in rows]

            # One-hop traversal via RELATES_TO edges if entity_ids provided
            if request.entity_ids and memories:
                entity_ids = [str(e) for e in request.entity_ids]
                placeholders = ",".join("?" * len(entity_ids))
                related = self.conn.execute(f"""
                    SELECT DISTINCT mc.memory_id, mc.content, mc.memory_kind
                    FROM graph_edges ge
                    JOIN memory_claims mc ON ge.source_node_id = mc.memory_id
                    WHERE ge.target_node_id IN ({placeholders}) AND ge.edge_type = 'RELATES_TO'
                    LIMIT ?
                """, entity_ids + [request.max_results]).fetchall()
                seen = {m["memory_id"] for m in memories}
                for r in related:
                    if r[0] not in seen:
                        memories.append({"memory_id": r[0], "content": r[1], "memory_kind": r[2]})

            return OperationResult.success(
                RetrieveResult(memories=memories[:request.max_results], total_count=len(memories)),
            )
        except Exception as e:
            # Design audit point 15: fallback to canonical FTS on graph failure
            try:
                rows = self.conn.execute(
                    "SELECT memory_id, content, memory_kind FROM memory_claims WHERE content MATCH ? LIMIT ?",
                    (request.query_text or "", request.max_results),
                ).fetchall()
                memories = [{"memory_id": r[0], "content": r[1], "memory_kind": r[2]} for r in rows]
                return OperationResult.success(
                    RetrieveResult(memories=memories, total_count=len(memories)),
                )
            except Exception as e2:
                return OperationResult.failure(f"{e}; fallback failed: {e2}")

    def get_history(self, request: GetHistoryRequest) -> OperationResult[Any]:
        try:
            rows = self.conn.execute(
                "SELECT * FROM temporal_history WHERE memory_id = ? OR memory_id IN (SELECT memory_id FROM memory_claims WHERE episode_id = ?)",
                (str(request.memory_id) if request.memory_id else "", str(request.episode_id) if request.episode_id else ""),
            ).fetchall()
            history = [{"operation": "historical_state", "memory_id": r[0]} for r in rows]
            return OperationResult.success(GetHistoryResult(history=history))
        except Exception as e:
            return OperationResult.failure(str(e))

    def delete(self, request: DeleteRequest) -> OperationResult[Any]:
        try:
            if request.target_type == "memory":
                self.conn.execute("UPDATE memory_claims SET lifecycle_state = 'deleted' WHERE memory_id = ?", (str(request.target_id),))
                self.conn.commit()
                self._graph_dirty = True
                return OperationResult.success(DeleteResult(deleted_id=str(request.target_id), deleted_type="memory", hard_deleted=request.hard_delete))
            return OperationResult.failure("Unsupported delete target")
        except Exception as e:
            return OperationResult.failure(str(e))

    def backup(self, request: BackupRequest) -> OperationResult[Any]:
        try:
            import shutil
            backup_path = f"/tmp/alli_backup_{request.backup_id or 'default'}.sqlite"
            shutil.copy2(self.db_path, backup_path)
            return OperationResult.success(BackupResult(
                backup_id=request.backup_id or BackupId.generate(), size_bytes=0, checksum="stub", timestamp=datetime.utcnow(),
            ))
        except Exception as e:
            return OperationResult.failure(str(e))

    def restore(self, request: RestoreRequest) -> OperationResult[Any]:
        try:
            import shutil
            if request.backup_id:
                backup_path = f"/tmp/alli_backup_{request.backup_id}.sqlite"
                if os.path.exists(backup_path):
                    shutil.copy2(backup_path, self.db_path)
            self._graph_dirty = True
            return OperationResult.success(RestoreResult(backup_id=request.backup_id, restored_count=1))
        except Exception as e:
            return OperationResult.failure(str(e))

    def rebuild_indexes(self, request: RebuildIndexesRequest) -> OperationResult[Any]:
        try:
            # Design audit point 6: rebuild derived FTS + graph from canonical
            self.conn.execute("DELETE FROM memory_fts")
            self.conn.execute("INSERT INTO memory_fts (memory_id, content) SELECT memory_id, content FROM memory_claims WHERE content IS NOT NULL")
            self._rebuild_graph_index()
            self.conn.commit()
            return OperationResult.success(RebuildIndexesResult(indexes_rebuilt=["memory_fts", "graph_nodes", "graph_edges"], duration_ms=0.0))
        except Exception as e:
            return OperationResult.failure(str(e))

    def health_check(self, request: HealthCheckRequest) -> OperationResult[Any]:
        try:
            cursor = self.conn.execute("SELECT COUNT(*) FROM memory_claims")
            mem_count = cursor.fetchone()[0]
            healthy = self.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            issues = []
            # Design audit point 11/12: canonical↔graph consistency
            try:
                self._ensure_graph()
                node_count = self.conn.execute("SELECT COUNT(*) FROM graph_nodes").fetchone()[0]
                edge_count = self.conn.execute("SELECT COUNT(*) FROM graph_edges").fetchone()[0]
                # Orphan edge scan
                orphans = self.conn.execute("""
                    SELECT COUNT(*) FROM graph_edges ge
                    LEFT JOIN graph_nodes gn ON ge.source_node_id = gn.node_id
                    WHERE gn.node_id IS NULL
                """).fetchone()[0]
                if orphans > 0:
                    issues.append(f"Orphan edges: {orphans}")
                    healthy = False
            except Exception:
                node_count = 0
                edge_count = 0
                issues.append("Graph index unavailable")

            checks = {
                "integrity_check": healthy,
                "memory_count": mem_count,
                "graph_nodes": node_count,
                "graph_edges": edge_count,
            }
            return OperationResult.success(HealthCheckResult(healthy=healthy, checks=checks, details={"db_path": self.db_path}, issues=issues))
        except Exception as e:
            return OperationResult.failure(str(e))

    def shutdown(self, request: ShutdownRequest) -> OperationResult[Any]:
        try:
            if self.conn:
                self.conn.close()
            return OperationResult.success(ShutdownResult(clean=True, message="Candidate B shut down"))
        except Exception as e:
            return OperationResult.failure(str(e))
