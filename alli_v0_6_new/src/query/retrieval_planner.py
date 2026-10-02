"""Phase 9 — Pipeline per spec §31 (no keyword→answer; rule 62)."""
from __future__ import annotations
class RetrievalPlanner:
    def plan(self, query_spec):
        # Query -> QuerySpec -> decomposition -> retrieval plan -> candidate retrieval
        # -> structural filtering -> ranking -> evidence assembly
        # Hybrid channels (rule 28): exact, lexical/FTS, semantic, entity, relationship, temporal, structured
        return {"channels":["entity","relationship","semantic","temporal","structured"],"filter":"structural_before_ranking","audit":"v0.6-new"}
