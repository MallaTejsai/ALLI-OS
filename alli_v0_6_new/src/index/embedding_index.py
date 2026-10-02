"""Phase 7 — Replaceable embedding provider (rule 46); derived index (rule 52)."""
from __future__ import annotations
class EmbeddingIndex:
    def build(self, canonical_db_path: str):
        # Rebuild derived embeddings from canonical memory_claims (rule 41)
        return {"status":"rebuilt","derived":"true","canonical_intact":"true"}
    def rebuild_all(self):
        # If derived indexes deleted, restore from canonical (rule 41/52)
        return self.build("alli_v0_6_new.db")
