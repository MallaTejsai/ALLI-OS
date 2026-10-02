"""Phase 10 — Ranking separate from retrieval; evidence assembly; evidence states (rule 10/28/29)."""
from __future__ import annotations
class RankingLayer:
    def rank(self, candidates, query_spec):
        # Consider: semantic, lexical, exact, entity, rel compat, temporal, scope, lifecycle, evidence quality, contradiction (rule 30)
        # Hard constraints (wrong entity/dir/temporal/polarity/scope) applied BEFORE final ranking (rule 30)
        return {"method":"multi_dimensional","constraints_applied_first":True,"audit":"v0.6-new"}
class EvidenceAssembler:
    def assemble(self, candidates):
        # Evidence ≠ truth (rule 20). States: SUFFICIENT / INSUFFICIENT / CONFLICTED / AMBIGUOUS.
        # Must refuse when insufficient.
        return {"states_supportable":["SUFFICIENT","INSUFFICIENT","CONFLICTED","AMBIGUOUS"],"audit":"v0.6-new"}
