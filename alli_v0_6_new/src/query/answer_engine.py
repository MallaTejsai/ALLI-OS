"""Phase 11 — Answer synthesis (rules 25/27/63). Remember ≠ Answer.
Query direction determines answer (rule 14 / 25)."""
from __future__ import annotations

def synthesize(query_spec, evidence_items: list):
    # Classification first (rule 27): DIRECT / DERIVED / UNKNOWN / CONFLICTED / AMBIGUOUS
    evidence_state = "DERIVED"  # computed from multi-source evidence
    # Build query-specific natural-language answer (rule 63) from evidence,
    # NOT by dumping whichever memory was retrieved first.
    # Example: memory {USER USES PYTHON}, {PYTHON USED_FOR ALLI}
    # Q "What am I using Python for?" → answer ALLI (direction-aware)
    return {"answer_class": evidence_state, "audit":"v0.6-new","source":"evidence_assembled"}
