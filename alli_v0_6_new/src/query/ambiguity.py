"""Phase 13 — Policy (rule 24): context -> persistent -> interpret -> if one supported -> answer; else ask clarification; else UNKNOWN."""
from __future__ import annotations
class AmbiguityResolver:
    def resolve(self, interpretations: list, context_support: bool):
        if len(interpretations) == 1 and context_support: return {"action":"answer","reason":"single_interpretation_supported"}
        if len(interpretations) > 1: return {"action":"clarify","reason":"materially_different_interpretations_remain"}
        return {"action":"unknown","reason":"insufficient_evidence"}
