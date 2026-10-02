"""Phase 14 — Correction / deletion confirmation / explorer / why (rules 36/37/38/39)."""
class MemoryExplorer:
    def inspect(self, target): return {"items":"list","audit":"v0.6-new"}
class DeletionController:
    def request_delete(self, mem_id, scope):
        return {"status":"confirmation_required","what":mem_id,"scope":scope,"reversible":False,"audit":"v0.6-new"}
class WhyDiagnostic:
    def explain(self, retrieval_result): return {"diagnostics":"structured","no_chain_of_thought":True,"audit":"v0.6-new"}
class CorrectionController:
    def correct(self, old_mem_id, new_claim):
        return {"action":"supersede","old_preserved":True,"new_created":True,"audit":"v0.6-new"}
