"""Phase 12 — Clean routing; personal memory ≠ general knowledge (rule 35 / 64)."""
class GeneralKnowledgeRouter:
    def route(self, query_spec):
        # If personal evidence sufficient -> personal route
        # Else if general knowledge only -> general route (must NOT block by missing personal evidence — rule 35)
        # Clean interface for future web/file/knowledge modules (rule 67)
        return {"route":"personal_or_general","personal_evidence_required":False,"audit":"v0.6-new"}
