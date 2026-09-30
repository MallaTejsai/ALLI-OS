"""STEP 14D-OBSERVABILITY — Adapter result mapping (TASK 4).
Only repairs adapter→BenchmarkResult mapping for EXISTING Candidate B output.
Candidate B behavior unchanged; no new retrieval/query logic; no score fabrication.
"""
from benchmark.v0_6_architecture.candidates.adapter_b import CandidateBAdapter
from benchmark.v0_6_architecture.results.schema import BenchmarkResult, RetrievalResultItem, EvidenceReference, AnswerResult
from benchmark.v0_6_architecture.candidates.contract import RetrieveResult, QueryResult, OperationResult

class CandidateBObservableAdapter(CandidateBAdapter):
    """Wraps Candidate B adapter to expose real operation result to benchmark."""

    def observe_retrieve(self, request) -> list[RetrievalResultItem]:
        """Map real RetrieveResult to BenchmarkResult retrieval_results.
        Candidate B returns RetrieveResult(memories=[...]) ordered by FTS.
        NO scores exist in Candidate B result; rank assigned by observed sequence.
        score left None (not fabricated)."""
        op = self.retrieve(request)
        if not op or not op.is_success or not hasattr(op, "data") or op.data is None:
            return []
        result = op.data  # RetrieveResult
        memories = getattr(result, "memories", []) if hasattr(result, "memories") else (result if isinstance(result, list) else [])
        items = []
        for rank, mem in enumerate(memories, start=1):
            mem_id = mem.get("memory_id") if isinstance(mem, dict) else (getattr(mem, "memory_id", str(mem)) if hasattr(mem, "memory_id") else str(mem))
            # Channel: Candidate B uses FTS lexical + graph; report observed
            channel = "lexical"  # FTS5; graph traversal also used but not scored separately
            # Score: Candidate B genuinely has none — documented missing, not invented
            items.append(RetrievalResultItem(
                memory_id=str(mem_id) if mem_id else "",
                rank=rank,
                retrieval_channel=channel,
                score=None,  # NOT CALCULABLE — no relevance/similarity in B output
                score_type=None,
                metadata={"source":"candidate_b_retrieve", "ordered_by":"fts_sequence", "score_available":False},
                evidence_relationship_id=None,
            ))
        return items

    def observe_query(self, request) -> AnswerResult:
        """Map real QueryResult to BenchmarkResult answer_result."""
        op = self.query(request)
        if not op or not op.is_success or not hasattr(op, "data") or op.data is None:
            return AnswerResult(answer_text="", answer_class="UNKNOWN")
        qr = op.data  # QueryResult
        text = getattr(qr, "answer", "") if hasattr(qr, "answer") else (qr.get("answer") if isinstance(qr, dict) else "")
        cls = getattr(qr, "answer_class", "UNKNOWN") if hasattr(qr, "answer_class") else (qr.get("answer_class") if isinstance(qr, dict) else "UNKNOWN")
        conf = getattr(qr, "confidence", None) if hasattr(qr, "confidence") else (qr.get("confidence") if isinstance(qr, dict) else None)
        # Evidence: Candidate B provides evidence list (memory_id + content) — expose
        ev = getattr(qr, "evidence", []) if hasattr(qr, "evidence") else (qr.get("evidence") if isinstance(qr, dict) else [])
        ev_ids = [e.get("memory_id") for e in ev if isinstance(e, dict) and e.get("memory_id")] if isinstance(ev, list) else []
        return AnswerResult(
            answer_text=str(text) if text is not None else "",
            answer_class=str(cls) if cls else "UNKNOWN",
            confidence=float(conf) if conf is not None else None,
            evidence_references=ev_ids,
            claims=[], unsupported_claims=[], ambiguity_notes=[], conflict_notes=[],
        )

    def observe_evidence(self, request) -> list[EvidenceReference]:
        """Map query evidence to EvidenceReference."""
        op = self.query(request)
        if not op or not op.is_success or not hasattr(op, "data") or op.data is None:
            return []
        qr = op.data
        ev = getattr(qr, "evidence", []) if hasattr(qr, "evidence") else (qr.get("evidence") if isinstance(qr, dict) else [])
        refs = []
        for e in ev if isinstance(ev, list) else []:
            if isinstance(e, dict):
                refs.append(EvidenceReference(
                    evidence_id=str(e.get("memory_id", "")),
                    source_memory_id=str(e.get("memory_id", "")) if e.get("memory_id") else None,
                    evidence_type="MEMORY",
                    support_relationship="SUPPORTS",
                    relevance_observation=f"Candidate B evidence (content length={len(str(e.get('content','')))}); structured evaluation limited.",
                ))
        return refs
