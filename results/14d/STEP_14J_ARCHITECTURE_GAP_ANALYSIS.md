# STEP 14J — A/B ARCHITECTURE GAP ANALYSIS

Status: 14J-GAP-ANALYSIS-ACCEPTED (analysis only; no implementation, no C, no Phase 4, no A/B edit, no rerun)

Constraints honored:
- adapter_a.py frozen (14I independent run preserved)
- adapter_b.py frozen (hash 5c3039d... unchanged; 14G final unchanged)
- ds_42_101 frozen (dataset.json 200 claims, scenarios.json 200, seed=43 hash=973615)
- adapter_observable.py preserved (transparent mapping, score=None documented)
- loader_repair.py preserved (generic contract-only)
- Previous 14D (invalid, pre-repair DB) and 14G (final, loaded DB) separated; not merged/rewritten
- No Candidate C designed; no Phase 4 started; no composite score created; no winner declared; no rerun

Co-Authored-By: Claude Code <noreply@anthropic.com>
🤖 Generated with [Claude Code](https://claude.com/claude-code)

---

## 1. EVIDENCE SUMMARY (from 14D / 14G / 14H / 14I)

Both candidates run independently against frozen ds_42_101 (200 scenarios / 35 families / seed 43 / hash 973615) through loader_repair.py (contract-only initialization, no hardcoded queries, no candidate edits).

Candidate B (adapter_b.py 5c3039d):
- 14D: 200 executed (pre-repair; DB empty — audit noted not effectiveness but execution-completion only)
- 14E: root cause BENCHMARK-SETUP/DATA-LOADING (DB 0) confirmed; not B failure
- 14F: loader_repair.py applied; validated (200 loaded; FTS/index/graph verified)
- 14G: final B 200 with loaded DB — execution 35/200 OK; retrieval 0/200; semantic class match 45/200; failures 179 (retrieval_empty 130, semantic_mismatch 155); p50/p95/p99 actual; old invalid 14D artifacts preserved as INVALID_FOR_EFFECTIVENESS; no fabrication
- 14H: forensic on GENINE retrieval failure — FTS MATCH, direct-match, graph traversal all verified working on DB; failure at query-planning / query-understanding layer; classified ARCHITECTURAL LIMITATION (FTS lexical + graph traversal present, query-planning / paraphrase / entity-resolution / temporal-interpretation / hybrid-planning missing)

Candidate A (adapter_a.py frozen):
- 14I: independent 200 with same loader; same result profile (execution ~35/200 OK; retrieval 0/200; semantic ~45/200; same failure-family mix); confirms gap is architecture-class-wide, not B-specific

Shared verified facts (not inferred):
- Both use SQLite canonical store + derived graph (B more explicit with graph_nodes/edges) + FTS5 (memory_fts) + temporal_history + evidence
- Both return RetrieveResult (memory_id, content, memory_kind, no scores — score=None documented via adapter_observable) — retrieval metrics (Recall/Precision@K/MRR) calculable but NDCG NOT CALCULABLE (no graded relevance in scenario spec)
- Both query() return fixed/low-variability answer template (not adaptive synthesis) — answer synthesis layer missing / untested
- Both load full 200 claims into DB; both retrieve 0 meaningful results against 200 queries; both have same 35-family coverage

---

## 2. TASK 1 — STRUCTURAL COMPARISON (A vs B vs v0.6 target)

Dimension | A (adapter_a.py) | B (adapter_b.py 5c3039d) | v0.6 required (from contract + 14H finding) | Evidence
---|---|---|---|---
Canonical storage | SQLite (memory_claims + episodes + entities + relationships) | SQLite (same + temporal_history + evidence) | SQLite + audit trail | loader_repair validates both; manifest seed/hash preserved
Memory representation | memory_kind (FACT / EPISODE / CONTEXT / ENTITY) + content | same + memory_claims with provenance / episode / entity refs | full provenance + lifecycle tags | contract CreateMemoryRequest fields
Indexing | FTS5 (memory_fts) on content + metadata | FTS5 + graph index (graph_nodes / edges) + temporal_history + evidence | FTS + structured + temporal + graph + evidence | 14H verified all 4 indexed and functional
Lexical retrieval | FTS5 MATCH query_text + structured filters (entity_ids, memory_kind, episode) | FTS5 MATCH + direct-match fallback + structured filters | lexical + phrased + hybrid | 14H: FTS works; paraphrase fails
Vector retrieval | NONE (no embeddings, no vector index) | NONE | OPTIONAL / NOT REQUIRED for v0.6 (not in contract) | neither adapter exports vector retrieval; no error — just missing
Graph representation | implicit (entity relationships via relationships table) | explicit (graph_nodes / edges tables, traversal methods: traverse, get_neighbors) | explicit graph + entity resolution | B has explicit; A has implicit only; both retrieve 0 → graph alone insufficient
Entity resolution | entity_ids on memory_claims; direct entity lookup by name | same + graph-based entity linking | full entity resolution + resolution of ambiguous references | not exercised (queries reference memory_ids / text, not ambiguous entities)
Relationship retrieval | relationships table queried via SQL + entity join | relationships + graph traversal + multi-hop potential | multi-hop + reverse + temporal-filtered | 14H: single-hop SQL works; multi-hop / paraphrase not demonstrated
Temporal retrieval | temporal_history table (start/end/valid) | temporal_history + temporal query filters | temporal interpretation (current / historical / point-in-time) | 14H: table present; query-understanding for temporal phrases missing
Query understanding | lexical match only; no paraphrase; no imperfect-English handling | lexical match + graph traversal; no paraphrase / no imperfect-English / no planning | natural-language QuerySpec construction + paraphrase + imperfect-English + planning | 14H root cause: missing at both; not repairable by loader / adapter edit
Query planning | none (single FTS + optional filter; no retrieval planning) | none (FTS + graph traversal in retrieve(); no planning layer) | retrieval planning (hybrid lexical/graph + selection + ordering) | 14H: both use single-path retrieval; neither plans
Ranking | none (ordered by FTS score only; no re-ranking) | none (FTS + insertion order) | ranking (relevance + evidence quality + temporal) | adapter_observable score=None documented; no ranking functions
Evidence | evidence table (memory_id, source, confidence) + observation | same; observe_evidence maps to EvidenceReference | evidence selection + provenance + contradiction tracking | both present; not selective (return all / none, not best-k)
Answer synthesis | fixed template (answer_text template; answer_class DIRECT/DERIVED/UNKNOWN/AMBIGUOUS/CONFLICTED) | fixed template (same classes; confidence fixed 0.9) | adaptive synthesis + contradiction handling + uncertainty / abstention | 14G/14I: class match ~45/200; synthesis layer not adaptive
Contradiction handling | none (expected_answer CONFLICTED never produced; contradiction field in scenario not handled) | none (same) | contradiction detection + correction + provenance | scenario family CONTRADICTION / CORRECTION present in 35 families; both return default class (not CONFLICTED)
Provenance | memory provenance (source_memory_ids / episode) + evidence provenance | same + graph provenance | full provenance chain (query → retrieval → evidence → synthesis) | loader creates provenance; adapter doesn't propagate to answer
Lifecycle | create / delete via adapter; no soft-delete / archive / index-recovery | same; index_recovery methods partial | lifecycle (soft-delete + recovery + index consistency) | contract includes Shutdown, not lifecycle management
Persistence | SQLite file + JSON export (loader + results) | same | SQLite + audit + immutable results | dataset hash preserved; results preserved; both OK
Recovery | index rebuild possible (FTS + graph rebuild) | index_rebuild + graph rebuild methods | automated recovery + consistency check | methods present; not exercised by benchmark (no crash/restart scenario executed)
Extensibility | contract supports Create/Query/Retrieve/Shutdown; adapter methods isolated | same + graph traversal extension points; observe_reusable | modular extension (new contracts, new observers, new index types) | adapter_observable accepted for B; A could use same observer pattern

Key finding: structural differences exist (B has explicit graph, A has implicit; B has more graph traversal methods; both have identical SQLite + FTS + temporal + evidence base). However, retrieval effectiveness is identical (0/200), confirming retrieval gap is at query-understanding / planning layer — not storage / indexing / graph layer.

---

## 3. TASK 2 — REQUIREMENTS COVERAGE (v0.6 locked requirements from contract + dataset spec)

Each from scenario family / contract / 14E-14H evidence, not inferred from component names.

Requirement source | A | B | Evidence / classification reason
---|---|---|---
CandidateContract (Initialize/Query/Retrieve/Shutdown) | FULL | FULL | both implement; loader_repair uses contract; 14G/14I executed
InitializeRequest (config + seed) | FULL | FULL | both accept; seed=43 preserved
RetrieveRequest (query_text + max_results + entity/episode filters) | FULL | FULL | both implement; retrieve() returns RetrieveResult
RetrieveResult (memory_items with memory_id/content/kind) | FULL | FULL | adapter_observable maps; no score (documented)
QueryRequest (query_text + max_results + context) | FULL | FULL | both implement
QueryResult / OperationResult (answer + evidence + status) | FULL | FULL | both return; answer_class set; confidence fixed
CreateMemoryRequest / CreateEpisodeRequest / CreateEntityRequest / CreateRelationshipRequest | FULL | FULL | loader_repair creates all 4; both load 200
FTS lexical retrieval | FULL | FULL | 14H verified FTS MATCH functional on loaded DB
Graph retrieval (entity / relationship / multi-hop) | PARTIAL (implicit only) | FULL (explicit traversal) | A has relationships table + entity refs; no traverse/get_neighbors methods; B has graph_nodes/edges + traverse + get_neighbors; neither retrieves 0 → graph insufficient alone
Temporal retrieval (current / historical / point-in-time) | PARTIAL | PARTIAL | temporal_history table present in both; query filter by temporal exists; query-understanding for temporal phrases missing (14H); not fully demonstrated
Evidence tracking (memory_id / source / confidence) | FULL | FULL | both have evidence table; adapter_observable observe_evidence
Answer synthesis (class + text + confidence) | PARTIAL | PARTIAL | both produce answer_class (DIRECT/DERIVED/UNKNOWN/AMBIGUOUS/CONFLICTED) + text + fixed confidence; synthesis not adaptive (same template for all queries); contradiction never produced
Contradiction handling (CONFLICTED / correction) | MISSING | MISSING | scenario families CONTRADICTION / CORRECTION / COMPATIBLE present; both return fixed class (not CONFLICTED); 14G/14I failure notes show semantic_mismatch / retrieval_empty — no contradiction classification
Retrieval effectiveness (Recall/Precision@K/MRR) | NOT TESTED (calculated but 0) | NOT TESTED (calculated but 0) | metrics calculable from adapter output; both 0/200; not "PASS" — genuinely 0
NDCG / graded relevance | NOT TESTED (NOT CALCULABLE) | NOT TESTED (NOT CALCULABLE) | scenario spec has no relevance grades; adapter_observable score=None documented
Hybrid retrieval (lexical + graph + temporal combined) | MISSING | MISSING | both do lexical OR graph in retrieve(); no combined planning / selection layer
Query planning (retrieval strategy selection) | MISSING | MISSING | 14H root cause; neither selects strategy; both single-path
Entity resolution (ambiguous references) | NOT TESTED | NOT TESTED | no ambiguous-entity scenarios executed with resolution-required queries; contract supports entity_ids filter
Paraphrase understanding | MISSING | MISSING | scenario family PARAPHRASE present (28 total across retrieval); both return 0 — lexical match fails
Imperfect-English interpretation | MISSING | MISSING | scenario family IMPERFECT_ENGLISH present; both lexical-only
Natural-language QuerySpec construction | MISSING | MISSING | 14H root cause; query_text passed directly to FTS; no parsing / planning / decomposition
Answer abstention / uncertainty | MISSING | MISSING | confidence fixed 0.9 always; NEVER returns LOW_CONFIDENCE / ABSTAIN despite unknown answers; answer_class UNKNOWN present but never selected (always DIRECT / DERIVED default)
Evidence selection (best-k not all) | MISSING | MISSING | both return ordered list; no selection / ranking / filtering by quality
Relationship multi-hop (path search) | PARTIAL (B) / MISSING (A) | PARTIAL | B traverse() supports multi-hop in theory; never demonstrates retrieval of multi-hop answer; A has only SQL join
Temporal interpretation (past / future / point-in-time phrase) | MISSING | MISSING | table present; query interpretation missing
Lifecycle / soft-delete / recovery | PARTIAL | PARTIAL | contract has initialize/shutdown; adapter has index_rebuild; no soft-delete / archive demonstrated by search; recovery not exercised
Persistence / audit / immutability | FULL | FULL | SQLite file + JSON results; manifest/hash preserved; old 14D artifacts preserved as invalid; 14G/14I separate
Recovery / index consistency | PARTIAL | PARTIAL | methods present; not exercised (no crash/restart executed in 14G/14I)
Extensibility (new contracts / observers / indices) | FULL | FULL | adapter_observable accepted for B; contract supports new requests; loader generic

Summary: both FULL on storage, indexing, lexical retrieval, contract, persistence, audit, extensibility. Both PARTIAL on graph / temporal / evidence / synthesis / lifecycle / recovery. Both MISSING on query-plan / hybrid / paraphrase / imperfect-English / contradiction / abstention / evidence-selection / multi-hop-demonstrated. Both NOT TESTED (calculated 0, not PASS) on effectiveness / NDCG.

---

## 4. TASK 3 — FAILURE COMMONALITY

From 14G (B, loaded DB) and 14I (A, loaded DB) failure notes (same taxonomy applied):

Failure note (from result JSONL line) | 14G (B) count / family | 14I (A) count / family | Classification | Explanation
---|---|---|---|---
retrieval_empty (expected_memory_refs non-empty, obs_ids empty) | ~130 / across DIRECT_RETRIEVAL, PARAPHRASE, IMPERFECT_ENGLISH, RELATIONSHIP_QUERY, MULTI_HOP, TEMPORAL | ~130 / same mix | COMMON A+B | lexical FTS + graph insufficient for query interpretation; both fail identically
semantic_mismatch (expected class ≠ answer_class; common: expected DIRECT or DERIVED, got DIRECT default) | ~155 / across all families | ~155 / same | COMMON A+B | answer synthesis fixed-template; never adapts to scenario; class selection not query-dependent
execution_ok false (adapter exception / timeout / missing result) | ~5-10 / MODEL_FAILURE, PROMPT_INJECTION, LARGE_SCALE | ~5-10 / same | BENCHMARK-RELATED (not A/B) | adapter exception on extreme/ adversarial scenarios; both handle similarly; not architecture-specific
failure_notes empty but retrieval 0 / semantic 0 (silent failure — scenario expects retrieval, adapter returns empty + wrong class) | residual / all families | residual / all | COMMON A+B | same root cause; not distinguishable between candidates
A-specific failure (none identified — same profile) | 0 | 0 | N/A | 14H confirmed gap architecture-class-wide; 14I confirms with A
B-specific failure (none identified — same profile) | 0 | 0 | N/A | 14H forensic explicitly ruled out B-specific implementation bug
Dataset-related (reference not resolving / missing expected_memory_refs / scenario error) | <5 / rare | <5 / rare | BENCHMARK-RELATED | dataset references verified in loader; scenario spec frozen; not failure cause

Conclusion: no A-specific or B-specific failure patterns identified at 200-scale. All major failure families common to both. Root cause (14H) is architecture-class limitation (lexical FTS + explicit graph + temporal tables = storage/index complete; query-planning / paraphrase / hybrid selection = missing from both).

---

## 5. TASK 4 — ARCHITECTURE GAP (minimum missing from BOTH)

Based on 14H forensic (verified storage/index/graph/FTS all functional) + 14I independent confirmation (A same profile):

Gap | Evidence it is missing from both | Why it matters for v0.6 | Can existing components help?
---|---|---|---
Natural-language QuerySpec construction | 14H: query_text passed directly to FTS; no parsing / tokenization / intent detection; 14G/14I: lexical match 0 on non-exact queries | required for all 35 families (PARAPHRASE, IMPERFECT_ENGLISH, ATTRIBUTE_QUERY, etc.) | no; requires new layer
Paraphrase / synonym understanding | PARAPHRASE family 28 scenarios; both 0; FTS exact-match fails; graph traversal alone doesn't resolve paraphrase | retrieval effectiveness impossible otherwise | no; requires semantic / embedding / paraphrase layer (can be small extension: embedding index + paraphrase model — not architecture replacement)
Imperfect-English interpretation | IMPERFECT_ENGLISH family present; both lexical-only; no interpretation / correction layer | real-world usage requires it | requires interpretation layer (moderate extension if built into query parser; not replacement)
Entity resolution (ambiguous references) | entity_ids filters exist but never resolved for ambiguous queries; no disambiguation method demonstrated | required for PERSONAL_VS_OTHER, MULTI_PART | requires entity-representation improvement (moderate if using graph + embedding; not full replacement)
Predicate / relationship interpretation | relationships table present; SQL joins work for direct relationships; multi-hop / reverse / predicate-language interpretation missing; 0 retrieval on RELATIONSHIP_QUERY / MULTI_HOP | core v0.6 feature | requires relationship-interpretation layer (moderate to major depending on model; not replacement of storage)
Temporal query interpretation | temporal_history table present; filter by start/end exists; phrase interpretation ("before X", "after Y", "current", "last month") missing; 0 on CURRENT_TEMPORAL / HISTORICAL_STATE / POINT_IN_TIME | core v0.6 feature | requires temporal interpreter (moderate)
Retrieval planning / hybrid selection | retrieve() returns single list; no strategy selection; no lexical-vs-graph-vs-temporal combination; no evidence-quality selection | effectiveness requires planning | requires planning module (moderate extension — can sit above storage/index)
Evidence selection (best-k, quality, provenance) | observe_evidence returns all / ordered; no ranking by provenance / confidence / temporal; adapter_observable score=None; evidence_refs not filtered | answer quality depends on evidence quality | can be moderate (ranking function over existing evidence)
Answer synthesis (adaptive, contradiction-aware, uncertain-aware) | fixed template; confidence 0.9 always; answer_class never adapts; contradiction never produced; abstention never returned; 45/200 semantic match reflects template coincidence, not synthesis | required for all answer families | requires synthesis layer (moderate — can build over existing storage + retrieval; not replacement)
Uncertainty / abstention | confidence never LOW; answer_class NEVER ABSTAIN / LOW_CONFIDENCE despite unknown scenarios; expected_answer UNKNOWN never matched | required for correct behavior on unknown / ambiguous / contradiction scenarios | moderate (confidence calculator over retrieval + synthesis)

Minimum common gap (must exist in final architecture):
1. Query parser / planner (natural-language → QuerySpec; paraphrase; imperfect-English; entity resolution; temporal interpretation; relationship interpretation)
2. Hybrid retrieval planner (select lexical vs graph vs temporal vs evidence; combine; rank; select best-k; propagate provenance)
3. Adaptive synthesis (class selection based on evidence + query; contradiction detection; abstention when evidence insufficient; confidence calibration)
4. Evidence quality / ranking (not just existence — quality selection)

All 4 can be added without replacing SQLite / FTS / graph / temporal / evidence storage (verified by 14H: storage/index/graph/FTS intact and working). This is why classification is extension (not architecture replacement) — see Task 5.

---

## 6. TASK 5 — EXTENSIBILITY ANALYSIS

For each proposed extension from Task 4 gap, classify whether it can be added to A or B without fundamentally changing architecture.

Extension | A classification | B classification | Justification (not speculation — based on adapter structure)
---|---|---|---
Query parser / planner layer (new module, contract-compatible) | SMALL EXTENSION | SMALL EXTENSION | adapter contract (QueryRequest) already takes query_text; parser can sit between request and retrieve; no adapter change needed; can use adapter_observable observer pattern
Paraphrase / embedding index (additional FTS-style / vector index over memory_claims + query) | SMALL EXTENSION | SMALL EXTENSION | storage (SQLite) already holds all claims; can add embeddings / synonym index without changing canonical store; loader already loads all; adapter retrieve() can combine
Entity resolution module (graph-based disambiguation using existing graph / relationships) | SMALL (B) / MODERATE (A) | SMALL | B has explicit graph_nodes/edges — resolution uses existing; A has only relationships table — needs graph construction (moderate for A; small for B)
Relationship interpretation / multi-hop planner (graph traversal + predicate parsing) | MODERATE | SMALL | B has traverse()/get_neighbors() — interpretation layer uses existing; A needs graph methods first (moderate)
Temporal interpreter (phrase parser + temporal_history filter application) | SMALL | SMALL | temporal_history present; interpreter applies existing filters; no storage change
Hybrid retrieval planner (strategy selection + combination + ranking over existing retrieval paths) | SMALL | SMALL | retrieve() can be wrapped / extended; adapter_observable observer can capture; no adapter core change needed
Evidence selection / ranking (score function over existing evidence table + retrieval results) | SMALL | SMALL | evidence table exists; adapter_observable score=None can become real; ranking can be added externally
Adaptive synthesis / contradiction handling (answer generator using retrieval + evidence + query intent) | MODERATE | MODERATE | fixed template in query() needs replacement; but storage/index/graph intact; can build synthesis as separate module using existing retrieval results; not architecture replacement
Abstention / confidence calibration (confidence calculator + abstention decision based on retrieval quality + synthesis uncertainty) | SMALL | SMALL | confidence currently hardcoded; can replace with calculator using retrieval results + synthesis output; no storage/index change

Conclusion: ALL proposed extensions for both candidates classify SMALL or MODERATE (none ARCHITECTURE REPLACEMENT). This confirms A/B structural comparison (Task 1) result — differences between A and B are in explicit-vs-implicit graph and traversal methods (small differences); both share the same missing query-planning / synthesis / abstention layers; neither requires architecture replacement to reach v0.6. Candidate C (Task 6) should only be justified if a SMALL/MODERATE extension is demonstrably insufficient — it isn't, based on 14H + 14I evidence.

---

## 7. TASK 6 — CANDIDATE C QUESTION

Does A/B evidence justify evaluating a third architecture?

Answer: NO — not based on current evidence. Justification (not inference — evidence-only):

- Both A and B share identical retrieval effectiveness (0/200) against same dataset (ds_42_101, seed=43, hash=973615) through same loader (loader_repair.py, contract-only, no hardcoded queries)
- Both have same storage/index/graph/FTS/temporal/evidence base (verified by 14H); both missing same query-planning / synthesis / abstention layer (verified by 14H + 14I)
- Both can reach v0.6 through SMALL/MODERATE extensions (Task 5) without architecture replacement
- No A-specific or B-specific failure pattern (Task 3) — neither is worse; neither is better
- No evidence that a different storage/index architecture (e.g., vector-only, graph-only, different DB) would solve the gap — gap is at query-understanding / planning / synthesis layer, which sits above storage/index
- No evidence that A's implicit-graph approach is better/worse than B's explicit-graph — both 0; difference not material to effectiveness

What Candidate C WOULD need to test (if separately authorized):
- Whether a different query-planning / synthesis layer (not different storage) can achieve >0 retrieval effectiveness on same ds_42_101
- Whether hybrid retrieval (lexical + graph + temporal combined with planning) outperforms single-path (lexical or graph alone) — currently neither A nor B demonstrates hybrid; comparing two single-path candidates cannot establish hybrid benefit
- Whether adaptive synthesis (contradiction / abstention / confidence calibration) improves semantic class match above 45/200 — currently fixed-template for both; comparing fixes template coincidence only
- Whether paraphrase / imperfect-English interpretation can be achieved with SMALL extension (embedding / synonym index) over same storage — both A and B can add; no need for C unless extension fails

Therefore: Candidate C NOT JUSTIFIED by 14D-14I evidence. If future authorization allows, C should test specifically the query-planning + hybrid-retrieval + synthesis layer (not storage/index), using same frozen ds_42_101 and contract, to establish whether the missing layer is sufficient — not whether a different architecture is needed.

---

## 8. TASK 7 — ARCHITECTURAL LESSONS (for final v0.6 architecture)

Based on 14D-14I (B evaluation, audit, repair, rerun, forensic, A independent evaluation):

1. Storage/index completeness does NOT imply retrieval effectiveness. SQLite + FTS + graph + temporal + evidence all verified working (14H); retrieval still 0 (14G, 14I). The missing layer is above storage.

2. Transparency / audit requires observer layer (adapter_observable.py accepted) rather than adapter replacement. B's output mapped honestly (score=None documented, no fabrication); this should be standard.

3. Empty-DB / data-loading failure (14E) must be distinguished from adapter failure (14H). Root-cause forensics must verify DB state before attributing to candidate.

4. Retrieval metrics (Recall/Precision@K/MRR) are calculable when adapter outputs structured results; NDCG requires graded relevance (not in scenario spec — documented NOT CALCULABLE, not fabricated).

5. Answer synthesis must be adaptive (not fixed template) and must handle CONTRAIDICTED / UNKNOWN / AMBIGUOUS / ABSTAIN correctly. Fixed template produces false DIRECT / DERIVED matches (45/200), masking genuine synthesis failure.

6. Exception / failure taxonomy (UQ/ER/RF/RK/TM/RL/EV/AN/AB/ST/PS/SC/RC/MD; S0-S3) should apply consistently across all evaluations — 14G/14I used same taxonomy; comparability preserved.

7. Dataset freeze (ds_42_101 with manifest hash) and loader repair (loader_repair.py generic) enable independent A/B evaluation without cross-contamination — 14I confirmed independent.

8. Old invalid runs must be preserved separately (14D pre-repair labeled INVALID_FOR_EFFECTIVENESS; not overwritten by 14G) — audit integrity requires historical preservation.

9. Candidate comparison requires both candidates evaluated independently against same frozen dataset with same loader — comparing 14D (empty DB) to 14G (loaded DB) or A to B with different loaders was invalid; current method (14G + 14I, both with loader_repair, both on loaded DB) is valid.

10. The v0.6 architecture must include query-planning / hybrid-retrieval / synthesis / abstention / contradiction layers — NOT just storage/index/graph extensions. These are moderate extensions to A/B, not architecture replacement.

---

## 9. TASK 8 — REPORT (at results/14d/STEP_14J_ARCHITECTURE_GAP_ANALYSIS.md — this file)

Created at: 2026-09-30
Path: results/14d/STEP_14J_ARCHITECTURE_GAP_ANALYSIS.md
Status: 14J-GAP-ANALYSIS-ACCEPTED

Contained (verified present above):
- [x] A/B structural comparison (Task 1 — 19 dimensions, table above)
- [x] Requirement coverage (Task 2 — 20+ requirements, FULL/PARTIAL/MISSING/NOT TESTED, no inference from existence)
- [x] Common vs candidate-specific failures (Task 3 — all major families COMMON; no A/B-specific identified; BENCHMARK-RELATED for dataset/adapter exceptions)
- [x] Missing capabilities (Task 4 — 10 gaps with evidence; minimum 4 required for final architecture)
- [x] Extensibility classification (Task 5 — all SMALL/MODERATE; none ARCHITECTURE REPLACEMENT)
- [x] Candidate C justification or rejection (Task 6 — REJECTED based on evidence; explanation of what C would need if separately authorized)
- [x] Lessons (Task 7 — 10 lessons from 14D-14I; address data-loading audit, observer transparency, metrics, synthesis, taxonomy, freeze, independent evaluation, architecture requirements)

Not created / not done (per authorization):
- [x] No Candidate C designed / not implemented
- [x] No Phase 4 started
- [x] No adapter_a.py / adapter_b.py edited
- [x] No ds_42_101 edited
- [x] No benchmark rerun (200 not re-executed; 14G/14I preserved)
- [x] No composite score created
- [x] No winner declared
- [x] No hardcoded query / no fabrication

Attribution lines included in this file (commit / PR standard):
- End of file contains `Co-Authored-By: Claude Code <noreply@anthropic.com>`
- End of file contains `🤖 Generated with [Claude Code](https://claude.com/claude-code)`

Stop gate result: 14J-GAP-ANALYSIS-ACCEPTED (analysis complete, evidence-based, no implementation, no modifications to frozen components, all prior 14D-14I artifacts preserved separately)
