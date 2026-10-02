=== LEVEL B — REAL ENVIRONMENT VALIDATION ===
FEATURE: Entity/Relation Engine (Phase 3) + Conversation Context (6) + Embeddings/FTS (7) + Retrieval Planner (9) + Ranking/Evidence (10) + General Knowledge Routing (12) + Ambiguity (13) + Control Plane (14) + Health/Repair (15) + Security (16) + CLI (17)
REQUIREMENT: direction-aware relationships; entity detected≠resolved; evidence states; data-only security; confirmation-required deletion; ambiguity policy; rebuild from canonical
IMPLEMENTED: alli_v0_6_new/src/model/entity.py, relationship.py, conversation_context.py; index/*.py; query/*.py; control/*.py; health/*.py; security/*.py; cli/cli.py; DB tables entities/relationships
AUTOMATED: PASS (import/structure verified; no old refs; DB version v0.6.0-new)
Real CLI: query direction-aware
---
FEATURE RECEIPT (rule 16):
  feature: entity_rel_engine + context + index + planner + ranking + general_knowledge + ambiguity + control + health + security + cli
  automated: PASS
  real_cli: PASS (exit 0; answer ALLI from rel direction; DB verified; stdout captured)
  persistence: NOT TESTED (restart + re-query required — not executed yet per rule 9)
  multi_turn: NOT TESTED (write/correct/query/historical sequence required — rule 8)
  unseen_queries: NOT TESTED (paraphrase/typo variants required — rule 13/14)
  adversarial/prompt_injection: PARTIAL (security model implemented; live stored-malicious-text test not executed — rule 43/16/57)
  database: PASS (canonical intact; entities/rel verified; derived rebuildable)
  final_status: REAL-WORLD PASS for direction-aware synthesis + entity + DB; NOT TESTED for persistence/multi-turn/unseen/adversarial — do NOT claim full feature PASS until those complete (rule 26/16/22).
---
NO AGGREGATE CHEATING (rule 23): Individual statuses reported separately; not summarized as "all passed".
NO PYTEST=PASS CLAIM (rule 6): Only real CLI evidence cited above counts for REAL-CLI column.
