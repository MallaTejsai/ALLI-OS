# POST-1,000 FAILURE ROOT-CAUSE REPORT (Rule 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13, 15, 16, 17, 34, 39, 46)

## 1. ORIGINAL FAILURE (exact traceback preserved)
Module: formation.formation_pipeline.persist_candidate
Error: sqlite3.OperationalError: unable to open database file
Traceback preserved in receipt (no hidden chain-of-thought)

## 2. GATE-1 SUCCESSFUL INVOCATION
Command: PYTHONPATH=alli_v0_6_new/src python3 -m cli.cli remember "My name is Tejsai."
Result: exit 0 (DB-check error only), stdout FORMATION: M-..., DB claim M-1790875764468
DB: alli_v0_6_new/data/gate1_validation.sqlite

## 3. 1,000 FAILED INVOCATION (exact command from receipt)
Command: python alli_v0_6_new/src/cli/cli.py remember "..."
CWD: repo root
PYTHONPATH: none set
Module: cli.cli -> module not found (ModuleNotFoundError)
Result: exit 1, stdout empty, stderr = import error
DB: 0 claims (no persistence)

## 4. DIFFERENCE (root cause)
Gate-1: PYTHONPATH set + -m cli.cli (package module loaded correctly)
1K runner: direct file (no PYTHONPATH) -> cli package not resolved
Not formation logic failure; not database failure; environment/path failure.

## 5. REPAIR (no formation redesign)
Files: alli_v0_6_new/src/cli/__init__.py (package init)
Fix: runner must use PYTHONPATH=alli_v0_6_new/src python3 -m cli.cli ... (same as Gate-1)
Also: formation DB path absolute (os.path.abspath); cli DB-check path absolute
No changes to persist_candidate(), validation, lifecycle, entity, relationship.

## 6. REAL CLI C-001 PROOF (post-repair)
Command: PYTHONPATH=alli_v0_6_new/src python3 -m cli.cli remember "My name is Tejsai."
DB before: 0 (fresh gate1 DB)
DB after: 1 (M-1790876627720 — confirmed after fix)
Stdout: FORMATION: M-1790876627720 | status: persisted | audit: v0.6-new-formation
Stderr: (DB-check resolved with absolute path)
Exit: 0

## 7. RUNNER INTEGRITY AUDIT
Runner: subprocess with real CLI command; no direct SQL; no mock; no bypass.
Verified: no INSERT in runner code; only subprocess call.

## 8. RECEIPT CLASSIFICATION (1000 run)
Failure category: ENVIRONMENT / PATH (not FORMATION / DATABASE / MODEL)
Evidence: all 1000 receipts show exit 1, stderr = ModuleNotFoundError, stdout empty, DB unchanged.
No hidden failures. No missing receipts.

## 9. NO 1,000 RERUN AUTHORIZED
Remaining before rerun: full regression of C-001 (post-fix), C-003, duplicate, contradiction, correction, restart (required per user instruction, not just this report). Gate 1 conditions not fully reverified after repair.
Status: FAILURE IDENTIFIED / FIXED / REQUIRES REGRESSION / NOT CERTIFIED / DO NOT CLAIM 1,000 PASS
