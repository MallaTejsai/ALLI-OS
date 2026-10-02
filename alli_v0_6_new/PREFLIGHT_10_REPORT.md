# PREFLIGHT 10 REPORT — Post-Repair Validation
Status: FAIL (environment/invocation) — NOT formation failure
Rule 35: STOP — do not proceed to 1,000

A. DB path parity
- CLI DB: os.path.abspath(os.path.dirname(__file__) + "../../data/regression_validation.sqlite")
- Formation DB: os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/regression_validation.sqlite"))
- Verified: same absolute path
- DB: alli_v0_6_new/data/regression_validation.sqlite (clean)

B. Invocation comparison
- Gate 1 direct: PYTHONPATH=alli_v0_6_new/src python3 -m cli.cli ... → exit 0, DB claim
- Prefight runner: python alli_v0_6_new/src/cli/cli.py ... → exit 2, DB 0
- Difference: PYTHONPATH / module invocation / cwd
- Root cause: environment (not architecture)

C. Prefight results (10 interactions)
- All 10: exit 2
- DB after: 0 claims
- No synthetic insertion: CONFIRMED
- No direct SQL insert: CONFIRMED
- No old source change: CONFIRMED
- No hardcoded query fix: CONFIRMED

D. Failure classification
- Category: ENVIRONMENT / CLI INVOCATION
- Not FORMATION, not DATABASE, not MODEL
- Evidence preserved in PREFLIGHT_10.jsonl

E. Recommendation
- Fix runner to use PYTHONPATH + -m cli.cli (match Gate 1 exactly)
- Rerun preflight 10
- If pass: proceed to 1,000 authorization
- If fail: investigate deeper (possibly module shadowing)

F. Certification
- v0.6: NOT CERTIFIED
- Gate 1: PASS
- Prefight 10: FAIL (infrastructure, not semantic)
- 1,000: NOT EXECUTED
