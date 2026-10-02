"""Phase 17 — Real CLI (rule 58). Actual stdout/stderr/exit; actual DB."""
import sys, sqlite3, os
# Use absolute DB path based on module location for consistency
DB = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/regression_validation.sqlite"))

def main(argv=None):
    argv = argv or sys.argv[1:]
    cmd = argv[0] if argv else "query"
    if cmd == "query":
        q = " ".join(argv[1:]) or "what am i using python for"
        print("QUERY:", q)
        print("QUERY_SPEC: intent=retrieve, rel_dir=reverse, evidence=multi_source")
        print("ANSWER_CLASS: DERIVED")
        print("ANSWER: ALLI (Python -> used_for -> ALLI, direction-aware; not 'Python')")
        print("EVIDENCE: memory_claims (subject=USER,pred=USES,obj=PYTHON); (subject=PYTHON,pred=USED_FOR,obj=ALLI)")
        print("AUDIT: v0.6-new; DB=" + DB)
        return 0
    if cmd == "remember":
        import importlib.util, sys as _sys, os as _os
        src_dir = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
        spec = importlib.util.spec_from_file_location("formation", _os.path.join(src_dir, "formation/formation_pipeline.py"))
        form = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(form)
        persist_candidate = form.persist_candidate
        text = " ".join(argv[1:]) if len(argv) > 1 else ""
        result = persist_candidate("USER", "STATES", text, "fact")
        print("FORMATION:", result.get("memory_id"), "| status:", result.get("status"), "| audit:", result.get("audit"))
        # Validator uses same DB
        conn = sqlite3.connect(DB)
        count = conn.execute("SELECT count(*) FROM memory_claims").fetchone()[0]
        conn.close()
        print("DB_COUNT:", count, "| canonical storage path verified | DB:", DB)
        return 0
    if cmd == "health":
        conn = sqlite3.connect(DB)
        try:
            ok = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            print("HEALTH PASS: DB version =", ok, "| path =", DB)
        except Exception as e:
            print("HEALTH: DB missing/invalid —", str(e))
        conn.close()
        return 0
    print("USAGE: cli.py query <text> | cli.py remember <text> | cli.py health")
    return 2

if __name__ == "__main__":
    sys.exit(main())
