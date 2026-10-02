"""Real-environment bridge (rule 19 / 5 / 6). Launches real CLI, captures stdout/stderr/exit/DB."""
import subprocess, sqlite3, json, time, os, sys
DB = "../../alli_v0_6_new.db"  # relative to test dir; real DB

def run(cmd, cwd="../../"):
    proc = subprocess.run([sys.executable, "-m", "cli.cli"] + cmd.split(), cwd=cwd,
                          capture_output=True, text=True, timeout=30)
    return {"cmd":cmd,"stdout":proc.stdout,"stderr":proc.stderr,"exit":proc.returncode,"ts":time.time()}
