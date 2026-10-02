"""Real interaction runner (rules 18/21). Feeds user input through actual CLI interface; formation pipeline decides persistence."""
import subprocess, sys, time, os, sqlite3, json
DB = "../alli_v0_6_new.db"

def interact(turn_text: str, session_id: str = "s1"):
    """Real CLI interaction — NOT direct DB insertion (rule 26/21)."""
    proc = subprocess.run([sys.executable, "-m", "cli.cli", "remember", turn_text],
                          cwd="..", capture_output=True, text=True, timeout=10)
    # Memory formation pipeline handles candidate -> validated -> lifecycle -> canonical persist
    return {"turn":turn_text,"stdout":proc.stdout,"stderr":proc.stderr,"exit":proc.returncode,"session":session_id,"ts":time.time()}
