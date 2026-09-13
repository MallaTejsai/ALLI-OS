import sys,os,tempfile,shutil
sys.path.insert(0,"src")
from memory.foundation.memory_engine import MemoryEngine
for label,q,mem in [
    ("temp_info","What am I debugging?","I am debugging this right now."),
    ("restart_color","What is my favorite color?","My favorite color is blue."),
    ("goal","What is my goal?","My goal is AI engineer."),
    ("project","What project am I working on?","I am building ALLI."),
    ("wording","What mobile development technology am I interested in?","I am learning Flutter for mobile development.")
]:
    td=tempfile.mkdtemp(); db=os.path.join(td,"t.sqlite")
    try:
        eng=MemoryEngine(db_path=db,allow_qwen=False)
        eng.remember(mem)
        ans=eng.recall(q)
        status="PASS" if (label=="goal" and "goal" in ans.lower()) or (label=="project" and "alli" in ans.lower()) or (label=="temp_info" and "debug" in ans.lower()) or (label=="restart_color" and "blue" in ans.lower()) or (label=="wording" and "flutter" in ans.lower()) else "FAIL"
        print(f"  {status} {label}: \"{ans[:50]}\"")
        eng.close()
    finally: shutil.rmtree(td)
