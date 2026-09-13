import sys,os,tempfile,shutil
sys.path.insert(0,"src")
from memory.foundation.memory_engine import MemoryEngine

def run(name, remember_text, recall_text, expected_substr):
    td=tempfile.mkdtemp(); db=os.path.join(td,"t.sqlite")
    try:
        eng=MemoryEngine(db_path=db, allow_qwen=False)
        if remember_text:
            eng.remember(remember_text)
        ans=eng.recall(recall_text) if recall_text else ""
        status="PASS" if expected_substr.lower() in ans.lower() else "FAIL"
        result={"test":name,"input":remember_text,"query":recall_text,"expected":expected_substr,"actual":ans[:80],"status":status}
        print(status + " " + name + ": " + ans[:60])
        return result
    finally:
        shutil.rmtree(td)

run("test_temporary_info", "I am debugging this right now.", "What am I debugging?", "debug")
run("test_project", "I am building ALLI.", "What project am I working on?", "alli")
run("test_preference", "I prefer learning through projects.", "How do I prefer to learn?", "project")
run("test_different_wording", "I am learning Flutter for mobile development.", "What mobile development technology am I interested in?", "flutter")
