import sys,os,tempfile,shutil
sys.path.insert(0,'src')
from memory.foundation.memory_engine import MemoryEngine
from memory.foundation.query_understanding import understand_query

td=tempfile.mkdtemp(); db=os.path.join(td,"t.sqlite")
eng=MemoryEngine(db_path=db,allow_qwen=False)
eng.remember("I am building ALLI.")
sq = understand_query("What project am I working on?")
ans = eng.recall("What project am I working on?")
print("INPUT: I am building ALLI.")
print("QUERY: What project am I working on?")
print("StructuredQuery: relation=%s object=%s object_hint=%s domain=%s entity=%s" % (sq.relation, sq.object, getattr(sq,'object_hint','N/A'), sq.domain, sq.entity))
print("STORED: relation=project, category=project, content='I am building ALLI.'")
print("ANSWER: %s" % str(ans)[:80])
print("FAIL? %s" % ("YES" if "alli" not in ans.lower() else "NO"))
eng.close(); shutil.rmtree(td)
