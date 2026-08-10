import sys
import json
sys.path.insert(0, "./src")
from memory.manager import SafeMemoryManager
from memory.store import MemoryStore
from memory.extractor import MemoryExtractor

def get_manager():
    store = MemoryStore("memory_test_alli")
    store.memories = [
        {
            "id": "mem_1",
            "category": "interest",
            "fact_type": "interest",
            "subject": "web development and editing",
            "value": "web development and editing",
            "content": "User is interested in web development and editing",
            "importance": "medium",
            "status": "active"
        },
        {
            "id": "mem_2",
            "category": "interest",
            "fact_type": "interest",
            "subject": "flutter also",
            "value": "flutter also",
            "content": "User is interested in flutter also",
            "importance": "medium",
            "status": "active"
        }
    ]
    return SafeMemoryManager(store), store

msgs = [
    "I don't like web development and editing",
    "I don't want web development and editing",
    "I'm not interested in web development and editing",
    "I no longer like web development and editing",
    "I stopped being interested in web development and editing",
    "I don't want to be interested in web development and editing"
]

for msg in msgs:
    mgr, store = get_manager()
    print(f"\n--- Testing: '{msg}' ---")
    res = mgr.process_user_message(msg)
    print("Result:", json.dumps(res, indent=2))
    print("Memories left:", [m["content"] for m in store.memories if m["status"] == "active"])

