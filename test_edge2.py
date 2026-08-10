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
            "subject": "web development",
            "value": "web development",
            "content": "User is interested in web development",
            "importance": "medium",
            "status": "active"
        },
        {
            "id": "mem_2",
            "category": "interest",
            "fact_type": "interest",
            "subject": "editing",
            "value": "editing",
            "content": "User is interested in editing",
            "importance": "medium",
            "status": "active"
        },
        {
            "id": "mem_3",
            "category": "interest",
            "fact_type": "interest",
            "subject": "flutter",
            "value": "flutter",
            "content": "User is interested in flutter",
            "importance": "medium",
            "status": "active"
        }
    ]
    return SafeMemoryManager(store), store

msgs = [
    "I am no longer interested in web development and editing"
]

for msg in msgs:
    mgr, store = get_manager()
    print(f"\n--- Testing: '{msg}' ---")
    res = mgr.process_user_message(msg)
    print("Result:", json.dumps(res, indent=2))
    print("Memories left:", [m["content"] for m in store.memories if m["status"] == "active"])

