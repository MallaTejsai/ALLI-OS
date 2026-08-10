import sys
sys.path.insert(0, "./src")
from memory.manager import SafeMemoryManager
from memory.store import MemoryStore
from memory.extractor import MemoryExtractor

def test(msg):
    store = MemoryStore("memory_test_bug_2")
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
    manager = SafeMemoryManager(store)
    return manager.process_user_message(msg)

print("res1:", test("I don't interested in web development and editing"))
print("res2:", test("I am not interested in web development and editing"))
print("res3:", test("I don't intrested in web developement and editing"))

