import sys
sys.path.insert(0, "./src")
from memory.manager import SafeMemoryManager
from memory.store import MemoryStore

store = MemoryStore("memory_test_4")
# Setup data
store.memories = [
    {"id": "mem_1", "category": "interest", "content": "web development", "status": "active", "fact_type": "interest"},
    {"id": "mem_2", "category": "interest", "content": "video editing", "status": "active", "fact_type": "interest"},
    {"id": "mem_3", "category": "interest", "content": "Flutter", "status": "active", "fact_type": "interest"}
]

manager = SafeMemoryManager(store)
res = manager.process_user_message("I am no longer interested in web development and video editing")
print("Result 2:", res)
print("Memories 2:", store.memories)
