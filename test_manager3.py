import sys
sys.path.insert(0, "./src")
from memory.manager import SafeMemoryManager
from memory.store import MemoryStore

store = MemoryStore("memory_test_3")
# Setup data
store.memories = [
    {"id": "mem_1", "category": "interest", "content": "User is interested in web development and editing", "status": "active", "fact_type": "interest"},
    {"id": "mem_2", "category": "interest", "content": "User is interested in flutter also", "status": "active", "fact_type": "interest"}
]

manager = SafeMemoryManager(store)
res = manager.process_user_message("I don't intrested in web developement and editing")
print("Result 1:", res)
print("Memories 1:", store.memories)

store.memories = [
    {"id": "mem_1", "category": "interest", "content": "User is interested in web development and editing", "status": "active", "fact_type": "interest"},
    {"id": "mem_2", "category": "interest", "content": "User is interested in flutter also", "status": "active", "fact_type": "interest"}
]
manager = SafeMemoryManager(store)
res = manager.process_user_message("I am no longer interested in web development and video editing")
print("Result 2:", res)
print("Memories 2:", store.memories)
