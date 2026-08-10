import sys
sys.path.insert(0, "./src")
from memory.manager import SafeMemoryManager
from memory.store import MemoryStore
import json
import os

store = MemoryStore("memory_test")
# Setup data
store.memories = [
    {"id": "mem_1", "category": "interest", "content": "User is interested in web development and editing", "status": "active"},
    {"id": "mem_2", "category": "interest", "content": "User is interested in flutter also", "status": "active"}
]

manager = SafeMemoryManager(store)
res = manager.process_user_message("I don't interested in web development and editing")
print(res)
print(store.memories)
