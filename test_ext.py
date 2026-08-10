import sys
sys.path.insert(0, "./src")
from memory.extractor import MemoryExtractor, extract_interest_and_learning_deletes

ext = MemoryExtractor()
msg = "I don't interested in web development and editing"
print("extract:", ext.extract(msg))
print("interest_deletes:", extract_interest_and_learning_deletes(msg))
