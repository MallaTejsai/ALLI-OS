"""ALLI v0.6 memory configuration — single place for tunable settings."""

import os

# Chat model (answer generation)
CHAT_MODEL = "qwen2.5:7b"

# Embedding model for semantic memory (must support ollama.embed)
EMBEDDING_MODEL = "all-minilm:latest"

# Retrieval
MEMORY_SIMILARITY_THRESHOLD = 0.30
FILTERED_SIMILARITY_THRESHOLD = 0.18
PREFERENCE_CONTENT_GATE = 0.32
PREFERENCE_SUBJECT_GATE = 0.25
MAX_RETRIEVED_MEMORIES = 8
SEMANTIC_WEIGHT = 0.80
CATEGORY_WEIGHT = 0.20
IMPORTANCE_WEIGHT = 0.0

# Query intent detection
INTENT_SIMILARITY_THRESHOLD = 0.30
INTENT_SCORE_MARGIN = 0.06

# Duplicate detection
DUPLICATE_SIMILARITY_THRESHOLD = 0.82

# Embedding text recipe version — bump forces re-embed on load
EMBEDDING_TEXT_VERSION = 4

# Optional debug (also ALLI_MEMORY_DEBUG=1)
MEMORY_DEBUG = False

# Conversation context
RECENT_MESSAGE_LIMIT = 12

# Paths
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_DIR = os.path.dirname(_SRC_DIR)
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
CONVERSATION_FILE = os.path.join(MEMORY_DIR, "conversation.json")
LONG_TERM_MEMORY_FILE = os.path.join(MEMORY_DIR, "long_term_memory.json")
BACKUP_DIR = os.path.join(MEMORY_DIR, "backups")

VALID_CATEGORIES = frozenset({
    "identity",
    "education",
    "goal",
    "ambition",
    "dream",
    "interest",
    "preference",
    "skill",
    "learning",
    "project",
    "routine",
    "task",
    "priority",
    "other",
})

VALID_IMPORTANCE = frozenset({"high", "medium", "low", "critical"})
VALID_STATUS = frozenset({"active", "inactive", "expired"})

IMPORTANCE_SCORES = {
    "critical": 1.0,
    "high": 0.85,
    "medium": 0.6,
    "low": 0.25,
}

UNIQUE_IDENTITY_FACT_TYPES = frozenset({"name", "nickname"})

# Priority tracking: explicit user priority signals
PRIORITY_KEYWORDS = frozenset({
    "main priority", "current priority", "top priority",
    "most important", "currently", "right now",
    "primary focus", "main focus",
})
