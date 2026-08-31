# ALLI Changelog

## v0.6 - Complete Memory Management

Added:
- Active/inactive memory status with historical snapshots on updates.
- `subject` field for stable preference/goal/skill matching.
- Preference fact types `like` / `dislike` with contradiction resolution.
- Active-only semantic retrieval and preference subject relevance gates.
- Clear missing answers for unknown personal facts (e.g. favorite food).
- Expanded automated v0.6 memory test suite (25+ cases + manual scenario bundle).
- Migration repair for misclassified learning goals (`learn` content → `learning_goal`).

Changed:
- Memory updates archive previous active values instead of silently overwriting history.
- Learning goals remain distinct from skills.
- Direct memory questions answered from verified active memories.
- Public runtime version banner set to ALLI v0.6.

## v0.5 - Semantic Memory Retrieval

Added:
- Modular memory package under `src/memory/`
  (MemoryStore, MemoryExtractor, SafeMemoryManager, EmbeddingStore,
  MemoryRetriever, QueryIntentDetector, grounded answers).
- Ollama embedding support via configurable `EMBEDDING_MODEL` (default: `all-minilm`).
- Semantic query-intent detection via prototype embeddings (no keyword question rules).
- Category-aware retrieval: intent filters by category/fact_type before ranking.
- Structured memories with `fact_type`, `value`, `embedding`, timestamps.
- Automatic migration of v0.4 memories (preserve IDs, backfill fields/embeddings).
- Deterministic name/nickname upsert, multi-goal coexistence, preference reversal.
- Separate MEMORY WRITE vs MEMORY READ vs general chat pipelines.
- Optional memory debug via `ALLI_MEMORY_DEBUG=1`.
- Atomic JSON writes and corrupt-file backups.
- Automated semantic memory tests.

Changed:
- Hard-coded / keyword personal Q&A removed.
- Keyword metadata boosts removed from retrieval.
- Learning goals (`learning_goal`) are distinct from skills.

## v0.4 - Smart Memory

Added:
- Recent conversation window: only the latest 12 user/assistant messages are sent to Qwen.
- Long-term memory recommendations with multiple ADD, UPDATE, DELETE, or NONE operations.
- Python-side validation before changing `long_term_memory.json`.
- Simple relevance retrieval using question and memory word overlap.
- Safe handling for missing, empty, invalid, and Markdown-fenced JSON responses.
- Configurable conversation consolidation using active-character and estimated-token thresholds.
- `conversation_archive.json` for summarized older conversation chunks.
- Numeric 1-to-10 memory importance, timestamps, and optional expiration fields.
- Expired-memory cleanup and invalid-JSON backup files.

Fixed:
- Unrelated stored interests, such as Flutter, are no longer automatically included in every prompt.

## v0.0

- Installed Python
- Installed Git
- Installed VS Code
- Installed Ollama
- Downloaded Qwen

---

## v0.1

Added:
- Python connected to Ollama
- Interactive Chat
- Infinite Loop
- Exit Command
- English System Prompt

Performance:
- Simple Question: 3 sec
- Explanation: 12 sec
- Programming: 15 sec

Known Limitations:
- No Memory
- No Voice
- No GUI
- No Browser Automation
