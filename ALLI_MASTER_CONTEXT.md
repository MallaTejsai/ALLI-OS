# ALLI MASTER CONTEXT

## Purpose
Permanent source of truth for ALLI-OS development and the version-based ALLI content series.

## Project
- Assistant: ALLI
- Project: ALLI-OS
- Stack: Python + Ollama + Qwen + local structured memory
- Current development line: v0.6
- Safe Git checkpoint: `bbda0f4` — `ALLI v0.6 working memory checkpoint`
- Coding-agent setup: Aider + Ollama/Qwen

## Version history

### v0.0 — Foundation
Python → Ollama → Qwen. Established the local assistant architecture.

### v0.1 — Conversation loop
Added continuous interaction with Python, `while`, input/output, lists, dictionaries and exit handling.

### v0.2 — Persistent conversation
Problem: conversation existed only in RAM and disappeared after restart.
Solution: persistent `memory/conversation.json`.

### v0.3 — Reliable conversation persistence
Fixed conversation saving/loading and made persistent chat history reliable.

### v0.4 — Long-term memory
Introduced `memory/long_term_memory.json` for durable personal facts such as identity, goals, skills, interests and learning.

### v0.5 — Structured memory extraction/management
Added structured memory subsystems:
`extractor.py`, `manager.py`, `store.py`, `retriever.py`, `query_intent.py`, `answers.py`, `embeddings.py`, `config.py`.
Started separating categories and ADD/UPDATE/DELETE/conflict operations.

### v0.6 — Memory management
Core idea: ALLI should not merely remember facts; it should manage changing facts.
Important capabilities: structured memories, deletion, goal management, preference conflict resolution, interest management, retrieval and embeddings.

## v0.6 source structure
```text
src/
├── main.py
└── memory/
    ├── answers.py
    ├── config.py
    ├── embeddings.py
    ├── extractor.py
    ├── manager.py
    ├── query_intent.py
    ├── retriever.py
    └── store.py
```

Runtime memory:
```text
memory/
├── conversation.json
├── conversation_archive.json
└── long_term_memory.json
```

## v0.6 proven working

### Identity
Name/nickname storage and retrieval work.

### Goals
Goal creation and explicit deletion work.
Example:
`my goal is to become a software engineer`
then:
`i don't want to become a software engineer`
correctly deletes the software-engineer goal while preserving other goals.

### Preferences
Preference reversal works.
Test:
`I like Java`
→ `like`
`I don't like Java`
→ `[Memory Conflict Resolved] memory_009`
→ `You dislike java.`
`I like Java`
→ `[Memory Conflict Resolved] memory_009`
→ `You like java...`

### Interests
Adding and deleting interests has been demonstrated.
Example:
`i am interested in web development`
→ memory added
`i am not interested in web development`
→ memory deleted

## v0.6 bug fixed

`I don't like Java` was incorrectly captured by an interest-deletion pattern:
```python
r"i\s+(?:do not|don't)\s+like\s+(.+)$",
```
Removing that pattern from the interest rejection section allowed the existing preference polarity + preference manager to correctly handle like/dislike reversals.

## v0.6 unresolved problems

### 1. Learning creation/deletion — FAIL
Observed:
`i am learning C`
→ ALLI says `You are learning Python.` and does not create C.
Then:
`i don't want to learn C anymore`
cannot delete C because C was never stored.

Expected:
`I am learning C` → add C
`I don't want to learn C anymore` → delete C
`what am i learning` → remaining learning memories.

Likely area: `src/memory/extractor.py`, then manager/query/answer behavior.

### 2. Interest answer formatting — FAIL
Observed:
`You are interested in I am interested in video editing and artificial intelligence.`
Likely answer-layer composition problem: a prefix is being combined with memory content that already contains a similar prefix.
Likely area: `src/memory/answers.py`.

### 3. Broad profile/introduction query — FAIL
`give a brief introduction about me` does not reliably synthesize relevant memories across identity, goals, skills, interests and learning.
This is a query-planning + retrieval + answer-composition problem, not just storage.

### 4. Semantic retrieval — NEEDS AUDIT
Embeddings exist, but semantic retrieval needs explicit tests for differently worded queries.

### 5. Git/runtime hygiene — NEEDS CLEANUP
The v0.6 checkpoint tracks runtime files such as `conversation.json`, `long_term_memory.json` and `__pycache__/*.pyc`. `.gitignore` currently contains `.aider*`.
Do not blindly restore runtime memory; inspect and clean tracking deliberately.

## Desired future memory architecture

```text
USER MESSAGE
    ↓
Intent Analyzer
    ↓
Fact Extractor / Query Planner
    ↓
Memory Operation
    ↓
Memory Manager
    ↓
Memory Repository
    ↓
Current State + History
    ↓
Retriever
    ↓
Answer Planner
    ↓
Qwen
    ↓
ALLI
```

Responsibilities:
- Extractor: determine what the user's message means.
- Memory Manager: decide ADD/UPDATE/DELETE/REPLACE/ARCHIVE/RESTORE.
- Retriever: find relevant memories using structured, lexical and semantic search.
- Answer Planner: combine memories into a natural response.

Future memory records should ideally have stable structured fields such as:
`id`, `category`, `subject`, `predicate/fact_type`, `value`, `polarity`, `status`, `confidence`, `importance`, `source`, timestamps and version/history.

Important invariants:
- No contradictory active states for the same logical fact.
- Deleted memories must not appear in normal retrieval.
- Explicit user statements outrank AI inference.
- Category boundaries must be respected.
- Every memory has a valid ID.
- Memory changes should be auditable.

Long-term storage may eventually move from JSON to SQLite + vector index + history/event log, but only after the conceptual system and tests are stable.

## Development-agent problem

Current debugging is inefficient because ChatGPT does not directly inspect the user's local repository; the user has been sending terminal screenshots/output repeatedly.

Desired local development-agent workflow:
1. Inspect repository.
2. Read project rules.
3. Understand architecture.
4. Reproduce bug.
5. Identify root cause.
6. Propose minimal fix.
7. Run tests.
8. Review result.
9. Commit only with approval.

Aider + local Ollama/Qwen is the current candidate foundation, but should be controlled with strict agent rules. No destructive Git actions, broad rewrites, or commits without approval.

## Series continuity problem

The ALLI series is published as version-based parts. A new ChatGPT conversation should not require the user to explain the entire previous version.

The permanent series context must record:
- what each version built,
- the problem before it,
- why the version was created,
- what actually changed,
- bugs discovered,
- what was fixed,
- what remains broken,
- current capabilities,
- current limitations,
- next logical problem,
- demo evidence,
- story continuity.

### Episode framework
```text
PREVIOUS VERSION
↓
What problem remained?
↓
CURRENT VERSION
↓
What are we solving?
↓
BUILD
↓
What actually changed?
↓
DISCOVERY / FAILURE
↓
FIX
↓
DEMO
↓
RESULT
↓
NEXT PROBLEM
```

Content should be natural, beginner-friendly, honest, low-jargon, story-driven, and based only on capabilities that actually exist.

## Long-term ALLI vision

The target is a personal computer agent:
Memory → context understanding → planning → tools → file operations → computer interaction → automation → reliable personal agent.

Example future request:
“Tomorrow I have my OS exam. Gather my reel resources, organize them, open CapCut and import them, remind me after one hour to study, give me important OS questions and make a quick-revision PDF.”

This is a future target, not a v0.6 capability.

## Version completion rule
A version is complete only when:
- feature implemented,
- relevant tests pass,
- critical bugs resolved,
- documentation updated,
- Git checkpoint created,
- demo behavior verified.

## Current next steps
1. Maintain this master context.
2. Build the controlled development-agent workflow.
3. Finish v0.6:
   - learning add/delete,
   - interest answer formatting,
   - profile/introduction synthesis,
   - semantic retrieval audit,
   - automated acceptance tests,
   - Git/runtime cleanup.
4. Create final v0.6 checkpoint.
5. Only then define v0.7 from the actual remaining problem.

## Important debugging rule
Work one problem at a time:
inspect → reproduce → root cause → minimal fix → test → checkpoint.

Do not use giant agent prompts or broad rewrites.

## Recent ground-truth tests

Preference:
`I don't like Java` → conflict resolved → dislike Java.
`I like Java` → same memory reversed → like Java.

Interest:
`I am interested in web development` → memory added.
`I am not interested in web development` → memory deleted.

Learning failure:
`I am learning C` → C not stored; Python returned.
`I don't want to learn C anymore` → C cannot be removed.

Goal:
Software-engineer goal was added/already present and then explicitly deleted successfully; AI engineer + researcher remained.

## Instruction for future ChatGPT series chats

When starting a new ALLI planning chat, provide this document and say:

> “This is the ALLI Master Context. Read it before planning the episode. Determine the core concept from the actual previous-version problem and current implementation. Do not invent capabilities that are not marked as working.”

The new chat should first identify:
- previous version,
- previous problem,
- what was built,
- what was fixed,
- what remains,
- current capability,
- next logical problem,
- episode core concept.

Only then should it create the hook/story/script.
