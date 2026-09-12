# ALLI v0.6 — Cognitive Memory Architecture

**Session 0 — Architecture Freeze**
**Date:** 2026-09-04
**Status:** Design Complete — Awaiting Review

---

## 1. Vision

ALLI v0.6 is a local personal AI assistant with a Cognitive Memory Engine that understands natural language, represents personal knowledge as entity-scoped facts with temporal validity, and answers questions only from supported evidence.

The engine separates language understanding (LLM) from memory semantics (deterministic Python). The LLM extracts structured meaning. Python owns validation, lifecycle, storage, retrieval, and evidence boundaries. This separation ensures that memory state is never at the mercy of a language model's hallucination.

---

## 2. Design Principles

### 2.1 Generic Meaning, Not One-Off Rules

Every component must handle paraphrases, grammar mistakes, slang, and varied wording. The system processes *meaning*, not *sentences*. No hardcoded example sentences in logic.

### 2.2 Entity-Scoped Facts

Every fact belongs to an entity. A query about MOM must never return USER's data. Entity resolution is generic — derived from syntactic role (possessive, self-reference), not from a list of known names.

### 2.3 Memory Is a Graph, Not a List

Facts are connected to entities through relationships. The model supports: entity→fact, fact→fact (contradiction, supersession), entity→entity (relationship). Categories (goal, skill, interest) are *derived tags* on facts, not the primary organizing structure.

### 2.4 Current vs. Historical State

Old information is never destroyed. State changes produce lifecycle transitions (active→inactive) with provenance. The system can always answer "what did I used to believe?" alongside "what do I believe now?"

### 2.5 Contradictions Are Detected, Not Ignored

Contradiction detection operates on semantic similarity of fact content, entity scope, polarity, temporal context, and explicit correction signals — not exact string matching.

### 2.6 Temporal Awareness

The system distinguishes persistent facts, temporary state, events, and historical facts. Temporary statements do not become permanent without explicit user intent.

### 2.7 Uncertainty Is Represented, Not Stored as Fact

Uncertain statements ("I think I might...") are stored with low confidence or as intentions, never as confirmed facts.

### 2.8 Hybrid Retrieval

No single retrieval method is sufficient. The system combines structured filtering, entity filtering, lexical matching, semantic similarity, and temporal filtering. Embeddings are one signal among many, not truth.

### 2.9 Evidence Before Answer

Retrieved memories are candidates. An evidence-validation stage confirms that retrieved facts actually answer the question before they are presented. If evidence is insufficient, the system says "I don't know."

### 2.10 Write and Read Are Separate Pipelines

Write path and read path are designed independently. They share the memory store but have distinct understanding, processing, and output stages.

### 2.11 LLM Is a Language Parser, Not a Memory Controller

The LLM understands natural language and produces structured output. It never directly reads or writes memory state. Python validates every LLM output before it touches memory.

### 2.12 Storage Is Abstracted

The memory engine depends on a storage interface, not a file format. JSON is the current implementation. SQLite + vector index is the future target. The interface contract ensures the engine does not need redesign when storage changes.

---

## 3. Terminology

| Term | Definition |
|------|-----------|
| **Entity** | A thing that facts can be about: a person (self, mom, friend), organization, or concept. Identified by name + type. |
| **Fact** | A discrete piece of knowledge: subject + predicate + object/value. Scoped to an entity. Has lifecycle state. |
| **Relationship** | A connection between entities (e.g., USER is-child-of MOM). Separate from facts about entities. |
| **Episode** | A recorded interaction event: user message + timestamp + extracted operations + memory changes. |
| **Memory** | The combination of a fact + its lifecycle state + provenance + temporal validity + confidence. |
| **Fact Type** | A semantic classification of a fact within its domain (e.g., "name", "like", "career_goal", "learning"). |
| **Domain** | A high-level grouping of related fact types (e.g., "identity", "preference", "goal", "learning"). |
| **Lifecycle State** | The current status of a memory: active, inactive, superseded, expired. |
| **Provenance** | The origin and history of a memory: source, extraction method, confidence, creation time. |
| **Temporal Validity** | The time window during which a fact is considered current: valid_from, valid_until. |
| **Supersession** | The event where a new fact replaces an old one as the current truth. The old fact becomes historical. |
| **Entity Resolution** | The process of determining which entity a user message or query refers to. |
| **Query Intent** | The structured representation of what a user is asking: domain, entity scope, cardinality, temporal scope. |
| **Evidence** | A retrieved memory that has passed validation and is considered reliable enough to include in an answer. |
| **Synthesis** | The deliberate combination of multiple facts across domains to produce a coherent response. |

---

## 4. Memory Model

### 4.1 Canonical Memory Representation

Every memory in ALLI v0.6 is represented as a record with these fields:

```
Memory {
    id:                 string          # unique identifier (memory_NNN)
    entity:             EntityRef       # who/what this is about
    fact:               Fact            # the knowledge content
    domain:             string          # derived domain tag (identity, goal, preference, etc.)
    lifecycle:          Lifecycle       # state machine: active → inactive/expired
    provenance:         Provenance      # origin, extraction method, confidence
    temporal:           TemporalScope   # valid_from, valid_until, is_persistent
    confidence:         float [0.0-1.0] # how certain this fact is
    embedding:          list[float]     # semantic vector (nullable)
    embedding_version:  int             # recipe version for re-embedding
    schema_version:     int             # for migration
    created_at:         ISO timestamp
    updated_at:         ISO timestamp
}
```

**WHY:** This representation captures everything needed for entity-scoping, temporal filtering, contradiction detection, history preservation, and provenance tracking — without requiring a graph database. The fields are flat enough for JSON storage now, but map naturally to relational + vector storage later.

### 4.2 Entity Reference

```
EntityRef {
    name:           string          # canonical entity name (lowercase, normalized)
    entity_type:    string          # "self" | "person" | "organization" | "concept"
    relation:       string?         # relationship to user: "self", "parent", "friend", etc.
    entity_id:      string?         # future: link to entity record
}
```

The default entity is `self` (the user). Third-party entities are extracted from possessive patterns ("mom's", "my friend's") or explicit naming.

### 4.3 Fact

```
Fact {
    fact_type:      string          # semantic type within domain (e.g., "name", "like", "career_goal")
    subject:        string          # normalized topic key for matching
    predicate:      string          # semantic relation: "has_name", "likes", "is_learning", "has_goal"
    value:          string          # compact value of the fact
    content:        string          # natural language statement of the fact
    polarity:       string?         # "positive" | "negative" | null (for preferences)
}
```

**WHY `subject` + `predicate` + `value`:** This triple structure enables generic matching. Two facts about the same subject with the same predicate are candidates for contradiction or update. The `content` field preserves the human-readable form.

**WHY `fact_type`:** Derived from domain + content analysis. Enables fine-grained filtering (e.g., "learning" vs. "skill" within the knowledge domain) without requiring the user to categorize their statements.

### 4.4 Derived Domain Tags

Domains are NOT the primary organizing structure. They are derived tags applied during extraction:

| Domain | Contains | Example Fact Types |
|--------|----------|-------------------|
| identity | Personal identification | name, nickname |
| preference | Likes and dislikes | like, dislike |
| knowledge | Skills and competencies | skill |
| learning | Active learning/studying | learning, studying |
| goal | Aspirations and targets | career_goal, learning_goal, project_goal |
| ambition | Professional aspirations | ambition |
| dream | Long-term visions | dream |
| education | Formal education | course, diploma, degree |
| project | Active projects | project |
| task | Temporary time-bound items | task, appointment |
| possession | Owned items | owned_item |
| priority | Current focus | priority |
| routine | Habits and patterns | habit |
| relationship | Facts about other entities | any fact_type (scoped to non-self entity) |

**WHY derived, not structural:** The same underlying fact model (entity + fact + lifecycle) works for all domains. Domain tags enable retrieval filtering without requiring different data structures per domain. New domains can be added by defining new tags, not new schemas.

---

## 5. Entity Model

### 5.1 Entity Resolution Architecture

Entity resolution happens in two places:

**Write-side:** Determines which entity a user statement is about.
**Read-side:** Determines which entity a query refers to.

Both use the same resolution strategy but operate on different input types (statements vs. questions).

### 5.2 Resolution Strategy

Resolution is层次化 (hierarchical), applied in priority order:

1. **Self-identification:** "I am X", "My name is X" → entity = self
2. **Possessive third-party:** "Mom's favorite color" → entity = mom (person)
3. **My-person possessive:** "My friend's name" → entity = friend (person)
4. **Self-reference:** "I like X", "My goal is..." → entity = self
5. **Third-party named:** "Rahul likes Java" → entity = rahul (person)
6. **Unresolved pronoun:** "their favorite" → entity = unknown (blocked)
7. **Default:** → entity = self (for clearly personal queries)

### 5.3 Entity Name Normalization

Entity names are normalized for consistent matching:
- Lowercased
- Whitespace-trimmed
- Common relationship words mapped to canonical types (mom→parent, friend→friend)

No hardcoded name list. Resolution is purely syntactic. The system learns entity names from user messages.

### 5.4 Entity-Scoped Storage

Every memory record stores its `entity` reference. Retrieval filters by entity before ranking. This prevents cross-entity contamination (USER's facts never appear in MOM's query results).

---

## 6. Fact/Relationship Model

### 6.1 Fact Lifecycle States

```
         ┌──────────┐
         │ proposed │ ← extraction produces this
         └────┬─────┘
              │ validation passes
              ▼
         ┌──────────┐
         │ active   │ ← current truth
         └────┬─────┘
              │ superseded by newer fact
              ▼
         ┌──────────┐
         │ inactive │ ← historical, preserved
         └────┬─────┘
              │ temporal expiry
              ▼
         ┌──────────┐
         │ expired  │ ← time-bound facts that passed their window
         └──────────┘
```

**WHY inactive, not deleted:** History must be preserved. The user may ask "what did I used to want to become?" Inactive facts carry supersession links so the system can reconstruct history.

### 6.2 Supersession

When a new fact contradicts or updates an existing one:

1. The old fact's lifecycle transitions: active → inactive
2. The old fact records `superseded_by: <new_fact_id>`
3. The new fact becomes active
4. Both records persist

The supersession chain enables: "What do I believe now?" (active) vs. "What did I used to believe?" (inactive, following supersession links).

### 6.3 Relationships Between Entities

Relationships are first-class facts about entity-to-entity connections:

```
Relationship {
    entity_a:        EntityRef
    entity_b:        EntityRef
    relation_type:   string     # "parent", "sibling", "friend", "colleague", etc.
    confidence:      float
    provenance:      Provenance
    temporal:        TemporalScope
}
```

Relationships are extracted from possessive patterns ("mom's"), kinship terms, and explicit statements ("my friend is Rahul"). They enable entity-scoped retrieval across the relationship graph.

---

## 7. Temporal Model

### 7.1 Temporal Scope

Every memory has temporal validity:

```
TemporalScope {
    valid_from:     ISO timestamp     # when this became true
    valid_until:    ISO timestamp?    # when this stopped being true (null = persistent)
    is_persistent:  bool              # true for durable facts, false for temporary
    temporal_hint:  string?           # natural language hint ("tomorrow", "next week")
}
```

### 7.2 Temporal Categories

| Category | Example | valid_until | is_persistent |
|----------|---------|-------------|---------------|
| Persistent fact | "My name is Rahul" | null | true |
| Temporary state | "I am hungry" | computed from hint | false |
| Task/Event | "I have exam tomorrow" | computed from hint | false |
| Historical fact | "I used to own Oppo A53" | when superseded | true (historical) |
| Uncertain intention | "I might learn Java" | null | false (low confidence) |

### 7.3 Temporal Hint Resolution

Natural language temporal hints are resolved to concrete timestamps:

- "tomorrow" → tomorrow 23:59:59
- "next week" → end of next week
- "in 3 days" → now + 3 days
- "this Friday" → upcoming Friday 23:59:59
- ISO date → direct parse

### 7.4 Expired Memory Handling

Expired memories are not returned by default during retrieval. They can be explicitly requested via temporal queries ("what did I have to do last week?"). Expired memories remain in storage for history.

---

## 8. Memory Lifecycle

### 8.1 Add

New fact → entity resolution → validation → embedding → lifecycle=active → persist → memory event recorded.

### 8.2 Update (Same Entity, Same Subject)

Existing fact found → semantic similarity confirms same topic → content/value updated → old version archived (active→inactive, superseded_by=new) → embedding regenerated → memory event recorded.

### 8.3 Supersession (Explicit Correction)

User says "Actually, I want to become X instead of Y" → new fact created → old fact superseded → supersession link recorded → memory event recorded.

### 8.4 Deactivation (User-initiated Deletion)

User says "Forget that I like X" → matching fact found → lifecycle transitions active→inactive → superseded_by=null (user-initiated) → memory event recorded. The fact is NOT deleted from storage.

### 8.5 Expiration (Automatic)

Temporal fact passes valid_until → lifecycle transitions active→expired during retrieval-time filtering → memory event recorded.

### 8.6 Consolidation (Duplicate Prevention)

Before adding a new fact, the system checks for existing active facts with:
- Same entity
- Same fact_type
- Same subject (normalized)
- Semantic similarity above threshold

If found: update existing instead of adding new. No duplicates.

---

## 9. Write Pipeline

The write pipeline processes user statements (non-questions) into memory operations.

```
USER MESSAGE
    │
    ▼
┌─────────────────────┐
│  1. MESSAGE UNDERSTANDING  │
│  - Is this a question?     │
│  - Is this a statement?    │
│  - Is this a command?      │
│  - Is this temporary?      │
│  - Is this uncertain?      │
│  - Is this third-party?    │
│  - Is this a correction?   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. ENTITY RESOLUTION     │
│  - Self or third-party?   │
│  - Which entity?          │
│  - Entity confidence      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  3. FACT EXTRACTION       │
│  - Domain/fact_type       │
│  - Subject (normalized)   │
│  - Value                  │
│  - Content (natural lang) │
│  - Polarity               │
│  - Confidence             │
│  - Temporal hints         │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  4. VALIDATION            │
│  - Garbage/vague check    │
│  - Temporal-only check    │
│  - Uncertainty check      │
│  - Domain validity        │
│  - Value normalization    │
│  - Subject normalization  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. DEDUPLICATION         │
│  - Find existing active   │
│    facts same entity +    │
│    same subject + same    │
│    fact_type              │
│  - Semantic similarity    │
│  - Decision: add new /    │
│    update existing /      │
│    unchanged              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. CONTRADICTION CHECK   │
│  - Cross-entity conflicts │
│  - Polarity reversal      │
│  - Explicit correction    │
│  - Temporal supersession  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  7. LIFECYCLE DECISION    │
│  - New fact → active      │
│  - Update → archive old   │
│    + activate new         │
│  - Delete → deactivate    │
│  - Cancel → deactivate    │
│    matching task          │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  8. PERSISTENCE           │
│  - Generate embedding     │
│  - Compute temporal       │
│  - Write to storage       │
│  - Record memory event    │
└──────────┬──────────┘
           │
           ▼
     WRITE RESULT
     (status, memory_id, details)
```

### 9.1 Extraction Method

Extraction uses a dual approach:

**Primary: LLM-constrained extraction.** The LLM receives the user message + a JSON schema + strict rules. It produces structured operations. Python validates every field before accepting.

**Fallback: Deterministic pattern extraction.** When LLM is unavailable or produces invalid output, a regex/pattern-based extractor handles common patterns. This ensures the system degrades gracefully.

**WHY dual:** The LLM handles natural language understanding (paraphrases, grammar mistakes, slang). The pattern extractor provides reliability when the LLM fails. Neither alone is sufficient.

### 9.2 Validation Rules

Every extracted operation must pass validation:

| Check | Rule | Action on failure |
|-------|------|-------------------|
| Garbage | Value is symbols-only or empty | Reject |
| Vague | Value is universal word (nothing, something, anything) | Reject |
| Temporal-only | Value contains only time words | Reject |
| Uncertain | Statement expresses uncertainty ("I think I might...") | Store with low confidence, not as fact |
| Too short | Value < minimum length | Reject |
| Too long | Content > maximum length | Truncate or reject |
| Invalid domain | Domain not in valid set | Map to "other" |

### 9.3 Memory Events

Every write operation produces a memory event:

```
MemoryEvent {
    event_type:     string      # "add", "update", "supersede", "deactivate", "expire"
    memory_id:      string      # affected memory
    timestamp:      ISO time
    trigger:        string      # "user_statement", "expiration", "consolidation"
    details:        dict        # operation-specific data
}
```

Events enable debugging, audit trails, and undo capabilities.

---

## 10. Read Pipeline

The read pipeline processes user queries into answers.

```
USER QUERY
    │
    ▼
┌─────────────────────┐
│  1. QUERY UNDERSTANDING    │
│  - Personal or general?    │
│  - Which domain(s)?        │
│  - Which entity?           │
│  - Singular or plural?     │
│  - Temporal scope?         │
│  - Synthesis request?      │
│  - Ambiguity?              │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  2. ENTITY RESOLUTION     │
│  - Self or third-party?   │
│  - Resolved or unknown?   │
│  - Entity scope for filter│
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────┐
│  3. HYBRID RETRIEVAL                      │
│                                           │
│  ┌──────────────┐  ┌──────────────┐       │
│  │  STRUCTURAL   │  │   ENTITY     │       │
│  │  FILTER       │  │   FILTER     │       │
│  │  (domain,     │  │  (entity     │       │
│  │   fact_type)  │  │   scope)     │       │
│  └──────┬───────┘  └──────┬───────┘       │
│         │                 │               │
│         ▼                 ▼               │
│  ┌────────────────────────────────┐       │
│  │      CANDIDATE SET             │       │
│  └────────────┬───────────────────┘       │
│               │                           │
│  ┌────────────┼────────────┐              │
│  │            │            │              │
│  ▼            ▼            ▼              │
│  LEXICAL    SEMANTIC    TEMPORAL          │
│  MATCH      MATCH       MATCH             │
│  (keyword   (embedding  (valid_from/      │
│   overlap)  cosine)     valid_until)      │
│  │            │            │              │
│  └────────────┼────────────┘              │
│               │                           │
│               ▼                           │
│  ┌────────────────────────────────┐       │
│  │      SCORED CANDIDATES         │       │
│  │  (combined ranking score)      │       │
│  └────────────┬───────────────────┘       │
│               │                           │
└───────────────┼───────────────────────────┘
                │
                ▼
┌─────────────────────┐
│  4. EVIDENCE VALIDATION  │
│  - Does retrieved fact   │
│    actually answer the   │
│    question?             │
│  - Is confidence high    │
│    enough?               │
│  - Is temporal scope     │
│    correct?              │
│  - Is entity match       │
│    confirmed?            │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  5. ANSWER PLANNING       │
│  - Singular: one fact?    │
│  - Plural: all matching?  │
│  - Synthesis: combine?    │
│  - Clarify: ambiguous?    │
│  - Unknown: no evidence?  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  6. RESPONSE GENERATION   │
│  - Format from evidence   │
│  - Do not hallucinate     │
│  - Say "I don't know"     │
│    when evidence missing  │
│  - Ask for clarification  │
│    when ambiguous         │
└──────────┬──────────┘
           │
           ▼
       ANSWER
```

---

## 11. Query Representation

### 11.1 Query Intent

Every user query is parsed into a structured QueryIntent:

```
QueryIntent {
    primary_domain:     string          # main domain being queried
    domains:            list[string]    # all requested domains (for compound queries)
    entity:             EntityRef       # who the query is about
    cardinality:        string          # "singular" | "plural" | "unknown"
    temporal_scope:     string?         # "current" | "historical" | "expired" | null
    is_memory_read:     bool            # requires personal memory?
    is_general_knowledge: bool          # general question, no memory needed?
    is_synthesis:       bool            # cross-domain summary needed?
    preference_polarity: string?        # "like" | "dislike" | null
    needs_clarification: bool           # ambiguous, ask user?
    clarification_message: string?      # what to ask
    normalized_query:   string          # cleaned query text
    confidence:         float           # intent detection confidence
}
```

### 11.2 Intent Detection Strategy

Intent detection uses a layered approach:

1. **Lexical cues:** Regex patterns detect explicit domain keywords ("my goal", "what do I like", "tell me about me"). Fast, deterministic, handles exact phrasings.

2. **Semantic prototypes:** Each domain has 4+ prototype phrases. The query is embedded and compared to all prototypes. Highest cosine similarity determines domain. Handles paraphrases.

3. **Disambiguation:** When multiple domains score similarly, lexical cues break ties. When semantic and lexical disagree, the higher-confidence signal wins.

4. **Special routing:** "tell me about me" → synthesis. "what is Python?" → general knowledge (no personal pronouns). "what should I learn?" → clarification (advice, not memory).

### 11.3 Personal vs. General Separation

A query is classified as personal if:
- Contains personal pronouns (I, my, me, myself)
- Contains domain-specific keywords (goal, skill, preference, etc.)
- Matches a memory-read prototype above threshold
- Contains "about me" or "about myself"

A query is classified as general if:
- Asks for definitions, explanations, how-to
- Contains no personal pronouns
- Matches general-knowledge prototypes
- Is clearly about external topics ("What is Python?")

General questions go to Ollama without memory retrieval. Personal questions go through the memory pipeline.

---

## 12. Hybrid Retrieval Architecture

### 12.1 Retrieval Stages

Retrieval is a pipeline of filters and scorers:

**Stage 1: Entity Filter**
- Filter memories by entity scope
- If query entity is resolved → only matching entity
- If query entity is unresolved → block all results
- If query entity is self → all self-scoped memories

**Stage 2: Structural Filter**
- Filter by domain and fact_type based on query intent
- Strict mode: only exact domain matches
- Relaxed mode: allow related domains with penalty

**Stage 3: Temporal Filter**
- Filter by temporal validity
- Current queries: only active, non-expired
- Historical queries: include inactive
- Expired queries: only expired

**Stage 4: Lexical Scoring**
- Token overlap between query and fact content/subject
- Normalized for stop words
- Handles exact keyword matches

**Stage 5: Semantic Scoring**
- Embed query, compare to all candidate embeddings
- Cosine similarity score
- Weighted by domain relevance

**Stage 6: Combined Ranking**

```
final_score = (semantic_score × SEMANTIC_WEIGHT)
            + (lexical_score × LEXICAL_WEIGHT)
            + (category_match × CATEGORY_WEIGHT)
            - (entity_mismatch_penalty)
            - (temporal_mismatch_penalty)
```

**Stage 7: Top-K Selection**
- Select top-K candidates by final_score
- K varies by cardinality (singular: 1-3, plural: up to 8)

### 12.2 Why Hybrid, Not Pure Semantic

Pure semantic retrieval has known failure modes:
- Paraphrases with different vocabulary may score low
- Short queries have weak embedding signals
- Domain-irrelevant memories with similar topic may rank high
- No understanding of entity scope

Hybrid retrieval addresses all of these:
- Lexical matching catches exact keyword hits
- Structural filtering ensures domain relevance
- Entity filtering prevents cross-contamination
- Semantic matching handles paraphrases
- Combined ranking balances all signals

### 12.3 Retrieval for Synthesis

For synthesis queries ("tell me about me"), retrieval is broad:
- No domain filter (retrieve from all domains)
- Entity filter still applies (only self)
- Temporal filter: current/active only
- All active memories are candidates
- Ranking prioritizes importance + recency

---

## 13. Evidence Architecture

### 13.1 Evidence Validation Rules

Retrieved candidates pass through evidence validation before being used in answers:

| Check | Rule | Action |
|-------|------|--------|
| Relevance | Fact actually addresses the query topic | Exclude if not |
| Confidence | Confidence ≥ minimum threshold (0.5) | Exclude if below |
| Temporal | Fact is current for the query's temporal scope | Exclude if expired |
| Entity | Entity match confirmed (not just filtered) | Exclude if mismatch |
| Completeness | Fact has enough detail to be useful | Include with caveat if partial |

### 13.2 Evidence Sufficiency

After validation, the system checks if evidence is sufficient to answer:

- **Sufficient:** At least one high-confidence, relevant, current fact exists → answer from evidence
- **Partial:** Some evidence exists but incomplete → answer with available evidence + note limitations
- **Insufficient:** No relevant evidence → "I don't have that information stored"
- **Ambiguous:** Multiple contradictory facts with equal confidence → "I have conflicting information about X"

### 13.3 Anti-Hallucination Boundary

This is the most critical architectural boundary:

**The answer generation stage MUST NOT:**
- Invent personal facts not in retrieved evidence
- Infer unstated preferences, goals, or skills
- Combine facts in ways not supported by evidence
- Present low-confidence information as certain
- Guess when evidence is missing

**The answer generation stage MUST:**
- Only use facts that passed evidence validation
- Clearly state when information is unknown
- Note when evidence is partial or low-confidence
- Ask for clarification when ambiguous

---

## 14. Answer Architecture

### 14.1 Answer Types

| Query Type | Answer Strategy |
|-----------|----------------|
| Singular personal ("What is my name?") | Retrieve top-1, format directly |
| Plural personal ("What are my goals?") | Retrieve all matching, format as list |
| Compound ("What are my skills and goals?") | Retrieve per-domain, format sections |
| Synthesis ("Tell me about me") | Retrieve broad, synthesize across domains |
| Preference polarity ("What do I dislike?") | Filter by polarity, format |
| Unknown ("What is my X?") | "I don't have that stored" |
| Ambiguous ("What is my priority?") | Clarify if multiple candidates |

### 14.2 Answer Formatting

Answers are constructed from evidence, not templates:

- Single fact: "Your name is {value}."
- Multiple facts: "Your goals are {A}, {B}, and {C}."
- Missing: "I don't currently have any stored information about your {domain}."
- Synthesis: Structured summary with domain sections, each sourced from evidence.

### 14.3 Synthesis Architecture

For "tell me about me" / "give me an introduction":

1. Retrieve all active self-scoped memories
2. Group by domain
3. For each domain with evidence:
   - Select most important/recent facts
   - Format concisely
4. Compose structured response:
   - Identity section (if present)
   - Goals section (if present)
   - Skills section (if present)
   - ... (only domains with evidence)
5. Do NOT include domains with no evidence
6. Do NOT fabricate connections between facts

**WHY not LLM synthesis:** The LLM would need to see all memories and produce a summary. This risks hallucination — the LLM might invent facts not in memory. Instead, the memory engine produces a structured summary from validated evidence. The LLM is only used if the user asks for a "natural language" rewrite of the structured summary.

---

## 15. Storage Abstraction

### 15.1 Storage Interface

The memory engine depends on an abstract storage interface:

```
MemoryStorage interface:
    load() → list[Memory]
    save(memories: list[Memory]) → None
    add_memory(memory: Memory) → Memory
    update_memory(id: str, changes: dict) → Memory?
    delete_memory(id: str) → bool
    get_by_id(id: str) → Memory?
    get_active() → list[Memory]
    get_by_entity(entity_name: str) → list[Memory]
    get_by_domain(domain: str) → list[Memory]
    search(query_vector: list[float], top_k: int) → list[ScoredMemory]
    count() → int
```

### 15.2 Current Implementation: JSON Storage

The JSON backend implements this interface:
- `load()` reads from a single JSON file
- `save()` writes atomically (temp file + replace)
- `search()` does brute-force cosine similarity over all embeddings
- All data in one file, loaded into memory

### 15.3 Future Implementation: SQLite + Vector Index

The SQLite backend will implement the same interface:
- `load()` reads from SQLite tables
- `save()` writes to SQLite with transactions
- `search()` uses vector index (FAISS or similar) for fast similarity search
- Data persisted on disk, loaded on demand

### 15.4 Why Interface-Based

The memory engine code never touches JSON files or SQLite tables directly. It calls interface methods. This means:
- Switching backends requires only implementing the interface
- No engine code changes when storage changes
- Testing can use in-memory storage (no file I/O)
- Future backends (Postgres, Redis) can be added without engine changes

---

## 16. LLM Boundaries

### 16.1 What the LLM Does

| Task | LLM Role | Validation |
|------|----------|------------|
| Understanding user statements | Parse natural language into structured operations | Schema validation, field validation |
| Understanding user queries | (Optional) Disambiguate complex queries | Intent confidence check |
| Generating natural responses | Rewrite structured answers in natural language | Only when user requests "natural" response |

### 16.2 What the LLM Must NOT Do

- Read memory files directly
- Write memory files directly
- Decide what to store without validation
- Decide what to retrieve without the retrieval pipeline
- Invent personal facts
- Override entity resolution
- Bypass lifecycle rules
- Modify temporal validity

### 16.3 Structured Output Schema

When the LLM extracts facts from a user message, it MUST produce output matching a strict JSON schema:

```
ExtractionResult {
    intent:           string      # "statement" | "question" | "greeting" | "command"
    is_temporary:     bool
    temporal_hint:    string?
    is_negation:      bool
    is_third_party:   bool
    is_uncertain:     bool
    operations:       list[MemoryOperation]
}

MemoryOperation {
    action:           string      # "ADD" | "UPDATE" | "DELETE" | "CANCEL"
    domain:           string
    fact_type:        string
    content:          string      # natural language statement
    subject:          string      # normalized topic
    value:            string
    polarity:         string?     # "like" | "dislike" | null
    importance:       string      # "high" | "medium" | "low"
    confidence:       float       # 0.0 - 1.0
    entity_name:      string
    entity_type:      string
    temporal_hint:    string?
}
```

### 16.4 Malformed Output Handling

If the LLM produces output that fails schema or field validation:
1. Log the malformed output for debugging
2. Fall back to deterministic pattern extraction
3. Never silently accept invalid data
4. Never partially apply an operation

---

## 17. Embedding Boundaries

### 17.1 What Embeddings Do

- Generate semantic vectors for facts and queries
- Enable cosine similarity scoring for retrieval
- Enable semantic duplicate detection
- Enable intent detection via prototype comparison

### 17.2 What Embeddings Must NOT Do

- Be the sole basis for retrieval decisions (hybrid, not pure semantic)
- Be treated as ground truth (one signal among many)
- Be required for the system to function (graceful degradation when unavailable)
- Override structural or entity-based filtering

### 17.3 Embedding Generation

- Model: `all-minilm:latest` (384 dimensions)
- Input text: Structured recipe combining domain, fact_type, and content
- Recipe version tracked; bumping forces re-embedding
- Embeddings cached in-memory during session
- Missing/null embeddings handled gracefully (skip semantic scoring, rely on other signals)

### 17.4 Embedding Text Recipe

To improve retrieval quality, embedding input is NOT raw content. It is a structured recipe:

```
"Durable personal memory about the user. Domain {domain}. Fact type {fact_type}. {domain_hint}. Fact: {content}"
```

Domain hints provide semantic context that all-minilm needs for good similarity scores. This addresses the known weakness of small embedding models with short, decontextualized text.

---

## 18. Observability / Debug Trace

### 18.1 Trace Structure

Every write and read operation produces a debug trace:

```
DebugTrace {
    input:              string      # original user message
    interpretation:     dict        # parsed intent, entities, operations
    entity_resolution:  dict        # resolved entity, confidence
    operations:         list[dict]  # extracted operations
    validation:         dict        # validation results
    deduplication:      dict        # duplicate detection results
    contradictions:     dict        # contradiction check results
    lifecycle:          dict        # state transitions
    persistence:        dict        # what was stored/updated
    retrieval:          dict        # retrieval query, candidates, scores
    evidence:           dict        # validated evidence
    answer_decision:    dict        # why this answer was chosen
    final_answer:       string      # the response
    timing:             dict        # duration of each stage
}
```

### 18.2 Debug Output

When `ALLI_MEMORY_DEBUG=1`:

```
[Write Pipeline]
  Input: "I am learning C"
  → Intent: statement (confidence=0.95)
  → Entity: self (resolved)
  → Operation: ADD domain=learning fact_type=learning subject="c" value="C"
  → Validation: PASS (not garbage, not vague, not temporal-only)
  → Dedup: no existing active learning fact for subject "c"
  → Lifecycle: new memory → active
  → Persisted: memory_042
  → Embedding: generated (384 dims)

[Read Pipeline]
  Input: "What am I learning?"
  → Intent: learning (score=0.87, lexical cue: "learning")
  → Entity: self (resolved)
  → Cardinality: plural
  → Retrieval:
    - Entity filter: 15 self-scoped → 15
    - Domain filter: learning → 3 candidates
    - Semantic: memory_042 score=0.82, memory_019 score=0.45, memory_031 score=0.31
    - Temporal: all current
    - Top-3 selected
  → Evidence: 3 facts validated (confidence ≥ 0.5, current, relevant)
  → Answer: "You are learning C, Python, and electronics."
```

---

## 19. Module / Component Boundaries

### 19.1 Module Map

```
src/
  main.py                    # CLI entry point, orchestrates pipelines
  memory/
    __init__.py              # Package exports
    config.py                # Tunable constants, paths, model names
    
    # WRITE PIPELINE
    message_understanding.py # Classify message type, detect correction/negation
    entity_resolution.py     # Extract entities from statements and queries
    fact_extraction.py       # LLM + pattern extraction of structured operations
    validation.py            # Validate extracted operations
    lifecycle.py             # Lifecycle state machine, supersession, expiry
    memory_store.py          # Storage interface + JSON implementation
    
    # READ PIPELINE
    query_understanding.py   # Parse query into QueryIntent
    retrieval.py             # Hybrid retrieval pipeline
    evidence.py              # Evidence validation and sufficiency
    answer.py                # Answer generation from evidence
    
    # SHARED
    embeddings.py            # Embedding generation and similarity
    models.py                # Data classes: Memory, Entity, Fact, etc.
    events.py                # Memory event logging
    debug.py                 # Debug trace infrastructure
```

### 19.2 Dependency Direction

```
main.py
  → message_understanding / query_understanding
  → entity_resolution
  → fact_extraction
  → validation
  → lifecycle
  → memory_store
  → retrieval
  → evidence
  → answer

memory_store ← (implements interface)
embeddings ← (used by retrieval, lifecycle, fact_extraction)
models ← (used by everything)
events ← (used by lifecycle, memory_store)
debug ← (used by everything)
```

No circular dependencies. Each module has a single responsibility.

---

## 20. API / Interface Contracts

### 20.1 MemoryStore Interface

```python
class MemoryStore(Protocol):
    def load(self) -> list[Memory]: ...
    def save(self, memories: list[Memory]) -> None: ...
    def add(self, memory: Memory) -> Memory: ...
    def update(self, memory_id: str, changes: dict) -> Memory | None: ...
    def deactivate(self, memory_id: str) -> bool: ...
    def supersede(self, winner_id: str, loser_id: str) -> bool: ...
    def get_by_id(self, memory_id: str) -> Memory | None: ...
    def get_active(self) -> list[Memory]: ...
    def get_by_entity(self, entity_name: str) -> list[Memory]: ...
    def search_semantic(self, query_vector: list[float], top_k: int) -> list[ScoredMemory]: ...
```

### 20.2 Extractor Interface

```python
class FactExtractor(Protocol):
    def extract(self, message: str) -> list[MemoryOperation]: ...
```

### 20.3 Retriever Interface

```python
class Retriever(Protocol):
    def retrieve(self, query: QueryIntent) -> list[ScoredMemory]: ...
```

### 20.4 Evidence Validator Interface

```python
class EvidenceValidator(Protocol):
    def validate(self, query: QueryIntent, candidates: list[ScoredMemory]) -> list[ScoredMemory]: ...
    def is_sufficient(self, query: QueryIntent, evidence: list[ScoredMemory]) -> bool: ...
```

### 20.5 Answer Generator Interface

```python
class AnswerGenerator(Protocol):
    def generate(self, query: QueryIntent, evidence: list[ScoredMemory]) -> str: ...
```

---

## 21. Testing Architecture

### 21.1 Test Layers

| Layer | Scope | Tools | Count Target |
|-------|-------|-------|--------------|
| Unit | Individual functions, pure logic | pytest | 200+ |
| Integration | Module interactions, pipeline stages | pytest + fixtures | 50+ |
| End-to-end | Full write/read cycles | pytest + CLI runner | 30+ |
| Real CLI | Actual user interaction | Manual + automated | 20+ scenarios |

### 21.2 Test Data Strategy

Tests MUST NOT rely on exact example sentences. Each test class includes:

1. **Canonical form:** "I am learning Python"
2. **Paraphrases:** "Python is what I'm studying", "Currently learning Python"
3. **Broken grammar:** "i am learn python", "learnin python rn"
4. **Slang:** "yo i'm picking up python"
5. **Short:** "learning python"
6. **Long:** "So basically I've been spending a lot of time learning Python lately"
7. **Multiple facts:** "I like Python and Java, but I'm learning C"
8. **Corrections:** "Actually I'm learning Java now, not Python"
9. **Contradictions:** "I like Python" → "I hate Python"
10. **Uncertainty:** "I think I might learn Python"
11. **Temporary:** "I'm studying for my Python exam tomorrow"
12. **Historical:** "When I was in first year, I studied C"
13. **Entity ambiguity:** "My mom likes blue" vs. "I like blue"
14. **Singular/plural:** "What is my goal?" vs. "What are my goals?"
15. **Unknown:** "What is my salary?" (not stored)
16. **General:** "What is Python?" (not personal)
17. **Synthesis:** "Tell me about myself"

### 21.3 Regression Test Classes

Based on known v0.6 failures:

```python
class TestLearningAddDelete:
    """Must handle all phrasings of learning statements and deletions."""

class TestInterestFormatting:
    """Must never produce duplicate prefixes in answers."""

class TestProfileSynthesis:
    """Must synthesize across domains for 'introduction' type queries."""

class TestSemanticRetrieval:
    """Must retrieve facts for paraphrased queries."""

class TestEntityScoping:
    """Must never return MOM's facts for USER's queries."""

class TestContradictionDetection:
    """Must detect polarity reversals, corrections, supersessions."""

class TestTemporalHandling:
    """Must distinguish temporary vs. persistent facts."""

class TestUncertaintyHandling:
    """Must not store uncertain statements as confirmed facts."""

class TestCardinality:
    """Must handle singular vs. plural queries correctly."""

class TestEvidenceBoundary:
    """Must never hallucinate when evidence is missing."""
```

### 21.4 Test Execution Commands

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# End-to-end tests
pytest tests/e2e/ -v

# All tests with coverage
pytest tests/ --cov=src/memory --cov-report=html

# Specific regression class
pytest tests/regression/ -v -k "TestLearningAddDelete"
```

---

## 22. Failure Handling

### 22.1 Extraction Failures

| Failure | Handling |
|---------|----------|
| LLM unavailable | Fall back to pattern extraction |
| LLM produces invalid JSON | Log, fall back to patterns |
| LLM produces valid JSON, invalid fields | Validate per-field, reject invalid ops |
| Pattern extraction finds nothing | Return empty operations (not an error) |
| Both fail | Return empty operations, acknowledge statement |

### 22.2 Retrieval Failures

| Failure | Handling |
|---------|----------|
| Embeddings unavailable | Use lexical-only scoring |
| No candidates found | Return "I don't have that stored" |
| Multiple contradictory candidates | Note conflict, present both |
| Evidence below confidence threshold | Return "I'm not certain about that" |
| Entity unresolved | Block results, ask for clarification |

### 22.3 Storage Failures

| Failure | Handling |
|---------|----------|
| File corrupt | Backup, start fresh, log error |
| Write fails | Retry once, then log and continue (in-memory state preserved) |
| Read fails | Return empty list, log error |

### 22.4 What Must NEVER Happen

1. **Memory corruption:** A failed write must never leave partial state
2. **Cross-entity contamination:** MOM's facts must never appear in USER's answers
3. **Hallucination:** The system must never invent personal facts
4. **Silent data loss:** Every state change must be logged as a memory event
5. **Unvalidated LLM output:** LLM output must never bypass validation
6. **Infinite loops:** Deduplication and consolidation must terminate
7. **Storage without interface:** Engine code must never touch files directly

---

## 23. Security / Safety Boundaries

### 23.1 Data Safety

- Memory files are local only, never transmitted
- No external API calls for memory operations
- Embeddings generated locally via Ollama
- No user data leaves the machine

### 23.2 Input Safety

- User input is sanitized before processing
- No code injection through memory values
- File paths are never constructed from user input
- Memory IDs are system-generated, not user-supplied

### 23.3 Operational Safety

- Destructive operations (deactivation) require explicit user intent signals
- No automatic deletion of memories
- History is always preserved
- Backup before migration

---

## 24. Migration Considerations

### 24.1 From Current v0.6

The existing `long_term_memory.json` contains memories in the old format. Migration:

1. Read old format records
2. Map fields to new Memory model
3. Infer entity references (all current memories are self-scoped)
4. Derive domain tags from old category field
5. Generate missing embeddings
6. Validate all migrated records
7. Write new format (keeping old file as backup)

### 24.2 Migration Safety

- Migration is idempotent (running twice produces same result)
- Old file is backed up before any changes
- Migration produces a dry-run report before applying
- Any validation failure stops migration and reports the issue

---

## 25. Future Extensibility

### 25.1 Adding New Domains

To add a new domain (e.g., "health"):

1. Add domain tag to config
2. Add domain hint to embedding recipe
3. Add domain to query intent prototypes
4. Add domain to retrieval filters
5. Add domain to answer formatting
6. No schema changes, no engine changes

### 25.2 Adding New Entity Types

To support new entity types (e.g., "location"):

1. Add entity_type to EntityRef valid values
2. Add resolution patterns to entity_resolution
3. No storage changes, no engine changes

### 25.3 Adding New Retrieval Signals

To add a new retrieval signal (e.g., recency scoring):

1. Add scorer to retrieval pipeline
2. Add weight to config
3. No storage changes, no schema changes

### 25.4 Storage Backend Swap

To swap JSON → SQLite:

1. Implement MemoryStore protocol for SQLite
2. Update config to point to new backend
3. No engine code changes
4. Run migration tool

---

## 26. Architectural Risks

### 26.1 High Risk

| Risk | Mitigation |
|------|-----------|
| LLM extraction quality varies | Dual extraction (LLM + pattern fallback); strict validation |
| Embedding model too small for paraphrases | Hybrid retrieval; lexical fallback; low threshold |
| Entity resolution ambiguity | Hierarchical resolution; unresolved entities blocked |
| Profile synthesis hallucination | Evidence-only synthesis; structured format; no LLM in synthesis path |

### 26.2 Medium Risk

| Risk | Mitigation |
|------|-----------|
| Single JSON file performance | Interface abstraction enables SQLite swap |
| Regex pattern maintenance | Patterns are fallback only; LLM is primary extractor |
| Temporal hint parsing errors | Conservative parsing; invalid hints → persistent |
| Cross-session embedding inconsistency | Embedding version tracking; re-embed on version bump |

### 26.3 Low Risk

| Risk | Mitigation |
|------|-----------|
| Memory file corruption | Atomic writes; backup on corruption |
| Storage interface contract drift | Protocol-based; type checking at dev time |
| Test coverage gaps | Regression test classes for known failures |

---

## 27. Implementation Sequence for Future Sessions

### Session 1: Foundation
- Define data models (Memory, Entity, Fact, QueryIntent, etc.)
- Implement storage interface + JSON backend
- Implement config
- Implement embedding generation
- Unit tests for models and storage

### Session 2: Write Pipeline Core
- Implement message understanding
- Implement entity resolution (write-side)
- Implement validation
- Implement lifecycle state machine
- Unit tests for each stage

### Session 3: Fact Extraction
- Implement LLM-constrained extraction
- Implement pattern fallback extraction
- Implement structured output schema + validation
- Integration tests for extraction pipeline

### Session 4: Read Pipeline Core
- Implement query understanding
- Implement entity resolution (read-side)
- Implement hybrid retrieval
- Implement evidence validation
- Unit tests for each stage

### Session 5: Answer + Synthesis
- Implement answer generation
- Implement profile synthesis
- Implement cardinality handling
- End-to-end tests for read pipeline

### Session 6: Integration + CLI
- Wire write + read pipelines in main.py
- Implement debug trace
- Implement memory events
- Real CLI testing

### Session 7: Regression + Hardening
- Run all regression test classes
- Fix identified issues
- Performance testing with large memory sets
- Edge case hardening

### Session 8: Migration + Polish
- Implement migration from old format
- Clean up git hygiene (update .gitignore)
- Documentation
- Final review

---

## 28. Definition of Done for v0.6

ALLI v0.6 is complete when:

### Functional Requirements
- [ ] Can store facts from natural language statements (including paraphrases and grammar mistakes)
- [ ] Can update existing facts when user provides corrections
- [ ] Can deactivate facts when user requests deletion (history preserved)
- [ ] Can detect and handle contradictions (polarity reversal, explicit correction)
- [ ] Can retrieve relevant facts for queries with different wording than stored statements
- [ ] Can distinguish personal queries from general knowledge questions
- [ ] Can handle singular vs. plural queries appropriately
- [ ] Can synthesize information across domains for "introduction" queries
- [ ] Can say "I don't know" when evidence is insufficient
- [ ] Can resolve entities generically (self, third-party via possessive patterns)
- [ ] Can distinguish current vs. historical facts
- [ ] Can handle temporary/time-bound facts
- [ ] Can represent and handle uncertainty
- [ ] Can preserve provenance for every memory

### Technical Requirements
- [ ] Write and read pipelines are separate modules
- [ ] Storage is abstracted via interface (JSON backend works, SQLite-ready)
- [ ] LLM output is validated before touching memory
- [ ] Embeddings are one retrieval signal, not the sole basis
- [ ] Every write produces a memory event
- [ ] Debug trace available for every operation
- [ ] No runtime files tracked in git
- [ ] No circular module dependencies

### Quality Requirements
- [ ] 200+ unit tests passing
- [ ] 50+ integration tests passing
- [ ] 30+ end-to-end tests passing
- [ ] All regression test classes passing
- [ ] No known data loss scenarios
- [ ] No hallucination in answers
- [ ] Response time < 2 seconds for typical queries

---

*End of Architecture Document — Session 0*
