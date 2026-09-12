# ALLI v0.6 — Final Architecture Specification

**Session 0 — Architecture Freeze**
**Date:** 2026-09-08
**Status:** Implementation-Ready — Pending Review

---

## 1. Purpose

This document is the final, implementation-ready architecture specification for ALLI v0.6: a grounded personal memory intelligence system.

ALLI v0.6 understands what the user says, determines what should be retained, stores information with structure/time/provenance, updates it when reality changes, retrieves relevant evidence, distinguishes unknown from misunderstood, and produces natural answers without inventing personal information.

---

## 2. Scope

### 2.1 In Scope

- Grounded personal memory intelligence
- Entity-scoped facts with temporal validity
- Hybrid retrieval with evidence grounding
- Natural answer generation from validated evidence
- Conversation-aware personal/general routing
- Entity discovery and relationship resolution
- Multi-entity aggregation
- Memory admission control
- Decision receipts for auditability

### 2.2 Out of Scope

- Full Knowledge Base
- File Intelligence
- Tool System
- PC automation
- Web Intelligence
- Study Assistant
- Coding Agent
- Content Intelligence
- Content Creation Agent
- Video Workflow Agent
- Creative/Motion Assistant
- Web Development Agent
- Mobile Control
- Voice ALLI
- Computer Vision
- Computer Operator
- Autonomous Workflows
- Huge ontology
- Multi-user social graph
- Fancy ML importance model
- "Remember everything then prune later"
- Autonomous workflow executor

---

## 3. Official Roadmap Reference

The official ALLI v0.6 roadmap is EXACTLY:

| Session | Name |
|---------|------|
| S0 | Architecture Specification |
| S1 | Foundation / Data Model |
| S2 | Memory Understanding / Extraction |
| S3 | Memory Lifecycle |
| S4 | Persistent Storage |
| S5 | Query Understanding |
| S6 | Hybrid Retrieval |
| S7 | Evidence Grounding |
| S8 | Answer Engine |
| S9 | Memory Intelligence |
| S10 | Real Conversation Tests |
| S11 | Adversarial / Stress Test |
| S12 | Persistence + Long Conversation |
| S13 | Performance / Cleanup |
| S14 | Final v0.6 Certification |

**DO NOT CHANGE THIS STRUCTURE.**

---

## 4. Core Principles

1. **UNDERSTANDING:** Every message is classified by intent and information content before any processing.
2. **GROUNDING:** Answers come ONLY from validated evidence, never from LLM invention.
3. **CORRECTNESS:** Contradictions are detected and resolved, not ignored.
4. **ENTITY ISOLATION:** Facts about one entity never contaminate another.
5. **TEMPORAL CORRECTNESS:** Current vs. historical vs. temporary facts are never flattened.
6. **EXPLAINABILITY:** Every admission decision is auditable via decision receipts.
7. **PERSISTENCE:** Memory survives restarts; history is preserved.
8. **GENERALIZATION:** All mechanisms handle arbitrary paraphrases, grammar mistakes, and slang. No sentence-specific logic.

---

## 5. High-Level Architecture

```
USER
  |
  v
FAST TRIAGE
  |
  v
UNDERSTANDING + INTENT
  |
  +---------------------------+
  |                           |
MESSAGE                     QUESTION
  |                           |
  v                           v
COMPLETENESS              QUERY UNDERSTANDING
  |                           |
  v                           v
EXTRACTION               ENTITY RESOLUTION
  |                           |
  v                           v
NORMALIZATION             PERSONAL / GENERAL
  |                          ROUTING
  v                            |
ENTITY                         v
  |                       RETRIEVAL
  v                            |
TEMPORAL                       v
  |                     STRUCTURED
  v                     KEYWORD/BM25
CONFIDENCE               SEMANTIC
  |                     RELATIONSHIP
  v                     TEMPORAL/METADATA
A-MAC                          |
  |                            v
  +-------------------> RRF FUSION
  |                            |
  v                            v
STORE / TEMP /           DETERMINISTIC
IGNORE / PENDING           RERANK
                               |
                               v
                        RETRIEVAL BUDGET
                               |
                               v
                        EVIDENCE GROUNDING
                               |
                               v
                              QWEN
                               |
                               v
                    PERSONAL CLAIM VERIFICATION
                               |
                               v
                        NATURAL ANSWER
```

---

## 6. Memory Model

### 6.1 Canonical Memory Representation

```
Memory {
    id:                 string          # unique identifier (memory_NNN)
    entity:             EntityRef       # who/this is about
    fact:               Fact            # the knowledge content
    domain:             string          # derived domain tag
    lifecycle:          Lifecycle       # state machine: proposed → active → inactive → expired
    provenance:         Provenance      # origin, extraction method, confidence
    temporal:           TemporalScope   # valid_from, valid_until, is_persistent
    confidence:         float [0.0-1.0]
    importance:         string          # critical | high | medium | low
    embedding:          list[float]     # semantic vector (nullable)
    embedding_version:  int
    schema_version:     int
    created_at:         ISO timestamp
    updated_at:         ISO timestamp
}
```

### 6.2 EntityRef

```
EntityRef {
    name:           string          # canonical entity name (lowercase, normalized)
    entity_type:    string          # "self" | "person" | "organization" | "concept" | "unknown"
    relation:       string?         # relationship to user
    entity_id:      string?         # link to entity record
}
```

### 6.3 Fact

```
Fact {
    fact_type:      string          # semantic type (e.g., "learning", "preference", "goal")
    subject:        string          # normalized topic key for matching
    predicate:      string          # semantic relation
    value:          string          # compact value
    content:        string          # natural language statement
    polarity:       string?         # "positive" | "negative" | null
}
```

### 6.4 Derived Domain Tags

| Domain | Contains |
|--------|----------|
| identity | Personal identification |
| preference | Likes and dislikes |
| knowledge | Skills and competencies |
| learning | Active learning/studying |
| goal | Aspirations and targets |
| ambition | Professional aspirations |
| education | Formal education |
| project | Active projects |
| possession | Owned items |
| relationship | Facts about other entities |
| general | Uncategorized |

---

## 7. Entity Model

### 7.1 Entity Types

| Type | Description |
|------|-------------|
| self | The user |
| person | Other people |
| organization | Companies, institutions |
| concept | Abstract concepts |
| unknown | Unresolved references |

### 7.2 Entity Resolution Strategy

Resolution is hierarchical and purely syntactic:

1. **Self-identification:** "I am X", "My name is X" → entity = self
2. **Possessive third-party:** "Mom's favorite color" → entity = mom (person)
3. **Named third-party:** "Rahul likes Java" → entity = rahul (person)
4. **Self-reference:** "I like X" → entity = self
5. **Unknown pronoun:** "their favorite" → entity = unknown (BLOCKED)
6. **Default:** → entity = self

### 7.3 Entity Name Normalization

- Lowercased
- Whitespace-trimmed
- Common relationship words mapped to canonical types (mom→person, friend→person)

No hardcoded name list. Resolution derives from syntactic role.

### 7.4 Entity Confidence

| Resolution Method | Confidence |
|-------------------|------------|
| self_identification | 1.0 |
| possessive_third_party | 0.9 |
| named_third_party | 0.8 |
| self_reference | 0.95 |
| default_self | 0.7 |

---

## 8. Fact Model

### 8.1 Single-Value vs Multi-Value Facts

**SINGLE-CURRENT-VALUE facts:**
- Current phone
- Current city
- Favourite colour

When changed:
- Old fact → inactive/superseded
- New fact → active

**MULTI-VALUE facts:**
- Likes
- Dislikes
- Learning
- Skills
- Projects

Adding one does NOT automatically invalidate others.
Negating one modifies that specific fact.

### 8.2 Semantic Distinctions

The system preserves these distinctions:

| Concept A | Concept B | Are they equal? |
|-----------|-----------|-----------------|
| LEARNING | STUDYING | Similar but distinct relations |
| USING | OWNING | Different relations |
| LIKING | DISLIKING | Opposite polarities |
| GOAL | PROJECT | Different domains |

---

## 9. Relationship Model

### 9.1 Relationship Structure

```
Relationship {
    id:             string
    entity_a:       EntityRef
    entity_b:       EntityRef
    relation_type:  string     # "brother_of", "parent_of", "friend_of", etc.
    confidence:     float
    provenance:     Provenance
    temporal:       TemporalScope
    lifecycle:      Lifecycle
}
```

### 9.2 Relationship Extraction

Relationships are extracted from:
- Possessive patterns ("my brother Devsai")
- Kinship terms
- Explicit statements ("He is my brother")

### 9.3 Entity Discovery

When an unknown entity is mentioned:
1. ALLI asks: "Who is Devsai?"
2. User responds: "He is my brother"
3. System establishes: Devsai → brother_of → user
4. Subsequent facts about Devsai are stored with entity = devsai

---

## 10. Episode Model

```
Episode {
    id:                     string
    user_message:           string
    timestamp:              ISO timestamp
    entities_involved:      list[EntityRef]
    extracted_facts:        list[string]    # memory IDs
    extracted_relationships: list[string]
    memory_events:          list[string]    # event IDs
    provenance:             Provenance
    metadata:               dict
}
```

---

## 11. Temporal Model

### 11.1 TemporalScope

```
TemporalScope {
    valid_from:     ISO timestamp
    valid_until:    ISO timestamp?    # null = persistent
    is_persistent:  bool
    temporal_hint:  string?           # natural language hint
}
```

### 11.2 Temporal Categories

| Category | valid_until | is_persistent |
|----------|-------------|---------------|
| Persistent fact | null | true |
| Temporary state | computed from hint | false |
| Task/Event | computed from hint | false |
| Historical fact | when superseded | true (historical) |
| Uncertain intention | null | false (low confidence) |

### 11.3 Temporal Hint Resolution

- "tomorrow" → tomorrow 23:59:59
- "next week" → end of next week
- "in 3 days" → now + 3 days
- "this Friday" → upcoming Friday 23:59:59
- ISO date → direct parse

### 11.4 Temporal Interpretation

| Expression | Interpretation |
|------------|----------------|
| "used to" | historical, inactive |
| "anymore" | superseded/negated |
| "currently" | current, active |
| "tomorrow" | future/temporary |
| "yesterday" | historical |
| "stopped" / "ceased" | inactive, superseded |
| "might" / "maybe" | uncertain, low confidence |

### 11.5 Expired Memory Handling

Expired memories are not returned by default. They can be explicitly requested via temporal queries. Expired memories remain in storage for history.

---

## 12. Provenance

### 12.1 Provenance Sources

| Source | Description |
|--------|-------------|
| USER_ASSERTED | Direct user statement |
| SYSTEM_INFERRED | System inference from other facts |
| MODEL_GENERATED | LLM-generated content (NOT stored as personal fact) |

### 12.2 Provenance Rules

- Only USER_ASSERTED facts become explicit personal memories
- MODEL_GENERATED content is NEVER silently converted to user memory
- SYSTEM_INFERRED facts require explicit evidence chain

---

## 13. Fast Triage

### 13.1 Purpose

General pre-memory triage layer that identifies:

- Greetings
- Acknowledgements
- Low-information conversational statements
- Obvious conversational requests
- Memory-bearing statements
- Questions
- General knowledge questions

### 13.2 Classification Rules

**NOT a phrase cache.** Uses general intent/information-content classification.

Mixed messages must still be processed:

- "Okay, remember my exam is tomorrow." → MUST process (contains memory instruction + temporal)
- "Hi, I like Python." → MUST process (contains memory content)
- "Hello!" → GREETING (no memory)
- "Okay." → ACKNOWLEDGEMENT (no memory)
- "Can you help me solve this problem?" → CONVERSATIONAL (no memory)

### 13.3 Triage Output

```
TriageResult {
    category:           string   # "greeting" | "acknowledgement" | "conversational" |
                                 # "memory_statement" | "question" | "general_knowledge" |
                                 # "mixed" | "malformed"
    has_memory_content: bool
    has_instruction:    bool     # "remember", "forget", etc.
    confidence:         float
}
```

---

## 14. Understanding

### 14.1 Message Understanding

```
MessageUnderstanding {
    message_type:       MessageType   # STATEMENT | QUESTION | COMMAND | GREETING
    is_temporary:       bool
    is_uncertain:       bool
    is_third_party:     bool
    is_correction:      bool
    is_negation:        bool
    is_deletion:        bool
    temporal_hint:      string?
    confidence:         float
}
```

### 14.2 Query Understanding

```
StructuredQuery {
    subject:            string          # who the query is about
    relation:           string          # the type of relationship sought
    object:             string          # the specific thing asked about
    state:              string          # "current" | "historical" | "future"
    time:               string?         # temporal qualifier
    cardinality:        string          # "one" | "many" | "existence" | "count"
    entity:             string?         # resolved entity
    domain:             string          # knowledge domain
    constraints:        list[string]    # additional filters
    negated:            bool
    uncertain:          bool
    is_personal:        bool
    is_general:         bool
    raw_query:          string
    normalized_query:   string
}
```

---

## 15. Completeness Gate

### 15.1 Three Situations

| Situation | Condition | Action |
|-----------|-----------|--------|
| UNDERSTOOD + EVIDENCE | Query understood, evidence exists | ANSWER |
| UNDERSTOOD + NO EVIDENCE | Query understood, no evidence | UNKNOWN |
| NOT UNDERSTOOD | Query ambiguous or malformed | CLARIFY |

### 15.2 Rules

- Do NOT convert misunderstanding into UNKNOWN
- Do NOT convert UNKNOWN into guessing
- Ask at most ONE clarification question per turn
- Malformed input → do not store garbage

### 15.3 Examples

| Input | Action |
|-------|--------|
| "What is what?" | CLARIFY |
| "I don't like like" | REJECT (malformed) |
| "I like don't" | REJECT (malformed) |
| "Xyzzor" | CLARIFY (unknown entity) |

---

## 16. Normalization

### 16.1 Lexical Normalization

- Contraction expansion: "I'm" → "I am"
- Slang removal: "bro" → ""
- Typo correction: "phne" → "phone"
- Whitespace normalization
- Case normalization

### 16.2 Grammar Normalization

The system handles imperfect human language through general mechanisms:

- Pattern-based extraction with fallback
- Lexical similarity scoring
- Semantic embedding comparison

No exact string patches. Same mechanism generalizes to unseen variations.

---

## 17. Entity Resolution

### 17.1 Write-Side Resolution

Determines which entity a user statement is about.

### 17.2 Read-Side Resolution

Determines which entity a query refers to.

### 17.3 Resolution Priority

1. Self-identification
2. Possessive third-party
3. Named third-party
4. Self-reference
5. Unknown pronoun → BLOCKED
6. Default → self

### 17.4 Unknown Entity Handling

When an unknown entity is encountered:
- Store with entity = "unknown"
- Prompt user for clarification
- On response, establish relationship and update entity

---

## 18. A-MAC (Memory Admission Controller)

### 18.1 Purpose

Decides PER EXTRACTED CANDIDATE whether to STORE, TEMP, IGNORE, or PENDING.

Operates PER CANDIDATE, not once per message.

### 18.2 Admission Outcomes

| Outcome | Meaning |
|---------|---------|
| STORE | Durable personal fact, stored with lifecycle |
| TEMP | Temporary context, stored with expiry |
| IGNORE | Not stored, not rescanned across sessions |
| PENDING | Needs resolution before final decision |

### 18.3 Decision Factors (Deterministic)

A-MAC uses these auditable factors:

| Factor | Weight | Description |
|--------|--------|-------------|
| explicit_remember | 0.30 | User said "remember", "store", etc. |
| completeness | 0.15 | Fact has enough detail |
| factual_confidence | 0.10 | LLM extraction confidence |
| semantic_novelty | 0.10 | Not duplicate of existing |
| category_importance | 0.10 | Domain importance (identity=high, hobby=low) |
| context_relevance | 0.10 | Active conversation context |
| temporal_characteristics | 0.05 | Temporary vs persistent |
| provenance | 0.10 | User-asserted vs inferred |

### 18.4 Decision Precedence

1. Explicit "remember" instruction → STORE (if confidence ≥ 0.5)
2. Explicit "forget" instruction → DEACTIVATE
3. Garbage/malformed → IGNORE
4. Uncertain/hedged → TEMP (low confidence)
5. Complete factual statement → STORE
6. Incomplete/ambiguous → PENDING

### 18.5 Example

Message: "I am building ALLI, I have a 16GB PC, and I have a blue shirt."

| Candidate | Decision | Reason |
|-----------|----------|--------|
| building ALLI | STORE | complete fact, active project |
| 16GB PC | STORE | complete possession fact |
| blue shirt | TEMP | low durable importance |

---

## 19. Admission Outcomes

### 19.1 STORE

- Durable personal fact
- Active lifecycle state
- Full provenance tracking
- Embedded for retrieval

### 19.2 TEMP

- Temporary context
- Active lifecycle with expiry
- Not promoted automatically
- Expires after temporal window

### 19.3 IGNORE

- Not stored
- Not rescanned across sessions
- Can be promoted only through:
  - User explicitly says "remember"
  - User restates the information
  - Contextual promotion trigger

### 19.4 PENDING

- Reason documented
- Confidence recorded
- Unresolved information noted
- Resolution trigger defined
- Expiry behavior defined

---

## 20. Decision Receipts

### 20.1 Structure

```
DecisionReceipt {
    decision:           string   # "STORE" | "TEMP" | "IGNORE" | "PENDING"
    candidate:          string   # the fact content
    entity:             string   # resolved entity
    factors:            dict     # factor_name → score
    reason:             string   # human-readable explanation
    confidence:         float    # overall confidence
    timestamp:          ISO time
}
```

### 20.2 Example Receipts

```
Decision: IGNORE
Reason: low durable importance, no active contextual relevance, no explicit remember request

Decision: STORE
Reason: explicit remember instruction, high confidence, complete fact, personally useful category
```

---

## 21. Lifecycle

### 21.1 States

```
proposed → active → inactive → expired
```

### 21.2 Operations

| Operation | Description |
|-----------|-------------|
| ADD | New fact → active |
| UPDATE | Existing fact found → supersede old, activate new |
| MERGE | Compatible addition → new fact linked to existing |
| SUPERSEDE | Explicit correction → old inactive, new active |
| INVALIDATE | User-initiated deletion → deactivate |
| IGNORE | Duplicate/unchanged → no state change |

### 21.3 Supersession Rules

When a new fact contradicts an existing one:
1. Old fact → inactive
2. Old fact records `superseded_by: <new_id>`
3. New fact → active
4. Both records persist

### 21.4 Promotion Rules

Promotion from IGNORE/TEMP to STORE occurs only through:
- User explicitly says "remember"
- User restates the information
- Approved contextual promotion trigger
- Optional user confirmation

Promotion preserves provenance/history and avoids duplicates.

---

## 22. Contradiction Handling

### 22.1 Detection

Contradiction detection uses:
- Semantic similarity of fact content
- Entity scope
- Polarity
- Temporal context
- Explicit correction signals

NOT exact string matching.

### 22.2 Resolution

| Type | Action |
|------|--------|
| Polarity reversal | Supersede old, activate new |
| Value change | Supersede old, activate new |
| Uncertain vs certain | Keep certain, note uncertainty |
| Compatible addition | Merge (new fact linked) |

---

## 23. Query Understanding

### 23.1 Question Types

| Type | Example |
|------|---------|
| Singular personal | "What is my goal?" |
| Plural personal | "What are my ambitions?" |
| Existence | "Do I like Python?" |
| Count | "How many languages do I know?" |
| Reverse entity | "Who likes sweets?" |
| Temporal | "What was my old phone?" |
| Open-ended | "What am I interested in?" |
| General knowledge | "What is Python?" |
| Ambiguous | "What am I leaving?" |
| Malformed | "I don't like like" |

### 23.2 Personal vs General Routing

**PERSONAL if:**
- Contains personal pronouns (I, my, me, myself)
- Contains domain-specific keywords (goal, skill, preference)
- Matches memory-read prototype
- Contains "about me"

**GENERAL if:**
- Asks for definitions, explanations
- Contains no personal pronouns
- Matches general-knowledge prototypes
- Is clearly about external topics

---

## 24. Personal/General Routing

### 24.1 Strict Boundary

| Query | Route |
|-------|-------|
| "What is Python?" | GENERAL |
| "What is Java?" | GENERAL |
| "What phone do I use?" | PERSONAL |
| "What am I learning?" | PERSONAL |
| "What Python-related things am I learning?" | PERSONAL |

### 24.2 Rules

- General knowledge questions must NOT be answered using unrelated personal memory
- Personal memory must NOT contaminate general explanations
- Conversational requests (help, assistance) → CONVERSATIONAL path

---

## 25. Hybrid Retrieval

### 25.1 Retrieval Channels

| Channel | Description |
|---------|-------------|
| STRUCTURED | Entity/relation/object matching |
| KEYWORD/BM25 | Token overlap scoring |
| SEMANTIC | Embedding cosine similarity |
| RELATIONSHIP | Entity relationship matching |
| TEMPORAL/METADATA | Lifecycle state and time filtering |

### 25.2 Candidate Retrieval

- Entity filter → scope to relevant entity
- Structural filter → scope to relevant domain/relation
- Temporal filter → scope to relevant time range
- Lexical match → token overlap scoring
- Semantic match → embedding similarity

---

## 26. RRF (Reciprocal Rank Fusion)

### 26.1 Formula

```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```

where:
- `k` = 60 (constant)
- `rank_i(d)` = rank of document d in channel i

### 26.2 Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| k | 60 | RRF constant |
| candidate_retrieval_K | 50 | Candidates per channel |
| reranking_K | 20 | After fusion |
| final_evidence_budget | 5 | Sent to Qwen |

### 26.3 Channel Weights

| Channel | Weight |
|---------|--------|
| structured | 0.30 |
| keyword | 0.15 |
| semantic | 0.35 |
| relationship | 0.10 |
| temporal | 0.10 |

---

## 27. Deterministic Reranking

### 27.1 Precedence

Semantic similarity does NOT automatically override stronger structured evidence.

Deterministic reranking accounts for:

1. **Relation compatibility** (0.30 weight)
2. **Entity compatibility** (0.25 weight)
3. **Temporal/state compatibility** (0.15 weight)
4. **Query constraints** (0.15 weight)
5. **Semantic/lexical relevance** (0.15 weight)

### 27.2 Bonus Signals

| Signal | Bonus |
|--------|-------|
| entity_match | +0.05 |
| relation_match | +0.05 |
| object_match | +0.05 |
| multi_source (3+) | +0.05 |
| multi_source (2) | +0.03 |

---

## 28. Retrieval Budget

### 28.1 Pipeline

```
broad candidate discovery (50 per channel)
    → precise ranking (top 20 after fusion)
        → small evidence set (5 final)
            → evidence grounding
                → Qwen
```

### 28.2 Budget

- Final evidence set: **5 candidates maximum**
- Never send entire memory database to Qwen
- Qwen receives ONLY accepted evidence

---

## 29. Evidence Grounding

### 29.1 Validation Checks

| Check | Rule |
|-------|------|
| entity | Memory entity matches query entity |
| relation | Memory fact_type compatible with query relation |
| object | Memory content compatible with query object |
| domain | Memory domain compatible with query domain |
| temporal | Temporal state matches query state |
| lifecycle | Memory lifecycle valid |
| negation | Negation preserved |
| uncertainty | Uncertainty preserved |

### 29.2 Evidence Strength

```
strength = (retrieval_score × 0.3)
         + (entity_valid × 0.25)
         + (relation_valid × 0.20)
         + (object_valid × 0.10)
         + (domain_valid × 0.05)
         + (temporal_valid × 0.05)
         + (lifecycle_valid × 0.05)
```

Penalties:
- negated: ×0.5
- uncertain: ×0.7
- inactive/superseded: ×0.6

### 29.3 Status Determination

| Status | Condition |
|--------|-----------|
| SUPPORTED | max_strength ≥ 0.6 AND avg_strength ≥ 0.4 |
| WEAK | max_strength ≥ 0.3 |
| CONFLICTING | Conflicting evidence detected |
| NONE | No accepted evidence |

---

## 30. Qwen Answer Generation

### 30.1 Input

Qwen receives ONLY:
- System prompt with grounding instructions
- The user's question
- Accepted evidence (NOT rejected candidates, NOT full database)

### 30.2 System Prompt Contract

```
You are ALLI, a personal AI assistant.
Answer the user's question using ONLY the evidence provided below.
Rules:
- Answer ONLY from the supplied evidence.
- Do NOT invent, infer, or fabricate personal facts.
- Do NOT use information not present in the evidence.
- If the evidence does not contain the answer, say so clearly.
- Keep answers concise and natural.
- Do not mention evidence scores, metadata, or internal details.
- If a fact is NEGATED, respect the negation.
- If a fact is UNCERTAIN, preserve the uncertainty.
- Always reply in English.
```

### 30.3 Evidence Formatting

```
- I like Python. [entity=user, fact=preference, status=active]
- I don't like Java. [entity=user, fact=preference, status=active, NEGATED]
- My mom likes pink. [entity=mom, fact=preference, status=active]
```

---

## 31. Personal Claim Verification

### 31.1 Process

After Qwen generates a response:

1. Identify atomic personal claims in the answer
2. Verify each claim against evidence
3. Reject unsupported claims
4. If no supported answer remains → UNKNOWN

### 31.2 Example

Evidence: "user likes Python"

Qwen says: "You like Python and use it professionally."

Verification:
- "You like Python" → SUPPORTED by evidence
- "use it professionally" → NOT SUPPORTED by evidence → REJECTED

Final answer: "You like Python."

---

## 32. Natural Answer Generation

### 32.1 Answer Templates

| Query Type | Template |
|------------|----------|
| "What is my goal?" | "Your goal is {value}." |
| "What phone do I use?" | "You use {value}." |
| "Do I like Python?" | "Yes, you like Python." / "No, you don't like Python." |
| "Who likes sweets?" | "{Entity} likes sweets." |
| Unknown | "I don't have that information stored." |
| Clarification | One clear clarification question |

### 32.2 Relationship-Aware Answers

If entity has known relationship to user:
- Use relationship naturally: "Your brother likes sweets."
- NOT: "Devsai likes sweets." (unless relationship unknown)

### 32.3 Multi-Entity Aggregation

"I like sweets."
"My brother likes sweets."
→ "Who likes sweets?"
→ "You and your brother like sweets."

---

## 33. Conversational Path

### 33.1 Purpose

Handle low-information conversation and general requests without forcing personal memory retrieval.

### 33.2 Conversational Types

| Type | Example | Response |
|------|---------|----------|
| Greeting | "Hi!" | "Hello!" |
| Acknowledgement | "Okay." | (acknowledge) |
| Assistance request | "Can you help me solve this problem?" | "Yeah, sure. Give me the problem and we'll solve it together." |
| General conversation | "I am fine." | Conversational response |

### 33.3 Coexistence

The conversational path coexists with the memory pipeline:
- Triage identifies conversational messages
- They are routed to Ollama without memory retrieval
- No memory operations are triggered

---

## 34. Persistence

### 34.1 Storage Backend

**SQLite** is the structured memory persistence layer.

### 34.2 Tables

| Table | Purpose |
|-------|---------|
| entities | Unique entities (people, devices, concepts) |
| memories | All facts/knowledge with full lifecycle |
| relationships | Connections between entities |
| memory_events | Audit trail for all operations |
| episodes | User interaction records |
| schema_info | Schema version tracking |

### 34.3 Transaction Boundaries

- All writes within transactions
- Atomic operations for supersession/deactivation
- WAL mode for concurrent reads
- Foreign keys enabled

### 34.4 Schema Versioning

- Schema version tracked in `schema_info` table
- Migration scripts for version upgrades
- Backward compatibility maintained

### 34.5 Data Safety

- Previous v0.6 data protected
- No production memory modification during architecture sessions
- Backup before migration

---

## 35. Embeddings / Index Metadata

### 35.1 Required Metadata

| Field | Description |
|-------|-------------|
| schema_version | Schema version |
| embedding_model | Model name (e.g., "all-minilm:latest") |
| embedding_dimension | Vector dimension (e.g., 384) |
| index_version | Index version |
| created_at | Creation timestamp |
| updated_at | Last update timestamp |

### 35.2 Mismatch Behavior

- Detect mismatch between stored and expected versions
- Re-embed where required
- Invalidate/flag incompatible vectors
- Never silently reuse incompatible embeddings

---

## 36. Retention

### 36.1 Retention Rules

| Factor | Influence |
|--------|-----------|
| Recency | Affects retrieval utility |
| Access frequency | Affects retrieval utility |
| Lifecycle state | Determines visibility |
| Temporal validity | Determines expiration |

### 36.2 Important Rule

Recency/access frequency must NOT determine truth.

A durable memory does NOT become false merely because it is old or rarely accessed.

---

## 37. Module Boundaries

### 37.1 Module Map

```
src/
  main.py                    # CLI entry point
  memory/
    config.py                # Constants, paths
    embeddings.py            # Embedding generation
    
    # WRITE PIPELINE
    understanding.py         # Message classification
    entity_resolution.py     # Entity extraction
    extraction.py            # Fact extraction
    operations.py            # Lifecycle operations
    validation.py            # Structural validation
    
    # READ PIPELINE
    query_understanding.py   # Query parsing
    hybrid_retriever.py      # Multi-channel retrieval
    evidence_grounding.py    # Evidence validation
    answer_engine.py         # Answer generation
    
    # STORAGE
    sqlite_store.py          # SQLite persistence
    adapter.py               # Format bridge
    
    # INTEGRATION
    memory_engine.py         # Unified API
    
    # SHARED
    models.py                # Data classes
    storage.py               # Storage interface
    ids.py                   # ID generation
```

### 37.2 Responsibility per Module

| Module | Responsibility |
|--------|----------------|
| understanding.py | Classify message type, detect signals |
| entity_resolution.py | Extract and resolve entities |
| extraction.py | Extract structured operations |
| operations.py | Lifecycle decisions, contradiction handling |
| validation.py | Structural validation |
| query_understanding.py | Parse queries into StructuredQuery |
| hybrid_retriever.py | Multi-channel retrieval + fusion |
| evidence_grounding.py | Validate evidence correctness |
| answer_engine.py | Orchestrate S5→S6→S7→Qwen |
| sqlite_store.py | SQLite persistence |
| adapter.py | Bridge foundation ↔ old format |
| memory_engine.py | Unified remember()/recall() API |

### 37.3 Dependency Direction

```
main.py → memory_engine.py → write_pipeline → understanding → extraction → operations → storage
       → answer_engine → query_understanding → hybrid_retriever → evidence_grounding

storage ← models (used by everything)
ids ← operations, storage
embeddings ← hybrid_retriever, answer_engine
```

No circular dependencies.

---

## 38. APIs

### 38.1 MemoryEngine Public API

```python
class MemoryEngine:
    def remember(message: str) -> RememberResult
    def recall(question: str) -> str
    def recall_detailed(question: str) -> AnswerResult
    def close() -> None
```

### 38.2 WritePipeline API

```python
class WritePipeline:
    def process(message: str) -> WriteResult
```

### 38.3 AnswerEngine API

```python
class AnswerEngine:
    def answer(question: str) -> AnswerResult
```

### 38.4 HybridRetriever API

```python
class HybridRetriever:
    def retrieve(query: StructuredQuery) -> HybridRetrievalResult
```

### 38.5 EvidenceGrounder API

```python
class EvidenceGrounder:
    def ground(query: StructuredQuery, retrieval: HybridRetrievalResult) -> EvidenceAssessment
```

---

## 39. Testing Architecture

### 39.1 Test Layers

| Layer | Scope | Count Target |
|-------|-------|--------------|
| Unit | Individual functions, pure logic | 200+ |
| Integration | Module interactions, pipeline stages | 50+ |
| Behavioral | Full write/read cycles | 30+ |
| Adversarial | Edge cases, stress tests | 20+ |
| Regression | Known failure classes | 10+ |
| Persistence | Cross-session data survival | 5+ |
| Real CLI | python3 src/main.py | 10+ scenarios |
| Real Ollama/Qwen | Live LLM integration | 10+ scenarios |
| Real Embeddings | Live embedding generation | 5+ scenarios |

### 39.2 Test Data Strategy

Tests MUST NOT rely on exact example sentences. Each test class includes:
- Canonical form
- Paraphrases
- Broken grammar
- Slang
- Short form
- Long form
- Multiple facts
- Corrections
- Contradictions
- Uncertainty
- Temporary statements
- Historical statements
- Entity ambiguity
- Singular/plural
- Unknown queries
- General knowledge
- Synthesis

### 39.3 Test Execution

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Behavioral tests
pytest tests/behavioral/ -v

# All tests
pytest tests/ -v

# Specific regression class
pytest tests/regression/ -v -k "TestClassName"
```

### 39.4 PASS Criteria

PASS requires actual execution/evidence. Never fabricate results.

If a path is not connected yet: NOT TESTABLE.

---

## 40. Frozen Certification Matrix

### 40.1 MEMORY

| Input | Expected |
|-------|----------|
| "I use Oppo A53." | STORE |
| "I switched to Samsung S24." | SUPERSEDE |
| "What phone do I use?" | "You use a Samsung S24." |
| "What was my old phone?" | "You used to have an Oppo A53." |

### 40.2 PREFERENCES

| Input | Expected |
|-------|----------|
| "I like Python." | STORE (positive) |
| "I like Java." | STORE (positive) |
| "I don't like Java anymore." | SUPERSEDE (negative) |
| "Do I like Python?" | "Yes, you like Python." |
| "Do I like Java?" | "No, you don't like Java." |

### 40.3 GOALS

| Input | Expected |
|-------|----------|
| "What is my goal?" | "Your goal is ..." |
| "What are my ambitions?" | List of ambitions |

### 40.4 STUDYING / LEARNING

| Input | Expected |
|-------|----------|
| "What am I studying?" | Studying facts |
| "What am I learning?" | Learning facts |

### 40.5 UNKNOWN

| Input | Expected |
|-------|----------|
| "What's my favourite movie?" | "I don't have that information stored." |

### 40.6 GENERAL

| Input | Expected |
|-------|----------|
| "What is Python?" | General knowledge (not memory) |

### 40.7 ENTITY

| Input | Expected |
|-------|----------|
| "Devsai likes sweets." | ASK "Who is Devsai?" |
| User says "brother" | STORE relationship |
| "Who likes sweets?" | "Your brother likes sweets." |

### 40.8 MULTI-ENTITY

| Input | Expected |
|-------|----------|
| "I like sweets." | STORE |
| "My brother likes sweets." | STORE |
| "Who likes sweets?" | "You and your brother like sweets." |

### 40.9 AMBIGUITY

| Input | Expected |
|-------|----------|
| "What is what?" | CLARIFY |
| "What am I leaving?" | CLARIFY (insufficient confidence) |

### 40.10 MALFORMED

| Input | Expected |
|-------|----------|
| "I don't like like." | REJECT |
| "I like don't." | REJECT |

### 40.11 TYPO

| Input | Expected |
|-------|----------|
| "wat phne do i hav" | RECOVER if confidence sufficient, else CLARIFY |
| "wht clr does mom lyk" | RECOVER if confidence sufficient, else CLARIFY |

### 40.12 FUTURE

| Input | Expected |
|-------|----------|
| "How is my future?" | GROUNDED INFERENCE from stored facts |
| "What will I become in future?" | GROUNDED INFERENCE, NOT prediction as fact |

### 40.13 CONVERSATIONAL

| Input | Expected |
|-------|----------|
| "I am fine." | Conversational response |
| "Can you help me solve this problem?" | "Yeah, sure. Give me the problem..." |

### 40.14 UNKNOWN ENTITY

| Input | Expected |
|-------|----------|
| "Xyzzor" | CLARIFY (unknown entity) |

### 40.15 AMBIGUOUS PRONOUN

| Input | Expected |
|-------|----------|
| "She likes pink." (no known "she") | CLARIFY |

---

## 41. S11 Root-Cause Mapping

### 41.1 Root Causes and Architectural Solutions

| # | S11 Problem | Root Cause | Architectural Solution |
|---|-------------|------------|----------------------|
| 1 | General knowledge leakage | No personal/general boundary | Fast triage + strict routing (§13, §24) |
| 2 | Reverse entity queries | Entity resolution limited | Entity discovery + relationship-aware retrieval (§17, §9) |
| 3 | Open-ended personal questions | No relation/category understanding | StructuredQuery relation extraction (§14) |
| 4 | Semantic retrieval overpowering structured meaning | Pure semantic ranking | Deterministic reranking with precedence (§27) |
| 5 | Severe typo handling | Limited normalization | Lexical normalization + typo correction (§16) |
| 6 | Temporal "stopped/ceased" semantics | Temporal flattening | Explicit temporal interpretation (§11.4) |
| 7 | Direct entity validation | Entity mismatch not caught | Entity validation in evidence grounding (§29.1) |
| 8 | Unsupported personal claims | LLM hallucination | Personal claim verification (§31) |
| 9 | Entity relationship resolution | No relationship graph | Relationship model + forward/reverse retrieval (§9) |
| 10 | Multi-entity aggregation | No aggregation logic | Multi-entity answer generation (§32.3) |
| 11 | Malformed input | No malformed handling | Completeness gate + rejection (§15) |
| 12 | Conversational requests entering memory retrieval | No conversational path | Fast triage + conversational routing (§13, §33) |

### 41.2 General Mechanisms

All solutions are GENERAL mechanisms:
- No sentence-specific patches
- No object-specific logic
- No person-specific rules
- Same mechanism handles arbitrary paraphrases, grammar, slang

---

## 42. Out-of-Scope

Items frozen as outside v0.6:

- Full Knowledge Base
- File Intelligence
- Tool System
- PC automation
- Web Intelligence
- Study Assistant
- Coding Agent
- Content Intelligence
- Content Creation Agent
- Video Workflow Agent
- Creative/Motion Assistant
- Web Development Agent
- Mobile Control
- Voice ALLI
- Computer Vision
- Computer Operator
- Autonomous Workflows
- Huge ontology
- Multi-user social graph
- Fancy ML importance model
- "Remember everything then prune later"
- Autonomous workflow executor

These remain later roadmap work.

---

## 43. Change Control Rules

### 43.1 Architecture Freeze

Once S0 is approved, implementation sessions must follow the frozen architecture.

### 43.2 Change Process

If implementation discovers a genuine architectural defect:

1. STOP
2. Document the issue
3. Propose the minimum architecture correction
4. Obtain review
5. Update architecture
6. Continue

Do not silently redesign during implementation.

### 43.3 Backward Compatibility

Any replacement must have:
- Reason
- Architecture requirement
- Regression protection
- Migration strategy where needed

---

## Appendix A: File Inventory

### A.1 Existing Files (Preserved)

| File | Status |
|------|--------|
| src/main.py | PRESERVED |
| src/memory/config.py | PRESERVED |
| src/memory/embeddings.py | PRESERVED |
| src/memory/store.py | PRESERVED |
| src/memory/manager.py | PRESERVED |
| src/memory/retriever.py | PRESERVED |
| src/memory/query_intent.py | PRESERVED |
| src/memory/answers.py | PRESERVED |
| src/memory/extractor.py | PRESERVED |
| src/memory/foundation/models.py | PRESERVED |
| src/memory/foundation/pipeline.py | PRESERVED |
| src/memory/foundation/understanding.py | PRESERVED |
| src/memory/foundation/entity_resolution.py | PRESERVED |
| src/memory/foundation/extraction.py | PRESERVED |
| src/memory/foundation/operations.py | PRESERVED |
| src/memory/foundation/validation.py | PRESERVED |
| src/memory/foundation/storage.py | PRESERVED |
| src/memory/foundation/sqlite_store.py | PRESERVED |
| src/memory/foundation/adapter.py | PRESERVED |
| src/memory/foundation/query_understanding.py | PRESERVED |
| src/memory/foundation/hybrid_retriever.py | PRESERVED |
| src/memory/foundation/evidence_grounding.py | PRESERVED |
| src/memory/foundation/answer_engine.py | PRESERVED |
| src/memory/foundation/memory_engine.py | PRESERVED |
| memory/long_term_memory.json | PRESERVED |
| memory/conversation.json | PRESERVED |
| memory.json | PRESERVED |

### A.2 Production Memory Safety

| File | Status |
|------|--------|
| memory/long_term_memory.json | SAFE |
| memory/conversation.json | SAFE |
| memory/v06_dev.sqlite | SAFE |

---

## Appendix B: Implementation Sequence

### B.1 Session Mapping

| Session | Focus | Key Deliverables |
|---------|-------|------------------|
| S1 | Foundation / Data Model | Data models, storage interface, config, embeddings |
| S2 | Memory Understanding / Extraction | Message understanding, entity resolution, extraction |
| S3 | Memory Lifecycle | Lifecycle state machine, contradiction detection |
| S4 | Persistent Storage | SQLite persistence, schema, migration |
| S5 | Query Understanding | StructuredQuery, personal/general routing |
| S6 | Hybrid Retrieval | Multi-channel retrieval, RRF fusion |
| S7 | Evidence Grounding | Evidence validation, strength calculation |
| S8 | Answer Engine | Qwen integration, claim verification |
| S9 | Memory Intelligence | MemoryEngine API, integration |
| S10 | Real Conversation Tests | Behavioral testing, CLI verification |
| S11 | Adversarial / Stress Test | Edge cases, stress testing |
| S12 | Persistence + Long Conversation | Cross-session survival |
| S13 | Performance / Cleanup | Optimization, code cleanup |
| S14 | Final v0.6 Certification | Full certification matrix |

---

*End of Architecture Document — Session 0*
