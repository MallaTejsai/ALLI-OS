import json
import os
import re
import shutil
from datetime import datetime, timedelta, timezone

from ollama import chat


# ==========================================
# FILE PATHS AND SETTINGS
# ==========================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_DIR = os.path.join(BASE_DIR, "memory")
CONVERSATION_FILE = os.path.join(MEMORY_DIR, "conversation.json")
LONG_TERM_MEMORY_FILE = os.path.join(MEMORY_DIR, "long_term_memory.json")
CONVERSATION_ARCHIVE_FILE = os.path.join(MEMORY_DIR, "conversation_archive.json")

MODEL_NAME = "qwen2.5:7b"
RECENT_MESSAGE_LIMIT = 12  # Six user/assistant turns.
MAX_RETRIEVED_MEMORIES = 5

# Conversation consolidation settings. Change these values as ALLI grows.
MAX_ACTIVE_CONVERSATION_CHARACTERS = 80_000
MAX_ACTIVE_ESTIMATED_TOKENS = 20_000  # Simple estimate: one token is about four characters.
ARCHIVE_CHUNK_MESSAGE_LIMIT = 20
ACTIVE_MESSAGES_TO_KEEP = 24
MAX_ARCHIVE_STORAGE_BYTES = 5 * 1024 * 1024

# Temporary facts are still supported by the data format, but remain disabled
# by default because deadlines and one-time events should normally be NONE.
STORE_TEMPORARY_MEMORIES = False

SYSTEM_MESSAGE = {
    "role": "system",
    "content": "You are ALLI, a helpful local personal AI assistant. Always reply in English.",
}

VALID_CATEGORIES = {
    "identity", "education", "goal", "interest", "preference",
    "skill", "project", "routine", "other",
}
VALID_IMPORTANCE = {"high", "medium", "low", 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
IMPORTANCE_SCORES = {"high": 9, "medium": 7, "low": 3}

# UPDATE and DELETE are intentionally conservative. A related new fact is still ADD.
CHANGE_SIGNAL_PHRASES = (
    "my goal changed", "i changed my goal", "instead of", "replace ",
    "i no longer want", "i stopped", "i changed from", "forget my previous",
    "my name changed", "i am no longer studying", "i don't want to learn",
    "i do not want to learn", "changed to", "no longer", "anymore",
    "used to", "but now",
)

TEMPORARY_PREFERENCE_PHRASES = (
    "this particular", "this specific", "this one", "right now",
    "at the moment", "just today", "for today",
)

TEMPORARY_INFORMATION_PHRASES = (
    "deadline", "assignment", "upcoming exam", "mid exam", "mid exams",
    "final exam", "final exams", "test tomorrow", "exam tomorrow",
    "due tomorrow", "next week", "this week", "today", "tomorrow", "tonight",
)

# These words are too general to prove that a memory is relevant to a question.
STOP_WORDS = {
    "about", "also", "and", "are", "can", "could", "does", "for", "from",
    "have", "how", "into", "is", "learn", "learning", "like", "more", "my",
    "need", "should", "that", "the", "their", "they", "this", "user", "want",
    "what", "when", "where", "which", "with", "would", "you", "your",
}

os.makedirs(MEMORY_DIR, exist_ok=True)


# ==========================================
# JSON HELPERS
# ==========================================

def save_json(path, data):
    """Save JSON with readable indentation."""
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)


def utc_now():
    """Return a timezone-aware timestamp suitable for JSON storage."""
    return datetime.now(timezone.utc).isoformat()


def create_invalid_json_backup(path):
    """Keep a copy of malformed JSON before starting a replacement list."""
    if not os.path.exists(path):
        return

    backup_path = f"{path}.invalid-{datetime.now().strftime('%Y%m%d-%H%M%S')}.bak"
    try:
        shutil.copy2(path, backup_path)
        print(f"[Warning] Invalid JSON backed up to {os.path.basename(backup_path)}")
    except OSError:
        print(f"[Warning] Could not create a backup for {os.path.basename(path)}")


def load_json_list(path):
    """Load a JSON list, safely repairing missing, empty, or invalid files."""
    if not os.path.exists(path):
        save_json(path, [])
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, list):
            raise ValueError("JSON root must be a list")

        return data
    except (OSError, json.JSONDecodeError, ValueError):
        print(f"[Warning] {os.path.basename(path)} was invalid. Starting with an empty list.")
        create_invalid_json_backup(path)
        save_json(path, [])
        return []


def importance_score(value):
    """Read both older text importance and the new adjustable 1-to-10 score."""
    if isinstance(value, int) and 1 <= value <= 10:
        return value
    return IMPORTANCE_SCORES.get(value, 5)


def normalize_long_term_memories(memories):
    """Upgrade old memory records without changing their facts or IDs."""
    changed = False

    for memory in memories:
        if not isinstance(memory, dict):
            continue

        score = importance_score(memory.get("importance"))
        if memory.get("importance") != score:
            memory["importance"] = score
            changed = True
        if "created_at" not in memory:
            memory["created_at"] = None
            changed = True
        if "updated_at" not in memory:
            memory["updated_at"] = None
            changed = True
        if "expires_at" not in memory:
            memory["expires_at"] = None
            changed = True

    return changed


def remove_expired_memories(memories):
    """Remove only memories with an explicit, already-passed expiration date."""
    now = datetime.now(timezone.utc)
    active_memories = []
    removed_count = 0

    for memory in memories:
        expires_at = memory.get("expires_at") if isinstance(memory, dict) else None
        try:
            expired = expires_at and datetime.fromisoformat(expires_at) <= now
        except (TypeError, ValueError):
            expired = False

        if expired:
            removed_count += 1
        else:
            active_memories.append(memory)

    if removed_count:
        print(f"[Memory Cleanup] Removed {removed_count} expired memories")

    return active_memories, removed_count


def clean_json_response(text):
    """Remove optional Markdown fences before parsing Qwen's JSON response."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def create_memory_id(memories):
    """Create the next ID even if older files use memory_1 instead of memory_001."""
    highest_number = 0

    for memory in memories:
        match = re.fullmatch(r"memory_(\d+)", str(memory.get("id", "")))
        if match:
            highest_number = max(highest_number, int(match.group(1)))

    return f"memory_{highest_number + 1:03d}"


# ==========================================
# LONG-TERM MEMORY ANALYSIS AND UPDATES
# ==========================================

def analyze_memory(user_message, memories):
    """Ask Qwen for memory recommendations. Python still owns every file change."""
    existing_memories = json.dumps(memories, indent=2, ensure_ascii=False)

    prompt = f"""
You are ALLI's long-term memory analyzer. Analyze ONLY the user's current message.

Recommend zero or more memory operations. ADD is the default for every new
durable personal fact. UPDATE only when the user explicitly says that one
specific existing fact changed or was replaced. DELETE only when the user
explicitly says that one specific existing fact should be removed or is no
longer true. Use NONE when nothing should be remembered.

Important rules:
- Store only short, factual information about the user. Never store a whole conversation.
- Treat clear personal information as long-term memory when it is likely to
  remain useful: identity, education, goals, skills, interests, projects,
  routines, and preferences. A new durable personal fact should normally be ADD.
- A casual one-time question is normally NONE.
- Temporary, time-limited information must be NONE, not a low-importance
  long-term memory. Do not save deadlines, upcoming exams, assignments,
  one-time events, or short-lived situations. For example, "I have one week
  for my mid exams" is NONE.
- For UPDATE and DELETE, use an exact ID from Existing memories.
- Compare the meaning and factual content of existing memories, not just their
  category. Multiple facts in the same category can coexist.
- Categories are not unique slots. Many goals, skills, interests, identity
  facts, and education facts may coexist. Never use the same category, the same
  importance, related concepts, or similar wording as a reason to UPDATE.
- UPDATE requires BOTH: (1) the message refers to the same specific fact as an
  exact existing memory ID, and (2) the user explicitly indicates a change with
  wording such as "my goal changed", "instead of", "replace", "I stopped", or
  "no longer". If either condition is uncertain, use ADD.
- Before UPDATE or DELETE, identify the exact existing memory ID. Do not update
  or delete a different memory in the same category.
- Example: "I also want to become an influencer" is ADD even if a creator goal
  already exists. "My goal changed ... researcher instead" can UPDATE the exact
  old goal. "I stopped learning Flutter" can DELETE the exact Flutter memory.
- Remember durable positive and negative preferences. General statements such as
  "I like ...", "I don't like ...", "I dislike ...", "I hate ...", "I prefer
  ...", "I prefer not to ...", "I avoid ...", and "I'm not interested in ..."
  must normally use ADD with category "preference". Do not return NONE merely
  because a preference is negative. Keep the content factual, such
  as "User likes AI" or "User dislikes complicated explanations".
- Do not save a temporary opinion about one situation as a long-term preference.
  Phrases such as "this particular", "this specific", "right now", or "today"
  are usually NONE.
- Positive and negative preferences about different subjects can coexist. A
  preference changes only when the user explicitly describes the change, such as
  "I used to dislike this, but now I like it"; then UPDATE the exact old
  preference. If "I don't want to learn X anymore" refers to an existing
  learning memory, use UPDATE or DELETE for that exact learning memory.
- For the same subject, a like and a dislike are opposites, not separate facts.
  If an existing preference says the user likes a subject and the user later
  dislikes that same subject (or the reverse), UPDATE that exact preference so
  contradictory memories are not created. Different subjects must still ADD
  separate preference memories.
- Allowed categories: identity, education, goal, interest, preference, skill,
  project, routine, other.
- Use this importance scale from 1 to 10:
  * 10 = extremely important, permanent identity or core information.
  * 9 = very important, long-term personal information.
  * 8 = strong long-term preference or important goal.
  * 7 = important current skill, project, or ongoing interest.
  * 6 = useful personal interest or preference.
  * 5 = moderately useful information.
  * 4 = minor useful information.
  * 3 = short-term information.
  * 2 = very temporary information.
  * 1 = almost not worth remembering.
  Identity and education are normally 10. Important goals are normally at least
  8. Current skills/projects are normally at least 7. Preferences and interests
  are normally at least 6. Temporary information is normally NONE.
- Return only JSON, with no Markdown fences.

Existing memories:
{existing_memories}

Return exactly this shape:
{{
  "operations": [
    {{
      "action": "ADD",
      "memory_id": null,
      "category": "skill",
      "content": "User wants to learn Java",
      "importance": 7,
      "expires_days": null
    }}
  ]
}}

Use an empty operations list when there is nothing to remember.

User message:
{user_message}
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )

    try:
        data = json.loads(clean_json_response(response["message"]["content"]))
    except (KeyError, TypeError, json.JSONDecodeError):
        print("[Warning] Memory analysis returned invalid JSON. No memory was changed.")
        return []

    # Accept the old single-operation shape too, so a small model mistake is harmless.
    if isinstance(data, dict) and "action" in data:
        return [data]
    if isinstance(data, dict) and isinstance(data.get("operations"), list):
        return data["operations"]
    if isinstance(data, list):
        return data

    print("[Warning] Memory analysis had an unexpected format. No memory was changed.")
    return []


def valid_memory_fields(operation):
    """Validate model-provided fields before trusting them in the JSON memory file."""
    category = operation.get("category")
    content = operation.get("content")
    importance = operation.get("importance")

    return (
        category in VALID_CATEGORIES
        and importance in VALID_IMPORTANCE
        and isinstance(content, str)
        and 1 <= len(content.strip()) <= 240
    )


def memory_fact_type(memory):
    """Distinguish separate identity facts such as a name and a nickname."""
    content = str(memory.get("content", "")).lower()

    if "nickname" in content or "nick name" in content:
        return "nickname"
    if "name" in content:
        return "name"
    return "other"


def final_importance(operation):
    """Enforce the clear importance rules while leaving less-clear cases to Qwen."""
    content = operation["content"].lower()
    category = operation["category"]

    requested_score = importance_score(operation["importance"])

    if any(word in content for word in ("today", "tomorrow", "tonight", "this week", "one-time")):
        return min(requested_score, 3)
    if category in {"identity", "education"}:
        return 10
    if category == "goal":
        return max(8, requested_score)
    if category in {"skill", "project"}:
        return max(7, requested_score)
    if category in {"interest", "preference"}:
        return max(6, requested_score)

    return requested_score


def normalized_memory_content(content):
    """Create a simple, case-insensitive form for exact duplicate checks."""
    words = re.findall(r"[a-zA-Z0-9]+", content.lower())
    ignored_words = {"a", "an", "the", "user"}
    return " ".join(word for word in words if word not in ignored_words)


def is_duplicate_memory(operation, memories):
    """Return True only for the same stored fact, never merely the same category."""
    if not isinstance(operation.get("content"), str):
        return False

    new_content = normalized_memory_content(operation["content"])
    return any(
        normalized_memory_content(memory.get("content", "")) == new_content
        for memory in memories
        if isinstance(memory, dict)
    )


def preference_polarity(content):
    """Return the simple positive/negative direction of a like or dislike."""
    text = content.lower()

    if any(phrase in text for phrase in ("dislikes", "does not like", "don't like", "hates")):
        return "dislike"
    if any(phrase in text for phrase in ("likes", "prefers")):
        return "like"
    return None


def preference_subject(content):
    """Extract the subject after 'likes' or 'dislikes' for simple comparison."""
    text = content.lower()
    match = re.search(r"\b(?:dislikes|likes|hates|prefers)\s+(.+)", text)
    if not match:
        return ""

    generic_words = {"a", "an", "the", "programming", "language", "subject"}
    words = re.findall(r"[a-zA-Z0-9]+", match.group(1))
    return " ".join(word for word in words if word not in generic_words)


def find_preference_reversal(operation, memories):
    """Find the exact stored preference with the same subject and opposite view."""
    if operation.get("category") != "preference":
        return None

    new_polarity = preference_polarity(operation.get("content", ""))
    new_subject = preference_subject(operation.get("content", ""))
    if not new_polarity or not new_subject:
        return None

    for index, memory in enumerate(memories):
        if memory.get("category") != "preference":
            continue
        if preference_subject(memory.get("content", "")) != new_subject:
            continue
        if preference_polarity(memory.get("content", "")) != new_polarity:
            return index

    return None


def update_memory(memory, operation):
    """Replace one existing fact while preserving its stable memory ID."""
    memory.update({
        "category": operation["category"],
        "content": operation["content"].strip(),
        "importance": final_importance(operation),
        "updated_at": utc_now(),
    })


def add_memory(operation, memories):
    """Create one validated memory record with a Python-generated ID."""
    new_memory = {
        "id": create_memory_id(memories),
        "category": operation["category"],
        "content": operation["content"].strip(),
        "importance": final_importance(operation),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "expires_at": None,
    }

    expires_days = operation.get("expires_days")
    if STORE_TEMPORARY_MEMORIES and isinstance(expires_days, int) and expires_days > 0:
        new_memory["expires_at"] = (datetime.now(timezone.utc) + timedelta(days=expires_days)).isoformat()

    memories.append(new_memory)
    print("[Memory Added]", new_memory["id"])


def has_explicit_change_signal(user_message):
    """Only explicit user language can authorize UPDATE or DELETE."""
    message = user_message.lower()
    return any(phrase in message for phrase in CHANGE_SIGNAL_PHRASES)


def is_temporary_preference(user_message):
    """Avoid saving opinions clearly limited to one short-lived situation."""
    message = user_message.lower()
    return any(phrase in message for phrase in TEMPORARY_PREFERENCE_PHRASES)


def is_temporary_information(user_message):
    """Identify deadlines and short-lived events that do not belong in long-term memory."""
    message = user_message.lower()
    has_short_duration = bool(re.search(r"\b(?:\d+|one|two|three|four|five)\s+(?:day|days|week|weeks|month|months)\b", message))
    return has_short_duration or any(
        phrase in message for phrase in TEMPORARY_INFORMATION_PHRASES
    )


def process_memory_operations(operations, memories, user_message, from_consolidation=False):
    """Apply valid recommendations. The model can suggest changes but cannot edit files."""
    changed = False

    for operation in operations:
        if not isinstance(operation, dict):
            print("[Memory Skipped] Invalid operation format")
            continue

        action = str(operation.get("action", "")).upper()
        memory_id = operation.get("memory_id")

        if action == "NONE":
            continue

        if action == "ADD":
            if not valid_memory_fields(operation):
                print("[Memory Skipped] Invalid ADD data")
                continue

            if not from_consolidation and not STORE_TEMPORARY_MEMORIES and is_temporary_information(user_message):
                print("[Memory Not Added] Temporary information")
                continue

            if not from_consolidation and operation["category"] == "preference" and is_temporary_preference(user_message):
                print("[Memory Not Added] Temporary preference")
                continue

            if is_duplicate_memory(operation, memories):
                print("[Memory Not Added] Exact duplicate already exists")
                continue

            reversal_index = find_preference_reversal(operation, memories)
            if reversal_index is not None:
                update_memory(memories[reversal_index], operation)
                changed = True
                print("[Memory Preference Reversed]", memories[reversal_index]["id"])
                continue

            add_memory(operation, memories)
            changed = True
            continue

        # A new fact must not disappear just because Qwen mistakenly selected
        # UPDATE. Without explicit change language, use NONE for a duplicate
        # and ADD for a new fact.
        if action in {"UPDATE", "DELETE"} and not has_explicit_change_signal(user_message):
            if action == "UPDATE" and valid_memory_fields(operation):
                if not from_consolidation and not STORE_TEMPORARY_MEMORIES and is_temporary_information(user_message):
                    print("[Memory Update Changed To None] Temporary information")
                elif not from_consolidation and operation["category"] == "preference" and is_temporary_preference(user_message):
                    print("[Memory Update Changed To None] Temporary preference")
                elif is_duplicate_memory(operation, memories):
                    print("[Memory Update Changed To None] Exact duplicate already exists")
                else:
                    reversal_index = find_preference_reversal(operation, memories)
                    if reversal_index is not None:
                        update_memory(memories[reversal_index], operation)
                        changed = True
                        print("[Memory Preference Reversed]", memories[reversal_index]["id"])
                    else:
                        add_memory(operation, memories)
                        changed = True
                        print("[Memory Update Changed To Add] No explicit change signal")
            else:
                print(f"[Memory {action} Skipped] No explicit change signal in user message")
            continue

        if action in {"UPDATE", "DELETE"}:
            if not isinstance(memory_id, str):
                print(f"[Memory {action} Skipped] Invalid memory ID")
                continue

            memory_index = next(
                (index for index, memory in enumerate(memories) if memory.get("id") == memory_id),
                None,
            )
            if memory_index is None:
                print(f"[Memory {action} Skipped] ID not found: {memory_id}")
                continue

            if action == "UPDATE":
                if not valid_memory_fields(operation):
                    print("[Memory Skipped] Invalid UPDATE data")
                    continue

                existing_memory = memories[memory_index]

                # A preference reversal updates the matching subject, even if
                # Qwen selected the wrong memory ID from another preference.
                reversal_index = find_preference_reversal(operation, memories)
                if reversal_index is not None:
                    memory_index = reversal_index
                    existing_memory = memories[memory_index]

                # A model must never overwrite one category with another.
                # Treat that recommendation as a new fact instead.
                if existing_memory.get("category") != operation["category"]:
                    add_memory(operation, memories)
                    changed = True
                    print("[Memory Update Changed To Add] Different category")
                    continue

                # Same category does not mean the same fact. In particular,
                # never let a nickname overwrite a name, or the reverse.
                if (
                    existing_memory.get("category") == "identity"
                    and operation["category"] == "identity"
                    and memory_fact_type(existing_memory) != memory_fact_type(operation)
                ):
                    add_memory(operation, memories)
                    changed = True
                    print("[Memory Update Changed To Add] Separate identity fact")
                    continue

                update_memory(memories[memory_index], operation)
                print("[Memory Updated]", memory_id)
            else:
                memories.pop(memory_index)
                print("[Memory Deleted]", memory_id)

            changed = True
            continue

        print(f"[Memory Skipped] Unknown action: {action}")

    if changed:
        save_json(LONG_TERM_MEMORY_FILE, memories)


# ==========================================
# CONVERSATION ARCHIVE AND CONSOLIDATION
# ==========================================

def conversation_character_count(conversation):
    """Measure the active conversation without relying on a fixed message count."""
    return len(json.dumps(conversation, ensure_ascii=False))


def estimated_conversation_tokens(conversation):
    """Use a simple local token estimate so no extra dependency is required."""
    return conversation_character_count(conversation) // 4


def needs_consolidation(conversation):
    """Consolidate only when configurable size or token limits are reached."""
    return (
        conversation_character_count(conversation) >= MAX_ACTIVE_CONVERSATION_CHARACTERS
        or estimated_conversation_tokens(conversation) >= MAX_ACTIVE_ESTIMATED_TOKENS
    )


def conversation_text(messages):
    """Convert saved chat messages into a clear transcript for Qwen's summary."""
    lines = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = message.get("role", "unknown").capitalize()
        content = message.get("content", "")
        if isinstance(content, str):
            lines.append(f"{role}: {content}")
    return "\n\n".join(lines)


def analyze_archive_chunk(messages, memories):
    """Summarize old chat and request memory changes before it leaves active context."""
    transcript = conversation_text(messages)
    existing_memories = json.dumps(memories, indent=2, ensure_ascii=False)
    prompt = f"""
You are ALLI's conversation consolidation assistant.

Summarize the older conversation below for a future archive. Keep the summary
short and useful. Then recommend long-term memory operations only for durable
user facts that are useful in future conversations.

Use ADD as the default for a new fact. UPDATE or DELETE requires an exact memory
ID plus clear evidence in the transcript that the same fact changed or ended.
Do not save ordinary questions, detailed answers, deadlines, upcoming exams,
assignments, or one-time events as long-term memory. Categories are not unique.
Return only JSON in this exact shape:
{{
  "summary": "Short factual archive summary.",
  "operations": [
    {{
      "action": "ADD",
      "memory_id": null,
      "category": "project",
      "content": "User is building ALLI",
      "importance": 8,
      "expires_days": null
    }}
  ]
}}

Existing memories:
{existing_memories}

Older conversation:
{transcript}
"""

    response = chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Return only valid JSON."},
            {"role": "user", "content": prompt},
        ],
        format="json",
    )

    try:
        data = json.loads(clean_json_response(response["message"]["content"]))
    except (KeyError, TypeError, json.JSONDecodeError):
        raise ValueError("Conversation consolidation returned invalid JSON")

    if not isinstance(data, dict) or not isinstance(data.get("summary"), str):
        raise ValueError("Conversation consolidation returned an invalid summary")

    operations = data.get("operations", [])
    if not isinstance(operations, list):
        operations = []

    return data["summary"].strip(), operations


def create_archive_id(archives):
    """Create a readable unique ID for an archived conversation chunk."""
    return f"archive_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(archives) + 1:03d}"


def compact_archive_details(archives):
    """Keep archive summaries when storage grows, pruning oldest raw details first."""
    while len(json.dumps(archives, ensure_ascii=False)) > MAX_ARCHIVE_STORAGE_BYTES:
        archive_with_details = next(
            (archive for archive in archives if archive.get("messages")),
            None,
        )
        if archive_with_details is None:
            break
        archive_with_details["messages"] = []
        archive_with_details["details_pruned"] = True


def consolidate_conversation_if_needed(conversation, memories):
    """Archive one old chunk after safely summarizing it and extracting memories."""
    if not needs_consolidation(conversation):
        return False

    messages_available = len(conversation) - ACTIVE_MESSAGES_TO_KEEP
    if messages_available <= 0:
        return False

    chunk_size = min(ARCHIVE_CHUNK_MESSAGE_LIMIT, messages_available)
    messages_to_archive = conversation[:chunk_size]

    try:
        summary, operations = analyze_archive_chunk(messages_to_archive, memories)
    except Exception as error:
        print(f"[Conversation Consolidation Error] {error}")
        return False

    # Reuse the existing validation rules before any conversation detail is moved.
    process_memory_operations(
        operations,
        memories,
        conversation_text(messages_to_archive),
        from_consolidation=True,
    )

    archives = load_json_list(CONVERSATION_ARCHIVE_FILE)
    archives.append({
        "id": create_archive_id(archives),
        "archived_at": utc_now(),
        "summary": summary,
        "message_count": len(messages_to_archive),
        "details_pruned": False,
        "messages": messages_to_archive,
    })
    compact_archive_details(archives)
    save_json(CONVERSATION_ARCHIVE_FILE, archives)

    del conversation[:chunk_size]
    print(f"[Conversation Consolidated] Archived {chunk_size} messages")
    return True


# ==========================================
# MEMORY RETRIEVAL
# ==========================================

def meaningful_words(text):
    words = re.findall(r"[a-zA-Z0-9+#.]+", text.lower())
    return {word for word in words if len(word) >= 3 and word not in STOP_WORDS}


def retrieve_relevant_memories(question, memories):
    """Use simple word overlap and a few personal-question patterns; no embeddings yet."""
    question_words = meaningful_words(question)
    lower_question = question.lower()
    scored_memories = []

    for memory in memories:
        if not isinstance(memory, dict) or not isinstance(memory.get("content"), str):
            continue

        memory_words = meaningful_words(memory["content"])
        score = len(question_words & memory_words) * 10
        category = memory.get("category")

        # These patterns make questions such as "What is my name?" work even
        # though "name" is not always part of the user's exact remembered wording.
        if category == "identity" and ("my name" in lower_question or "nickname" in lower_question):
            score += 20
        if category in {"skill", "interest", "goal"} and any(
            phrase in lower_question
            for phrase in ("what should i learn", "what to learn", "learn next", "career path")
        ):
            score += 5
        if category == "education" and any(
            word in lower_question
            for word in ("education", "study", "studies", "subject", "semester", "course", "diploma")
        ):
            score += 5

        if score > 0:
            scored_memories.append((score, importance_score(memory.get("importance")), memory))

    scored_memories.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [item[2] for item in scored_memories[:MAX_RETRIEVED_MEMORIES]]


def direct_memory_request(question):
    """Identify questions that should be answered from stored memory exactly."""
    text = question.lower().strip()

    if any(phrase in text for phrase in ("what do you remember about me", "what do you know about me")):
        return "all"
    if "my nickname" in text or "my nick name" in text:
        return "nickname"
    if "my name" in text:
        return "name"

    category_requests = {
        "goal": ("my goal", "my goals"),
        "skill": ("my skill", "my skills"),
        "interest": ("my interest", "my interests"),
        "preference": ("my preference", "my preferences", "what do i like", "what do i dislike"),
        "education": ("my education", "what am i studying"),
    }

    for category, phrases in category_requests.items():
        if any(phrase in text for phrase in phrases):
            return category

    return None


def direct_memory_answer(question, memories):
    """Answer a direct memory question without invoking Qwen or changing memory."""
    request = direct_memory_request(question)
    if request is None:
        return None

    valid_memories = [memory for memory in memories if isinstance(memory, dict)]

    if request == "all":
        selected = valid_memories
        heading = "Here is what I currently have stored about you:"
    elif request == "name":
        selected = [memory for memory in valid_memories if memory_fact_type(memory) == "name"]
        heading = "Your stored name:"
    elif request == "nickname":
        selected = [memory for memory in valid_memories if memory_fact_type(memory) == "nickname"]
        heading = "Your stored nickname:"
    else:
        selected = [memory for memory in valid_memories if memory.get("category") == request]
        display_names = {
            "goal": "goals",
            "skill": "skills",
            "interest": "interests",
            "preference": "preferences",
            "education": "education information",
        }
        heading = f"Here are the {display_names[request]} I currently have stored:"

    if not selected:
        if request == "all":
            return "I currently do not have any long-term information stored about you."
        return f"I currently do not have your {request} information stored."

    lines = [f"- {memory.get('content', '')}" for memory in selected]
    return heading + "\n" + "\n".join(lines)


def is_follow_up_question(current_question, previous_user_message):
    """Return True only when the new question appears connected to the previous topic."""
    current_lower = current_question.lower().strip()

    follow_up_starts = (
        "and ", "also ", "what about", "why ", "how ", "can you explain",
        "tell me more", "give me an example", "what do you mean", "then ",
    )
    if current_lower.startswith(follow_up_starts):
        return True

    # Shared meaningful words are a simple, useful signal that the topic continued.
    return bool(meaningful_words(current_question) & meaningful_words(previous_user_message))


def select_conversation_context(conversation):
    """Keep recent context for follow-ups, but drop it when the user clearly changes topic."""
    chat_messages = [
        message for message in conversation
        if message.get("role") in {"user", "assistant"} and isinstance(message.get("content"), str)
    ]

    current_question = chat_messages[-1]
    previous_user_messages = [message for message in chat_messages[:-1] if message["role"] == "user"]

    if not previous_user_messages:
        return [current_question]

    previous_question = previous_user_messages[-1]["content"]
    if not is_follow_up_question(current_question["content"], previous_question):
        # A fresh topic should not inherit a nearby Flutter (or any other) discussion.
        return [current_question]

    return chat_messages[-RECENT_MESSAGE_LIMIT:]


def build_chat_messages(conversation, relevant_memories):
    """Build a small prompt from relevant memories and topic-aware recent chat."""
    recent_messages = select_conversation_context(conversation)

    messages = [SYSTEM_MESSAGE]

    if relevant_memories:
        memory_lines = [
            f"- ({memory['id']}, {memory.get('category', 'other')}) {memory['content']}"
            for memory in relevant_memories
        ]
        messages.append({
            "role": "system",
            "content": "Relevant long-term user memories for this answer:\n" + "\n".join(memory_lines)
            + "\nUse them only when they help answer the current question. Do not mention unrelated memories.",
        })

    messages.extend(recent_messages)
    return messages


# ==========================================
# ALLI START
# ==========================================

def main():
    conversation = load_json_list(CONVERSATION_FILE)
    long_term_memory = load_json_list(LONG_TERM_MEMORY_FILE)

    memories_changed = normalize_long_term_memories(long_term_memory)
    long_term_memory, expired_count = remove_expired_memories(long_term_memory)
    if memories_changed or expired_count:
        save_json(LONG_TERM_MEMORY_FILE, long_term_memory)

    print("=" * 40)
    print("           ALLI v0.4")
    print("=" * 40)

    # ==========================================
    # MAIN LOOP
    # ==========================================
    while True:
        question = input("\nYou: ").strip()

        if question.lower() == "exit":
            print("\nALLI: Goodbye!")
            break

        if not question:
            print("\nALLI: Please type a message.")
            continue

        # Save the complete history, but later send only a small recent window to Qwen.
        conversation.append({"role": "user", "content": question})

        # Direct memory questions are read-only. They must not call the memory
        # analyzer because asking about a fact is not a new fact.
        answer = direct_memory_answer(question, long_term_memory)

        if answer is None:
            try:
                operations = analyze_memory(question, long_term_memory)
                process_memory_operations(operations, long_term_memory, question)
            except Exception as error:
                # Memory failures should not stop normal conversation.
                print(f"[Memory Analyzer Error] {error}")

            relevant_memories = retrieve_relevant_memories(question, long_term_memory)
            messages = build_chat_messages(conversation, relevant_memories)

            print("\nALLI: Thinking...")

            try:
                response = chat(model=MODEL_NAME, messages=messages)
                answer = response["message"]["content"]
            except Exception as error:
                print(f"\nALLI: I could not contact Ollama. {error}")
                # Keep the user's message on disk even when Qwen is unavailable.
                save_json(CONVERSATION_FILE, conversation)
                continue

        conversation.append({"role": "assistant", "content": answer})
        save_json(CONVERSATION_FILE, conversation)

        # The detailed active transcript stays small; older details are moved
        # into a summarized archive only when a size/token threshold is reached.
        if consolidate_conversation_if_needed(conversation, long_term_memory):
            save_json(CONVERSATION_FILE, conversation)

        print("\nALLI:")
        print(answer)


if __name__ == "__main__":
    main()
