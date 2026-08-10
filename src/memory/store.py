"""Safe long-term memory persistence and v0.4 → v0.5 migration."""

from __future__ import annotations

import json
import os
import re
import shutil
from datetime import datetime, timezone
from typing import Any, Optional

from . import config
from .embeddings import EmbeddingError, EmbeddingStore


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def save_json(path: str, data: Any) -> None:
    """Atomic JSON write: temp file then replace."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    os.replace(temp_path, path)


def backup_corrupt_file(path: str) -> Optional[str]:
    """Copy a corrupt file to long_term_memory.corrupt.<timestamp>.json style."""
    if not os.path.exists(path):
        return None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    directory = os.path.dirname(path)
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    backup_name = f"{name}.corrupt.{stamp}{ext or '.json'}"
    backup_path = os.path.join(directory, backup_name)

    try:
        shutil.copy2(path, backup_path)
        return backup_path
    except OSError:
        return None


def load_json_list(path: str, *, create_if_missing: bool = True) -> list:
    """Load a JSON list; back up corrupt files instead of silently destroying them."""
    if not os.path.exists(path):
        if create_if_missing:
            save_json(path, [])
        return []

    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
        if not isinstance(data, list):
            raise ValueError("JSON root must be a list")
        return data
    except (OSError, json.JSONDecodeError, ValueError) as error:
        backup = backup_corrupt_file(path)
        message = (
            f"[Memory Error] {os.path.basename(path)} is invalid: {error}."
        )
        if backup:
            message += f" Backup created: {os.path.basename(backup)}"
        print(message)
        # Recover with empty list only after backup; do not delete original until
        # we can write a valid replacement.
        save_json(path, [])
        return []


def create_memory_id(memories: list) -> str:
    highest = 0
    for memory in memories:
        if not isinstance(memory, dict):
            continue
        match = re.fullmatch(r"memory_(\d+)", str(memory.get("id", "")))
        if match:
            highest = max(highest, int(match.group(1)))
    return f"memory_{highest + 1:03d}"


def normalize_importance(value: Any) -> str:
    if value in config.VALID_IMPORTANCE:
        return value
    if isinstance(value, int):
        if value >= 8:
            return "high"
        if value >= 4:
            return "medium"
        return "low"
    return "medium"


def infer_fact_type(category: str, content: str) -> str:
    """Best-effort fact_type for migrated or new memories."""
    text = str(content).lower()
    category = (category or "other").lower()

    if category == "identity":
        if "nickname" in text or "nick name" in text:
            return "nickname"
        if re.search(r"\bname\b", text):
            return "name"
        return "identity_other"

    if category == "goal":
        if re.search(r"\blearn\b|\blearning\b", text):
            return "learning_goal"
        if re.search(r"\bbuild\b|\bproject\b", text):
            return "project_goal"
        return "career_goal"

    if category == "learning":
        return "learning"

    if category == "preference":
        if any(
            phrase in text
            for phrase in (
                "dislikes",
                "dislike",
                "doesn't like",
                "does not like",
                "don't like",
                "hates",
                "hate",
            )
        ):
            return "dislike"
        return "like"

    if category == "education":
        if "college" in text or "university" in text:
            return "college"
        if "course" in text or "diploma" in text:
            return "course"
        return "education"

    if category == "skill":
        return "skill"
    if category == "interest":
        return "interest"
    if category == "project":
        return "project"
    if category == "routine":
        return "habit"
    return "other"


def extract_memory_value(
    category: str,
    fact_type: str,
    content: str,
    explicit_value: Optional[str] = None,
) -> str:
    """Extract the compact value of a structured memory."""
    if isinstance(explicit_value, str) and explicit_value.strip():
        return explicit_value.strip()

    text = str(content or "").strip()
    fact_type = (fact_type or "").lower()

    if fact_type in {"like", "dislike"}:
        return fact_type

    patterns = [
        r"^User's name is\s+(.+)$",
        r"^User's nickname is\s+(.+)$",
        r"^User wants to become\s+(.+)$",
        r"^User wants to learn\s+(.+)$",
        r"^User wants to build\s+(.+)$",
        r"^User wants to\s+(.+)$",
        r"^User likes\s+(.+)$",
        r"^User dislikes\s+(.+)$",
        r"^User knows\s+(.+)$",
        r"^User can\s+(.+)$",
        r"^User is interested in\s+(.+)$",
        r"^User is studying\s+(.+)$",
        r"^User is learning\s+(.+)$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(".")

    if fact_type in {"name", "nickname"} and " is " in text.lower():
        return text.split(" is ", 1)[-1].strip().rstrip(".")

    return text


def extract_memory_subject(
    category: str,
    fact_type: str,
    content: str,
    value: str,
    explicit_subject: Optional[str] = None,
) -> str:
    """Extract a comparable subject key for duplicate/update matching."""
    if isinstance(explicit_subject, str) and explicit_subject.strip():
        return explicit_subject.strip().lower()

    category = (category or "").lower()
    fact_type = (fact_type or "").lower()
    content = str(content or "")
    value = str(value or "")

    if category == "identity":
        return fact_type or "identity"

    if category == "preference":
        match = re.search(
            r"\b(?:likes|dislikes|hates|prefers)\s+(.+)$",
            content,
            flags=re.IGNORECASE,
        )
        subject = match.group(1).strip() if match else value
        # Reuse the shared preference normalizer (imported lazily to avoid cycles).
        from .extractor import normalize_preference_subject

        return normalize_preference_subject(subject)

    if category in {
        "goal", "skill", "interest", "learning", "project", "education", "routine"
    }:
        text = value or content
        text = re.sub(
            r"^user\s+(wants to become|wants to learn|wants to build|"
            r"wants to|knows|is interested in|is learning)\s+",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"[^a-z0-9+#\s]", " ", text.lower())
        words = [
            w for w in text.split()
            if w not in {"a", "an", "the", "to", "my", "user", "become", "be"}
        ]
        return " ".join(words)

    return (value or content).strip().lower()


def default_importance_for(category: str, fact_type: str) -> str:
    if category in {"identity", "education"}:
        return "high"
    if category == "goal":
        return "high"
    if fact_type in {"name", "nickname"}:
        return "high"
    if category in {"skill", "interest", "learning", "project", "preference", "routine"}:
        return "medium"
    return "medium"


def build_embedding_text(
    category: str,
    fact_type: str,
    content: str,
) -> str:
    """
    Build retrieval-oriented text for embedding.

    all-minilm matches questions to statements poorly on raw fact text alone.
    Prefixing durable category/fact cues improves semantic retrieval while the
    stored `content` field remains the clean factual statement shown to Qwen.
    """
    category = (category or "other").lower()
    fact_type = (fact_type or "").lower()
    content = (content or "").strip()

    hints = {
        "identity": {
            "name": (
                "personal identity; user's real given name; "
                "who the user is by name"
            ),
            "nickname": (
                "personal nickname or alias; informal name used to address "
                "the user"
            ),
            "default": "personal identity information about the user",
        },
        "goal": {
            "learning_goal": (
                "learning goal; topics the user wants to learn but does not "
                "necessarily know yet"
            ),
            "career_goal": (
                "life and career goals; dreams ambitions aims; what the user "
                "wants to become or achieve"
            ),
            "project_goal": (
                "goal to build or create something; project ambition"
            ),
            "default": (
                "personal goals dreams ambitions aims; what the user wants "
                "to become or achieve"
            ),
        },
        "preference": (
            "personal preferences likes dislikes enjoy hate"
        ),
        "skill": (
            "skills the user already has; things the user already knows "
            "how to do; competencies and abilities"
        ),
        "learning": (
            "topics the user is currently learning or wants to learn; "
            "learning progress distinct from completed skills"
        ),
        "interest": "personal interests hobbies",
        "education": "education college course major studying school background",
        "project": "personal projects building creating",
        "routine": "personal routines habits schedule",
        "other": "personal durable memory about the user",
    }

    if category == "identity":
        prefix = hints["identity"].get(fact_type, hints["identity"]["default"])
    elif category == "goal":
        prefix = hints["goal"].get(fact_type, hints["goal"]["default"])
    else:
        prefix = hints.get(category, hints["other"])

    return (
        f"Durable personal memory about the user. "
        f"Category {category}. Fact type {fact_type}. "
        f"{prefix}. Fact: {content}"
    )


class MemoryStore:
    """Load, migrate, and persist durable long-term memories."""

    def __init__(
        self,
        path: Optional[str] = None,
        embedding_store: Optional[EmbeddingStore] = None,
    ):
        self.path = path or config.LONG_TERM_MEMORY_FILE
        self.embedding_store = embedding_store or EmbeddingStore()
        self.memories: list[dict] = []

    def load(self, *, ensure_embeddings: bool = True) -> list[dict]:
        raw = load_json_list(self.path)
        self.memories = [m for m in raw if isinstance(m, dict)]
        changed = self.migrate_old_memories()
        if ensure_embeddings:
            try:
                if self.ensure_embeddings():
                    changed = True
            except EmbeddingError as error:
                print(f"[Warning] {error}")
        if changed:
            self.save()
        return self.memories

    def save(self) -> None:
        save_json(self.path, self.memories)

    def migrate_old_memories(self) -> bool:
        """Normalize v0.4 records without duplicating or dropping facts."""
        changed = False
        now = utc_now_iso()

        for memory in self.memories:
            if "id" not in memory or not memory["id"]:
                memory["id"] = create_memory_id(self.memories)
                changed = True

            category = str(memory.get("category", "other")).lower()
            # Migrate legacy learning facts stored under goal/interest.
            content_l = str(memory.get("content", "")).lower()
            fact_l = str(memory.get("fact_type", "")).lower()
            if (
                category in {"goal", "interest"}
                and (
                    fact_l == "learning_goal"
                    or re.search(r"\bis learning\b|\bwants to learn\b", content_l)
                )
            ):
                memory["category"] = "learning"
                memory["fact_type"] = "learning"
                memory["embedding"] = None
                category = "learning"
                changed = True

            if category not in config.VALID_CATEGORIES:
                memory["category"] = "other"
                changed = True
            else:
                if memory.get("category") != category:
                    memory["category"] = category
                    changed = True

            content = memory.get("content")
            if not isinstance(content, str):
                memory["content"] = str(content or "").strip()
                changed = True

            importance = normalize_importance(memory.get("importance"))
            if memory.get("importance") != importance:
                memory["importance"] = importance
                changed = True

            fact_type = memory.get("fact_type")
            if not isinstance(fact_type, str) or not fact_type.strip():
                memory["fact_type"] = infer_fact_type(
                    memory.get("category", "other"),
                    memory.get("content", ""),
                )
                changed = True
            else:
                # Correct clearly wrong goal subtypes after earlier migrations.
                inferred = infer_fact_type(
                    memory.get("category", "other"),
                    memory.get("content", ""),
                )
                if (
                    memory.get("category") == "goal"
                    and inferred == "learning_goal"
                    and memory.get("fact_type") != "learning_goal"
                ):
                    memory["fact_type"] = "learning_goal"
                    changed = True
                if (
                    memory.get("category") == "preference"
                    and inferred in {"like", "dislike"}
                    and memory.get("fact_type") not in {"like", "dislike"}
                ):
                    memory["fact_type"] = inferred
                    changed = True

            value = extract_memory_value(
                memory.get("category", "other"),
                memory.get("fact_type", "other"),
                memory.get("content", ""),
                memory.get("value") if memory.get("category") != "preference" else memory.get("value"),
            )
            # Preference values are polarity; keep like/dislike.
            if memory.get("category") == "preference":
                polarity = memory.get("fact_type")
                if polarity not in {"like", "dislike"}:
                    polarity = infer_fact_type("preference", memory.get("content", ""))
                    memory["fact_type"] = polarity
                    changed = True
                if memory.get("value") != polarity:
                    memory["value"] = polarity
                    changed = True
                value = polarity
            elif memory.get("value") != value:
                memory["value"] = value
                changed = True

            subject = extract_memory_subject(
                memory.get("category", "other"),
                memory.get("fact_type", "other"),
                memory.get("content", ""),
                memory.get("value", ""),
                memory.get("subject"),
            )
            if memory.get("subject") != subject:
                memory["subject"] = subject
                changed = True

            status = str(memory.get("status", "active")).lower()
            if status not in config.VALID_STATUS:
                status = "active"
            if memory.get("status") != status:
                memory["status"] = status
                changed = True

            if not memory.get("created_at"):
                memory["created_at"] = now
                changed = True
            if not memory.get("updated_at"):
                memory["updated_at"] = memory.get("created_at", now)
                changed = True

            embedding = memory.get("embedding")
            if embedding is not None and not isinstance(embedding, list):
                memory["embedding"] = None
                changed = True

            # Re-embed when the embedding text recipe changes.
            if memory.get("embedding_version") != config.EMBEDDING_TEXT_VERSION:
                memory["embedding"] = None
                memory["embedding_version"] = config.EMBEDDING_TEXT_VERSION
                changed = True

        return changed

    def active_memories(self) -> list[dict]:
        """Return only currently active memories for retrieval/answers."""
        result = []
        for memory in self.memories:
            if not isinstance(memory, dict):
                continue
            status = str(memory.get("status", "active")).lower()
            if status == "active":
                result.append(memory)
        return result

    def _embedding_text_for(self, memory: dict) -> str:
        return build_embedding_text(
            memory.get("category", "other"),
            memory.get("fact_type", "other"),
            memory.get("content", ""),
        )

    def ensure_embeddings(self) -> bool:
        """Generate missing embeddings for active memories that need retrieval."""
        missing = [
            memory
            for memory in self.memories
            if str(memory.get("status", "active")).lower() == "active"
            and (
                not isinstance(memory.get("embedding"), list)
                or not memory.get("embedding")
            )
        ]
        if not missing:
            return False

        texts = [self._embedding_text_for(m) for m in missing]
        vectors = self.embedding_store.generate_embeddings(texts)
        for memory, vector in zip(missing, vectors):
            memory["embedding"] = vector
            memory["embedding_version"] = config.EMBEDDING_TEXT_VERSION
            memory["updated_at"] = utc_now_iso()
        return True

    def get_by_id(self, memory_id: str) -> Optional[dict]:
        for memory in self.memories:
            if memory.get("id") == memory_id:
                return memory
        return None

    def _archive_snapshot(self, memory: dict) -> dict:
        """Keep an inactive historical copy before updating an active fact."""
        import copy

        snapshot = copy.deepcopy(memory)
        snapshot["id"] = create_memory_id(self.memories)
        snapshot["status"] = "inactive"
        snapshot["superseded_by"] = memory.get("id")
        snapshot["updated_at"] = utc_now_iso()
        # Inactive copies are not used for retrieval.
        snapshot["embedding"] = None
        self.memories.append(snapshot)
        return snapshot

    def add_memory(
        self,
        *,
        category: str,
        content: str,
        fact_type: Optional[str] = None,
        importance: Optional[str] = None,
        value: Optional[str] = None,
        subject: Optional[str] = None,
        embedding: Optional[list[float]] = None,
        status: str = "active",
    ) -> dict:
        now = utc_now_iso()
        category = category if category in config.VALID_CATEGORIES else "other"
        content = content.strip()
        fact_type = fact_type or infer_fact_type(category, content)
        importance = normalize_importance(
            importance or default_importance_for(category, fact_type)
        )

        if category == "preference" and fact_type in {"like", "dislike"}:
            resolved_value = fact_type
        else:
            resolved_value = extract_memory_value(
                category, fact_type, content, value
            )

        resolved_subject = extract_memory_subject(
            category,
            fact_type,
            content,
            resolved_value if category != "preference" else (value or resolved_value),
            subject,
        )

        if embedding is None:
            embedding = self.embedding_store.generate_embedding(
                build_embedding_text(category, fact_type, content)
            )

        memory = {
            "id": create_memory_id(self.memories),
            "category": category,
            "fact_type": fact_type,
            "subject": resolved_subject,
            "value": resolved_value,
            "content": content,
            "importance": importance,
            "status": status if status in config.VALID_STATUS else "active",
            "embedding": embedding,
            "embedding_version": config.EMBEDDING_TEXT_VERSION,
            "created_at": now,
            "updated_at": now,
        }
        self.memories.append(memory)
        self.save()
        return memory

    def update_memory(
        self,
        memory_id: str,
        *,
        content: Optional[str] = None,
        category: Optional[str] = None,
        fact_type: Optional[str] = None,
        importance: Optional[str] = None,
        value: Optional[str] = None,
        subject: Optional[str] = None,
        archive_history: bool = True,
    ) -> Optional[dict]:
        memory = self.get_by_id(memory_id)
        if memory is None:
            return None

        content_changed = False
        if content is not None and content.strip() != memory.get("content"):
            content_changed = True

        fact_changed = (
            fact_type is not None and fact_type != memory.get("fact_type")
        )
        value_will_change = False
        if value is not None and value != memory.get("value"):
            value_will_change = True
        if content_changed:
            value_will_change = True

        if archive_history and (
            content_changed or fact_changed or value_will_change
        ):
            if str(memory.get("status", "active")).lower() == "active":
                snapshot = self._archive_snapshot(memory)
                memory["replaces"] = snapshot["id"]

        if content is not None and content_changed:
            memory["content"] = content.strip()

        if category is not None and category in config.VALID_CATEGORIES:
            memory["category"] = category

        if fact_type is not None:
            memory["fact_type"] = fact_type
        elif content_changed:
            memory["fact_type"] = infer_fact_type(
                memory.get("category", "other"),
                memory.get("content", ""),
            )

        if importance is not None:
            memory["importance"] = normalize_importance(importance)

        if memory.get("category") == "preference":
            polarity = memory.get("fact_type")
            if polarity not in {"like", "dislike"}:
                polarity = infer_fact_type(
                    "preference", memory.get("content", "")
                )
                memory["fact_type"] = polarity
            memory["value"] = polarity
            memory["subject"] = extract_memory_subject(
                "preference",
                polarity,
                memory.get("content", ""),
                polarity,
                subject or memory.get("subject"),
            )
        elif content_changed or value is not None or fact_type is not None:
            memory["value"] = extract_memory_value(
                memory.get("category", "other"),
                memory.get("fact_type", "other"),
                memory.get("content", ""),
                value,
            )
            memory["subject"] = extract_memory_subject(
                memory.get("category", "other"),
                memory.get("fact_type", "other"),
                memory.get("content", ""),
                memory.get("value", ""),
                subject,
            )

        if content_changed or fact_type is not None or category is not None:
            memory["embedding"] = self.embedding_store.generate_embedding(
                self._embedding_text_for(memory)
            )
            memory["embedding_version"] = config.EMBEDDING_TEXT_VERSION

        memory["status"] = "active"
        memory["updated_at"] = utc_now_iso()
        self.save()
        return memory

    def delete_memory(self, memory_id: str) -> bool:
        for index, memory in enumerate(self.memories):
            if memory.get("id") == memory_id:
                self.memories.pop(index)
                self.save()
                return True
        return False

    def deactivate_memory(self, memory_id: str) -> bool:
        memory = self.get_by_id(memory_id)
        if memory is None:
            return False
        memory["status"] = "inactive"
        memory["embedding"] = None
        memory["updated_at"] = utc_now_iso()
        self.save()
        return True
