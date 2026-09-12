"""Stable identifier generation for ALLI v0.6 foundation objects.

Identity must not depend on the exact wording of a user's sentence.
IDs are generated deterministically and are stable across sessions.
"""

from __future__ import annotations

import threading
from typing import Optional


# Thread-safe counters for ID generation
_entity_counter = 0
_memory_counter = 0
_event_counter = 0
_episode_counter = 0
_relationship_counter = 0
_lock = threading.Lock()


def generate_entity_id(name: Optional[str] = None) -> str:
    """Generate a stable entity ID.

    For well-known entities (self), returns a deterministic ID.
    For others, generates a sequential ID.
    """
    global _entity_counter
    if name and name.strip().lower() == "self":
        return "entity_000"
    with _lock:
        _entity_counter += 1
        return f"entity_{_entity_counter:03d}"


def generate_memory_id() -> str:
    """Generate a unique memory ID."""
    global _memory_counter
    with _lock:
        _memory_counter += 1
        return f"memory_{_memory_counter:03d}"


def generate_event_id() -> str:
    """Generate a unique memory event ID."""
    global _event_counter
    with _lock:
        _event_counter += 1
        return f"event_{_event_counter:03d}"


def generate_episode_id() -> str:
    """Generate a unique episode ID."""
    global _episode_counter
    with _lock:
        _episode_counter += 1
        return f"episode_{_episode_counter:03d}"


def generate_relationship_id() -> str:
    """Generate a unique relationship ID."""
    global _relationship_counter
    with _lock:
        _relationship_counter += 1
        return f"rel_{_relationship_counter:03d}"


def reset_counters() -> None:
    """Reset all counters. For testing only."""
    global _entity_counter, _memory_counter, _event_counter
    global _episode_counter, _relationship_counter
    with _lock:
        _entity_counter = 0
        _memory_counter = 0
        _event_counter = 0
        _episode_counter = 0
        _relationship_counter = 0


def init_counters_from_db(conn) -> None:
    """Initialize counters from existing database IDs to prevent collisions.

    Must be called after the database schema exists. Reads the highest
    numeric suffix from each ID table and sets counters above that value.
    """
    global _memory_counter, _event_counter, _entity_counter
    import re

    def _max_suffix(table: str, id_column: str, prefix: str) -> int:
        try:
            row = conn.execute(
                f"SELECT MAX({id_column}) FROM {table}"
            ).fetchone()
            if row and row[0]:
                match = re.search(r"(\d+)$", str(row[0]))
                if match:
                    return int(match.group(1))
        except Exception:
            pass
        return 0

    with _lock:
        _memory_counter = max(_memory_counter, _max_suffix("memories", "memory_id", "memory_"))
        _event_counter = max(_event_counter, _max_suffix("memory_events", "event_id", "event_"))
        _entity_counter = max(_entity_counter, _max_suffix("entities", "entity_id", "entity_"))
