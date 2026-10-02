"""Phase 6 — Separate evidence layer; NOT auto-persisted (rule 9 / spec §3/§6)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ConversationContext:
    session_id: str
    recent_messages: List[str] = None     # working context, temporary
    current_entities: List[str] = None     # references, not durable
    current_topic: Optional[str] = None
    unresolved_ambiguity: bool = False
    provenance_tag: str = "conversation-context"  # distinguish from persistent (rule 9 / 19)
