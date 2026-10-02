"""Phase 4 — Canonical Memory Claim (rule 7 / ARCHITECTURE §6). NOT reduced to text+vector."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass
class MemoryClaim:
    memory_id: str
    subject: str          # entity ref
    predicate: str        # relation / attribute
    object_value: str     # value or related entity
    memory_kind: str      # fact/preference/goal/plan/state/event/etc
    polarity: str         # positive / negative (rule 16)
    modality: str         # fact/plan/possibility/intention/hypothetical/preference/uncertain (rule 17)
    temporal_scope: str   # current/historical/point/relative (rule 15)
    provenance: str       # source episode + extractor + timestamp (rule 19)
    evidence_refs: list[str]
    lifecycle_state: str  # candidate/validated/active/superseded/invalidated/archived (rule 12)
    scope: str            # conversation/persistent/project/personal (rule 18)
    created_at: str
    update_time: str
    supersession_ref: Optional[str]  # B supersedes A (rule 11, 12)
    episode_ref: Optional[str]
    source: str
