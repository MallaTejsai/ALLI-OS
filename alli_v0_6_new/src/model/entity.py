"""Phase 3 — First-class entity (rule 13 / ARCHITECTURE §6 / spec §3)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List

ENTITY_TYPES = set(["PERSON","PROJECT","TECHNOLOGY","ORGANIZATION","EDUCATION","PLACE","DEVICE","PRODUCT","CONCEPT","OTHER"])

@dataclass
class Entity:
    entity_id: str
    type: str                # from ENTITY_TYPES
    canonical_name: str
    aliases: List[str] = None
    detected: bool = True     # entity detected (separate from resolution)
    resolved: bool = False    # canonical identity established?
    ambiguous: bool = False   # ambiguous — ask clarification (rule 24/13)
    unresolved: bool = False  # detected but not resolved (must NOT invent identity — rule 13)
    provenance: Optional[str] = None
