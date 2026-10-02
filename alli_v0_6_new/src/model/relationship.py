"""Phase 3 — First-class directional relationships (rule 14 / spec §3)."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Relationship:
    rel_id: str
    subject_ref: str        # subject entity id
    predicate: str          # relation type / name
    object_ref: str         # object entity id
    direction: str          # "forward" | "reverse" (query may ask either direction)
    provenance: str = ""
