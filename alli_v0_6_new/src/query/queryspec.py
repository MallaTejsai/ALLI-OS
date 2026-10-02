"""Phase 8 — Semantic QuerySpec (rule 22). NO keyword→hardcoded-answer (rule 62)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class QuerySpec:
    intent: str                  # retrieve / compare / explain / broad / narrow
    subject: Optional[str] = None
    target_entity: Optional[str] = None
    memory_kind: Optional[str] = None
    predicate: Optional[str] = None
    requested_value: Optional[str] = None
    relationship_direction: Optional[str] = None  # forward / reverse (rule 14)
    temporal_constraints: list = field(default_factory=list)
    state_constraints: list = field(default_factory=list)
    polarity: Optional[str] = None
    modality: Optional[str] = None
    specificity: str = "specific"
    cardinality: str = "one"     # one / many
    scope: Optional[str] = None
    answer_type: str = "direct"  # direct / derived / unknown / ambiguous / conflicted (rule 27)
    semantic_representation: str = ""
