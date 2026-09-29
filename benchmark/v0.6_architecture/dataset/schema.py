"""
STEP 5 — Synthetic Benchmark Dataset Schema.

Defines structured types for synthetic dataset records.
Architecture-neutral; uses existing identity concepts from Step 2/3.
No production memory schema redesign.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Any, List
from uuid import uuid4


class EntityType(Enum):
    PERSON = "PERSON"
    PROJECT = "PROJECT"
    TECHNOLOGY = "TECHNOLOGY"
    ORGANIZATION = "ORGANIZATION"
    EDUCATION = "EDUCATION"
    PLACE = "PLACE"
    DEVICE = "DEVICE"
    PRODUCT = "PRODUCT"
    CONCEPT = "CONCEPT"
    OTHER = "OTHER"


class MemoryKind(Enum):
    FACT = "FACT"
    PREFERENCE = "PREFERENCE"
    GOAL = "GOAL"
    PLAN = "PLAN"
    STATE = "STATE"
    EVENT = "EVENT"
    EXPERIENCE = "EXPERIENCE"
    RELATIONSHIP = "RELATIONSHIP"
    OBSERVATION = "OBSERVATION"


class Polarity(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    UNKNOWN = "UNKNOWN"
    POSSIBLE = "POSSIBLE"


class Modality(Enum):
    ASSERTED = "ASSERTED"
    POSSIBLE = "POSSIBLE"
    PLANNED = "PLANNED"
    HYPOTHETICAL = "HYPOTHETICAL"


@dataclass(frozen=True)
class TemporalInfo:
    """Structured temporal information (not only text)."""
    point_in_time: Optional[str] = None  # ISO date or approximate
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    is_historical: bool = False
    superseded_by: Optional[str] = None  # MemoryClaim ID that supersedes
    temporal_precision: str = "exact"  # exact, approximate, open-ended


@dataclass(frozen=True)
class ProvenanceInfo:
    """Synthetic provenance — distinct from truth."""
    source_episode_id: Optional[str] = None
    source_type: str = "synthetic"
    generation_version: str = "0.1.0"
    generated_at: Optional[str] = None  # deterministic, not runtime


@dataclass(frozen=True)
class Entity:
    """Synthetic entity — benchmark-only, not production."""
    entity_id: str
    entity_type: EntityType
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)
    episode_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Episode:
    """Episode context for synthetic generation."""
    episode_id: str
    episode_type: str = "synthetic"
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryClaim:
    """Synthetic memory claim — compatible with canonical concepts."""
    memory_id: str
    episode_id: Optional[str] = None
    entity_ids: List[str] = field(default_factory=list)
    content: str
    memory_kind: MemoryKind
    polarity: Polarity = Polarity.POSITIVE
    modality: Modality = Modality.ASSERTED
    temporal_info: TemporalInfo = field(default_factory=TemporalInfo)
    provenance: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    confidence: float = 1.0
    scope: Optional[str] = None  # ownership / privacy / cross-user
    duplicate_of: Optional[str] = None  # reference to duplicate memory
    supersedes: Optional[str] = None  # reference to superseded memory
    contradictions: List[str] = field(default_factory=list)  # opposing memory IDs
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Relationship:
    """First-class relationship — not embedded in text."""
    relationship_id: str
    source_entity_id: str
    target_entity_id: str
    relationship_type: str  # e.g., PERSON -> uses -> TECHNOLOGY
    confidence: float = 1.0
    temporal_info: TemporalInfo = field(default_factory=TemporalInfo)
    provenance: ProvenanceInfo = field(default_factory=ProvenanceInfo)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetConfig:
    """Generator configuration — references Step 3 versions."""
    seed: int
    scale: int  # 100, 1000, 10000, 50000, 100000
    dataset_version: str = "0.1.0"
    generator_version: str = "0.1.0"
    benchmark_version: str = "0.6.0"
    composition_targets: dict[str, float] = field(default_factory=lambda: {
        "facts": 0.25,
        "preferences": 0.10,
        "goals_plans": 0.10,
        "states_events_experiences": 0.15,
        "relationships": 0.15,
        "temporal_history": 0.10,
        "contradictions_corrections": 0.05,
        "negation_modality": 0.05,
        "ambiguity_distractors": 0.05,
    })


@dataclass(frozen=True)
class DatasetMetadata:
    """Dataset identity — deterministic from config."""
    dataset_id: str
    dataset_version: str
    generator_version: str
    seed: int
    scale: int
    created_at: Optional[str] = None  # deterministic; not runtime timestamp
    entity_count: int = 0
    episode_count: int = 0
    memory_claim_count: int = 0
    relationship_count: int = 0
    fingerprint: str = ""  # Derive from deterministic content
