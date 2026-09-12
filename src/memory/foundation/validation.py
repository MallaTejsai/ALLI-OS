"""Structural validation for ALLI v0.6 foundation objects.

Rejects structurally invalid objects before they enter memory.
Does NOT implement natural-language validation (that belongs to
the extraction layer in a later session).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .models import (
    EntityRef,
    Fact,
    ImportanceLevel,
    Lifecycle,
    LifecycleState,
    Memory,
    MemoryEvent,
    MemoryType,
    Provenance,
    TemporalScope,
    EventType,
)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    errors: list[str]

    def __bool__(self) -> bool:
        return self.is_valid


def validate_entity_ref(ref: EntityRef) -> ValidationResult:
    """Validate that an EntityRef is structurally sound."""
    errors: list[str] = []

    if not ref.name or not ref.name.strip():
        errors.append("EntityRef.name must be non-empty")

    from .models import EntityType
    valid_types = {t.value for t in EntityType}
    if ref.entity_type not in valid_types:
        errors.append(f"EntityRef.entity_type must be one of {valid_types}, got '{ref.entity_type}'")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_fact(fact: Fact) -> ValidationResult:
    """Validate that a Fact is structurally sound."""
    errors: list[str] = []

    if not fact.fact_type or not fact.fact_type.strip():
        errors.append("Fact.fact_type must be non-empty")

    if not fact.content or not fact.content.strip():
        errors.append("Fact.content must be non-empty")

    if fact.polarity is not None and fact.polarity not in {"positive", "negative"}:
        errors.append(f"Fact.polarity must be 'positive', 'negative', or None, got '{fact.polarity}'")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_temporal(temporal: TemporalScope) -> ValidationResult:
    """Validate that a TemporalScope is structurally sound."""
    errors: list[str] = []

    if not temporal.valid_from:
        errors.append("TemporalScope.valid_from must be non-empty")

    if (
        temporal.valid_until is not None
        and temporal.valid_from
        and temporal.valid_until < temporal.valid_from
    ):
        errors.append("TemporalScope.valid_until must be >= valid_from")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_provenance(provenance: Provenance) -> ValidationResult:
    """Validate that a Provenance is structurally sound."""
    errors: list[str] = []

    from .models import ProvenanceSource
    valid_sources = {s.value for s in ProvenanceSource}
    if provenance.source not in valid_sources:
        errors.append(f"Provenance.source must be one of {valid_sources}, got '{provenance.source}'")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_lifecycle(lifecycle: Lifecycle) -> ValidationResult:
    """Validate that a Lifecycle is structurally sound."""
    errors: list[str] = []

    valid_states = {s.value for s in LifecycleState}
    if lifecycle.state not in valid_states:
        errors.append(f"Lifecycle.state must be one of {valid_states}, got '{lifecycle.state}'")

    if lifecycle.superseded_by and lifecycle.state != LifecycleState.INACTIVE.value:
        errors.append("Lifecycle.superseded_by requires state to be 'inactive'")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_memory(memory: Memory) -> ValidationResult:
    """Validate that a Memory record is structurally sound.

    This is the primary validation entry point for storage.
    Checks all sub-components and cross-field constraints.
    """
    errors: list[str] = []

    # ID
    if not memory.id or not memory.id.strip():
        errors.append("Memory.id must be non-empty")

    # Entity
    entity_result = validate_entity_ref(memory.entity)
    errors.extend(entity_result.errors)

    # Fact
    fact_result = validate_fact(memory.fact)
    errors.extend(fact_result.errors)

    # Domain
    if not memory.domain or not memory.domain.strip():
        errors.append("Memory.domain must be non-empty")

    # Confidence
    if not (0.0 <= memory.confidence <= 1.0):
        errors.append(f"Memory.confidence must be 0.0-1.0, got {memory.confidence}")

    # Importance
    valid_importance = {level.value for level in ImportanceLevel}
    if memory.importance not in valid_importance:
        errors.append(f"Memory.importance must be one of {valid_importance}, got '{memory.importance}'")

    # Memory type
    valid_memory_types = {t.value for t in MemoryType}
    if memory.memory_type not in valid_memory_types:
        errors.append(f"Memory.memory_type must be one of {valid_memory_types}, got '{memory.memory_type}'")

    # Lifecycle
    lifecycle_result = validate_lifecycle(memory.lifecycle)
    errors.extend(lifecycle_result.errors)

    # Temporal
    temporal_result = validate_temporal(memory.temporal)
    errors.extend(temporal_result.errors)

    # Provenance
    provenance_result = validate_provenance(memory.provenance)
    errors.extend(provenance_result.errors)

    # Cross-field: expired memory must not be active
    if memory.lifecycle.is_active and memory.temporal.is_expired_at():
        errors.append("Active memory must not be temporally expired")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)


def validate_event(event: MemoryEvent) -> ValidationResult:
    """Validate that a MemoryEvent is structurally sound."""
    errors: list[str] = []

    if not event.id or not event.id.strip():
        errors.append("MemoryEvent.id must be non-empty")

    valid_types = {e.value for e in EventType}
    if event.event_type not in valid_types:
        errors.append(f"MemoryEvent.event_type must be one of {valid_types}, got '{event.event_type}'")

    if not event.memory_id or not event.memory_id.strip():
        errors.append("MemoryEvent.memory_id must be non-empty")

    if not event.timestamp:
        errors.append("MemoryEvent.timestamp must be non-empty")

    return ValidationResult(is_valid=len(errors) == 0, errors=errors)
