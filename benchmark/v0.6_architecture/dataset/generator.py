"""Synthetic dataset generator — deterministic, seed-controlled, no external dependencies."""
from dataset.schema import (
    Entity, Episode, MemoryClaim, Relationship,
    TemporalInfo, ProvenanceInfo, DatasetConfig, DatasetMetadata,
    EntityType, MemoryKind, Polarity, Modality,
)
import random
from typing import List, Optional


class SyntheticDatasetGenerator:
    """
    Deterministic synthetic dataset generator.
    No model dependency. No external API. No cloud.
    Uses deterministic random seed for reproducibility.
    """

    def __init__(self, config: DatasetConfig):
        self.config = config
        self.rng = random.Random(config.seed)

    def generate(self, scale: Optional[int] = None) -> dict:
        """
        Generate synthetic dataset at requested scale.
        If scale is None, uses config.scale.
        Returns dict with entities, episodes, claims, relationships.
        """
        target = scale or self.config.scale
        entities = self._generate_entities(min(target // 10, 500))
        episodes = self._generate_episodes(min(len(entities) // 2, 100))
        claims = self._generate_memory_claims(entities, episodes, target)
        relationships = self._generate_relationships(entities, min(target // 20, 1000))
        return {
            "metadata": DatasetMetadata(
                dataset_id=f"ds_{self.config.seed}_{target}",
                dataset_version=self.config.dataset_version,
                generator_version=self.config.generator_version,
                seed=self.config.seed,
                scale=target,
                entity_count=len(entities),
                episode_count=len(episodes),
                memory_claim_count=len(claims),
                relationship_count=len(relationships),
            ),
            "entities": entities,
            "episodes": episodes,
            "memory_claims": claims,
            "relationships": relationships,
        }

    def _generate_entities(self, count: int) -> List[Entity]:
        types = list(EntityType)
        names = ["Person", "Project", "Tech", "Org", "Edu", "Place", "Device", "Product", "Concept", "Other"]
        result = []
        for i in range(count):
            result.append(Entity(
                entity_id=f"ent_{self.config.seed}_{i:05d}",
                entity_type=self.rng.choice(types),
                name=f"{names[i % len(names)]}_{i}",
            ))
        return result

    def _generate_episodes(self, count: int) -> List[Episode]:
        result = []
        for i in range(count):
            result.append(Episode(
                episode_id=f"ep_{self.config.seed}_{i:05d}",
                episode_type="synthetic",
            ))
        return result

    def _generate_memory_claims(self, entities, episodes, target: int) -> List[MemoryClaim]:
        kinds = list(MemoryKind)
        result = []
        for i in range(target):
            entity_ids = [entities[i % len(entities)].entity_id] if entities else ["ent_default"]
            result.append(MemoryClaim(
                memory_id=f"mem_{self.config.seed}_{i:05d}",
                episode_id=episodes[i % len(episodes)].episode_id if episodes else None,
                entity_ids=entity_ids,
                content=f"Synthetic claim {i} on {entity_ids[0]}",
                memory_kind=self.rng.choice(kinds),
                polarity=self.rng.choice(list(Polarity)),
                modality=self.rng.choice(list(Modality)),
            ))
        return result

    def _generate_relationships(self, entities, count: int) -> List[Relationship]:
        result = []
        for i in range(count):
            src = entities[i % len(entities)].entity_id if entities else "ent_0"
            tgt = entities[(i + 1) % len(entities)].entity_id if entities else "ent_1"
            result.append(Relationship(
                relationship_id=f"rel_{self.config.seed}_{i:05d}",
                source_entity_id=src,
                target_entity_id=tgt,
                relationship_type="uses",
            ))
        return result
