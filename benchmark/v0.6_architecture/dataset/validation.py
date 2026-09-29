"""STEP 5 — Dataset validation."""
from dataset.schema import DatasetConfig, MemoryClaim, Entity, Relationship

class DatasetValidator:
    def validate(self, dataset: dict) -> bool:
        required = ["metadata", "entities", "episodes", "memory_claims", "relationships"]
        for k in required:
            if k not in dataset:
                raise ValueError(f"Missing dataset key: {k}")
        meta = dataset["metadata"]
        if meta.scale <= 0:
            raise ValueError("Scale must be positive")
        return True
