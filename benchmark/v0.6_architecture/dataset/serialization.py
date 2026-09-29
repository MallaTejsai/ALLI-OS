"""STEP 5 — Dataset serialization."""
import json

def serialize(dataset: dict) -> str:
    return json.dumps(dataset, sort_keys=True, indent=2, default=str)
