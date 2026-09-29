"""STEP 6 — Golden Scenario Serialization."""
import json

class ScenarioSerializer:
    @staticmethod
    def to_json(s) -> str:
        return json.dumps(s.to_dict(), sort_keys=True, indent=2)

    @staticmethod
    def from_dict(data: dict):
        from scenarios.schema import GoldenScenario, ScenarioFamily, Difficulty, Severity, ExpectedAnswer, ExpectedEvidence
        return GoldenScenario(
            scenario_id=data["scenario_id"],
            family=ScenarioFamily(data["family"]),
            difficulty=Difficulty(data["difficulty"]),
            severity=Severity(data["severity"]),
            dataset_ref=data.get("dataset_ref", ""),
            initial_state_ref=data.get("initial_state_ref"),
            query_text=data["query_text"],
            expected_interpretation=data.get("expected_interpretation", ""),
            expected_memory_refs=data.get("expected_memory_refs", []),
            expected_evidence=ExpectedEvidence(**data["expected_evidence"]) if data.get("expected_evidence") else None,
            expected_answer=ExpectedAnswer(**data["expected_answer"]) if data.get("expected_answer") else None,
            expected_behavior=data.get("expected_behavior", []),
            failure_expectation=data.get("failure_expectation"),
            scope=data.get("scope"),
            provenance_ref=data.get("provenance_ref"),
            version=data.get("version", "0.1.0"),
            metadata=data.get("metadata", {}),
        )
