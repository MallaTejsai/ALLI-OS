"""STEP 6 — Golden Scenario Collection."""
from scenarios.schema import GoldenScenario, ScenarioFamily, Difficulty, Severity

class ScenarioCollection:
    def __init__(self):
        self._scenarios: dict[str, GoldenScenario] = {}

    def add(self, s: GoldenScenario) -> None:
        if s.scenario_id in self._scenarios:
            raise ValueError(f"Duplicate scenario ID: {s.scenario_id}")
        self._scenarios[s.scenario_id] = s

    def get(self, sid: str) -> GoldenScenario:
        if sid not in self._scenarios:
            raise KeyError(f"Scenario not found: {sid}")
        return self._scenarios[sid]

    def filter_family(self, family: ScenarioFamily) -> list:
        return [s for s in self._scenarios.values() if s.family == family]

    def filter_difficulty(self, difficulty: Difficulty) -> list:
        return [s for s in self._scenarios.values() if s.difficulty == difficulty]

    def all_ids(self) -> list:
        return sorted(self._scenarios.keys())
