"""STEP 6 — Golden Scenario Validation."""
from scenarios.schema import GoldenScenario, ScenarioFamily, Difficulty, Severity, AnswerClass

def validate_scenario(s: GoldenScenario) -> bool:
    if not s.scenario_id:
        raise ValueError("scenario_id required")
    if s.family is None or not isinstance(s.family, ScenarioFamily):
        raise ValueError("valid family required")
    if s.difficulty is None:
        raise ValueError("difficulty required")
    if s.severity is None:
        raise ValueError("severity required")
    if s.query_text == "":
        raise ValueError("query_text required")
    if s.expected_answer is None:
        raise ValueError("expected_answer required")
    if s.expected_answer.answer_class not in [a.value for a in AnswerClass]:
        raise ValueError("invalid answer_class")
    return True
