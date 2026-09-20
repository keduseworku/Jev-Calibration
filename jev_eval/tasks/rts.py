"""Demo #5: many units, each deciding every tick. Rank-based (argmax action), game outcome is the oracle.

The binding constraint is the rate limit (1,200 requests/min, 250k tokens/s, 32k state per request),
so units are packed into one state and asked per-unit questions via path references. Threat is a
named-bucket Score, never a number (docs: numeric magnitudes are weak).
"""
from typesafe_sdk import Choice, Score

ACTIONS = {"advance": "move toward the nearest objective", "retreat": "move away from the nearest enemy",
           "attack": "engage the nearest enemy", "hold": "stay in position", "regroup": "move toward the nearest friendly unit"}
THREAT_LEVELS = ["no enemies nearby", "enemies visible but distant", "enemies close, unit healthy",
                 "enemies close, unit damaged", "surrounded or nearly destroyed"]


def plan_batches(n_units: int, hz: float, tokens_per_unit: int = 60, overhead_tokens: int = 800,
                 questions_per_unit: int = 2, tokens_per_question: int = 45, state_limit: int = 32_000,
                 request_limit: int = 64_000, rpm: int = 1_200, tps: int = 250_000) -> dict:
    """How many units fit per request, and whether n_units at hz decisions/s is feasible under the limits."""
    by_state = (state_limit - overhead_tokens) // tokens_per_unit
    by_request = (request_limit - overhead_tokens) // (tokens_per_unit + questions_per_unit * tokens_per_question)
    upr = max(1, min(by_state, by_request))
    req_per_s = n_units * hz / upr
    tokens_per_req = overhead_tokens + upr * (tokens_per_unit + questions_per_unit * tokens_per_question)
    tok_per_s = req_per_s * tokens_per_req
    max_req_per_s = min(rpm / 60, tps / tokens_per_req)
    return {"units_per_request": upr, "requests_per_s_needed": req_per_s, "tokens_per_s_needed": tok_per_s,
            "max_requests_per_s": max_req_per_s, "feasible": req_per_s <= max_req_per_s,
            "max_hz_at_n_units": max_req_per_s * upr / n_units,
            "max_units_at_hz": int(max_req_per_s * upr / hz)}


def pack_units(units: list[dict]) -> dict:
    """units: [{"id", "hp", "position", "nearby_enemies", "nearby_allies", "objective"}] -> one state."""
    return {"units": units}


def questions(n_units: int) -> dict:
    qs = {}
    for i in range(n_units):
        qs[f"u{i}_action"] = Choice(instructions=f"What should `units[{i}]` do this tick?", criteria=ACTIONS)
        qs[f"u{i}_threat"] = Score(instructions=f"How threatened is `units[{i}]`?", criteria=THREAT_LEVELS)
    return qs


def decisions(answers: dict, n_units: int) -> list[dict]:
    return [{"action": answers[f"u{i}_action"]["choice"],
             "threat_level": int(round(answers[f"u{i}_threat"]["score"]))} for i in range(n_units)]
