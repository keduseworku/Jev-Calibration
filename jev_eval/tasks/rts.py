"""Demo #5: many units, each deciding every tick. Rank-based (argmax action); the game is the oracle.

The binding constraint is the published limits (1,200 requests/min, 250k tokens/s, 32k state and 64k
total per request), so units are packed into one state and asked per-unit questions via path
references. Threat is a named-level Score, never a number (docs: numeric magnitudes are weak).
"""
from typesafe_sdk import Choice, Score

ACTIONS = {"advance": "move toward the nearest objective", "retreat": "move away from the nearest enemy",
           "attack": "engage the nearest enemy", "hold": "stay in position", "regroup": "move toward the nearest friendly unit"}
THREAT_LEVELS = ["no enemies nearby", "enemies visible but distant", "enemies close, unit healthy",
                 "enemies close, unit damaged", "surrounded or nearly destroyed"]
TOKENS_PER_UNIT, OVERHEAD_TOKENS, TOKENS_PER_QUESTION = 60, 800, 45  # assumptions, not measured; the docs give no per-request question cap
STATE_LIMIT, REQUEST_LIMIT, RPM, TPS = 32_000, 64_000, 1_200, 250_000


def plan_batches(n_units: int, hz: float) -> dict:
    """Units per request under the published limits and the token assumptions above; whether n_units at hz is feasible."""
    if hz <= 0 or n_units <= 0:
        raise ValueError("n_units and hz must be positive")
    per_unit = TOKENS_PER_UNIT + 2 * TOKENS_PER_QUESTION
    upr = max(1, min((STATE_LIMIT - OVERHEAD_TOKENS) // TOKENS_PER_UNIT, (REQUEST_LIMIT - OVERHEAD_TOKENS) // per_unit))
    tokens_per_req = OVERHEAD_TOKENS + upr * per_unit
    max_req_per_s = min(RPM / 60, TPS / tokens_per_req)
    need = n_units * hz / upr
    return {"units_per_request": upr, "requests_per_s_needed": need, "max_requests_per_s": max_req_per_s,
            "feasible": need <= max_req_per_s, "max_hz_at_n_units": max_req_per_s * upr / n_units,
            "max_units_at_hz": int(max_req_per_s * upr / hz)}


def questions(n_units: int) -> dict:
    qs = {}
    for i in range(n_units):
        qs[f"u{i}_action"] = Choice(instructions=f"What should `units[{i}]` do this tick?", criteria=ACTIONS)
        qs[f"u{i}_threat"] = Score(instructions=f"How threatened is `units[{i}]`?", criteria=THREAT_LEVELS)
    return qs
