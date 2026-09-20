"""Counterfactual sampling for reject decisions (pruned branches, blocked tool calls, excluded fields).

If Jev says "reject" and you never look, you never learn the false-reject rate, and the
calibration set for that question never contains negatives. Pre-register the sampling rate.
"""
import numpy as np
from scipy.stats import norm


def sample_rejected(ids, rejected, rate: float = 0.10, seed: int = 0, min_n: int = 30) -> list:
    """Uniform sample of rejected ids to expand / execute / review anyway."""
    ids, rejected = np.asarray(ids), np.asarray(rejected, bool)
    pool = ids[rejected]
    k = min(len(pool), max(min_n, int(round(rate * len(pool)))))
    rng = np.random.default_rng(seed)
    return list(rng.choice(pool, size=k, replace=False)) if k else []


def wilson(k: int, n: int, z: float = 1.645) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, d = k / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def false_reject_rate(n_expanded: int, n_should_have_kept: int) -> dict:
    lo, hi = wilson(n_should_have_kept, n_expanded)
    return {"n_expanded": n_expanded, "false_rejects": n_should_have_kept,
            "rate": n_should_have_kept / n_expanded if n_expanded else None, "ci90": [lo, hi]}


def expansions_for_precision(p_guess: float, half_width: float, conf: float = 0.90) -> int:
    """How many rejected items to expand for a CI of +/- half_width around a guessed false-reject rate."""
    z = norm.ppf(0.5 + conf / 2)
    return int(np.ceil(z * z * p_guess * (1 - p_guess) / half_width ** 2))
