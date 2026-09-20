"""Counterfactual sampling for reject decisions: expand / execute / review a fixed share of what Jev
rejected, or the false-reject rate is never observed and the calibration set never sees negatives."""
import numpy as np
from scipy.stats import binomtest


def sample_rejected(rejected_ids, rate: float = 0.10, seed: int = 0, min_n: int = 30) -> list:
    ids = list(rejected_ids)
    k = min(len(ids), max(min_n, round(rate * len(ids))))
    return [ids[i] for i in np.random.default_rng(seed).choice(len(ids), size=k, replace=False)]


def false_reject_rate(n_expanded: int, n_should_have_kept: int) -> dict:
    """Entered by hand from review outcomes; Wilson 90% CI. Before any review: rate None, CI [0, 1]."""
    if n_expanded == 0:
        return {"n_expanded": 0, "false_rejects": 0, "rate": None, "ci90": [0.0, 1.0]}
    ci = binomtest(n_should_have_kept, n_expanded).proportion_ci(confidence_level=0.9, method="wilson")
    return {"n_expanded": n_expanded, "false_rejects": n_should_have_kept,
            "rate": n_should_have_kept / n_expanded, "ci90": [ci.low, ci.high]}
