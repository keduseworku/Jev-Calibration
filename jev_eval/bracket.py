"""Bracket evaluation: Jev between a random floor and an oracle ceiling, with bootstrap CIs.

Report Jev as a fraction of the random-to-oracle gap. This is the framing TypeSafe's own
skill-suggestion cookbook uses (baseline vs suggestion vs oracle, 488 requests) and it neutralises
the confound that the harness (state design, schema, wording) is doing most of the work.
"""
from collections.abc import Callable

import numpy as np


def bracket(items: list, deciders: dict[str, Callable], metric: Callable[[list, list], float],
            n_boot: int = 1000, seed: int = 0, floor: str = "random", ceiling: str = "oracle") -> dict:
    """deciders: name -> f(item) -> decision. metric(decisions, items) -> float (higher is better).
    Must include floor and ceiling keys. Every other decider is reported as gap fraction."""
    for k in (floor, ceiling):
        if k not in deciders:
            raise ValueError(f"deciders must include '{k}'")
    decisions = {name: [f(it) for it in items] for name, f in deciders.items()}
    rng = np.random.default_rng(seed)
    n = len(items)
    boots = {name: [] for name in deciders}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        sub_items = [items[i] for i in idx]
        for name in deciders:
            boots[name].append(metric([decisions[name][i] for i in idx], sub_items))
    out = {"n": n, "metrics": {}, "gap_fraction": {}}
    for name in deciders:
        b = np.asarray(boots[name])
        out["metrics"][name] = {"point": float(metric(decisions[name], items)),
                                "ci90": [float(np.percentile(b, 5)), float(np.percentile(b, 95))]}
    lo, hi = np.asarray(boots[floor]), np.asarray(boots[ceiling])
    for name in deciders:
        if name in (floor, ceiling):
            continue
        g = (np.asarray(boots[name]) - lo) / np.where(hi - lo == 0, np.nan, hi - lo)
        pt = out["metrics"]
        denom = pt[ceiling]["point"] - pt[floor]["point"]
        out["gap_fraction"][name] = {
            "point": float((pt[name]["point"] - pt[floor]["point"]) / denom) if denom else None,
            "ci90": [float(np.nanpercentile(g, 5)), float(np.nanpercentile(g, 95))]}
    return out


def accuracy_metric(decisions, items, truth_key="truth"):
    return float(np.mean([d == it[truth_key] for d, it in zip(decisions, items)]))
