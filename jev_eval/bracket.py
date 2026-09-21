"""Bracket: floor, Jev, oracle on the same items, with 90% bootstrap CIs.

Headline = Jev's fraction of the floor-to-oracle gap (the framing of TypeSafe's own skill-suggestion
cookbook). For tasks whose truth is by construction the oracle is 1 by definition and the floor
carries the harness information; for tree pruning the oracle is hindsight pruning and is not 1.
The default floor is the better of uniform-random and majority-class guessing. When the floor is
within MIN_GAP of the oracle the gap fraction is ill-conditioned and reported as None.
"""
from collections import Counter

import numpy as np

MIN_GAP = 0.05


def floor_scores(truth, n_options: int) -> np.ndarray:
    """Per-item expected accuracy of the better trivial baseline."""
    mode = Counter(truth).most_common(1)[0][0]
    majority = np.array([t == mode for t in truth], float)
    return majority if majority.mean() > 1 / n_options else np.full(len(majority), 1 / n_options)


def _boot(stat, arrs, n_boot, seed, paired):
    """Percentile bootstrap; resamples where stat is not finite are dropped, and the CI is None if
    fewer than half survive or any array has fewer than 2 items."""
    if min(len(a) for a in arrs) < 2:
        return None
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        if paired:
            idx = rng.integers(0, len(arrs[0]), len(arrs[0]))
            samples = [a[idx] for a in arrs]
        else:
            samples = [a[rng.integers(0, len(a), len(a))] for a in arrs]
        with np.errstate(divide="ignore", invalid="ignore"):
            vals.append(stat(*samples))
    ok = np.array(vals, float)
    ok = ok[np.isfinite(ok)]
    return [float(np.percentile(ok, 5)), float(np.percentile(ok, 95))] if len(ok) >= n_boot / 2 else None


def bracket(floor, jev, oracle, n_boot: int = 1000, seed: int = 0) -> dict:
    """Per-item score arrays (0/1 or expected). Points for all three, 90% CIs for Jev and the gap."""
    f, j, o = (np.asarray(a, float) for a in (floor, jev, oracle))
    gap = lambda f, j, o: (j.mean() - f.mean()) / (o.mean() - f.mean())
    out = {"n": int(len(j)), "floor": {"point": float(f.mean())}, "oracle": {"point": float(o.mean())},
           "jev": {"point": float(j.mean()), "ci90": _boot(np.mean, [j], n_boot, seed, True)}}
    if o.mean() - f.mean() >= MIN_GAP:
        out["gap"] = {"point": float(gap(f, j, o)), "ci90": _boot(gap, [f, j, o], n_boot, seed, True)}
    else:
        out["gap"] = {"point": None, "ci90": None, "note": f"floor within {MIN_GAP} of oracle"}
    return out


def difference(a, b, paired: bool, n_boot: int = 1000, seed: int = 0) -> dict:
    """mean(a) - mean(b) with a 90% bootstrap CI; paired resamples index a and b together."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return {"point": float(a.mean() - b.mean()), "paired": paired,
            "ci90": _boot(lambda a, b: a.mean() - b.mean(), [a, b], n_boot, seed, paired)}
