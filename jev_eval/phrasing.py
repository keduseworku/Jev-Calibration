"""Paraphrase-variance panel: N wordings of the same question, compared on identical examples.

Pre-register the panel and the baseline; pick the wording on the calibration split only. The
anth.us study found 72.3% vs 76.1% between two Noul wordings (p = 6e-6) on the same texts, so a
wording choice is a modelling decision and must be frozen before thresholds are set.
"""
import numpy as np
from scipy.stats import binomtest

from jev_calibration import metrics
from jev_calibration.calibrate import Isotonic

from .calibration import auroc


def mcnemar_exact(correct_a, correct_b) -> float:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    if n01 + n10 == 0:
        return 1.0
    return float(binomtest(n01, n01 + n10, 0.5).pvalue)


def phrasing_panel(variants: dict[str, np.ndarray], labels, is_calibration, baseline: str,
                   cutoff: float = 0.5) -> list[dict]:
    """variants: name -> P(yes) per example (same order). labels: 0/1. is_calibration: bool mask.
    Returns one row per variant with test-split accuracy, AUROC, raw and isotonic ECE, and an exact
    McNemar p-value against the baseline wording."""
    y, cal = np.asarray(labels, int), np.asarray(is_calibration, bool)
    test = ~cal
    base_correct = ((variants[baseline][test] >= cutoff).astype(int) == y[test])
    rows = []
    for name, p in variants.items():
        p = np.asarray(p, float)
        correct = ((p[test] >= cutoff).astype(int) == y[test])
        iso = Isotonic().fit(p[cal], y[cal])
        rows.append({"variant": name, "n_test": int(test.sum()),
                     "accuracy": float(correct.mean()),
                     "auroc": auroc(p[test], y[test]),
                     "ece_raw": metrics.ece(p[test], y[test]),
                     "ece_isotonic": metrics.ece(iso.predict(p[test]), y[test]),
                     "mcnemar_p_vs_baseline": mcnemar_exact(base_correct, correct) if name != baseline else None})
    return rows
