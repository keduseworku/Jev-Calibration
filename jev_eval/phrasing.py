"""Paraphrase panel: N wordings of the same question on identical examples, both splits reported.

Choose the wording on the `cal_*` columns only; `test_*` columns are report-only (PREREGISTRATION 4).
anth.us found 72.3% vs 76.1% between two Noul wordings on the same texts (p = 6e-6), so the wording
is a modelling decision and must be frozen before thresholds are set.
"""
import numpy as np
from scipy.stats import binomtest

from jev_calibration import metrics
from jev_calibration.calibrate import Isotonic

from .calibration import auroc


def mcnemar_exact(correct_a, correct_b) -> float:
    a, b = np.asarray(correct_a, bool), np.asarray(correct_b, bool)
    n01, n10 = int((a & ~b).sum()), int((~a & b).sum())
    return 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)


def phrasing_panel(variants: dict, labels, is_calibration, baseline: str) -> list[dict]:
    """variants: name -> P(yes) per example. labels: 0/1. is_calibration: bool mask."""
    y, cal = np.asarray(labels, int), np.asarray(is_calibration, bool)
    p_all = {k: np.asarray(v, float) for k, v in variants.items()}
    correct = lambda p, m: (p[m] >= 0.5) == y[m]
    rows = []
    for name, p in p_all.items():
        row = {"variant": name}
        for tag, m in (("cal", cal), ("test", ~cal)):
            c = correct(p, m)
            row.update({f"{tag}_accuracy": float(c.mean()), f"{tag}_auroc": auroc(p[m], y[m]),
                        f"{tag}_ece_raw": metrics.ece(p[m], y[m]),
                        f"{tag}_mcnemar_p_vs_baseline": None if name == baseline else mcnemar_exact(correct(p_all[baseline], m), c)})
        row["test_ece_isotonic"] = metrics.ece(Isotonic().fit(p[cal], y[cal]).predict(p[~cal]), y[~cal])
        rows.append(row)
    return rows
