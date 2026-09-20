"""Per-question calibration: calibrator and threshold from the fit split, every reported number
from the holdout.

Rank-based uses need no labels; any accept / reject / block threshold is set in calibrated
probability, per question, per primitive, per model id. Calibrators are never persisted: they are
recomputed each run from answers cached under the question fingerprint.
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from jev_calibration import metrics
from jev_calibration.calibrate import Isotonic, PlattLogit


def raw_score(answer: dict) -> float:
    """Top-label probability: the quantity calibrated against correctness."""
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())


def predicted(answer: dict):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    if answer["type"] == "score":
        return max(answer["probabilities"], key=answer["probabilities"].get)  # argmax level key, never the mean
    return answer["choice"]


def auroc(conf, correct):
    correct = np.asarray(correct, int)
    if len(correct) == 0 or correct.min() == correct.max():
        return None
    return float(roc_auc_score(correct, np.asarray(conf, float)))


def threshold_for_accuracy(conf, correct, target: float) -> dict:
    """Lowest t such that accuracy among conf >= t reaches target. Sweeps the distinct values with
    upstream coverage_curve, so ties (isotonic steps, 2-decimal rounding) count as applied."""
    rows = [r for r in metrics.coverage_curve(conf, correct, np.unique(conf))
            if r["accuracy"] is not None and r["accuracy"] >= target]
    return rows[0] if rows else {"threshold": None, "coverage": None, "accuracy": None}


def calibrate_question(raw, correct, target_acc: float = 0.95, holdout: float = 0.4, seed: int = 0,
                       min_n: int = 50, prereg_n: int = 200, groups=None):
    """60/40 split, stratified, or by group when `groups` is given (items sharing a group stay on one
    side). Isotonic and the accuracy-targeted threshold come from the fit split; ECE, AUROC and the
    delivered coverage / accuracy at that threshold come from the holdout.
    Returns (isotonic, report, holdout_indices)."""
    raw, correct = np.asarray(raw, float), np.asarray(correct, int)
    if len(raw) < min_n:
        raise ValueError(f"{len(raw)} labeled examples < {min_n}: below ~50 a calibrator is noise")
    if groups is not None:
        fit, ho = next(GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed).split(raw, groups=groups))
    else:
        both = min(correct.sum(), len(correct) - correct.sum()) >= 2
        fit, ho = train_test_split(np.arange(len(raw)), test_size=holdout, random_state=seed, stratify=correct if both else None)
    iso = Isotonic().fit(raw[fit], correct[fit])
    cal_fit, cal_ho = iso.predict(raw[fit]), iso.predict(raw[ho])
    single_class = correct[fit].min() == correct[fit].max()
    report = {"n_fit": int(len(fit)), "n_holdout": int(len(ho)), "below_prereg_floor": len(raw) < prereg_n,
              "single_class_fit": bool(single_class), "auroc_holdout": auroc(raw[ho], correct[ho]),
              "raw": metrics.summary(raw[ho], correct[ho]), "isotonic": metrics.summary(cal_ho, correct[ho])}
    chosen = {"threshold": None} if single_class else threshold_for_accuracy(cal_fit, correct[fit], target_acc)
    t = chosen["threshold"]
    delivered = metrics.coverage_curve(cal_ho, correct[ho], [t])[0] if t is not None else {"coverage": None, "accuracy": None}
    report["threshold"] = {"target": target_acc, "threshold": t, "chosen_on": "fit",
                           "holdout_coverage": delivered["coverage"], "holdout_accuracy": delivered["accuracy"]}
    if not single_class:
        report["platt"] = metrics.summary(PlattLogit().fit(raw[fit], correct[fit]).predict(raw[ho]), correct[ho])
    return iso, report, ho
