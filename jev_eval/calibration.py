"""Per-question calibration bound to a question fingerprint.

Rule from the anth.us study: rank-based uses need no labels; any threshold (accept / reject /
block / auto-approve) must be set in calibrated-probability terms, per question, per primitive,
per model version, on a few hundred labeled examples with a holdout.
"""
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from jev_calibration import metrics
from jev_calibration.calibrate import CALIBRATORS, Isotonic, PlattLogit


def raw_score(answer: dict) -> float:
    """Top-label probability for any primitive: the quantity to calibrate against correctness."""
    t = answer["type"]
    if t == "noul":
        p = answer["noul"]
        return max(p, 1 - p)
    return max(answer["probabilities"].values())


def predicted(answer: dict):
    t = answer["type"]
    if t == "noul":
        return answer["noul"] >= 0.5
    if t == "choice":
        return answer["choice"]
    return max(answer["probabilities"], key=answer["probabilities"].get)


def auroc(conf, correct) -> float | None:
    correct = np.asarray(correct, int)
    if correct.min() == correct.max():
        return None
    return float(roc_auc_score(correct, np.asarray(conf, float)))


@dataclass
class CalibratedQuestion:
    name: str
    fingerprint: str
    method: str
    x: list = field(default_factory=list)  # isotonic knots (or platt params in x=[a,b])
    y: list = field(default_factory=list)
    n_fit: int = 0
    report: dict = field(default_factory=dict)

    def predict(self, raw):
        raw = np.asarray(raw, float)
        if self.method == "isotonic":
            return np.interp(raw, self.x, self.y)
        a, b = self.x
        from jev_calibration.calibrate import logit
        return 1 / (1 + np.exp(-(a * logit(raw) + b)))

    def check(self, fingerprint: str):
        if fingerprint != self.fingerprint:
            raise ValueError(f"calibrator for {self.name} was fit on fingerprint {self.fingerprint}, "
                             f"got {fingerprint}: wording/options/model changed, refit before thresholding")

    def save(self, path):
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


def calibrate_question(raw, correct, name: str, fingerprint: str, method: str = "isotonic",
                       holdout: float = 0.4, seed: int = 0, min_n: int = 50):
    """Fit on (1-holdout) of the data, report raw / platt / isotonic on the holdout, return the bundle."""
    raw, correct = np.asarray(raw, float), np.asarray(correct, int)
    if len(raw) < min_n:
        raise ValueError(f"{name}: {len(raw)} labeled examples < {min_n}; below ~50 a calibrator is noise "
                         f"(anth.us calibration-size result)")
    idx = np.arange(len(raw))
    strat = correct if 0 < correct.mean() < 1 else None
    fit_i, ho_i = train_test_split(idx, test_size=holdout, random_state=seed, stratify=strat)
    report = {"n_fit": int(len(fit_i)), "n_holdout": int(len(ho_i)),
              "auroc_holdout": auroc(raw[ho_i], correct[ho_i]),
              "raw": metrics.summary(raw[ho_i], correct[ho_i])}
    fitted = {}
    for cname in ("platt_logit", "isotonic"):
        c = CALIBRATORS[cname]().fit(raw[fit_i], correct[fit_i])
        fitted[cname] = c
        report[cname] = metrics.summary(c.predict(raw[ho_i]), correct[ho_i])
    c = fitted[method]
    if method == "isotonic":
        x, y = [float(v) for v in c.iso.X_thresholds_], [float(v) for v in c.iso.y_thresholds_]
    else:
        x, y = [c.params["a"], c.params["b"]], []
    return CalibratedQuestion(name=name, fingerprint=fingerprint, method=method, x=x, y=y,
                              n_fit=int(len(fit_i)), report=report)


def threshold_for_accuracy(calibrated_conf, correct, target: float = 0.95) -> dict:
    """Lowest calibrated-confidence threshold whose auto-accepted set stays at or above target accuracy."""
    conf, correct = np.asarray(calibrated_conf, float), np.asarray(correct, float)
    order = np.argsort(-conf)
    cum_acc = np.cumsum(correct[order]) / np.arange(1, len(order) + 1)
    ok = np.where(cum_acc >= target)[0]
    if len(ok) == 0:
        return {"target": target, "threshold": None, "coverage": 0.0, "accuracy": None}
    k = int(ok.max())
    return {"target": target, "threshold": float(conf[order][k]), "coverage": float((k + 1) / len(conf)),
            "accuracy": float(cum_acc[k])}
