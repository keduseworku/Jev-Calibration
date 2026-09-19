"""Calibrators, always fit explicitly on the calibration split only."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 0.005  # Jev rounds probabilities to 2 decimals; clip at half that resolution


def logit(p, eps: float = EPS):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return np.log(p / (1 - p))


class PlattLogit:
    """Platt scaling on logit(score): P(y=1) = sigmoid(a * logit(s) + b)."""

    def __init__(self, eps: float = EPS):
        self.eps = eps

    def fit(self, scores, labels):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)  # effectively unregularized
        self.lr.fit(logit(scores, self.eps).reshape(-1, 1), np.asarray(labels, int))
        return self

    def predict(self, scores):
        return self.lr.predict_proba(logit(scores, self.eps).reshape(-1, 1))[:, 1]

    @property
    def params(self):
        return {"a": float(self.lr.coef_[0, 0]), "b": float(self.lr.intercept_[0])}


class PlattRaw:
    """Platt scaling on the raw score, as in Classification-with-Confidence (but on all fit data)."""

    def fit(self, scores, labels):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(np.asarray(scores, float).reshape(-1, 1), np.asarray(labels, int))
        return self

    def predict(self, scores):
        return self.lr.predict_proba(np.asarray(scores, float).reshape(-1, 1))[:, 1]

    @property
    def params(self):
        return {"a": float(self.lr.coef_[0, 0]), "b": float(self.lr.intercept_[0])}


class Isotonic:
    def fit(self, scores, labels):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.iso.fit(np.asarray(scores, float), np.asarray(labels, float))
        return self

    def predict(self, scores):
        return self.iso.predict(np.asarray(scores, float))

    @property
    def params(self):
        return {"n_knots": int(len(self.iso.X_thresholds_))}


CALIBRATORS = {"platt_logit": PlattLogit, "platt_raw": PlattRaw, "isotonic": Isotonic}
