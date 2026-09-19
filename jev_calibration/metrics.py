"""Calibration metrics (ECE/MCE/Brier follow the Classification-with-Confidence definitions)."""
import numpy as np


def _bins(conf, acc, edges):
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            yield m.mean(), conf[m].mean(), acc[m].mean()


def ece(conf, acc, n_bins: int = 10) -> float:
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    edges = np.linspace(0, 1, n_bins + 1)
    return float(sum(w * abs(c - a) for w, c, a in _bins(conf, acc, edges)))


def ece_equal_mass(conf, acc, n_bins: int = 10) -> float:
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    order = np.argsort(conf)
    total = 0.0
    for chunk in np.array_split(order, n_bins):
        if len(chunk):
            total += len(chunk) / len(conf) * abs(conf[chunk].mean() - acc[chunk].mean())
    return float(total)


def mce(conf, acc, n_bins: int = 10) -> float:
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    edges = np.linspace(0, 1, n_bins + 1)
    return float(max((abs(c - a) for _, c, a in _bins(conf, acc, edges)), default=0.0))


def brier(conf, acc) -> float:
    return float(np.mean((np.asarray(conf, float) - np.asarray(acc, float)) ** 2))


def log_loss(conf, acc, eps: float = 1e-6) -> float:
    p = np.clip(np.asarray(conf, float), eps, 1 - eps)
    y = np.asarray(acc, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def summary(conf, acc, n_bins: int = 10) -> dict:
    return {"n": int(len(conf)), "ece": ece(conf, acc, n_bins),
            "ece_equal_mass": ece_equal_mass(conf, acc, n_bins), "mce": mce(conf, acc, n_bins),
            "brier": brier(conf, acc), "log_loss": log_loss(conf, acc),
            "mean_confidence": float(np.mean(conf)), "accuracy": float(np.mean(acc))}


def coverage_curve(conf, acc, thresholds):
    """For each threshold: fraction auto-accepted (conf >= t) and accuracy among accepted."""
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    out = []
    for t in thresholds:
        m = conf >= t
        out.append({"threshold": float(t), "coverage": float(m.mean()),
                    "accuracy": float(acc[m].mean()) if m.any() else None})
    return out
