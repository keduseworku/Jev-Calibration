import numpy as np
from jev_calibration import metrics
from jev_calibration.calibrate import Isotonic, PlattLogit
from jev_calibration.dataset import load_dataset
from jev_calibration.splits import make_splits


def _overconfident(n=20000, seed=0):
    """True P(correct) = s**0.5-ish shrink of reported score: model claims more than it delivers."""
    rng = np.random.default_rng(seed)
    s = rng.beta(0.3, 0.3, n).clip(1e-4, 1 - 1e-4)
    true_p = 1 / (1 + np.exp(-0.5 * np.log(s / (1 - s)) - 0.3))
    return s, (rng.random(n) < true_p).astype(int)


def test_platt_reduces_ece_on_held_out():
    s, y = _overconfident()
    c = PlattLogit().fit(s[:10000], y[:10000])
    assert metrics.ece(c.predict(s[10000:]), y[10000:]) < metrics.ece(s[10000:], y[10000:]) / 3


def test_isotonic_monotone():
    s, y = _overconfident()
    p = Isotonic().fit(s, y).predict(np.linspace(0, 1, 200))
    assert np.all(np.diff(p) >= -1e-12)


def test_ece_zero_when_calibrated():
    rng = np.random.default_rng(1)
    p = rng.random(200000)
    y = (rng.random(200000) < p).astype(int)
    assert metrics.ece(p, y) < 0.01


def test_coverage_curve():
    c = metrics.coverage_curve([0.6, 0.9, 0.95], [0, 1, 1], [0.5, 0.9])
    assert c[0]["coverage"] == 1.0 and c[1]["accuracy"] == 1.0


def test_dataset_and_splits():
    rows = load_dataset()
    assert len(rows) == 8801  # 10,000 lines minus exact within-category repeats
    sp = make_splits(rows)
    assert len(sp) == len(rows)
    assert {v for v in sp.values()} == {"calibration", "test"}
