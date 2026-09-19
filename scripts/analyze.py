"""Fit calibrators on the calibration split, evaluate on the test split, write metrics + charts."""
import json
from pathlib import Path
import numpy as np
from jev_calibration import metrics, plots
from jev_calibration.calibrate import CALIBRATORS
from jev_calibration.dataset import load_dataset
from jev_calibration.features import load_features
from jev_calibration.splits import load_or_make_splits

rows = load_dataset(); splits = load_or_make_splits(rows)
df = load_features(rows, splits)
cal, test = df[df.split == "calibration"], df[df.split == "test"]
assert not set(cal.id) & set(test.id), "leak between calibration and test"
Path("results").mkdir(exist_ok=True); Path("images").mkdir(exist_ok=True)

# signal -> (score column, target column). Targets: correctness of the predicted label,
# except noul_p_pos, which is a class probability calibrated against the positive label.
SIGNALS = {
    "noul_top": ("noul_top", "noul_correct"),
    "choice_top": ("choice_top", "choice_correct"),
    "choice_confidence": ("choice_confidence", "choice_correct"),
    "noul_p_pos": ("noul_p_pos", "y_pos"),
}
out, curves = {"n_calibration": len(cal), "n_test": len(test), "signals": {}}, {}
for name, (score, target) in SIGNALS.items():
    res = {"raw": metrics.summary(test[score], test[target])}
    for cname, C in CALIBRATORS.items():
        c = C().fit(cal[score], cal[target])
        res[cname] = {**metrics.summary(c.predict(test[score]), test[target]), "params": c.params}
        if cname == "platt_logit":
            plots.before_after(test[score], c.predict(test[score]), test[target], name, f"images/{name}_reliability.png")
            if target != "y_pos":
                curves[f"{name} (Platt)"] = metrics.coverage_curve(c.predict(test[score]), test[target], np.linspace(0.5, 0.999, 60))
    plots.histogram(df[score], f"{name} (all examples)", f"images/{name}_hist.png")
    res["by_strength_accuracy"] = {s: float(g[target].mean()) for s, g in test.groupby("strength")} if target != "y_pos" else None
    out["signals"][name] = res
plots.coverage(curves, "images/coverage.png")
out["latency_s"] = {"median": float(df.latency_s.median()), "p95": float(df.latency_s.quantile(0.95))}
Path("results/metrics.json").write_text(json.dumps(out, indent=2))
for n, r in out["signals"].items():
    print(f"{n:18s} ECE raw {r['raw']['ece']:.4f} -> platt_logit {r['platt_logit']['ece']:.4f} | isotonic {r['isotonic']['ece']:.4f} | acc {r['raw']['accuracy']:.3f}")
