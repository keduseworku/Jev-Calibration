"""How much calibration data does each method need? Resample the calibration split at several sizes,
fit Platt and isotonic each time, and score on the fixed test split."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from jev_calibration import metrics, plots_variants as pv
from jev_calibration.calibrate import Isotonic, PlattLogit
from jev_calibration.dataset import load_dataset
from jev_calibration.splits import load_or_make_splits
from jev_calibration.variants import p_pos_signals

SIZES = [20, 30, 50, 100, 200, 500, 1000, 2000, 5280]
REPS, SEL = 200, ["noul_pos", "choice2", "score5_mean", "ensemble_noul"]
rows = load_dataset(); splits = load_or_make_splits(rows); meta = {r["id"]: r for r in rows}
recs = []
for line in Path("data/jev_variants.jsonl").read_text().splitlines():
    r = json.loads(line)
    recs.append({"id": r["id"], "split": splits[r["id"]], "y": int(meta[r["id"]]["expected"] == "positive"), **p_pos_signals(r["answers"])})
df = pd.DataFrame(recs)
cal, test = df[df.split == "calibration"].reset_index(drop=True), df[df.split == "test"]
rng = np.random.default_rng(0)

out = {"reps": REPS, "sizes": SIZES, "n_test": len(test), "variants": {}}
plot_res, raw = {}, {}
for v in SEL:
    raw[v] = metrics.ece(test[v], test.y)
    plot_res[v] = {"platt_logit": {}, "isotonic": {}}; out["variants"][v] = {"raw_ece": raw[v]}
    for n in SIZES:
        vals = {"platt_logit": [], "isotonic": []}; extra = {"platt_logit": [], "isotonic": []}
        for _ in range(REPS if n < len(cal) else 1):
            idx = rng.choice(len(cal), n, replace=False) if n < len(cal) else np.arange(len(cal))
            s, y = cal.loc[idx, v].values, cal.loc[idx, "y"].values
            if len(set(y)) < 2: continue
            for m, C in (("platt_logit", PlattLogit), ("isotonic", Isotonic)):
                p = C().fit(s, y).predict(test[v].values)
                vals[m].append(metrics.ece(p, test.y)); extra[m].append(metrics.brier(p, test.y))
        for m in vals:
            a = np.array(vals[m]); q = np.percentile(a, [10, 90])
            plot_res[v][m][n] = (float(a.mean()), float(q[0]), float(q[1]))
            out["variants"][v][f"{m}@{n}"] = {"ece_mean": float(a.mean()), "ece_p10": float(q[0]), "ece_p90": float(q[1]),
                                              "brier_mean": float(np.mean(extra[m])), "reps": len(a)}
pv.size_plot(plot_res, raw, "images/variants/calibration_set_size.png")
Path("results/calibration_size.json").write_text(json.dumps(out, indent=2))

pd.set_option("display.width", 200)
for v in SEL:
    t = pd.DataFrame({m: {n: plot_res[v][m][n][0] for n in SIZES} for m in ("platt_logit", "isotonic")})
    t["iso_p90"] = [plot_res[v]["isotonic"][n][2] for n in SIZES]; t["platt_p90"] = [plot_res[v]["platt_logit"][n][2] for n in SIZES]
    print(f"\n{v}  raw ECE {raw[v]:.3f}"); print(t.round(3).to_string())
