"""Raw confidence (top-label probability) vs. observed accuracy for Jev's Noul and Choice answers,
on all examples. No fitting: this is what the model's stated confidence looks like before any calibration."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from jev_calibration import plots_variants as pv
from jev_calibration.dataset import load_dataset

OUT = Path("images/raw"); OUT.mkdir(parents=True, exist_ok=True)
meta = {r["id"]: r for r in load_dataset()}
R = []
for line in open("data/jev_variants.jsonl"):
    r = json.loads(line); m = meta[r["id"]]; a = r["answers"]
    p = a["noul_pos"]["noul"]; c = a["choice2"]
    R.append({"tier": m["strength"], "noul_conf": max(p, 1 - p), "noul_ok": ("positive" if p >= .5 else "negative") == m["expected"],
              "choice_conf": c["probabilities"][c["choice"]], "choice_ok": c["choice"] == m["expected"]})
d = pd.DataFrame(R)
views = {"all_tiers": d, "no_neutral": d[d.tier != "neutral"]}
KINDS = {"noul": "Noul", "choice": "Choice"}
bands, labels = [0.5, 0.6, 0.8, 0.9, 0.95, 1.0], ["50-60%", "60-80%", "80-90%", "90-95%", "95-100%"]

out = {}
for vname, x in views.items():
    for k, kn in KINDS.items():
        conf, ok = x[f"{k}_conf"], x[f"{k}_ok"].astype(int)
        b = pd.cut(conf, bands, labels=labels, right=True, include_lowest=True)  # buckets include their upper edge, like the charts
        g = x.groupby(b, observed=True).agg(n=(f"{k}_ok", "size"), mean_confidence=(f"{k}_conf", "mean"), accuracy=(f"{k}_ok", "mean"))
        out[f"{k}_{vname}"] = {"n": int(len(x)), "mean_confidence": float(conf.mean()), "accuracy": float(ok.mean()),
                               "bands": {i: {c: float(v) for c, v in row.items()} for i, row in g.iterrows()}}
Path("results/raw_confidence_bands.json").write_text(json.dumps(out, indent=2))

for k, kn in KINDS.items():
    pv.raw_reliability(d[f"{k}_conf"], d[f"{k}_ok"], f"Raw Total Probability vs Actual Accuracy: Jev {kn}", OUT / f"reliability_raw_{k}.png")
    pv.raw_histogram(d[f"{k}_conf"], f"Total Probability Distribution: Jev {kn}", OUT / f"histogram_raw_{k}.png", subtitle=f"{len(d):,} sentiment predictions")
fig, axes = plt.subplots(2, 2, figsize=(13, 13))
for i, (k, kn) in enumerate(KINDS.items()):
    for j, (vname, x) in enumerate(views.items()):
        pv.raw_reliability(x[f"{k}_conf"], x[f"{k}_ok"].astype(int), f"Jev {kn}: {'all tiers' if j == 0 else 'neutral tier excluded'} (n={len(x):,})", None, ax=axes[i][j])
fig.suptitle("Raw total probability vs actual accuracy", fontsize=14, fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "reliability_raw_grid.png", dpi=120); plt.close(fig)

for key, v in out.items():
    print(f"\n{key}: n={v['n']} mean conf {v['mean_confidence']:.3f} accuracy {v['accuracy']:.3f}")
    print(pd.DataFrame(v["bands"]).T.round(3).to_string())
