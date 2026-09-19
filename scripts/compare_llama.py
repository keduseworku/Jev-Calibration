"""Jev vs. Llama 3.1-8B (from Classification-with-Confidence) on the identical 1,000 examples.

Both models are scored the same way: top-label probability ("total probability") and whether the predicted
label was correct. Calibrators are evaluated with repeated 5-fold cross-validation, never in-sample.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from jev_calibration import metrics, plots_variants as pv
from jev_calibration.calibrate import Isotonic, PlattLogit
from jev_calibration.dataset import example_id

OUT = Path("images/comparison"); OUT.mkdir(parents=True, exist_ok=True)
llama = json.loads(Path("data/reference/llama31_8b_base_results.json").read_text())["results"]
jev = {json.loads(l)["id"]: json.loads(l)["answers"] for l in Path("data/jev_variants.jsonl").read_text().splitlines()}

def col(f): return np.array([f(r) for r in llama])
ids = [example_id(r["text"]) for r in llama]
S = {}  # name -> (confidence, correct, platt eps)
S["Llama 3.1-8B (base)"] = (col(lambda r: r["confidence"]), col(lambda r: r["correct"]).astype(int), 1e-6)

def jev_noul(i, exp):
    p = jev[i]["noul_pos"]["noul"]; pred = "positive" if p >= .5 else "negative"
    return max(p, 1 - p), int(pred == exp)
def jev_choice(i, exp):
    a = jev[i]["choice2"]; return a["probabilities"][a["choice"]], int(a["choice"] == exp)
for name, f in (("Jev (Noul)", jev_noul), ("Jev (Choice)", jev_choice)):
    v = [f(i, r["expected"]) for i, r in zip(ids, llama)]
    S[name] = (np.array([x[0] for x in v]), np.array([x[1] for x in v]), 0.005)

def cv_ece(conf, correct, make, reps=20):
    out = []
    for seed in range(reps):
        pred = np.zeros(len(conf))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(conf, correct):
            pred[te] = make().fit(conf[tr], correct[tr]).predict(conf[te])
        out.append((metrics.ece(pred, correct), metrics.brier(pred, correct)))
    return np.mean(out, axis=0)

res = {}
for name, (c, y, eps) in S.items():
    pe, pb = cv_ece(c, y, lambda: PlattLogit(eps)); ie, ib = cv_ece(c, y, Isotonic)
    res[name] = {"n": len(c), "accuracy": float(y.mean()), "mean_confidence": float(c.mean()),
                 "ece_raw": metrics.ece(c, y), "mce_raw": metrics.mce(c, y), "brier_raw": metrics.brier(c, y),
                 "ece_platt_cv": float(pe), "ece_isotonic_cv": float(ie), "brier_platt_cv": float(pb), "brier_isotonic_cv": float(ib),
                 "auroc_confidence_vs_correct": float(roc_auc_score(y, c)),
                 **{f"n_conf>={t}": int((c >= t).sum()) for t in (.95, .9)},
                 **{f"acc_when_conf>={t}": float(y[c >= t].mean()) if (c >= t).any() else None for t in (.95, .9)},
                 "n_in_top_5pct_bucket": int(((c > .95) & (c <= 1)).sum())}
cats = np.array([r["category"] for r in llama]); keep = ~np.char.startswith(cats.astype(str), "neutral")
for name, (c, y, eps) in S.items():
    c2, y2 = c[keep], y[keep]
    res[name]["no_neutral"] = {"n": int(keep.sum()), "accuracy": float(y2.mean()), "mean_confidence": float(c2.mean()),
                               "ece_raw": metrics.ece(c2, y2), "auroc": float(roc_auc_score(y2, c2)),
                               "n_conf>=0.95": int((c2 >= .95).sum()), "acc_when_conf>=0.95": float(y2[c2 >= .95].mean())}
Path("results/comparison_llama.json").write_text(json.dumps(res, indent=2))

import matplotlib.pyplot as plt
names = list(S); tag = {"Llama 3.1-8B (base)": "llama", "Jev (Noul)": "jev_noul", "Jev (Choice)": "jev_choice"}
top = max(np.histogram(c, bins=np.linspace(.495, 1.005, 52))[0].max() for c, _, _ in S.values()) * 1.1
for n in names:
    c, y, _ = S[n]
    pv.raw_histogram(c, f"Total Probability Distribution: {n}", OUT / f"raw_confidence_histogram_{tag[n]}.png",
                     subtitle=f"{len(c):,} sentiment predictions")
    pv.raw_reliability(c, y, f"Raw Total Probability vs Actual Accuracy: {n}", OUT / f"reliability_raw_{tag[n]}.png")
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))
for ax, n in zip(axes, names): pv.raw_histogram(S[n][0], n, None, ax=ax, ylim=top)
fig.suptitle("Raw total probability, same 1,000 examples", fontsize=13, fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "histogram_comparison.png", dpi=140); plt.close(fig)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, n in zip(axes, names): pv.raw_reliability(S[n][0], S[n][1], n, None, ax=ax)
fig.suptitle("Raw total probability vs actual accuracy, same 1,000 examples", fontsize=13, fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "reliability_comparison.png", dpi=130); plt.close(fig)

import pandas as pd
pd.set_option("display.width", 220)
print(pd.DataFrame(res).T[["accuracy", "mean_confidence", "ece_raw", "ece_platt_cv", "ece_isotonic_cv", "brier_raw", "brier_isotonic_cv",
                           "auroc_confidence_vs_correct", "n_conf>=0.95", "acc_when_conf>=0.95", "n_conf>=0.9", "n_in_top_5pct_bucket"]].round(3).to_string())

print(); print(pd.DataFrame({n: r["no_neutral"] for n, r in res.items()}).T.round(3).to_string())
