"""Compare question setups (Noul/Choice/Score variants) and calibration methods (Platt vs isotonic)."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from jev_calibration import metrics, plots_variants as pv
from jev_calibration.calibrate import Isotonic, PlattLogit
from jev_calibration.dataset import load_dataset
from jev_calibration.splits import load_or_make_splits
from jev_calibration.variants import p_pos_signals

OUT = Path("images/variants"); OUT.mkdir(parents=True, exist_ok=True)
rows = load_dataset(); splits = load_or_make_splits(rows); meta = {r["id"]: r for r in rows}

recs = []
for line in Path("data/jev_variants.jsonl").read_text().splitlines():
    r = json.loads(line); m = meta[r["id"]]
    a = r["answers"]
    recs.append({"id": r["id"], "split": splits[r["id"]], "strength": m["strength"], "y": int(m["expected"] == "positive"),
                 "neutral_prob": a["choice3"]["probabilities"]["neutral"], **p_pos_signals(a)})
df = pd.DataFrame(recs).drop_duplicates("id")
VARS = [c for c in df.columns if c not in ("id", "split", "strength", "y", "neutral_prob")]
cal, test = df[df.split == "calibration"], df[df.split == "test"]
assert not set(cal.id) & set(test.id)
print(f"{len(df)} examples ({len(cal)} calibration / {len(test)} test)")

# determinism / independence check against the base run
base = {json.loads(l)["id"]: json.loads(l)["answers"] for l in Path("data/jev_raw.jsonl").read_text().splitlines()}
common = [i for i in df.id if i in base]
d = np.array([abs(base[i]["positive"]["noul"] - df.set_index("id").loc[i, "noul_pos"]) for i in common])
print(f"noul_pos vs base run (same question, different request): mean |diff| {d.mean():.4f}, max {d.max():.2f}")


def evaluate(sub_test, calibrators):
    res = {}
    for m in ("raw", "platt_logit", "isotonic"):
        p = sub_test[v] if m == "raw" else calibrators[m].predict(sub_test[v])
        res[m] = metrics.summary(p, sub_test.y) | {"auc": float(roc_auc_score(sub_test.y, p))}
    return res


results = {"n_calibration": len(cal), "n_test": len(test), "variants": {}}
grid, curves_done = {}, False
for v in VARS:
    fits = {"platt_logit": PlattLogit().fit(cal[v], cal.y), "isotonic": Isotonic().fit(cal[v], cal.y)}
    nn = test[test.strength != "neutral"]
    results["variants"][v] = {
        "all": evaluate(test, fits), "no_neutral": evaluate(nn, fits),
        "accuracy_at_0.5": float(((test[v] >= .5).astype(int) == test.y).mean()),
        "platt_params": fits["platt_logit"].params,
        "by_strength_accuracy": {s: float(((g[v] >= .5).astype(int) == g.y).mean()) for s, g in test.groupby("strength")},
    }
    grid[v] = {"y": test.y.values, "raw": test[v].values, "platt_logit": fits["platt_logit"].predict(test[v]),
               "isotonic": fits["isotonic"].predict(test[v]),
               "ece": {m: results["variants"][v]["all"][m]["ece"] for m in ("raw", "platt_logit", "isotonic")}}
    if v in ("noul_pos", "choice2", "score5"):
        pv.mapping_plot(cal[v], cal.y, fits["platt_logit"], fits["isotonic"], f"How each method recalibrates: {v}",
                        OUT / f"mapping_{v}.png")

tab = lambda metric, view="all": pd.DataFrame({m: {v: results["variants"][v][view][m][metric] for v in VARS}
                                               for m in ("raw", "platt_logit", "isotonic")})
pv.reliability_grid({v: grid[v] for v in VARS}, OUT / "reliability_grid.png")
for metric in ("ece", "brier", "log_loss"):
    pv.metric_bars(tab(metric), metric, OUT / f"{metric}_all.png", f"{metric} by question setup and calibration (test, all tiers)")
pv.metric_bars(tab("ece", "no_neutral"), "ece", OUT / "ece_no_neutral.png", "ECE, excluding the arbitrary-label neutral tier")
pv.hist_grid(df, VARS, OUT / "histograms.png")
pv.strength_bars(pd.DataFrame({v: results["variants"][v]["by_strength_accuracy"] for v in VARS}).T, OUT / "accuracy_by_strength.png")
pv.heatmap(test[VARS].corr(method="spearman"), OUT / "variant_correlation.png", "Spearman correlation between variants' P(positive)")
agree = pd.DataFrame({a: {b: float(((test[a] >= .5) == (test[b] >= .5)).mean()) for b in VARS} for a in VARS})
pv.heatmap(agree, OUT / "variant_agreement.png", "How often two variants make the same call (test)")

def accept_curve(sub, v, iso):
    conf = np.maximum(iso.predict(sub[v]), 1 - iso.predict(sub[v]))  # calibrated confidence in the predicted side
    correct = ((sub[v] >= .5).astype(int) == sub.y).values
    order = np.argsort(-conf, kind="stable"); c = correct[order]
    k = np.arange(1, len(c) + 1)
    return pd.DataFrame({"coverage": k / len(c), "accuracy": np.cumsum(c) / k})


SEL = ["noul_pos", "noul_favorable", "choice2", "choice3", "score5_mean", "ensemble_noul"]
for view, sub in (("all", test), ("no_neutral", test[test.strength != "neutral"])):
    curves = {v: accept_curve(sub, v, Isotonic().fit(cal[v], cal.y)) for v in SEL}
    pv.coverage_plot(curves, OUT / f"accuracy_vs_coverage_{view}.png",
                     "Accuracy vs. coverage" + (" (neutral tier excluded)" if view == "no_neutral" else " (all tiers)"))
    results[f"coverage_{view}"] = {v: {f"cov@acc>={t}": (float(c[c.accuracy >= t].coverage.max()) if (c.accuracy >= t).any() else 0.0)
                                        for t in (0.9, 0.95, 0.99)} for v, c in curves.items()}

Path("results/variants.json").write_text(json.dumps(results, indent=2))
pd.set_option("display.width", 200)
summ = pd.DataFrame({"acc@.5": {v: results["variants"][v]["accuracy_at_0.5"] for v in VARS},
                     "auc": {v: results["variants"][v]["all"]["raw"]["auc"] for v in VARS},
                     "ECE raw": tab("ece").raw, "ECE platt": tab("ece").platt_logit, "ECE iso": tab("ece").isotonic,
                     "Brier raw": tab("brier").raw, "Brier platt": tab("brier").platt_logit, "Brier iso": tab("brier").isotonic,
                     "ECE iso (no neutral)": tab("ece", "no_neutral").isotonic})
print(summ.round(3).to_string())

for view in ("all", "no_neutral"):
    print(view, "coverage at accuracy target:"); print(pd.DataFrame(results[f"coverage_{view}"]).T.round(3).to_string())
