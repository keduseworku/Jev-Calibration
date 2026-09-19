"""Tweet-sized (1600x900) charts, one message each. Fixed color per entity: Noul blue, Choice orange.

Usage: social_images.py            -> images/social  (with a repo footer, for posting)
       social_images.py --article  -> images/article (no footer; adds the Platt/isotonic and calibration-size charts)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ARTICLE = "--article" in sys.argv
OUT = Path("images/article" if ARTICLE else "images/social"); OUT.mkdir(parents=True, exist_ok=True)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
BLUE, ORANGE, AQUA, VIOLET, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#9a9994"
FOOT = "" if ARTICLE else "github.com/AnthusAI/Jev-Calibration"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

bands = json.load(open("results/raw_confidence_bands.json"))
var = json.load(open("results/variants.json"))["variants"]
BKEYS = ["50-60%", "60-80%", "80-90%", "90-95%", "95-100%"]  # keys in results/raw_confidence_bands.json
BLAB = [k.replace("-", "–") for k in BKEYS]


def frame(title, subtitle, foot_extra=""):
    fig = plt.figure(figsize=(8, 4.5), dpi=200)
    fig.text(.045, .945, title, fontsize=16, fontweight="bold", va="top")
    fig.text(.045, .875, subtitle, fontsize=10.5, color=INK2, va="top")
    if FOOT: fig.text(.955, .012, FOOT, fontsize=8, color=MUTED, ha="right")
    return fig


def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=.8); ax.set_axisbelow(True); ax.tick_params(length=0, labelsize=10)


def band_rows(kind):
    b = bands[f"{kind}_all_tiers"]["bands"]; return [b[k] for k in BKEYS]


def save(fig, name): fig.savefig(OUT / name, dpi=200, facecolor=SURF); plt.close(fig); print("wrote", OUT / name)


# 1. hero: stated confidence vs actual accuracy, both question types
fig = frame("When Jev is 90% confident, is it right 90% of the time?",
            "Raw (uncalibrated) confidence vs. actual accuracy, 8,801 labeled sentiment examples")
ax = fig.add_axes([.085, .14, .87, .66]); style(ax)
ax.plot([.4, 1], [.4, 1], "--", color=MUTED, lw=1.5, zorder=1)
ax.text(.755, .715, "perfect calibration", color=MUTED, fontsize=9.5, rotation=32, ha="center")
for kind, col, name in (("noul", BLUE, "Noul (yes/no)"), ("choice", ORANGE, "Choice (pick one)")):
    r = band_rows(kind); x = [v["mean_confidence"] for v in r]; y = [v["accuracy"] for v in r]
    ax.plot(x, y, "-o", color=col, lw=2.4, ms=9, mec=SURF, mew=2, label=name, zorder=3)
ax.legend(loc="upper left", frameon=False, fontsize=11, handlelength=1.6)
ax.annotate("Noul's top band: all 2,104\nanswers were right", (.978, 1.0), (.83, .93), fontsize=9.5, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.annotate("Choice: 63% of answers land here,\n99% stated, only 90% right", (.994, .902), (.87, .60), fontsize=9.5, color=INK2, ha="center",
            arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
ax.set(xlim=(.45, 1.0), ylim=(.4, 1.04), xlabel="Jev's stated confidence", ylabel="Actual accuracy")
ax.set_xticks([.5, .6, .7, .8, .9, 1.0]); ax.set_xticklabels([f"{int(t*100)}%" for t in [.5, .6, .7, .8, .9, 1.0]])
ax.set_yticks([.4, .6, .8, 1.0]); ax.set_yticklabels(["40%", "60%", "80%", "100%"])
save(fig, "1_stated_vs_actual.png")


# 2 & 3. per-band bars, stated vs actual
def bars(kind, col, title, subtitle, callout, name, hl):
    fig = frame(title, subtitle)
    ax = fig.add_axes([.085, .16, .87, .60]); style(ax)
    r = band_rows(kind); w = .36; total = sum(v["n"] for v in r)
    for i, v in enumerate(r):
        ax.bar(i - w / 2 - .01, v["mean_confidence"], w, color="#cfcec8", zorder=2)
        ax.bar(i + w / 2 + .01, v["accuracy"], w, color=col, zorder=2)
        if i in hl:
            ax.text(i - w / 2 - .01, v["mean_confidence"] + .02, f"{v['mean_confidence']*100:.0f}%", ha="center", fontsize=10, color=INK2)
            ax.text(i + w / 2 + .01, v["accuracy"] + .02, f"{v['accuracy']*100:.0f}%", ha="center", fontsize=10.5, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n{int(v['n']):,} ({v['n']/total*100:.0f}%)" for l, v in zip(BLAB, r)], fontsize=9.5)
    ax.set(ylim=(0, 1.12), xlim=(-.6, 4.6)); ax.set_yticks([0, .25, .5, .75, 1.0]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Jev's stated confidence (bucket)  ·  answers in bucket (share of all answers)", labelpad=8, fontsize=9.5)
    ax.legend(handles=[Line2D([], [], marker="s", ls="", ms=9, color="#cfcec8", label="Stated confidence"),
                       Line2D([], [], marker="s", ls="", ms=9, color=col, label="Actual accuracy")],
              loc="lower left", frameon=False, fontsize=10, ncol=2, bbox_to_anchor=(0, 1.0))
    ax.text(-.5, 1.1, callout, fontsize=10.5, color=INK, va="top", linespacing=1.45)
    save(fig, name)


bars("noul", BLUE, "Noul (yes/no): solid at the extremes, shaky in the middle",
     "Stated confidence vs. actual accuracy by bucket, raw values, 8,801 examples",
     "Above 95%: all 2,104 were right.\nAround 70%: only 54% were.", "2_noul_bars.png", hl={1, 4})
bars("choice", ORANGE, "Choice (pick one): close to a coin flip until 95%",
     "Stated confidence vs. actual accuracy by bucket, raw values, 8,801 examples",
     "Below 95%: 50–57% right,\nwhatever it states.\nAbove 95%: 90% right.", "3_choice_bars.png", hl={1, 3, 4})

# 4 & 5. dot plots across the setups
ROWS = [("noul_pos", "Noul: “is it positive?”", BLUE), ("noul_neg", "Noul: “is it negative?”", BLUE),
        ("noul_favorable", "Noul: different wording", BLUE), ("choice2", "Choice: 2 options", ORANGE),
        ("choice2_swapped", "Choice: options reversed", ORANGE), ("choice2_described", "Choice: with descriptions", ORANGE),
        ("choice3", "Choice: 3 options", ORANGE), ("score5", "Score: 5 levels", AQUA), ("score5_mean", "Score: expected value", AQUA),
        ("ensemble_noul", "Average of the 3 Noul", VIOLET), ("ensemble_all", "Average of 8 setups", VIOLET)]
ypos = list(range(len(ROWS)))[::-1]

fig = frame("Same task, different setup, different results",
            "Same 8,801 examples; only how the question is asked changes (raw values, test split)")
axA = fig.add_axes([.30, .14, .30, .61]); axB = fig.add_axes([.66, .14, .30, .61], sharey=axA)
for ax in (axA, axB): style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.8)
for (k, lab, col), y in zip(ROWS, ypos):
    a = var[k]["accuracy_at_0.5"] * 100; e = var[k]["all"]["raw"]["ece"] * 100
    axA.plot(a, y, "o", color=col, ms=10, mec=SURF, mew=2); axB.plot(e, y, "o", color=col, ms=10, mec=SURF, mew=2)
    if k in ("noul_pos", "noul_favorable"): axA.text(a + (-.35 if k == "noul_pos" else .35), y, f"{a:.1f}%", fontsize=10, fontweight="bold", va="center", ha="right" if k == "noul_pos" else "left")
    if k in ("ensemble_noul", "choice2_swapped"): axB.text(e + .5, y, f"{e:.1f}", fontsize=10, fontweight="bold", va="center")
axA.set_yticks(ypos); axA.set_yticklabels([r[1] for r in ROWS], fontsize=9.5)
axA.set_xlim(69, 79); axA.set_xticks([72, 74, 76, 78]); axA.set_xticklabels(["72%", "74%", "76%", "78%"])
axB.set_xlim(0, 18); axB.set_xticks([0, 5, 10, 15]); plt.setp(axB.get_yticklabels(), visible=False)
axA.set_title("Accuracy (default cutoff)", fontsize=11, loc="left", color=INK, pad=10)
axB.set_title("Calibration error (points)", fontsize=11, loc="left", color=INK, pad=10)
fig.legend(handles=[Line2D([], [], marker="o", ls="", ms=8, color=c, label=n) for n, c in (("Noul", BLUE), ("Choice", ORANGE), ("Score", AQUA), ("Averages", VIOLET))],
           loc="lower left", bbox_to_anchor=(.03, .0), ncol=4, frameon=False, fontsize=9.5)
save(fig, "4_setups_differ.png")

fig = frame("Calibrating on labeled examples cuts error to under 2 points",
            "Calibration error on held-out data, for every setup. Fit on 5,280 examples, tested on 3,521")
ax = fig.add_axes([.30, .17, .655, .60]); style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, lw=.8)
for (k, lab, col), y in zip(ROWS, ypos):
    v = var[k]["all"]; r, p, i = (v[m]["ece"] * 100 for m in ("raw", "platt_logit", "isotonic"))
    ax.plot([i, r], [y, y], color=GRID, lw=3, zorder=1)
    ax.plot(r, y, "o", color=GRAY, ms=10, mec=SURF, mew=2, zorder=3); ax.plot(p, y, "o", color=VIOLET, ms=10, mec=SURF, mew=2, zorder=3)
    ax.plot(i, y, "o", color=AQUA, ms=10, mec=SURF, mew=2, zorder=3)
ax.axvline(2, color=MUTED, ls="--", lw=1.2); ax.text(2.15, ypos[0] + .62, "2 points", fontsize=9, color=INK2)
ax.set_yticks(ypos); ax.set_yticklabels([r[1] for r in ROWS], fontsize=9.5); ax.set_xlim(0, 18)
ax.set_xticks([0, 5, 10, 15]); ax.set_xlabel("Calibration error (ECE, percentage points)", labelpad=4)
ax.legend(handles=[Line2D([], [], marker="o", ls="", ms=9, color=GRAY, label="Raw"), Line2D([], [], marker="o", ls="", ms=9, color=VIOLET, label="Platt scaling"),
                   Line2D([], [], marker="o", ls="", ms=9, color=AQUA, label="Isotonic regression")],
          loc="lower right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, .02))
save(fig, "5_calibration_fixes.png")


if ARTICLE:
    import numpy as np
    from jev_calibration.calibrate import Isotonic, PlattLogit
    from jev_calibration.dataset import load_dataset
    from jev_calibration.plots_variants import binned
    from jev_calibration.splits import load_or_make_splits
    from jev_calibration.variants import p_pos_signals

    rows = load_dataset(); splits = load_or_make_splits(rows); meta = {r["id"]: r for r in rows}
    S, Y = [], []
    for line in open("data/jev_variants.jsonl"):
        r = json.loads(line)
        if splits[r["id"]] == "calibration":
            S.append(p_pos_signals(r["answers"])["noul_pos"]); Y.append(int(meta[r["id"]]["expected"] == "positive"))
    S, Y = np.array(S), np.array(Y)
    platt, iso = PlattLogit().fit(S, Y), Isotonic().fit(S, Y)
    xs = np.linspace(0, 1, 600); flat = xs[(iso.predict(xs) > .63) & (iso.predict(xs) < .67)]
    print("isotonic plateau ~0.65 spans raw", round(flat.min(), 2), "to", round(flat.max(), 2))

    fig = frame("Platt bends a curve. Isotonic follows the data.",
                "Jev's Noul answer, mapped to how often the label was positive (5,280 calibration examples)")
    ax = fig.add_axes([.085, .15, .87, .62]); style(ax)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1.4, zorder=1, label="Raw, as stated")
    ax.plot(xs, platt.predict(xs), color=VIOLET, lw=2.6, zorder=3, label="Platt scaling")
    ax.step(xs, iso.predict(xs), where="post", color=AQUA, lw=2.6, zorder=3, label="Isotonic regression")
    g = binned(S, Y, n_bins=12)
    ax.scatter(g.s, g.y, s=np.sqrt(g.n) * 6, color=INK, alpha=.85, zorder=4, edgecolor=SURF, linewidth=1.5, label="What actually happened")
    ax.annotate("Raw scores from about %.2f to %.2f\nall landed near 65%% positive" % (flat.min(), flat.max()), (np.mean([flat.min(), flat.max()]), .65), (.62, .30),
                fontsize=10, color=INK2, ha="left", arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.set(xlim=(0, 1), ylim=(0, 1.04), xlabel="Jev's raw P(positive)", ylabel="How often the label was positive")
    ax.set_xticks([0, .25, .5, .75, 1]); ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"]); ax.set_yticks([0, .25, .5, .75, 1]); ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    save(fig, "6_platt_vs_isotonic.png")

    cs = json.load(open("results/calibration_size.json")); sizes = cs["sizes"]
    fig = frame("A few hundred labeled examples get you most of the way",
                "Held-out error by calibration-set size (200 draws per size; band = 10th–90th percentile)")
    for k, (setup, name) in enumerate((("noul_pos", "Noul (yes/no)"), ("choice2", "Choice (pick one)"))):
        ax = fig.add_axes([.085 + k * .455, .17, .42, .58]); style(ax)
        d = cs["variants"][setup]
        for m, col, lab in (("platt_logit", VIOLET, "Platt scaling"), ("isotonic", AQUA, "Isotonic regression")):
            mean = np.array([d[f"{m}@{n}"]["ece_mean"] for n in sizes]) * 100
            lo = np.array([d[f"{m}@{n}"]["ece_p10"] for n in sizes]) * 100; hi = np.array([d[f"{m}@{n}"]["ece_p90"] for n in sizes]) * 100
            ax.fill_between(sizes, lo, hi, color=col, alpha=.15, lw=0); ax.plot(sizes, mean, "-o", color=col, lw=2.4, ms=6, mec=SURF, mew=1.5, label=lab)
        ax.axhline(d["raw_ece"] * 100, color=MUTED, ls="--", lw=1.4, label="Uncalibrated")
        ax.set_xscale("log"); ax.set_xlim(17, 6500); ax.set_ylim(0, 18); ax.set_xticks([20, 50, 100, 200, 500, 1000, 5000]); ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "5k"])
        ax.set_title(name, fontsize=11.5, loc="left", color=INK, pad=8); ax.set_xlabel("Labeled examples used to calibrate", labelpad=6)
        if k == 0: ax.set_ylabel("Calibration error (percentage points)"); ax.legend(loc="upper right", frameon=False, fontsize=9.5)
        else: plt.setp(ax.get_yticklabels(), visible=False)
    save(fig, "7_calibration_set_size.png")
