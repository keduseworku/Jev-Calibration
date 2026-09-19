"""Visualizations for the calibration-method and question-variant comparison."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"raw": "#888888", "platt_logit": "#1f77b4", "isotonic": "#d62728"}
LABELS = {"raw": "Raw", "platt_logit": "Platt (logit)", "isotonic": "Isotonic"}


def binned(score, y, n_bins=10, min_n=8):
    """Equal-mass bins; ties collapse bins, so bin count adapts to how many distinct values exist."""
    s = pd.Series(np.asarray(score, float)); t = pd.Series(np.asarray(y, float))
    b = pd.qcut(s.rank(method="first"), n_bins, labels=False) if s.nunique() > n_bins else pd.factorize(s)[0]
    g = pd.DataFrame({"s": s, "y": t, "b": b}).groupby("b").agg(s=("s", "mean"), y=("y", "mean"), n=("y", "size"))
    return g[g.n >= min_n]


def mapping_plot(score, y, platt, iso, title, path):
    """What each method does: the learned score -> probability curves over the observed frequencies."""
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    xs = np.linspace(0, 1, 400)
    ax.plot([0, 1], [0, 1], "--", color="#aaa", label="identity (already calibrated)")
    ax.plot(xs, platt.predict(xs), color=COLORS["platt_logit"], lw=2.2, label="Platt: smooth sigmoid")
    ax.step(xs, iso.predict(xs), where="post", color=COLORS["isotonic"], lw=2.2, label="Isotonic: monotone steps")
    g = binned(score, y)
    ax.scatter(g.s, g.y, s=np.sqrt(g.n) * 7, color="black", alpha=.6, zorder=3, label="observed (calibration split)")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Jev's raw P(positive)", ylabel="P(label is positive)", title=title)
    ax.legend(loc="upper left", fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def reliability_grid(data: dict, path, cols=("raw", "platt_logit", "isotonic")):
    """data: {variant: {"y": array, "raw": array, "platt_logit": array, "isotonic": array, "ece": {method: float}}}."""
    rows = list(data)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(3.0 * len(cols), 2.7 * len(rows)), squeeze=False)
    for i, v in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i][j]; g = binned(data[v][c], data[v]["y"])
            ax.plot([0, 1], [0, 1], "--", color="#aaa", lw=1)
            ax.plot(g.s, g.y, "-", color=COLORS[c], alpha=.6)
            ax.scatter(g.s, g.y, s=np.sqrt(g.n) * 5, color=COLORS[c])
            ax.set(xlim=(0, 1), ylim=(0, 1)); ax.tick_params(labelsize=6)
            ax.text(.04, .86, f"ECE {data[v]['ece'][c]:.3f}", fontsize=7, transform=ax.transAxes)
            if i == 0: ax.set_title(LABELS[c], fontsize=9)
            if j == 0: ax.set_ylabel(v, fontsize=8)
    fig.suptitle("Reliability on the test split (x: predicted P(positive), y: observed fraction positive)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.985)); fig.savefig(path, dpi=130); plt.close(fig)


def metric_bars(table: pd.DataFrame, metric: str, path, title):
    """table: index variant, columns method -> value."""
    fig, ax = plt.subplots(figsize=(9, 4.4)); w = .27; x = np.arange(len(table))
    for k, m in enumerate(["raw", "platt_logit", "isotonic"]):
        ax.bar(x + (k - 1) * w, table[m], w, color=COLORS[m], label=LABELS[m])
    ax.set_xticks(x); ax.set_xticklabels(table.index, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel=metric, title=title); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def hist_grid(df: pd.DataFrame, cols, path):
    n = len(cols); r = int(np.ceil(n / 4)); fig, axes = plt.subplots(r, 4, figsize=(12, 2.4 * r), squeeze=False)
    for ax in axes.ravel(): ax.axis("off")
    for ax, c in zip(axes.ravel(), cols):
        ax.axis("on"); ax.hist(df[c], bins=np.linspace(0, 1, 21), color="#4c72b0"); ax.set_title(c, fontsize=8)
        ax.tick_params(labelsize=6)
    fig.suptitle("Distribution of each variant's P(positive) (all examples)", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def heatmap(mat: pd.DataFrame, path, title, fmt="{:.2f}"):
    fig, ax = plt.subplots(figsize=(7.2, 6)); im = ax.imshow(mat.values, vmin=mat.values.min(), vmax=1, cmap="viridis")
    ax.set_xticks(range(len(mat))); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(mat))); ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(len(mat)):
        for j in range(len(mat)):
            ax.text(j, i, fmt.format(mat.values[i, j]), ha="center", va="center", fontsize=6, color="white")
    ax.set_title(title, fontsize=10); fig.colorbar(im); fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def strength_bars(acc: pd.DataFrame, path):
    """acc: index variant, columns strength -> accuracy."""
    fig, ax = plt.subplots(figsize=(9, 4.4)); cols = list(acc.columns); w = .8 / len(cols); x = np.arange(len(acc))
    for k, c in enumerate(cols):
        ax.bar(x + (k - (len(cols) - 1) / 2) * w, acc[c], w, label=c)
    ax.axhline(.5, color="#aaa", ls="--", lw=1); ax.set_xticks(x); ax.set_xticklabels(acc.index, rotation=35, ha="right", fontsize=8)
    ax.set(ylabel="accuracy at 0.5", title="Accuracy by sentiment strength (test split)", ylim=(0, 1.05)); ax.legend(title="tier", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def coverage_plot(curves: dict, path, title):
    """curves: {name: DataFrame(coverage, accuracy)} - accuracy of auto-accepted decisions vs fraction accepted."""
    fig, ax = plt.subplots(figsize=(7.4, 5))
    for name, c in curves.items():
        ax.plot(c.coverage, c.accuracy, lw=1.8, label=name)
    ax.set(xlabel="fraction of decisions auto-accepted", ylabel="accuracy of accepted decisions", title=title, ylim=(0.5, 1.01), xlim=(0, 1))
    ax.grid(alpha=.3); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def size_plot(res: dict, raw: dict, path, metric="ece"):
    """res: {variant: {method: {n: (mean, lo, hi)}}}; raw: {variant: value}. One panel per variant."""
    names = list(res); fig, axes = plt.subplots(1, len(names), figsize=(4.2 * len(names), 4), sharey=True, squeeze=False)
    for ax, v in zip(axes[0], names):
        for m in ("platt_logit", "isotonic"):
            ns = sorted(res[v][m]); mean = np.array([res[v][m][n][0] for n in ns])
            lo = np.array([res[v][m][n][1] for n in ns]); hi = np.array([res[v][m][n][2] for n in ns])
            ax.plot(ns, mean, "-o", ms=3, color=COLORS[m], label=LABELS[m]); ax.fill_between(ns, lo, hi, color=COLORS[m], alpha=.18)
        ax.axhline(raw[v], color=COLORS["raw"], ls="--", label="Raw (uncalibrated)")
        ax.set_xscale("log"); ax.set_xlabel("calibration examples"); ax.set_title(v, fontsize=10); ax.grid(alpha=.3)
    axes[0][0].set_ylabel(f"{metric.upper()} on test (mean, 10-90% band)"); axes[0][0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
