import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def reliability(ax, conf, acc, title, n_bins=10):
    conf, acc = np.asarray(conf, float), np.asarray(acc, float)
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys, ns = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum() >= 5:
            xs.append(conf[m].mean()); ys.append(acc[m].mean()); ns.append(m.sum())
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.scatter(xs, ys, s=np.sqrt(ns) * 8, alpha=0.8)
    ax.plot(xs, ys, alpha=0.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="predicted probability", ylabel="observed accuracy", title=title)


def before_after(conf_before, conf_after, acc, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    reliability(axes[0], conf_before, acc, f"{title}: raw")
    reliability(axes[1], conf_after, acc, f"{title}: Platt scaled")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def histogram(conf, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(conf, bins=np.linspace(0, 1, 21)); ax.set(title=title, xlabel="value", ylabel="examples")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def coverage(curves: dict, path):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, c in curves.items():
        pts = [(x["coverage"], x["accuracy"]) for x in c if x["accuracy"] is not None]
        ax.plot(*zip(*pts), label=name)
    ax.set(xlabel="fraction auto-accepted", ylabel="accuracy of accepted", title="Accuracy vs. coverage (test)")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
