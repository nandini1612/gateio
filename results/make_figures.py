"""Regenerate the GateIO result figures from the per-sequence CSVs and path arrays.

Reads `val_all_results.csv`, `test_all_results.csv`, and `paths/*.npy` from this
directory and writes the three data figures into `figures/`. The two diagrams
(`figures/architecture.svg`, `figures/gate.svg`) are authored by hand and are not
regenerated here.

    python results/make_figures.py
"""

import csv
import os
import sys

import numpy as np

# The path dicts may have been pickled with numpy 2.x (module `numpy._core`); alias
# it to this env's numpy.core so they also unpickle under numpy 1.x.
try:
    import numpy.core as _npc
    sys.modules.setdefault("numpy._core", _npc)
    sys.modules.setdefault("numpy._core.multiarray", _npc.multiarray)
except Exception as _e:  # pragma: no cover
    print("numpy alias warning:", _e)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

RES = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)

# Colourblind-safe (Okabe-Ito) palette, one colour per system
C = {"GateIO": "#0072B2", "LSTM": "#D55E00", "EKF": "#009E73", "Const-v": "#666666"}
KEY = {"GateIO": "drift_m", "LSTM": "lstm_drift_m", "EKF": "ekf_drift_m", "Const-v": "naive_drift_m"}
GROUPS = ["straight-short", "straight-med", "TURN", "FALSE-ALARM", "long-outage"]

plt.rcParams.update({
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130,
})


def load(split):
    with open(os.path.join(RES, f"{split}_all_results.csv"), newline="") as f:
        return list(csv.DictReader(f))


def fig1_trajectories(split="test"):
    """Dead-reckoned trajectory during the outage for the turn / false-alarm / long
    examples (S41, S47, S53)."""
    def traj(ax, si, title):
        d = np.load(os.path.join(RES, "paths", f"{split}_path_seq{si}.npy"),
                    allow_pickle=True).item()
        pp, pt = d["pos_pred"], d["pos_true"]
        s = int(d["outage_start"]); e = min(s + 100, len(pp)); seg = slice(s, e)
        ax.plot(pt[seg, 1], pt[seg, 0], "-", color="black", lw=2.0, label="Ground truth")
        ax.plot(pp[seg, 1], pp[seg, 0], "-", color=C["GateIO"], lw=2.0, label="GateIO")
        ax.plot(pt[s, 1], pt[s, 0], "o", color="black", ms=6, zorder=5)
        ax.plot(pt[e - 1, 1], pt[e - 1, 0], "s", color="black", ms=6, mfc="white", zorder=5)
        ax.plot(pp[e - 1, 1], pp[e - 1, 0], "s", color=C["GateIO"], ms=6, mfc="white", zorder=5)
        ax.set_title(f"{title}\nGateIO drift {float(d['drift_m']):.1f} m  ·  "
                     f"const-v {float(d['naive_drift_m']):.0f} m", fontsize=9.5)
        ax.set_xlabel("East (m)"); ax.set_aspect("equal", adjustable="datalim")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    traj(axes[0], 41, "Turn (S41)")
    traj(axes[1], 47, "False-alarm (S47)")
    traj(axes[2], 53, "Long outage (S53)")
    axes[0].set_ylabel("North (m)")
    axes[0].legend(loc="best", fontsize=8.5, frameon=False)
    fig.suptitle(f"Dead-reckoned trajectory during a 10 s GPS outage — {split} set",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"fig1_trajectories_{split}.png"), bbox_inches="tight")
    plt.close(fig)


def _group_means(rows):
    out = {}
    for m, k in KEY.items():
        out[m] = [np.mean([float(r[k]) for r in rows if r["group"] == g]) for g in GROUPS]
    return out


def fig2_group_bars(val, test):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    x = np.arange(len(GROUPS)); w = 0.2
    for ax, rows, name in [(axes[0], val, "Validation (in-distribution)"),
                           (axes[1], test, "Test (held-out flights)")]:
        gm = _group_means(rows)
        for i, m in enumerate(["GateIO", "LSTM", "EKF", "Const-v"]):
            ax.bar(x + (i - 1.5) * w, gm[m], w, label=m, color=C[m])
        ax.set_yscale("log")
        ax.set_xticks(x); ax.set_xticklabels(GROUPS, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(name, fontsize=10.5)
        ax.axhline(5, color="crimson", ls="--", lw=1, alpha=0.7)
    axes[0].set_ylabel("Mean endpoint drift (m, log scale)")
    axes[1].legend(fontsize=8.5, frameon=False, ncol=2)
    axes[1].text(4.4, 5.6, "5 m", color="crimson", fontsize=8, ha="right")
    fig.suptitle("Per-group mean drift after a 10 s outage", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig2_group_bars.png"), bbox_inches="tight")
    plt.close(fig)


def fig3_drift_cdf(val, test):
    def cdf(ax, rows, title):
        xs = np.linspace(0, 50, 400)
        for m in ["GateIO", "LSTM", "EKF", "Const-v"]:
            a = np.array([float(r[KEY[m]]) for r in rows])
            ax.plot(xs, [(a <= t).mean() for t in xs], color=C[m], lw=2, label=m)
        ax.axvline(5, color="crimson", ls="--", lw=1, alpha=0.7)
        ax.set_xlim(0, 50); ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_xlabel("Endpoint drift threshold (m)"); ax.set_title(title, fontsize=10.5)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    cdf(axes[0], val, "Validation (in-distribution)")
    cdf(axes[1], test, "Test (held-out flights)")
    axes[0].set_ylabel("Outages at or under threshold")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=False)
    axes[0].text(5.4, 0.05, "5 m", color="crimson", fontsize=8)
    fig.suptitle("Cumulative distribution of endpoint drift", fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_drift_cdf.png"), bbox_inches="tight")
    plt.close(fig)


def main():
    val, test = load("val"), load("test")
    fig1_trajectories("test")
    fig2_group_bars(val, test)
    fig3_drift_cdf(val, test)
    print("Wrote:", sorted(f for f in os.listdir(FIG) if f.endswith(".png")))


if __name__ == "__main__":
    main()
