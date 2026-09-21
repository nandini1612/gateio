"""
Plot a held-out trajectory from one leave-one-flight-out fold: the GateIO-predicted
path vs. ground truth vs. constant-velocity, over a 10 s outage.

A real predicted-vs-truth path is more convincing than aggregate drift bars, so this
picks one representative test sequence (default: the sequence whose GateIO endpoint
drift is closest to the fold's median) and draws the integrated horizontal paths.

Layout expected (same as evaluate_lofo.py):
    <folds_dir>/fold{k}.npz
    <ckpt_dir>/fold{k}/gateio_v2_best.pt

Usage:
    python eval/plot_lofo_trajectory.py --folds-dir ./folds --ckpt-dir ./checkpoints_lofo \
        --fold 2 --out results/figures/fig_trajectory.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.gateio import SEQ_LEN, DT           # noqa: E402
from eval.evaluate import predict_sequence, load_model  # noqa: E402
from eval.evaluate_lofo import classify, OE_LEN  # noqa: E402

BLUE, ORANGE, GREY, INK = "#0072B2", "#D55E00", "#8aa0b2", "#12324a"


def constv_path(y_raw_seq, os_, oe_):
    """Integrate the last-known velocity held constant through the outage."""
    v_hold = y_raw_seq[max(os_ - 1, 0), :2]
    pos = np.zeros((SEQ_LEN, 2), dtype=float)
    for t in range(os_ + 1, SEQ_LEN):
        pos[t] = pos[t - 1] + v_hold * DT
    return pos


def main():
    ap = argparse.ArgumentParser(description="Plot one LOFO held-out trajectory.")
    ap.add_argument("--folds-dir", required=True)
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--fold", type=int, default=2, help="Which fold's test flight to draw.")
    ap.add_argument("--model", choices=["lstm", "gateio"], default="lstm",
                    help="Backbone to plot (lstm = GateIO-LSTM, the primary model).")
    ap.add_argument("--seq", type=int, default=None,
                    help="Sequence index to draw; default = the one nearest the fold's median drift.")
    ap.add_argument("--prefer", choices=["any", "turn", "straight"], default="any",
                    help="Restrict the auto-pick to turning / straight sequences.")
    ap.add_argument("--dual", action="store_true",
                    help="Two panels: a typical straight outage and a turn, each at its group median.")
    ap.add_argument("--out", default="results/figures/fig_trajectory.png")
    args = ap.parse_args()

    label = "GateIO-LSTM" if args.model == "lstm" else "GateIO-TCN"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold_npz = os.path.join(args.folds_dir, f"fold{args.fold}.npz")
    ckpt = os.path.join(args.ckpt_dir, f"fold{args.fold}", f"{args.model}_v2_best.pt")
    npz = np.load(fold_npz)
    Xt = npz["X_test"].astype(np.float32); Yt = npz["Y_test"].astype(np.float32)
    ti = npz["test_valid_idx"]
    Xmed = npz["X_median"].astype(np.float32); Xiq = npz["X_iqr"].astype(np.float32)
    dv_iqr = npz["Y_iqr"].astype(np.float32); dv_median = npz["Y_median"].astype(np.float32)
    flight = str(npz["split_test"]) if "split_test" in npz.files else f"fold{args.fold}"
    model = load_model(ckpt, args.model, device)

    os_ = SEQ_LEN // 3; oe_ = min(os_ + OE_LEN, SEQ_LEN)

    # Score every valid sequence so we can pick a representative one.
    recs = []
    for si in range(len(ti)):
        st = int(ti[si]); xrs = Xt[st:st + SEQ_LEN]; yrs = Yt[st:st + SEQ_LEN]
        if len(xrs) < SEQ_LEN:
            continue
        xns = (xrs - Xmed) / np.where(Xiq < 1e-6, 1.0, Xiq)
        out = predict_sequence(model, xns, yrs, os_, oe_, dv_iqr, dv_median, device)
        recs.append((si, out["drift_m"], classify(xrs, os_, oe_), xns, yrs))

    def pick(group_filter):
        """The sequence whose drift is nearest the median of its group (or all)."""
        pool = [r for r in recs if group_filter == "any" or r[2].lower() == group_filter]
        pool = pool or recs
        med = float(np.median([r[1] for r in pool]))
        return min(pool, key=lambda r: abs(r[1] - med))

    def draw(ax, chosen, show_ylabel=True):
        si, drift, grp, xns, yrs = chosen
        out = predict_sequence(model, xns, yrs, os_, oe_, dv_iqr, dv_median, device)
        pp, pt = out["pos_pred"], out["pos_true"]
        pc = constv_path(yrs, os_, oe_)
        ax.plot(pt[os_:oe_, 0], pt[os_:oe_, 1], color=INK, lw=3.0, label="Ground truth", zorder=3)
        ax.plot(pp[os_:oe_, 0], pp[os_:oe_, 1], color=BLUE, lw=2.4, label=label, zorder=4)
        ax.plot(pc[os_:oe_, 0], pc[os_:oe_, 1], color=GREY, lw=2.0, ls="--",
                label="Constant velocity", zorder=2)
        ax.scatter([0], [0], color="black", s=45, zorder=5, label="Outage onset")
        ax.plot([pt[oe_ - 1, 0], pp[oe_ - 1, 0]], [pt[oe_ - 1, 1], pp[oe_ - 1, 1]],
                color=ORANGE, lw=1.4, ls=":", zorder=4)
        ax.annotate(f"{drift:.1f} m", (pp[oe_ - 1, 0], pp[oe_ - 1, 1]),
                    textcoords="offset points", xytext=(8, 6), color=ORANGE, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("East (m)")
        if show_ylabel:
            ax.set_ylabel("North (m)")
        ax.set_title(f"{grp.title()} outage, {label} drift {drift:.1f} m", color=INK, fontsize=11)
        ax.legend(loc="best", frameon=False, fontsize=8.5)
        ax.grid(alpha=0.25)
        return si, grp, drift

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.dual:
        fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=150)
        s0 = draw(axes[0], pick("straight"), show_ylabel=True)
        s1 = draw(axes[1], pick("turn"), show_ylabel=False)
        fig.suptitle(f"Held-out {flight}: a typical straight outage and a turn (10 s)",
                     color=INK, fontsize=13, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(args.out, bbox_inches="tight", facecolor="white")
        print(f"fold {args.fold} ({flight}): straight seq {s0[0]} {s0[2]:.1f} m | "
              f"turn seq {s1[0]} {s1[2]:.1f} m")
        print(f"wrote {args.out}")
        return

    chosen = next(r for r in recs if r[0] == args.seq) if args.seq is not None else pick(args.prefer)
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=150)
    si, grp, drift = draw(ax, chosen)
    ax.set_title(f"Held-out {flight}, {grp.lower()} outage (10 s)\n"
                 f"{label} endpoint drift {drift:.1f} m", color=INK, fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(f"fold {args.fold} ({flight}): drew seq {si}  group={grp}  drift={drift:.2f} m")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
