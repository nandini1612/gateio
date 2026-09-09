"""
Leave-one-flight-out (LOFO) evaluation for GateIO v2.

Runs over the five bag-level folds produced by data/preprocess/combine_baglevel.py.
For each fold it evaluates the trained v2 GateIO checkpoint on that fold's held-out
TEST flight, alongside the constant-velocity and (per-fold-tuned) EKF baselines,
then pools every test sequence across all folds for a single leakage-free estimate.

Because held-out flights don't have the paper's fixed sequence groups, each test
sequence is classified data-driven as TURN vs STRAIGHT by its peak yaw rate during
the outage (|gyro_z| > TURN_GYRO_THR).

Layout expected:
    <folds_dir>/fold{0..4}.npz                     (combine_baglevel output)
    <ckpt_dir>/fold{0..4}/gateio_v2_best.pt        (per-fold v2 training output)

Usage:
    python eval/evaluate_lofo.py --folds-dir ./folds --ckpt-dir ./checkpoints_lofo
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.gateio import WIN_LEN, SEQ_LEN, DT  # noqa: E402
from eval.evaluate import predict_sequence, load_model  # noqa: E402
from eval import ekf_baseline as ekf  # noqa: E402

OE_LEN = 100
TURN_GYRO_THR = 0.10        # rad/s — per-sequence turn/straight classification
N_FOLDS = 5


def classify(xrs, os_, oe_):
    """TURN if the outage involves *sustained* yaw (mean |gyro_z| over the outage
    exceeds the threshold), else STRAIGHT. Mean, not max, so a brief yaw jitter in
    otherwise-level flight is not mislabelled a turn."""
    gz = np.abs(xrs[os_:oe_, WIN_LEN // 2, 5])
    return "TURN" if gz.mean() > TURN_GYRO_THR else "STRAIGHT"


def eval_fold(fold_npz, ckpt, device, tune_ekf=True):
    npz = np.load(fold_npz)
    Xt = npz["X_test"].astype(np.float32); Yt = npz["Y_test"].astype(np.float32)
    ti = npz["test_valid_idx"]
    Xmed = npz["X_median"].astype(np.float32); Xiq = npz["X_iqr"].astype(np.float32)
    dv_iqr = npz["Y_iqr"].astype(np.float32); dv_median = npz["Y_median"].astype(np.float32)
    test_flight = str(npz["split_test"]) if "split_test" in npz.files else "?"

    model = load_model(ckpt, "gateio", device)

    # EKF params: tune on this fold's val flight (never on test), else locked.
    if tune_ekf:
        Xv = npz["X_val"].astype(np.float32); Yv = npz["Y_val"].astype(np.float32)
        vi = npz["val_valid_idx"]
        q_v, q_b, r = ekf.tune_kf(Xv, Yv, vi, dv_iqr, dv_median, verbose=False)
    else:
        q_v, q_b, r = ekf.Q_V_OPT, ekf.Q_B_OPT, ekf.R_OPT

    rows = []
    for si in range(len(ti)):
        st = int(ti[si]); xrs = Xt[st:st + SEQ_LEN]; yrs = Yt[st:st + SEQ_LEN]
        if len(xrs) < SEQ_LEN:
            continue
        xns = (xrs - Xmed) / np.where(Xiq < 1e-6, 1.0, Xiq)
        os_ = SEQ_LEN // 3; oe_ = min(os_ + OE_LEN, SEQ_LEN)
        g = predict_sequence(model, xns, yrs, os_, oe_, dv_iqr, dv_median, device)["drift_m"]
        e = ekf.run_kf_sequence(si, q_v, q_b, r, Xt, Yt, ti, dv_iqr, dv_median)
        n = ekf.run_naive_sequence(si, Yt, ti, dv_iqr, dv_median)
        rows.append((classify(xrs, os_, oe_), g, e, n))
    return test_flight, rows


def summarize(tag, rows):
    a = np.array([[r[1], r[2], r[3]] for r in rows], dtype=float)  # GateIO, EKF, Const-v
    if not len(a):
        return
    def line(name, mask):
        b = a[mask]
        if not len(b):
            return
        print(f"  {name:<14}{b[:,0].mean():>9.2f}m{b[:,1].mean():>9.2f}m{b[:,2].mean():>9.2f}m"
              f"{100*np.mean(b[:,0]<5):>8.0f}%{len(b):>5}")
    grp = np.array([r[0] for r in rows])
    print(f"\n  {tag}   ({len(rows)} seqs)")
    print(f"  {'':<14}{'GateIO':>10}{'EKF':>10}{'Const-v':>10}{'G<5m':>8}{'N':>5}")
    line("STRAIGHT", grp == "STRAIGHT")
    line("TURN", grp == "TURN")
    line("all", np.ones(len(a), bool))


def main():
    ap = argparse.ArgumentParser(description="Leave-one-flight-out evaluation for GateIO v2.")
    ap.add_argument("--folds-dir", required=True, help="Dir with fold0.npz .. fold4.npz")
    ap.add_argument("--ckpt-dir", required=True,
                    help="Dir with fold0/gateio_v2_best.pt .. fold4/gateio_v2_best.pt")
    ap.add_argument("--no-tune-ekf", action="store_true", help="Use locked EKF params.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    pooled = []
    print("=" * 62)
    for k in range(N_FOLDS):
        fold_npz = os.path.join(args.folds_dir, f"fold{k}.npz")
        ckpt = os.path.join(args.ckpt_dir, f"fold{k}", "gateio_v2_best.pt")
        if not (os.path.exists(fold_npz) and os.path.exists(ckpt)):
            print(f"[fold {k}] missing ({fold_npz} / {ckpt}) — skipped")
            continue
        flight, rows = eval_fold(fold_npz, ckpt, device, tune_ekf=not args.no_tune_ekf)
        summarize(f"fold {k}: test={flight}", rows)
        pooled.extend(rows)

    print("\n" + "=" * 62)
    summarize("LOFO POOLED (all held-out flights)", pooled)
    print("=" * 62)


if __name__ == "__main__":
    main()
