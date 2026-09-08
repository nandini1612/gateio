"""
Evaluate all four GPS-outage-bridging systems and print the results table.

Runs, on the chosen split (val or test):

    GateIO      learned TCN+attention model         (checkpoint required)
    GateIO-LSTM learned recurrent baseline           (checkpoint required)
    EKF         quaternion gravity-compensated filter (eval.ekf_baseline)
    Const-v     hold last GPS velocity (naive baseline)

and reports mean / median / 90th-percentile / %-under-5m drift overall and per
sequence group, then saves a per-sequence CSV.

The learned-model inference (``predict_sequence``) is the debugged reference
implementation: it rebuilds the model input exactly as the training data loader
did (IMU channels 0:10, forward-filled last-known GPS velocity as ``v_prev``, and
the outage flag), and integrates the *denormalised* predicted velocity to get
position. Reintroducing any of the historical bugs — stripping the GPS channels,
using normalised v_prev, or skipping denormalisation — silently breaks it.

Usage
-----
    python eval/evaluate.py --data MARS_Master_Dataset.npz \
        --gateio-ckpt marsnet_r20_final.pt \
        --lstm-ckpt   marsnet_lstm_best.pt \
        --split val --out-dir results

By default the EKF uses the locked tuned parameters from the paper. Pass
``--tune-ekf`` to re-tune on the val set (never on test).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.gateio import GateIO, GateIOLSTM, N_IMU_CHAN, SEQ_LEN, WIN_LEN, DT  # noqa: E402
from eval import ekf_baseline as ekf  # noqa: E402

OE_LEN = 100  # outage length in windows (10 s) — matches the training simulation

GROUP_MAP = ekf.GROUP_MAP
GROUP_ORDER = ekf.GROUP_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Learned-model inference (reference implementation — do not modify)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_sequence(model, x_norm_seq, y_raw_seq, outage_start, outage_end,
                     dv_iqr, dv_median, device):
    """Run a trained GateIO / GateIOLSTM model on one sequence.

    Args:
        x_norm_seq: (S, WIN_LEN, 14) manually normalised input windows.
        y_raw_seq:  (S, 3) raw (Y_iqr-normalised) GPS velocity targets from the NPZ.
        outage_start/outage_end: window indices bounding the outage.
        dv_iqr, dv_median: Y_iqr / Y_median from the NPZ (for denormalisation).
    Returns dict: dv_pred_ms, dv_true_ms, pos_pred, pos_true, outage_start, drift_m.
    """
    model.eval()
    S = x_norm_seq.shape[0]
    W = x_norm_seq.shape[1]

    # Outage flag with a short fractional ramp into the outage
    om = torch.zeros(S, dtype=torch.float32)
    if outage_start >= 3:
        om[outage_start - 2] = 1 / 3
        om[outage_start - 1] = 2 / 3
    om[outage_start:outage_end] = 1.0

    # v_prev: forward-fill the last known GPS velocity (raw Y values, NOT normalised)
    vr = torch.from_numpy(y_raw_seq).float()
    vp = torch.zeros(S, 3)
    last = vr[0].clone()
    for t in range(S):
        if om[t] > 0.5:
            vp[t] = last
        else:
            vp[t] = vr[t]
            last = vr[t].clone()

    # Input: IMU channels 0:10 + v_prev (10:13) + outage flag (13) — all 14 channels,
    # so the backbone can look back into pre-outage windows and recover v_prev.
    xi = torch.from_numpy(x_norm_seq).float()[:, :, :N_IMU_CHAN]
    vpch = vp.unsqueeze(1).expand(-1, W, -1)
    fch = om.view(S, 1, 1).expand(-1, W, 1)
    xf = torch.cat([xi, vpch, fch], dim=-1)

    dvp = model(xf.unsqueeze(0).to(device),
                om.unsqueeze(0).to(device),
                vp.unsqueeze(0).to(device)).squeeze(0).cpu()

    iqr = torch.from_numpy(dv_iqr).float()
    med = torch.from_numpy(dv_median).float()
    pm = dvp * iqr + med         # predicted GPS velocity (physical m/s)
    tm = vr                       # true GPS velocity

    def integrate(dv, s):
        pos = torch.zeros(S, 2)
        for t in range(s + 1, S):
            pos[t] = pos[t - 1] + dv[t, :2] * DT
        return pos.numpy()

    pp = integrate(pm, outage_start)
    pt = integrate(tm, outage_start)
    drift = float(np.linalg.norm(pp[outage_end - 1] - pt[outage_end - 1]))
    return {"dv_pred_ms": pm.numpy(), "dv_true_ms": tm.numpy(),
            "pos_pred": pp, "pos_true": pt,
            "outage_start": outage_start, "drift_m": drift}


# ─────────────────────────────────────────────────────────────────────────────
# Persistence-substitution diagnostic (measures the v2 fix's ceiling, no retrain)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def persistence_check(model, X, Y, idx, dv_iqr, dv_median, Xmed, Xiq, device,
                      gate_thr=0.10, groups=("straight-short",)):
    """For each sequence, re-integrate the outage with the model's velocity replaced by
    ``v_prev`` on gated straight windows (|gyro_z| < gate_thr). This is what a model that
    correctly deferred to persistence would achieve — the ceiling the residual (v2) head
    reaches for. Reports, per group, normal vs persistence-substituted vs const-v drift.
    """
    mid = WIN_LEN // 2
    rows = {g: [] for g in groups}
    for si in range(len(idx)):
        g = GROUP_MAP.get(si)
        if g not in rows:
            continue
        st = int(idx[si]); xrs = X[st:st + SEQ_LEN]; yrs = Y[st:st + SEQ_LEN]
        if len(xrs) < SEQ_LEN:
            continue
        xns = (xrs - Xmed) / np.where(Xiq < 1e-6, 1.0, Xiq)
        os_ = SEQ_LEN // 3; oe_ = min(os_ + OE_LEN, SEQ_LEN)
        r = predict_sequence(model, xns, yrs, os_, oe_, dv_iqr, dv_median, device)
        pm = r["dv_pred_ms"].copy()                       # (S,3) physical model velocity
        gyro = np.abs(xrs[:, mid, 5])                      # raw yaw rate
        vprev = yrs[os_ - 1] if os_ > 0 else yrs[0]
        for t in range(os_, oe_):
            if gyro[t] < gate_thr:
                pm[t, :2] = vprev[:2]                      # substitute persistence
        pp = np.zeros((SEQ_LEN, 2)); pt = np.zeros((SEQ_LEN, 2))
        for t in range(os_ + 1, SEQ_LEN):
            pp[t] = pp[t - 1] + pm[t, :2] * DT
            pt[t] = pt[t - 1] + yrs[t, :2] * DT
        sub_drift = float(np.linalg.norm(pp[oe_ - 1] - pt[oe_ - 1]))
        naive = ekf.run_naive_sequence(si, Y, idx, dv_iqr, dv_median)
        rows[g].append((r["drift_m"], sub_drift, naive))

    print(f"\n{'='*66}\n  PERSISTENCE-SUBSTITUTION CHECK (gate |gyro_z| < {gate_thr})\n{'='*66}")
    print(f"  {'Group':<16}{'GateIO':>10}{'persist-sub':>13}{'Const-v':>10}{'N':>5}")
    for g in groups:
        a = np.array(rows[g])
        if not len(a):
            continue
        print(f"  {g:<16}{a[:,0].mean():>9.2f}m{a[:,1].mean():>12.2f}m{a[:,2].mean():>9.2f}m{len(a):>5}")
    print("  persist-sub ≈ Const-v  =>  deferring to v_prev on straight windows recovers"
          "\n  the loss; that is the gap the v2 residual head is designed to close.")


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: str, kind: str, device):
    """Load a GateIO ('gateio') or GateIOLSTM ('lstm') checkpoint.

    weights_only=False is required: these checkpoints bundle numpy arrays
    (DV_iqr etc.) alongside the model weights, which PyTorch >= 2.6 refuses to
    unpickle under the default weights_only=True.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    residual = bool(ck.get("residual", False)) if isinstance(ck, dict) else False
    ctor = GateIO if kind == "gateio" else GateIOLSTM
    model = ctor(persistence_residual=residual).to(device)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    # Drop the residual head's non-persistent normalisation buffers if an older
    # checkpoint stored them; they are restored below via set_normalization.
    state = {k: v for k, v in state.items()
             if k not in ("head.y_med", "head.y_iqr", "head.dv_scale")}
    model.load_state_dict(state)
    if residual:
        # The residual head needs the velocity normalisation stats it was trained with.
        model.set_normalization(ck["DV_median"], ck["DV_iqr"], ck["DV_IQR_TRUE"])
        print(f"  loaded v2 (residual) checkpoint: {os.path.basename(ckpt_path)}")
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────
def run_all_models(model_gateio, model_lstm, X, Y, idx, dv_iqr, dv_median,
                   Xmed, Xiq, ekf_params, device, split_name, out_dir):
    q_v, q_bias, r = ekf_params
    n_seqs = len(idx)
    os.makedirs(out_dir, exist_ok=True)
    all_res = []

    print(f"\n{'='*70}\n  {split_name} SET — {n_seqs} sequences\n{'='*70}")
    print(f"  {'S':>3}  {'GateIO':>9}  {'LSTM':>8}  {'EKF':>8}  {'Const-v':>8}  Group")
    print("  " + "-" * 62)

    for si in range(n_seqs):
        start = int(idx[si])
        xrs = X[start:start + SEQ_LEN]
        yrs = Y[start:start + SEQ_LEN]
        if len(xrs) < SEQ_LEN:
            continue
        xns = (xrs - Xmed) / np.where(Xiq < 1e-6, 1.0, Xiq)
        os_ = SEQ_LEN // 3
        oe_ = min(os_ + OE_LEN, SEQ_LEN)
        grp = GROUP_MAP.get(si, "?")

        r_g = predict_sequence(model_gateio, xns, yrs, os_, oe_, dv_iqr, dv_median, device)
        r_l = predict_sequence(model_lstm, xns, yrs, os_, oe_, dv_iqr, dv_median, device)
        d_ekf = ekf.run_kf_sequence(si, q_v, q_bias, r, X, Y, idx, dv_iqr, dv_median)
        d_naive = ekf.run_naive_sequence(si, Y, idx, dv_iqr, dv_median)

        all_res.append({
            "seq_idx": si, "group": grp, "outage_start": os_,
            "drift_m": r_g["drift_m"], "lstm_drift_m": r_l["drift_m"],
            "ekf_drift_m": d_ekf, "naive_drift_m": d_naive,
            "pos_pred": r_g["pos_pred"], "pos_true": r_g["pos_true"],
        })
        print(f"  {si:>3}  {r_g['drift_m']:>8.2f}m  {r_l['drift_m']:>7.2f}m  "
              f"{d_ekf:>7.2f}m  {d_naive:>7.2f}m  {grp}")

    m = np.array([x["drift_m"] for x in all_res])
    ls = np.array([x["lstm_drift_m"] for x in all_res])
    ek = np.array([x["ekf_drift_m"] for x in all_res])
    na = np.array([x["naive_drift_m"] for x in all_res])

    print(f"\n  {'Metric':<22}{'GateIO':>10}{'LSTM':>10}{'EKF':>10}{'Const-v':>10}")
    print("  " + "-" * 62)
    for label, fn in [
        ("Mean (m)", lambda a: f"{a.mean():.2f}"),
        ("Median (m)", lambda a: f"{np.median(a):.2f}"),
        ("90th pctile (m)", lambda a: f"{np.percentile(a, 90):.2f}"),
        ("% under 5m", lambda a: f"{100 * np.mean(a < 5):.1f}%"),
    ]:
        print(f"  {label:<22}{fn(m):>10}{fn(ls):>10}{fn(ek):>10}{fn(na):>10}")

    print(f"\n  {'Group':<20}{'GateIO':>10}{'LSTM':>10}{'EKF':>10}{'Const-v':>10}{'N':>4}")
    print("  " + "-" * 66)
    for grp in GROUP_ORDER:
        gi = [i for i, x in enumerate(all_res) if x["group"] == grp]
        if not gi:
            continue
        print(f"  {grp:<20}" + "".join(f"{a[gi].mean():>9.2f}m" for a in (m, ls, ek, na))
              + f"{len(gi):>4}")
    print("  " + "=" * 66)

    csv_path = os.path.join(out_dir, f"{split_name.lower()}_all_results.csv")
    pd.DataFrame([{k: x[k] for k in ("seq_idx", "group", "outage_start", "drift_m",
                                     "lstm_drift_m", "ekf_drift_m", "naive_drift_m")}
                  for x in all_res]).to_csv(csv_path, index=False)
    print(f"  Saved {csv_path}")
    return all_res


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate GateIO + baselines.")
    ap.add_argument("--data", required=True, help="Path to MARS_Master_Dataset.npz")
    ap.add_argument("--gateio-ckpt", required=True, help="GateIO checkpoint (marsnet_r20_final.pt)")
    ap.add_argument("--lstm-ckpt", required=True, help="LSTM checkpoint (marsnet_lstm_best.pt)")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--tune-ekf", action="store_true",
                    help="Re-tune the EKF on the val set instead of using locked params.")
    ap.add_argument("--persistence-check", action="store_true",
                    help="Also run the persistence-substitution diagnostic on straight groups "
                         "(replaces the model with v_prev on gated straight windows).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    npz = np.load(args.data)
    dv_iqr = npz["Y_iqr"].astype(np.float32)
    dv_median = npz["Y_median"].astype(np.float32)
    Xmed = npz["X_median"].astype(np.float32)
    Xiq = npz["X_iqr"].astype(np.float32)

    if args.split == "val":
        X, Y, idx = npz["X_val"].astype(np.float32), npz["Y_val"].astype(np.float32), npz["val_valid_idx"]
    else:
        X, Y, idx = npz["X_test"].astype(np.float32), npz["Y_test"].astype(np.float32), npz["test_valid_idx"]

    # EKF params: tune on val, otherwise the locked paper values.
    if args.tune_ekf:
        Xv, Yv, vi = npz["X_val"].astype(np.float32), npz["Y_val"].astype(np.float32), npz["val_valid_idx"]
        ekf_params = ekf.tune_kf(Xv, Yv, vi, dv_iqr, dv_median)
    else:
        ekf_params = (ekf.Q_V_OPT, ekf.Q_B_OPT, ekf.R_OPT)
        print(f"EKF locked params: Q_v={ekf_params[0]:.3e} Q_bias={ekf_params[1]:.3e} R={ekf_params[2]:.3e}")

    model_gateio = load_model(args.gateio_ckpt, "gateio", device)
    model_lstm = load_model(args.lstm_ckpt, "lstm", device)

    run_all_models(model_gateio, model_lstm, X, Y, idx, dv_iqr, dv_median,
                   Xmed, Xiq, ekf_params, device, args.split.upper(), args.out_dir)

    if args.persistence_check:
        persistence_check(model_gateio, X, Y, idx, dv_iqr, dv_median, Xmed, Xiq, device,
                          groups=("straight-short", "straight-med", "FALSE-ALARM"))


if __name__ == "__main__":
    main()
