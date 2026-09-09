"""
Train GateIO (or the GateIO-LSTM baseline) for UAV GPS-outage bridging.

Script version of the R20 training notebook, so training does not require Colab.
The model, loss, sampler, and schedule are ported verbatim from the notebook that
produced the paper's checkpoints; only the I/O (paths, CLI, checkpoint layout) is
cleaned up.

The loss centrepiece is the yaw-rate-gated velocity-persistence prior
``L_cvprior``:

    On outage windows with |gyro_z| < CVPRIOR_GYRO_THR (straight cruise), the model
    is pushed toward *zero velocity change* (constant-velocity prior). The gate
    switches OFF during turns (|gyro_z| >= threshold) so the dead-reckoning loss
    L_dr — and only L_dr — teaches turning dynamics. An earlier ungated velocity
    prior competed with L_dr on turns and was overwhelmed; gating is what makes the
    prior safe to weight heavily.

Usage
-----
    python train/train_gateio.py --data /path/to/MARS_Master_Dataset.npz \
        --model gateio --ckpt-dir ./checkpoints

    python train/train_gateio.py --data ... --model lstm --ckpt-dir ./checkpoints

The dataset loader (``data/data_loader.py``) and NPZ format are documented in the
project README.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

# Make sibling packages importable when run as a script from anywhere.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from models.gateio import (  # noqa: E402
    GateIO, GateIOLSTM, N_IMU_CHAN, SEQ_LEN, WIN_LEN, DT, DV_SCALE_FLOOR,
)
from data.data_loader import MARSDataset  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# Loss weights and gates (frozen R20 values — do not change for a fair comparison)
# ─────────────────────────────────────────────────────────────────────────────
LAM_DR = 0.90               # dead-reckoning (outage) Huber loss
LAM_CVPRIOR = 0.50          # gated constant-velocity prior
LAM_SMOOTH = 0.001          # jerk penalty
LAM_DRIFT = 0.002           # cumulative position-error penalty (warmed in)
LAM_PHYS_MAX = 0.01         # ZUPT physics penalty ceiling (warmed in)

CVPRIOR_GYRO_THR = 0.10     # rad/s — yaw-rate gate for L_cvprior (~87% straight / 13% turning)
TURN_GYRO_THR = 0.10        # rad/s — sequence-level turn detection for the sampler
ZUPT_GYRO_THR = 0.05        # rad/s — near-stationary gate for L_phys
HUBER_DELTA = 0.3
DR_AXIS_WEIGHTS = torch.tensor([1.0, 1.0, 3.0])   # up-weight vertical axis in L_dr

# Warmup schedules (epochs)
PHYS_WARM_START, PHYS_WARM_END = 30, 60
DRIFT_WARM_START, DRIFT_WARM_END = 30, 70

# Augmentation
TEMPORAL_DROPOUT_P = 0.25   # randomly drop whole windows during training
W_TURN = 8.0                # sequence-level upsampling weight for turning sequences

# Training
MAX_EPOCHS = 200
PATIENCE = 60
LR = 8e-4
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16

# Δv "true" IQR reference retained in the checkpoint for downstream diagnostics.
DV_IQR_TRUE = np.array([0.055, 0.319, 0.00078], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# v_prev: forward-filled last-known GPS velocity (physical units)
# ─────────────────────────────────────────────────────────────────────────────
def compute_v_prev(y_norm: torch.Tensor, outage_mask: torch.Tensor,
                   dv_iqr_t: torch.Tensor, dv_med_t: torch.Tensor):
    """Vectorised forward-fill of the last GPS velocity before/at each step.

    Returns (v_prev, v_raw), both physical (m/s). During an outage v_prev holds
    the velocity from the last GPS-aided window; otherwise it equals the current
    (denormalised) target.
    """
    device = y_norm.device
    B, S, _ = y_norm.shape
    v_raw = y_norm * dv_iqr_t + dv_med_t
    not_flag = ~(outage_mask > 0.5)
    t_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    gps_idx = torch.where(not_flag, t_idx, torch.zeros_like(t_idx))
    last_gps = gps_idx.cummax(dim=1).values
    v_prev = torch.stack([v_raw[:, :, ax].gather(1, last_gps) for ax in range(3)], dim=-1)
    return v_prev, v_raw


# ─────────────────────────────────────────────────────────────────────────────
# Combined loss (R20 "v10", gated L_cvprior)
# ─────────────────────────────────────────────────────────────────────────────
def combined_loss(dv_pred_norm, dv_true_norm, outage_mask, dv_iqr_t, dv_median_t,
                  v_prev=None, x_raw=None, lam_p=0.0, lam_d=0.0, cap_m=100.0,
                  residual_mode=False, dv_scale_t=None):
    """Total training loss and a dict of its components.

    Terms:
        L_data      Huber on GPS-aided windows (xy)
        L_dr        Huber on all outage windows, axis-weighted (the DR teacher)
        L_cvprior   gated constant-velocity prior: on straight outage windows,
                    push predicted velocity change toward zero
        L_smooth    jerk penalty over the whole sequence
        L_phys      ZUPT penalty (warmed in via lam_p)
        L_trans     penalty on the first outage window (suppress a boundary spike)
        L_drift     cumulative xy position error over the outage (warmed in via lam_d)
    """
    B, S, _ = dv_pred_norm.shape
    device = dv_pred_norm.device
    iqr = dv_iqr_t.to(device)
    med = dv_median_t.to(device)
    aw = DR_AXIS_WEIGHTS.to(device)
    out_mask = outage_mask > 0.5
    aid_mask = ~out_mask

    # L_data — GPS-aided windows, xy only
    L_data = (F.huber_loss(dv_pred_norm[aid_mask][:, :2], dv_true_norm[aid_mask][:, :2],
                           delta=HUBER_DELTA, reduction="mean")
              if aid_mask.any() else torch.tensor(0.0, device=device))

    # ── v2 residual mode: express L_dr / L_cvprior as the velocity *increment*
    # over v_prev, normalised by the true increment scale (DV_IQR_TRUE). This
    # un-compresses the wide-IQR forward axis and makes "hold velocity" (increment
    # = 0) the target of the prior — instead of "predict the training median".
    # dv_pred here equals the residual head's raw output r when the two are paired.
    if residual_mode:
        assert dv_scale_t is not None and v_prev is not None, "residual_mode needs dv_scale_t and v_prev"
        dvs = dv_scale_t.to(device)
        v_pred_phys = dv_pred_norm * iqr + med
        v_true_phys = dv_true_norm * iqr + med
        inc_pred = (v_pred_phys - v_prev) / dvs          # predicted increment (unit-scale)
        inc_true = (v_true_phys - v_prev) / dvs          # true increment (unit-scale)
    else:
        inc_pred = dv_pred_norm
        inc_true = dv_true_norm

    # L_dr — all outage windows, axis-weighted (on the increment in v2)
    L_dr = (F.huber_loss(inc_pred[out_mask] * aw, inc_true[out_mask] * aw,
                         delta=HUBER_DELTA, reduction="mean")
            if out_mask.any() else torch.tensor(0.0, device=device))

    # L_cvprior — gated constant-velocity prior (the paper's key term)
    L_cvprior = torch.tensor(0.0, device=device)
    n_straight_out = 0
    if x_raw is not None and out_mask.any() and v_prev is not None:
        gyro_z = x_raw[:, :, WIN_LEN // 2, 5].abs()          # yaw rate (channel 5)
        straight_out = (gyro_z < CVPRIOR_GYRO_THR) & out_mask
        n_straight_out = int(straight_out.sum().item())
        if straight_out.any():
            # Push the velocity increment toward zero on straight windows.
            # v1: increment == normalised absolute velocity, so "→0" pulls toward the
            #     training median (the bug). v2: increment is (v_pred - v_prev)/DV_scale,
            #     so "→0" means genuinely hold the last known velocity.
            zero_target = torch.zeros_like(inc_pred[straight_out][:, :2])
            L_cvprior = F.huber_loss(inc_pred[straight_out][:, :2], zero_target,
                                     delta=HUBER_DELTA, reduction="mean")

    # L_smooth — jerk penalty
    L_smooth = (dv_pred_norm[:, 1:, :] - dv_pred_norm[:, :-1, :]).pow(2).mean()

    # L_phys — ZUPT gate
    L_phys = torch.tensor(0.0, device=device)
    if lam_p > 0 and x_raw is not None:
        gm = x_raw[:, :, WIN_LEN // 2, 3:6].norm(dim=-1)
        zm = (gm < ZUPT_GYRO_THR) & out_mask
        if zm.any():
            L_phys = (dv_pred_norm[zm].norm(dim=-1) * gm[zm]).mean()

    # L_trans — suppress a spike at the outage boundary
    trans_list = []
    for b in range(B):
        oi = out_mask[b].nonzero(as_tuple=True)[0]
        if len(oi) > 0:
            trans_list.append(dv_pred_norm[b, oi[0]].pow(2).mean())
    L_trans = torch.stack(trans_list).mean() if trans_list else torch.tensor(0.0, device=device)

    # L_drift — cumulative position error during outage
    L_drift = torch.tensor(0.0, device=device)
    if lam_d > 0 and out_mask.any():
        dl = []
        for b in range(B):
            oi = out_mask[b].nonzero(as_tuple=True)[0]
            if len(oi) == 0:
                continue
            pm = dv_pred_norm[b, oi] * iqr + med
            tm = dv_true_norm[b, oi] * iqr + med
            pe = ((pm - tm) * DT).cumsum(0).norm(dim=-1)
            dl.append(pe[-1].clamp(max=cap_m))
        if dl:
            L_drift = torch.stack(dl).mean()

    total = (L_data + LAM_DR * L_dr + LAM_CVPRIOR * L_cvprior
             + LAM_SMOOTH * L_smooth + lam_p * L_phys
             + LAM_DRIFT * lam_d * L_drift + 0.05 * L_trans)

    comps = {"L_data": L_data.item(), "L_dr": L_dr.item(), "L_cvprior": L_cvprior.item(),
             "L_phys": L_phys.item(), "L_smooth": L_smooth.item(),
             "L_drift": L_drift.item(), "L_trans": L_trans.item(),
             "n_straight_out": float(n_straight_out)}
    return total, comps


# ─────────────────────────────────────────────────────────────────────────────
# Input assembly (shared by train and val)
# ─────────────────────────────────────────────────────────────────────────────
def build_model_input(xn, om, vp, temporal_dropout: bool, device):
    """Assemble the (B,S,W,14) model input: IMU 0:10 | v_prev | outage flag."""
    B, S, W, _ = xn.shape
    flag = om.float()
    xi = xn[:, :, :, :N_IMU_CHAN]
    if temporal_dropout and TEMPORAL_DROPOUT_P > 0:
        xi = xi * (torch.rand(B, S, 1, 1, device=device) > TEMPORAL_DROPOUT_P).float()
    vpch = vp.unsqueeze(2).expand(-1, -1, W, -1)
    fch = flag.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, W, 1)
    xf = torch.cat([xi, vpch, fch], dim=-1)
    return xf, flag


def train_one_epoch(model, loader, optimizer, scheduler, dv_iqr_t, dv_median_t,
                    device, lam_p=0.0, lam_d=0.0, residual_mode=False, dv_scale_t=None):
    model.train()
    totals = defaultdict(float)
    n = 0
    for batch in loader:
        xn = batch["x_norm"].to(device)
        yn = batch["y_norm"].to(device)
        om = batch["outage_mask"].to(device)
        xr = batch["x_raw"].to(device)
        vp, _ = compute_v_prev(yn, om, dv_iqr_t, dv_median_t)
        xf, flag = build_model_input(xn, om, vp, temporal_dropout=True, device=device)

        optimizer.zero_grad()
        dv_pred = model(xf, flag, vp)
        loss, comps = combined_loss(dv_pred, yn, om, dv_iqr_t, dv_median_t,
                                    v_prev=vp, x_raw=xr, lam_p=lam_p, lam_d=lam_d,
                                    residual_mode=residual_mode, dv_scale_t=dv_scale_t)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()

        totals["total"] += loss.item()
        for k, v in comps.items():
            totals[k] += v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_val_drift(model, loader, dv_iqr_t, dv_median_t, device) -> float:
    """Mean endpoint drift (m) over the val set — the model-selection metric."""
    model.eval()
    drifts = []
    for batch in loader:
        xn = batch["x_norm"].to(device)
        yn = batch["y_norm"].to(device)
        om = batch["outage_mask"].to(device)
        vp, _ = compute_v_prev(yn, om, dv_iqr_t, dv_median_t)
        xf, flag = build_model_input(xn, om, vp, temporal_dropout=False, device=device)
        dvp = model(xf, flag, vp)
        B = xn.shape[0]
        for b in range(B):
            oi = om[b].nonzero(as_tuple=True)[0]
            if len(oi) == 0:
                continue
            # Δv space: reconstruct absolute velocity as v_prev + Δv, then integrate.
            vp_b = vp[b]
            pm = vp_b + dvp[b] * dv_iqr_t + dv_median_t
            tm = vp_b + yn[b] * dv_iqr_t + dv_median_t
            pe = ((pm - tm) * DT)[oi].cumsum(0).norm(dim=-1)
            drifts.append(pe[-1].item())
    return float(np.mean(drifts)) if drifts else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Sampler
# ─────────────────────────────────────────────────────────────────────────────
def build_train_sampler(npz, n_seqs, train_valid_idx):
    """Weighted sequence sampler: upsample turning sequences by W_TURN."""
    X_tr = npz["X_train"]
    W_train_flat = npz["W_train"]
    weights = np.zeros(n_seqs, dtype=np.float32)
    for i, start in enumerate(train_valid_idx):
        end = min(int(start) + SEQ_LEN, len(X_tr))
        gyro = np.linalg.norm(X_tr[int(start):end, WIN_LEN // 2, 3:6], axis=-1)
        weights[i] = (W_TURN if np.any(gyro > TURN_GYRO_THR)
                      else float(np.mean(W_train_flat[int(start):end])))
    weights /= weights.sum()
    return WeightedRandomSampler(torch.from_numpy(weights).float(),
                                 num_samples=n_seqs, replacement=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description="Train GateIO / GateIO-LSTM.")
    ap.add_argument("--data", required=True, help="Path to MARS_Master_Dataset.npz")
    ap.add_argument("--model", choices=["gateio", "lstm"], default="gateio")
    ap.add_argument("--ckpt-dir", default="./checkpoints", help="Where to save checkpoints")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--v2", action="store_true",
                    help="Enable the residual/persistence parametrisation (v2 fix): the "
                         "DR head predicts a velocity increment over v_prev and the prior "
                         "is scaled by DV_IQR_TRUE. Default off = original v1 (reproduces the paper).")
    ap.add_argument("--lam-cvprior", type=float, default=None,
                    help="Override LAM_CVPRIOR (persistence prior weight). Raise it in v2 to "
                         "hold persistence on straight/long groups; turns are gated off so are unaffected.")
    ap.add_argument("--lam-drift", type=float, default=None,
                    help="Override LAM_DRIFT (cumulative-position-error weight).")
    args = ap.parse_args()

    # Optional loss-weight overrides (v2 tuning). Reassigning the module globals is
    # sufficient because combined_loss reads them at call time from this module.
    global LAM_CVPRIOR, LAM_DRIFT
    if args.lam_cvprior is not None:
        LAM_CVPRIOR = args.lam_cvprior
    if args.lam_drift is not None:
        LAM_DRIFT = args.lam_drift

    os.makedirs(args.ckpt_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  model: {args.model}  |  "
          f"LAM_CVPRIOR={LAM_CVPRIOR}  LAM_DRIFT={LAM_DRIFT}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    npz = np.load(args.data)
    dv_iqr = npz["Y_iqr"].astype(np.float32)
    dv_median = npz["Y_median"].astype(np.float32)
    dv_iqr_t = torch.from_numpy(dv_iqr).to(device)
    dv_med_t = torch.from_numpy(dv_median).to(device)

    train_ds = MARSDataset(args.data, split="train", outage_prob=0.8)
    val_ds = MARSDataset(args.data, split="val", outage_prob=0.0)
    train_valid_idx = npz["train_valid_idx"]
    n_seqs = len(train_valid_idx)

    sampler = build_train_sampler(npz, n_seqs, train_valid_idx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    n_batches = math.ceil(n_seqs / args.batch_size)
    print(f"Train seqs: {n_seqs}  Val seqs: {len(val_ds)}  Batches/epoch: {n_batches}")

    ctor = GateIO if args.model == "gateio" else GateIOLSTM
    model = ctor(persistence_residual=args.v2).to(device)
    dv_scale_t = (torch.from_numpy(DV_IQR_TRUE).to(device).clamp_min(DV_SCALE_FLOOR)
                  if args.v2 else None)
    if args.v2:
        model.set_normalization(dv_median, dv_iqr, DV_IQR_TRUE)  # floors internally
        print(f"v2 residual head ON  |  DV_scale (floored at {DV_SCALE_FLOOR}) = "
              f"{dv_scale_t.cpu().numpy()}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * n_batches,
        pct_start=0.05, anneal_strategy="cos", div_factor=25, final_div_factor=100)

    tag = "_v2" if args.v2 else ""
    ckpt_best = os.path.join(args.ckpt_dir, f"{args.model}{tag}_best.pt")
    ckpt_last = os.path.join(args.ckpt_dir, f"{args.model}{tag}_last.pt")

    history = defaultdict(list)
    best_drift = float("inf")
    patience_ctr = 0

    print(f"\n{'Ep':>4} {'Loss':>9} {'Drift':>8} {'L_cvp':>7} {'t(s)':>6}  Status")
    print("-" * 60)
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        lam_p = min(LAM_PHYS_MAX, LAM_PHYS_MAX * max(0, epoch - PHYS_WARM_START)
                    / max(1, PHYS_WARM_END - PHYS_WARM_START))
        lam_d = min(1.0, max(0.0, epoch - DRIFT_WARM_START)
                    / max(1, DRIFT_WARM_END - DRIFT_WARM_START))

        tc = train_one_epoch(model, train_loader, optimizer, scheduler,
                             dv_iqr_t, dv_med_t, device, lam_p, lam_d,
                             residual_mode=args.v2, dv_scale_t=dv_scale_t)
        vd = eval_val_drift(model, val_loader, dv_iqr_t, dv_med_t, device)

        history["train_loss"].append(tc["total"])
        history["val_drift_m"].append(vd)
        for k in ("L_data", "L_dr", "L_cvprior", "L_smooth", "L_drift"):
            history[k].append(tc.get(k, 0.0))

        is_best = (not math.isnan(vd)) and (vd < best_drift)
        if is_best:
            best_drift = vd
            patience_ctr = 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_drift,
                        "history": dict(history), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "DV_iqr": dv_iqr, "DV_median": dv_median,
                        "DV_IQR_TRUE": DV_IQR_TRUE, "run": args.model,
                        "residual": args.v2}, ckpt_best)
            status = f"* BEST {best_drift:.2f}m"
        else:
            patience_ctr += 1
            status = f"({patience_ctr}/{PATIENCE})"

        torch.save({"model": model.state_dict(), "epoch": epoch, "best_val": best_drift,
                    "history": dict(history), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "patience_ctr": patience_ctr,
                    "DV_iqr": dv_iqr, "DV_median": dv_median,
                    "DV_IQR_TRUE": DV_IQR_TRUE, "run": args.model,
                    "residual": args.v2}, ckpt_last)

        if epoch == 1 or epoch % 5 == 0 or is_best:
            print(f"{epoch:>4} {tc['total']:>9.4f} {vd:>7.2f}m {tc.get('L_cvprior', 0.):>7.4f} "
                  f"{time.time() - t0:>6.0f}  {status}")

        if patience_ctr >= PATIENCE:
            print(f"Early stop at epoch {epoch}")
            break

    print(f"\nBest val drift: {best_drift:.3f} m  ->  {ckpt_best}")


if __name__ == "__main__":
    main()
