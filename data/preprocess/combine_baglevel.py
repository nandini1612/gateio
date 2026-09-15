"""
Bag-level (leave-one-flight-out) dataset assembly for GateIO v2.

The paper's v2 dataset (combine_and_norm_v2.py) splits each flight 80/10/10
*chronologically*, so val/test sequences are temporally adjacent to training data
from the same flight. That within-flight leakage inflates validation numbers and
(as diagnosed) corrupts model selection. This script instead assigns *whole
flights* to train/val/test, so no flight contributes to more than one split.

With only five flights (two turn-heavy: gnss03, island_gnss03), a single held-out
split can't cover turns in train, val and test at once, so we use
leave-one-flight-out cross-validation: five folds, each holding out one flight as
test, one as val, training on the other three. Every flight is tested exactly once
with no leakage, and each fold trains on at least one turn-heavy flight.

Output is byte-compatible with the v2 pipeline (same keys/normalisation as
combine_and_norm_v2.py: raw X/Y, train-only robust scaling, IQR floors, quaternion
passthrough, per-flight sequence indices) plus DV_IQR_TRUE (the per-axis velocity-
increment scale) for the v2 residual head.

Usage:
    python data/preprocess/combine_baglevel.py --fold 0 --out fold0.npz
    # or explicit:
    python data/preprocess/combine_baglevel.py \
        --train gnss03 island_gnss02 island_gnss03 --val gnss02 --test gnss01 --out fold0.npz
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.stats import iqr

# ── Frozen channel map / constants (must match combine_and_norm_v2.py) ─────────
SEQ_LEN = 300
N_CHANNELS = 14
CH_GYR_START, CH_GYR_END = 3, 6
CH_QUAT_START, CH_QUAT_END = 6, 10
CH_GPS_START, CH_GPS_END = 10, 13
CH_MASK = 13
PASSTHROUGH_CHANNELS = list(range(CH_QUAT_START, CH_QUAT_END)) + [CH_MASK]

TURN_GYRO_THRESH = 0.5           # rad/s, on raw (bias-removed) gyro
TURN_WEIGHT_MULTIPLIER = 3.0
Y_IQR_FLOOR = 0.1
X_GPS_IQR_FLOOR = 0.1
TRAIN_STRIDE_STD = 30
TRAIN_STRIDE_TURN = 15
VAL_TEST_STRIDE = 10
TURN_HEAVY = ["gnss03", "island_gnss03"]

FLIGHTS = ["gnss01", "gnss02", "gnss03", "island_gnss02", "island_gnss03"]

# Leave-one-flight-out folds: test rotates through all five; val = next flight;
# train = the remaining three. Each train set contains a turn-heavy flight.
FOLDS = [
    {"test": "gnss01",        "val": "gnss02",        "train": ["gnss03", "island_gnss02", "island_gnss03"]},
    {"test": "gnss02",        "val": "gnss03",        "train": ["gnss01", "island_gnss02", "island_gnss03"]},
    {"test": "gnss03",        "val": "island_gnss02", "train": ["gnss01", "gnss02", "island_gnss03"]},
    {"test": "island_gnss02", "val": "island_gnss03", "train": ["gnss01", "gnss02", "gnss03"]},
    {"test": "island_gnss03", "val": "gnss01",        "train": ["gnss02", "gnss03", "island_gnss02"]},
]


def flight_path(processed_dir: str, name: str) -> str:
    return os.path.join(processed_dir, f"transformer_ds_{name}.npz")


def load_flight(processed_dir: str, name: str):
    p = flight_path(processed_dir, name)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Per-flight file not found: {p}\n"
            f"combine_baglevel needs transformer_ds_<flight>.npz for all 5 flights in "
            f"--processed-dir. Point --processed-dir at the folder that holds them, or "
            f"upload them there.")
    d = np.load(p, allow_pickle=True)
    return d["X_train"].astype(np.float32), d["y_train"].astype(np.float32)


def valid_indices(lengths, names, seq_len, stride):
    """Sequence start indices within the concatenated array, never crossing a
    flight boundary; turn-heavy flights use the tighter TRAIN_STRIDE_TURN."""
    idx, offset = [], 0
    for L, nm in zip(lengths, names):
        s = TRAIN_STRIDE_TURN if any(t in nm for t in TURN_HEAVY) else stride
        if L >= seq_len:
            idx.extend(range(offset, offset + L - seq_len + 1, s))
        offset += L
    return np.array(idx, dtype=np.int32)


def increment_iqr(Y_list):
    """Per-axis IQR of the per-step velocity increment (Δv), within-flight only."""
    incs = [np.diff(y, axis=0) for y in Y_list if len(y) > 1]
    allinc = np.concatenate(incs, axis=0)
    return iqr(allinc, axis=0).astype(np.float32)


def build(processed_dir, train, val, test, out):
    print(f"train={train}  val={val}  test={test}")
    Xtr_list = [load_flight(processed_dir, n)[0] for n in train]
    Ytr_list = [load_flight(processed_dir, n)[1] for n in train]
    Xv, Yv = load_flight(processed_dir, val)
    Xt, Yt = load_flight(processed_dir, test)

    X_train = np.concatenate(Xtr_list, 0)
    Y_train = np.concatenate(Ytr_list, 0)
    tr_lengths = [len(x) for x in Xtr_list]

    # ── train-only robust scaling ──────────────────────────────────────────────
    flat = X_train.reshape(-1, N_CHANNELS)
    X_median = np.median(flat, axis=0).astype(np.float32)
    X_iqr = iqr(flat, axis=0).astype(np.float32)
    X_iqr[X_iqr < 1e-4] = 1e-4
    for ch in range(CH_GPS_START, CH_GPS_END):
        X_iqr[ch] = max(X_iqr[ch], X_GPS_IQR_FLOOR)
    for ch in PASSTHROUGH_CHANNELS:            # quaternion + mask passthrough
        X_median[ch] = 0.0
        X_iqr[ch] = 1.0

    Y_median = np.median(Y_train, axis=0).astype(np.float32)
    Y_iqr = iqr(Y_train, axis=0).astype(np.float32)
    Y_iqr[Y_iqr < Y_IQR_FLOOR] = Y_IQR_FLOOR

    dv_iqr_true = increment_iqr(Ytr_list)      # per-fold velocity-increment scale

    # ── turn weighting (per train window) ───────────────────────────────────────
    max_gyro = np.max(np.abs(X_train[:, :, CH_GYR_START:CH_GYR_END]), axis=(1, 2))
    W_train = np.ones(len(X_train), dtype=np.float32)
    W_train[max_gyro > TURN_GYRO_THRESH] = TURN_WEIGHT_MULTIPLIER

    # ── sequence indices ─────────────────────────────────────────────────────────
    train_valid_idx = valid_indices(tr_lengths, train, SEQ_LEN, TRAIN_STRIDE_STD)
    val_valid_idx = valid_indices([len(Xv)], [val], SEQ_LEN, VAL_TEST_STRIDE)
    test_valid_idx = valid_indices([len(Xt)], [test], SEQ_LEN, VAL_TEST_STRIDE)

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    np.savez_compressed(
        out,
        X_train=X_train, Y_train=Y_train, W_train=W_train,
        W_sample=np.ones(len(X_train), dtype=np.float32),   # unused by the pipeline; kept for compat
        X_val=Xv, Y_val=Yv, X_test=Xt, Y_test=Yt,
        train_valid_idx=train_valid_idx,
        val_valid_idx=val_valid_idx,
        test_valid_idx=test_valid_idx,
        X_median=X_median, X_iqr=X_iqr, Y_median=Y_median, Y_iqr=Y_iqr,
        DV_IQR_TRUE=dv_iqr_true,
        seq_len=np.array([SEQ_LEN]), n_channels=np.array([N_CHANNELS]),
        ch_gyr_start=np.array([CH_GYR_START]), ch_gyr_end=np.array([CH_GYR_END]),
        ch_quat_start=np.array([CH_QUAT_START]), ch_quat_end=np.array([CH_QUAT_END]),
        ch_gps_start=np.array([CH_GPS_START]), ch_gps_end=np.array([CH_GPS_END]),
        ch_mask=np.array([CH_MASK]),
        # provenance (plain unicode scalars — no pickle needed to read back)
        split_train=np.array(",".join(train)),
        split_val=np.array(val),
        split_test=np.array(test),
    )
    print(f"  train windows {len(X_train):>6}  seqs {len(train_valid_idx):>4}")
    print(f"  val   windows {len(Xv):>6}  seqs {len(val_valid_idx):>4}")
    print(f"  test  windows {len(Xt):>6}  seqs {len(test_valid_idx):>4}")
    print(f"  Y_iqr={Y_iqr.round(3)}  Y_median={Y_median.round(3)}  DV_IQR_TRUE={dv_iqr_true.round(4)}")
    print(f"  saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Bag-level (LOFO) combine for GateIO v2.")
    ap.add_argument("--processed-dir", default="./data/processed",
                    help="Directory holding transformer_ds_<flight>.npz")
    ap.add_argument("--fold", type=int, choices=range(5),
                    help="Predefined leave-one-flight-out fold (0-4).")
    ap.add_argument("--train", nargs="+", help="Explicit train flight names.")
    ap.add_argument("--val", help="Explicit val flight name.")
    ap.add_argument("--test", help="Explicit test flight name.")
    ap.add_argument("--out", required=True, help="Output .npz path.")
    args = ap.parse_args()

    if args.fold is not None:
        f = FOLDS[args.fold]
        train, val, test = f["train"], f["val"], f["test"]
    else:
        assert args.train and args.val and args.test, "give --fold, or --train/--val/--test"
        train, val, test = args.train, args.val, args.test
    build(args.processed_dir, train, val, test, args.out)


if __name__ == "__main__":
    main()
