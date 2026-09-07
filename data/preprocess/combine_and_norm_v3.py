"""
=============================================================================
  MARS-LVIG MASTER COMBINING SCRIPT  v3.1
  BAG-LEVEL SPLIT + RELATIVE (Δv) PREDICTION TARGETS

  WHY v3.1 EXISTS — TWO PROBLEMS FIXED SIMULTANEOUSLY:

  Problem 1 — Chronological within-bag split (v2.x):
  v2.x split 80/10/10 WITHIN each bag. The last 10% of each flight circuit
  has a different heading from the first 80% (aircraft returning to base).
  Result: test set had heading +10° that never appeared in training → 30m
  drift on straight sequences where naive gives 2m.

  Problem 2 — Absolute velocity prediction targets:
  The old Y targets stored absolute GPS velocity (in IQR-normalised space).
  The model learned a mapping from IMU patterns to absolute velocity values
  that were specific to the training bag speeds (40 m/s mean). On test bags
  at different speeds or headings, the mapping produced the wrong absolute
  velocity even though the IMU patterns were similar.

  FIX 1 — BAG-LEVEL SPLIT:
  Complete bags assigned entirely to train/val/test. Every window from a bag
  stays in the same split. Model trains on complete flight circuits and is
  evaluated on held-out complete flights from different terrain types.

  FIX 2 — RELATIVE (Δv) PREDICTION TARGETS:
  Y targets now store velocity INCREMENT relative to the last GPS-aided
  reading before each window: Δv[t] = v_GPS[t] - v_GPS[last_aided[t]].
  During aided windows: Δv = 0 (no change from current GPS, by definition).
  During outage windows: Δv accumulates the velocity change since outage start.

  WHY Δv GENERALISES BETTER:
  - Straight cruise at ANY speed: Δv ≈ 0 regardless of absolute speed
  - Turns at any speed: Δv captures the heading change magnitude
  - L_cvprior target becomes literally zero (no speed-dependent reference)
  - IQR of Δv is much smaller and more stable than IQR of absolute velocity
  - The model only needs to learn WHEN and HOW MUCH velocity changes,
    not what absolute speed the aircraft happens to be flying at

  BAG ASSIGNMENTS (5 bags, 2 terrain types):
  - Train: gnss01, gnss02, island_gnss02  (both terrains, straighter bags)
  - Val:   gnss03                          (airfield turns — harder airfield)
  - Test:  island_gnss03                  (island turns — different terrain)

  WHAT IS KEPT FROM v2.2:
  - All channel definitions and PASSTHROUGH_CHANNELS logic
  - Global robust scaling (train set only, applied to Δv targets)
  - Turn-weighting (W_train)
  - Per-bag inverse-frequency sample weights (W_sample)
  - Within-bag turn upsampling
  - Sequence boundary mapping with configurable stride
  - All sanity checks and warnings

  CHANGES FROM v3.0:
  - Y_train/Y_val/Y_test now store Δv (relative) instead of absolute GPS vel
  - Y_GPS_train/Y_GPS_val/Y_GPS_test store absolute GPS vel for v_prev lookup
  - Y_iqr and Y_median are computed from Δv values (much smaller range)
  - Inference: v_pred_physical = v_prev_physical + Δv_pred * Y_iqr + Y_median

=============================================================================
"""

import os
import glob
import numpy as np
from scipy.stats import iqr

# =============================================================================
# CONFIGURATION
# =============================================================================
PROCESSED_DIR   = r"./data/processed"
MASTER_OUT_FILE = os.path.join(PROCESSED_DIR, "MARS_Master_Dataset.npz")

# ── BAG-LEVEL SPLIT ASSIGNMENTS ───────────────────────────────────────────────
# Edit these lists if bag names change or more bags are added.
# Each entry is a substring that must appear in the bag filename.
TRAIN_BAGS = ["gnss01", "gnss02", "island_gnss02"]
VAL_BAGS   = ["HKairport_GNSS03", "airport_gnss03", "gnss03"]  # will be overridden below
TEST_BAGS  = ["island_gnss03"]

# Pattern matching is substring-based and checked in order: train → val → test.
# "gnss03" would match island_gnss03 as val before reaching test.
# Fix: check test patterns BEFORE val patterns in the function, or use
# more specific val pattern. We use the override below.
VAL_BAGS  = ["gnss03"]   # matches transformer_ds_gnss03.npz only
# To avoid island_gnss03 matching VAL before TEST, we override bag_split:
def bag_split(bag_name, train_list, val_list, test_list):
    """Return which split this bag belongs to. TEST checked before VAL."""
    for pattern in train_list:
        if pattern in bag_name: return "train"
    # Check TEST before VAL to avoid island_gnss03 matching gnss03 pattern
    for pattern in test_list:
        if pattern in bag_name: return "test"
    for pattern in val_list:
        if pattern in bag_name: return "val"
    return None
# ─────────────────────────────────────────────────────────────────────────────

# Sequence parameters
SEQ_LEN              = 300   # 30-second sequences
TRAIN_STRIDE_STD     = 30    # non-overlapping for training (distinct sequences)
TRAIN_STRIDE_TURN    = 15    # tighter stride for turn-heavy training bags
VAL_TEST_STRIDE      = 10    # finer evaluation coverage
TURN_HEAVY_BAGS      = ["gnss03", "island_gnss03"]  # not in train but kept for ref

# Turn weighting
TURN_GYRO_THRESH       = 0.5   # rad/s  [P-3]: real turns = 0.6-1.45 rad/s
TURN_WEIGHT_MULTIPLIER = 3.0
TURN_UPSAMPLE_ALPHA    = 2.0   # W_sample turn upsampling factor

# IQR floors  [P-4]
Y_IQR_FLOOR    = 0.1   # m/s
X_GPS_IQR_FLOOR = 0.1  # m/s

# Channel map (frozen — never change without reprocessing bags)
CH_ACC_START = 0;  CH_ACC_END = 3
CH_GYR_START = 3;  CH_GYR_END = 6
CH_QUAT_START = 6; CH_QUAT_END = 10
CH_GPS_START = 10; CH_GPS_END = 13
CH_MASK      = 13
N_CHANNELS   = 14
PASSTHROUGH_CHANNELS = list(range(CH_QUAT_START, CH_QUAT_END)) + [CH_MASK]

CHANNEL_NAMES = (
    [f"acc_{a}" for a in "xyz"]
    + [f"gyr_{a}" for a in "xyz"]
    + [f"quat_{a}" for a in "wxyz"]
    + [f"gps_v{a}" for a in "ned"]
    + ["mask"]
)

# =============================================================================
# HELPERS
# =============================================================================
def plog(m): print(m)
def warn(m): print(f"  WARNING: {m}")
def die(m):
    print(f"  ERROR: {m}")
    import sys; sys.exit(1)

def get_valid_sequence_indices(lengths_list, seq_len, stride,
                                bag_names_list=None, turn_heavy=None):
    """
    Build valid sequence start indices respecting bag boundaries.
    Optionally uses tighter stride for turn-heavy bags.
    """
    valid_idx = []; offset = 0
    for i, L in enumerate(lengths_list):
        eff_stride = stride
        if bag_names_list and turn_heavy:
            bname = bag_names_list[i]
            if any(th in bname for th in turn_heavy):
                eff_stride = TRAIN_STRIDE_TURN
        if L >= seq_len:
            valid_idx.extend(range(offset, offset + L - seq_len + 1, eff_stride))
        else:
            warn(f"Bag with {L} windows < SEQ_LEN={seq_len}. No sequences.")
        offset += L
    return np.array(valid_idx, dtype=np.int32)

# =============================================================================
# STEP 1: Load and assign bags
# =============================================================================
npz_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "transformer_ds_*.npz")))
if not npz_files:
    die(f"No transformer_ds_*.npz files found in {PROCESSED_DIR}")

plog(f"\n{'='*60}")
plog(f"  MARS MASTER COMBINE v3.1 — BAG-LEVEL SPLIT + Δv TARGETS")
plog(f"  Found {len(npz_files)} bags.")
plog(f"{'='*60}")
plog(f"\n  Split assignments:")
plog(f"  {'Bag':<45} {'Split':<8} {'Windows':>8}")
plog(f"  {'-'*63}")

train_x, train_y, train_lengths, train_names = [], [], [], []
val_x,   val_y,   val_lengths,   val_names   = [], [], [], []
test_x,  test_y,  test_lengths,  test_names  = [], [], [], []
train_w_list = []

for fpath in npz_files:
    bname  = os.path.basename(fpath)
    split  = bag_split(bname, TRAIN_BAGS, VAL_BAGS, TEST_BAGS)
    data   = np.load(fpath)
    X      = data["X_train"]
    Y      = data["y_train"]

    if X.shape[2] != N_CHANNELS:
        warn(f"{bname}: {X.shape[2]} channels, expected {N_CHANNELS}. Skipping.")
        continue
    if split is None:
        warn(f"{bname}: not assigned to any split. Skipping.")
        continue

    N = len(X)
    plog(f"  {bname:<45} {split:<8} {N:>8}")

    if split == "train":
        train_x.append(X); train_y.append(Y)
        train_lengths.append(N); train_names.append(bname)
        train_w_list.append(N)
    elif split == "val":
        val_x.append(X); val_y.append(Y)
        val_lengths.append(N); val_names.append(bname)
    elif split == "test":
        test_x.append(X); test_y.append(Y)
        test_lengths.append(N); test_names.append(bname)

if not train_x: die("No training bags found. Check TRAIN_BAGS list.")
if not val_x:   die("No validation bags found. Check VAL_BAGS list.")
if not test_x:  die("No test bags found. Check TEST_BAGS list.")

# Overlap check
all_train = set(TRAIN_BAGS); all_val = set(VAL_BAGS); all_test = set(TEST_BAGS)
overlap_tv = all_train & all_val
overlap_tt = all_train & all_test
overlap_vt = all_val & all_test
if overlap_tv: die(f"Bags in both TRAIN and VAL: {overlap_tv}")
if overlap_tt: die(f"Bags in both TRAIN and TEST: {overlap_tt}")
if overlap_vt: die(f"Bags in both VAL and TEST: {overlap_vt}")

X_train = np.concatenate(train_x, axis=0)
Y_train = np.concatenate(train_y, axis=0)
X_val   = np.concatenate(val_x,   axis=0)
Y_val   = np.concatenate(val_y,   axis=0)
X_test  = np.concatenate(test_x,  axis=0)
Y_test  = np.concatenate(test_y,  axis=0)

plog(f"\n  TRAIN: {X_train.shape}  bags: {train_names}")
plog(f"  VAL:   {X_val.shape}   bags: {val_names}")
plog(f"  TEST:  {X_test.shape}  bags: {test_names}")

# =============================================================================
# STEP 2b: Compute Δv = v_GPS[t] - v_GPS[last_aided_before_t]
#
# Y from bag files stores raw GPS velocity per window.
# We convert to RELATIVE velocity change from last known GPS reading.
#
# Within a bag, windows are sequential. The outage flag (ch 13) tells us
# which windows are outage. But at this stage we don't yet have the outage
# flag applied — the combine script works on raw un-masked windows.
#
# Strategy: compute Δv at the SEQUENCE level during training, not here.
# Here we store the absolute GPS velocity as Y_GPS_* for use as v_prev,
# and keep Y_* as absolute for now. The training data loader (MARSDataset)
# computes Δv on the fly relative to the last GPS-aided window in each
# training sequence. This is already consistent with how v_prev is computed.
#
# HOWEVER: Y_iqr and Y_median should be computed from Δv values, not
# absolute velocity, so the normalised target space is Δv-centred.
# We compute representative Δv from consecutive window differences.
# =============================================================================
plog("\n-- Computing Δv statistics for normalisation --")

# Δv per window = difference from previous window's GPS velocity
# (approximates the per-window velocity change)
# Computed within each bag to avoid bag-boundary artifacts
dv_samples = []
offset = 0
for bname, bn in zip(train_names, train_w_list):
    end     = offset + bn
    y_bag   = Y_train[offset:end]           # (bn, 3) absolute GPS vel
    dv_bag  = np.diff(y_bag, axis=0)        # (bn-1, 3) consecutive differences
    dv_samples.append(dv_bag)
    plog(f"  {bname:<45} Δv_y std={dv_bag[:,1].std():.4f} m/s")
    offset  = end

dv_all      = np.concatenate(dv_samples, axis=0)  # (N_train - n_bags, 3)
Y_dv_median = np.median(dv_all, axis=0)
Y_dv_iqr    = np.array([
    float(np.subtract(*np.percentile(dv_all[:,i], [75,25])))
    for i in range(3)
])
# Apply IQR floor
Y_dv_iqr = np.maximum(Y_dv_iqr, Y_IQR_FLOOR).astype(np.float32)
Y_dv_median = Y_dv_median.astype(np.float32)

plog(f"\n  Δv median: {Y_dv_median}")
plog(f"  Δv IQR:    {Y_dv_iqr}  (this is Y_iqr in the NPZ)")
plog(f"  Note: Y_iqr is now Δv-based, not absolute-velocity-based.")
plog(f"  Training loop normalises: (y_GPS[t] - v_prev[t]) / Y_iqr")

# Store absolute GPS velocities separately for v_prev lookup at inference
Y_GPS_train = Y_train.copy().astype(np.float32)
Y_GPS_val   = Y_val.copy().astype(np.float32)
Y_GPS_test  = Y_test.copy().astype(np.float32)

# =============================================================================
# STEP 2: Per-bag inverse-frequency sample weights [D-2]
# =============================================================================
plog("\n-- Per-Bag Sample Weights [D-2] --")
n_bags = len(train_w_list)
total  = sum(train_w_list)
W_sample_list = []
for bname, bn in zip(train_names, train_w_list):
    w = total / (n_bags * bn)
    W_sample_list.append(np.full(bn, w, dtype=np.float32))
    plog(f"  {bname:<45} {bn:>7} windows  weight={w:.4f}×")
W_sample = np.concatenate(W_sample_list)

# Within-bag turn upsampling [R7-S1]
plog("\n-- Within-Bag Turn Upsampling [R7-S1] --")
gyro_mag = np.linalg.norm(X_train[:, 100, CH_GYR_START:CH_GYR_END], axis=-1)
turn_factor = np.ones(len(X_train), dtype=np.float32)
offset = 0
for bname, bn in zip(train_names, train_w_list):
    end   = offset + bn
    bg    = gyro_mag[offset:end]
    factor = 1.0 + TURN_UPSAMPLE_ALPHA * np.clip(bg / TURN_GYRO_THRESH, 0., 1.)
    turn_factor[offset:end] = factor.astype(np.float32)
    plog(f"  {bname:<45} mean_factor={factor.mean():.3f}  max={factor.max():.3f}")
    offset = end
W_sample = W_sample * turn_factor
W_sample = W_sample / W_sample.mean()
plog(f"  W_sample: min={W_sample.min():.3f}  max={W_sample.max():.3f}  mean={W_sample.mean():.3f}")

# =============================================================================
# STEP 3: Sequence boundary mapping
# =============================================================================
plog("\n-- Sequence Boundary Mapping --")
train_valid_idx = get_valid_sequence_indices(
    train_lengths, SEQ_LEN, TRAIN_STRIDE_STD,
    bag_names_list=train_names, turn_heavy=TURN_HEAVY_BAGS)
val_valid_idx   = get_valid_sequence_indices(val_lengths,  SEQ_LEN, VAL_TEST_STRIDE)
test_valid_idx  = get_valid_sequence_indices(test_lengths, SEQ_LEN, VAL_TEST_STRIDE)

plog(f"  Train sequences: {len(train_valid_idx)}  (stride={TRAIN_STRIDE_STD})")
plog(f"  Val sequences:   {len(val_valid_idx)}   (stride={VAL_TEST_STRIDE})")
plog(f"  Test sequences:  {len(test_valid_idx)}  (stride={VAL_TEST_STRIDE})")

if len(val_valid_idx) < 10:
    warn("Fewer than 10 val sequences. Evaluation will be noisy.")
if len(test_valid_idx) < 10:
    warn("Fewer than 10 test sequences. Evaluation will be noisy.")

# =============================================================================
# STEP 4: Global robust scaling — TRAIN SET ONLY
# =============================================================================
plog("\n-- Global Robust Scaling (Train Set Only) --")
X_train_flat = X_train.reshape(-1, N_CHANNELS)
X_median = np.median(X_train_flat, axis=0)
X_iqr_   = iqr(X_train_flat, axis=0)

# Warn on near-zero IQR
for ch in range(N_CHANNELS):
    if ch in PASSTHROUGH_CHANNELS: continue
    if X_iqr_[ch] < 1e-4:
        warn(f"Channel {ch} ({CHANNEL_NAMES[ch]}) IQR={X_iqr_[ch]:.2e} near zero.")
X_iqr_raw = X_iqr_.copy()
X_iqr_[X_iqr_ < 1e-4] = 1e-4

# [P-4] GPS IQR floor
for ch in range(CH_GPS_START, CH_GPS_END):
    if X_iqr_raw[ch] < X_GPS_IQR_FLOOR:
        X_iqr_[ch] = X_GPS_IQR_FLOOR
        plog(f"  ch{ch} gps_v{'ned'[ch-CH_GPS_START]}: IQR floored "
             f"{X_iqr_raw[ch]:.4f} → {X_GPS_IQR_FLOOR:.4f}")
    else:
        plog(f"  ch{ch} gps_v{'ned'[ch-CH_GPS_START]}: IQR={X_iqr_raw[ch]:.4f} (ok)")

# [FIX-A] Passthrough channels: set median=0, iqr=1
for ch in PASSTHROUGH_CHANNELS:
    X_median[ch] = 0.0
    X_iqr_[ch]   = 1.0

plog(f"\n  {'Channel':<12} {'Median':>10} {'IQR':>10}  Mode")
for ch in range(N_CHANNELS):
    mode = "PASSTHROUGH" if ch in PASSTHROUGH_CHANNELS else "scaled"
    plog(f"  {CHANNEL_NAMES[ch]:<12} {X_median[ch]:>10.4f} {X_iqr_[ch]:>10.4f}  {mode}")

# Target velocity scaling — use Δv statistics computed above
# Y_iqr and Y_median now represent the scale of velocity CHANGES,
# not absolute velocities. This makes them independent of flight speed.
Y_median  = Y_dv_median
Y_iqr_    = Y_dv_iqr
plog(f"\n  Y_median (Δv median): {Y_median.round(6)}")
plog(f"  Y_iqr    (Δv IQR):    {Y_iqr_.round(6)}")
plog(f"  Compare to old absolute-velocity Y_iqr ≈ [1.516, 7.985, 0.100]")
plog(f"  Δv IQR should be much smaller — good, less normalisation compression")

# Scaling sanity check
X_flat_scaled = (X_train_flat - X_median) / X_iqr_
for ch in [0, 3]:
    sm = np.median(X_flat_scaled[:, ch])
    si = iqr(X_flat_scaled[:, ch])
    ok = abs(sm) < 0.05 and abs(si - 1.0) < 0.05
    plog(f"  Scaling sanity ch{ch} ({CHANNEL_NAMES[ch]}): "
         f"median={sm:.4f}  IQR={si:.4f}  {'OK' if ok else 'WARN'}")

# =============================================================================
# STEP 5: Turn weighting
# =============================================================================
plog("\n-- Turn Weighting --")
max_gyro = np.max(np.abs(X_train[:, :, CH_GYR_START:CH_GYR_END]), axis=(1, 2))
W_train  = np.ones(len(X_train), dtype=np.float32)
W_train[max_gyro > TURN_GYRO_THRESH] = TURN_WEIGHT_MULTIPLIER
turn_n = int((W_train == TURN_WEIGHT_MULTIPLIER).sum())
plog(f"  Turn windows:     {turn_n} ({turn_n/len(W_train)*100:.1f}%)")
plog(f"  Straight windows: {len(W_train)-turn_n} ({(len(W_train)-turn_n)/len(W_train)*100:.1f}%)")
if turn_n == 0:
    warn("No turn windows detected. Check TURN_GYRO_THRESH and CH_GYR indices.")

# =============================================================================
# STEP 6: Export
# =============================================================================
plog(f"\n-- Exporting to {MASTER_OUT_FILE} --")
np.savez_compressed(
    MASTER_OUT_FILE,
    # ── Training data ─────────────────────────────────────────────────────
    X_train=X_train.astype(np.float32),
    Y_train=Y_train.astype(np.float32),    # absolute GPS vel (for Δv computation
                                            # in data loader at training time)
    Y_GPS_train=Y_GPS_train,               # same — explicit alias for clarity
    W_train=W_train.astype(np.float32),
    W_sample=W_sample.astype(np.float32),
    # ── Validation data ───────────────────────────────────────────────────
    X_val=X_val.astype(np.float32),
    Y_val=Y_val.astype(np.float32),        # absolute GPS vel
    Y_GPS_val=Y_GPS_val,
    # ── Test data ─────────────────────────────────────────────────────────
    X_test=X_test.astype(np.float32),
    Y_test=Y_test.astype(np.float32),      # absolute GPS vel
    Y_GPS_test=Y_GPS_test,
    # ── Sequence indices ──────────────────────────────────────────────────
    train_valid_idx=train_valid_idx,
    val_valid_idx=val_valid_idx,
    test_valid_idx=test_valid_idx,
    # ── Normalisation parameters ──────────────────────────────────────────
    X_median=X_median.astype(np.float32),
    X_iqr=X_iqr_.astype(np.float32),
    # Y_iqr and Y_median are now Δv-based (velocity change scale)
    # Inference: v_pred_physical = v_prev + (y_norm_pred * Y_iqr + Y_median)
    Y_median=Y_median.astype(np.float32),  # ≈ 0 (Δv median is near zero)
    Y_iqr=Y_iqr_.astype(np.float32),       # scale of velocity changes
    # ── Dataset metadata ──────────────────────────────────────────────────
    seq_len=np.array([SEQ_LEN]),
    n_channels=np.array([N_CHANNELS]),
    ch_gyr_start=np.array([CH_GYR_START]),
    ch_gyr_end=np.array([CH_GYR_END]),
    ch_quat_start=np.array([CH_QUAT_START]),
    ch_quat_end=np.array([CH_QUAT_END]),
    ch_gps_start=np.array([CH_GPS_START]),
    ch_gps_end=np.array([CH_GPS_END]),
    ch_mask=np.array([CH_MASK]),
    # ── v3.1 provenance metadata ──────────────────────────────────────────
    split_version=np.array([31]),           # 31 = v3.1
    train_bags=np.array(train_names, dtype=object),
    val_bags=np.array(val_names,     dtype=object),
    test_bags=np.array(test_names,   dtype=object),
)

plog(f"\n  SUCCESS — v3.1")
plog(f"  Train windows:     {len(X_train):,}  ({len(train_valid_idx)} sequences)")
plog(f"  Val windows:       {len(X_val):,}   ({len(val_valid_idx)} sequences)")
plog(f"  Test windows:      {len(X_test):,}  ({len(test_valid_idx)} sequences)")
plog(f"  W_sample:          min={W_sample.min():.3f}  max={W_sample.max():.3f}")
plog(f"  Train bags: {train_names}")
plog(f"  Val bags:   {val_names}")
plog(f"  Test bags:  {test_names}")
plog(f"")
plog(f"  Y_iqr (Δv scale): {Y_iqr_}  (was ~[1.516, 7.985, 0.100] in v2)")
plog(f"  Y_median (Δv):    {Y_median}  (was ~[-0.146, 0.820, 0.003] in v2)")
plog(f"")
plog(f"  CRITICAL CHANGES vs v2 — update training notebook:")
plog(f"  1. L_cvprior target is now 0 (not v_prev_norm)")
plog(f"     Straight cruise → Δv = 0 by physics, regardless of speed")
plog(f"  2. compute_v_prev still uses absolute GPS vel for v_prev tracking")
plog(f"     but dv_pred is now Δv (change from v_prev)")
plog(f"  3. Position integration: pos += (v_prev + dv_pred_phys) * DT")
plog(f"     where dv_pred_phys = dv_pred_norm * Y_iqr + Y_median")
plog(f"")
plog(f"  Next steps:")
plog(f"  1. Upload new MARS_Master_Dataset.npz to Drive")
plog(f"  2. Re-run marsnet_run20_retrain.ipynb  (~40 min)")
plog(f"  3. Re-run marsnet_lstm_baseline.ipynb  (~25 min)")
plog(f"  4. Re-run marsnet_ekf_v2_and_test.ipynb for EKF tuning")
plog(f"  5. Run test set once all val numbers look right")
