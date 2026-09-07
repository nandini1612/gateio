"""
=============================================================================
  MARS-LVIG MASTER COMBINING SCRIPT  v2.2
  - 80/10/10 Stratified Chronological Split with Dead Zones
  - Global Robust Scaling (Median & IQR, Train Set Only)
  - Turn-Weighting & Sequence Boundary Mapping

  CHANGES FROM v2.1 → v2.2:

  [R7-S1] WITHIN-BAG TURN UPSAMPLING added to W_sample
          Root cause: D-2 inverse-frequency weighting equalises bag contribution
          but leaves within-bag distribution unchanged. gnss01's 3309 effective
          windows per epoch are ~95% straight-flight (only ~165 turn windows).
          gnss03's 3309 effective windows are ~40% turns (1324 turn windows).
          Adding turn_factor = 1 + TURN_UPSAMPLE_ALPHA * clip(gyro_mag/p90, 0, 1)
          to W_sample increases how often turn-heavy windows are sampled without
          breaking the bag-level balance from D-2. Turn windows in gnss03 and
          island_gnss03 get ~3× the sampling weight of their own straight windows.

  [R7-S2] PER-BAG TURN STRIDE for gnss03 and island_gnss03
          These two bags have the highest yaw activity (gyr_z_std 0.169-0.215
          rad/s). Using stride=15 instead of 30 for these bags approximately
          doubles the number of turn-containing sequences available during
          training. gnss01 keeps stride=30 (its windows are near-duplicate anyway).
          Bag-specific stride is applied in get_valid_sequence_indices.

  [R7-S3] W_sample now saved per-train-window (combined bag + turn weight)
          and passed to WeightedRandomSampler in the model notebook.

  CHANGES FROM v2.0 → v2.1:

  [P-3] TURN_GYRO_THRESH raised 0.3 → 0.5 rad/s
        Root cause of over-classification: raw Livox gyro noise floor
        reaches 2σ ≈ 0.38–0.41 rad/s on the noisiest bags (gnss03,
        island_gnss03). With DESIGN-FIX-1 in preprocess_vg.py v5.3
        the stored gyro is now LP-filtered, so 2σ max is 0.415 rad/s.
        Real banked turns produce 0.6–1.45 rad/s — well above 0.5.
        Threshold 0.5 keeps a margin of 0.085 rad/s on the noisiest bag
        while eliminating false turn classifications during straight flight.

  [P-4] IQR FLOORS — prevents gradient imbalance across velocity axes
        Y_IQR_FLOOR = 0.1 m/s applied to all target velocity channels.
        X_GPS_IQR_FLOOR = 0.1 m/s applied to GPS input channels 10-12.
        Root cause: vD IQR is 0.016–0.032 m/s (MAV rarely changes altitude)
        vs vE IQR up to 7.87 m/s. After scaling, a 1 m/s vD error becomes
        30–60 normalised units vs 0.13 normalised units for vE — a 400× loss
        gradient imbalance that makes vD dominate training despite being
        the easiest axis to predict. The 0.1 m/s floor caps this ratio at 80×.

  [D-1] STRIDE PARAMETER IN SEQUENCE INDEX BUILDER
        Train sequences: stride=30 (non-overlapping 3s steps).
        Val/Test sequences: stride=10 (finer coverage for evaluation).
        Root cause: stride=1 (old default) produced 8,505 training sequences
        where adjacent sequences share 299/300 windows — 99.7% overlap.
        These are not independent training examples; they are near-duplicate
        trajectories that waste GPU compute and inflate the apparent dataset
        size. Stride-30 gives ~393 genuinely distinct base sequences.

  [D-2] PER-BAG INVERSE-FREQUENCY SAMPLE WEIGHTS
        Each bag gets W_sample = total_train_windows / (n_bags × bag_windows).
        gnss01 has 39% of windows but only 0–3.55 m/s coverage (slow only).
        Without rebalancing it dominates every epoch. The per-bag weight
        down-weights gnss01 (≈0.45×) and up-weights island_gnss03 (≈2.1×),
        ensuring the fast/aggressive bags get proportional training exposure.
        Used with WeightedRandomSampler in the training DataLoader.

  CHANGES FROM v1.0 → v2.0:

  [FIX-A] Quaternion channels (6-9) are now EXCLUDED from robust scaling.
          WHY: Robust scaling destroys quaternion unit-norm geometry.
          After (q - median) / IQR, ||q|| != 1 and the four components no
          longer represent a valid rotation on S³. The physics loss cannot
          build a valid rotation matrix from scaled quaternions.
          Quaternions are already bounded in [-1, +1] and have a well-defined
          zero-mean property (over a diverse flight, mean ≈ 0). We pass them
          through unchanged by setting median=0 and iqr=1 for those channels,
          identical to how the mask channel is already handled.

  [FIX-B] n_train negative-value guard.
          WHY: For short bags, the formula
               n_train = N - n_val - n_test - 2*DEAD_ZONE_WINDOWS
          can silently go negative if n_val and n_test are clamped to
          min_required_len while N is small. Python slicing X[:negative]
          silently returns almost the entire array from the wrong end,
          giving the test split the bag's training data with no error.
          We now explicitly check n_train > SEQ_LEN and skip the bag
          with a clear warning if it's too short to be usable.

  [FIX-C] Channel index constants used throughout.
          WHY: Hardcoded magic numbers like [:, :, 3:6] for gyro channels
          will silently compute turn weights from the wrong channels if
          the column ordering changes. All indices now reference the
          frozen CHANNEL_MAP defined at the top of this file.

  [FIX-D] Cross-bag EDA consistency check.
          WHY: Before scaling, we verify that all bags' per-channel means
          are within a reasonable range of each other. A large outlier in
          one bag's mean (e.g. >3 sigma from the cross-bag mean of means)
          indicates a unit conversion failure or axis ordering mismatch in
          that specific bag's preprocessing run. This catches problems early
          rather than letting a corrupt bag silently pollute the training data.

  [FIX-E] Robust scaler IQR floor raised to 1e-4 (was 1e-8).
          WHY: An IQR of 1e-8 means a feature has essentially zero variance
          across the training set. This produces scaled values in the range
          of 1e8 rather than zero-centering the feature. A channel with
          near-zero IQR should be investigated, not silently divided by a
          tiny number. We now warn explicitly and apply a sensible floor.
=============================================================================
"""

import os
import glob
import numpy as np
from scipy.stats import iqr

# =============================================================================
# CONFIGURATION
# =============================================================================
PROCESSED_DIR = r"./data/processed"
MASTER_OUT_FILE = os.path.join(PROCESSED_DIR, "MARS_Master_Dataset.npz")

TRAIN_PCT = 0.80
VAL_PCT = 0.10
# [DESIGN-FIX-1-COMBINE] Define TEST_PCT explicitly to avoid float64 erosion.
# 1.0 - 0.80 - 0.10 = 0.09999999999999998 in IEEE 754 float64, which causes
# int(N * 0.0999...) = N*0.1 - 1 on bags where N*0.1 is a near-integer.
# The result is a test split one window shorter than intended — silently wrong.
TEST_PCT = round(1.0 - TRAIN_PCT - VAL_PCT, 10)  # exactly 0.1

# Sequence & Safety Parameters
DEAD_ZONE_WINDOWS = 10  # 1.0 seconds at 0.1s stride — prevents data leakage
SEQ_LEN = 300  # 30-second sequences (300 windows × 0.1s stride)
MIN_WINDOWS_PER_SPLIT = SEQ_LEN + 50  # Minimum windows needed per split

TURN_GYRO_THRESH = 0.5  # [P-3] raised from 0.3 rad/s. Livox filtered 2σ max
# = 0.415 rad/s; real turns produce 0.6–1.45 rad/s.
# Margin = 0.085 rad/s on the noisiest bag (island_gnss03).
TURN_WEIGHT_MULTIPLIER = 3.0  # Penalty multiplier for turn windows in loss function

# [R7-S1] Within-bag turn upsampling for W_sample
# turn_factor = 1 + TURN_UPSAMPLE_ALPHA * clip(gyro_mag / bag_p90_gyro, 0, 1)
# Straight windows (gyro≈0):   factor ≈ 1.0×  (unchanged)
# Strong turn (gyro = p90):     factor ≈ 3.0×  (sampled 3× more often)
TURN_UPSAMPLE_ALPHA = 2.0

# [R7-S2] Per-bag stride for sequence index builder
# Turn-heavy bags use stride=15 (doubles their sequence count vs stride=30)
# Determined by bag name substring matching — update if bag names change
TURN_HEAVY_BAGS = ["gnss03", "island_gnss03"]  # substrings to match
TRAIN_STRIDE_TURN = 15  # stride for turn-heavy bags
TRAIN_STRIDE_STD = 30  # stride for straight/slow bags (unchanged)

# [P-4] IQR floors — prevent gradient imbalance on low-variance axes
Y_IQR_FLOOR = 0.1  # m/s — applied to all 3 target velocity channels.
# vD IQR is only 0.016–0.032 m/s; without this floor
# a 1 m/s vD error becomes ~40 normalised units vs
# 0.13 for vE — a 300× gradient imbalance.
X_GPS_IQR_FLOOR = 0.1  # m/s — applied to GPS input channels 10-12 (vn,ve,vd)
# for the same reason as Y_IQR_FLOOR.

# =============================================================================
# [FIX-C] FROZEN CHANNEL MAP
# ─────────────────────────────────────────────────────────────────────────────
# These indices MUST match the feature_cols ordering in preprocess_vg.py.
# Never use magic numbers like [:,:,3:6] anywhere below — always use these.
# Changing these requires re-running all preprocessing bags from scratch.
# =============================================================================
CH_ACC_START = 0  # channels 0,1,2: raw specific force x,y,z (gravity included) [P-1]
CH_ACC_END = 3
CH_GYR_START = 3  # channels 3,4,5: bias-corrected angular velocity x,y,z
CH_GYR_END = 6
CH_QUAT_START = 6  # channels 6,7,8,9: VQF quaternion w,x,y,z
CH_QUAT_END = 10
CH_GPS_START = 10  # channels 10,11,12: GPS velocity vn,ve,vd
CH_GPS_END = 13
CH_MASK = 13  # channel 13: outage mask flag (0=active, 1=denied)
N_CHANNELS = 14

# Channels that should NOT be robustly scaled:
# - Quaternions: geometric meaning requires unit norm preservation
# - Mask: binary flag, must stay 0/1
PASSTHROUGH_CHANNELS = list(range(CH_QUAT_START, CH_QUAT_END)) + [CH_MASK]


# =============================================================================
# HELPERS
# =============================================================================
def plog(m):
    print(m)


def warn(m):
    print(f"  WARNING: {m}")


def die(m):
    print(f"  ERROR: {m}")
    import sys

    sys.exit(1)


# =============================================================================
# STEP 1: Load and Validate All Preprocessed Bags
# =============================================================================
npz_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "transformer_ds_*.npz")))

if not npz_files:
    die(
        f"No transformer_ds_*.npz files found in {PROCESSED_DIR}. "
        f"Run preprocess_vg.py for each bag first."
    )

plog(f"\n======================================================")
plog(f"  MARS MASTER COMBINE v2.0")
plog(f"  Found {len(npz_files)} preprocessed bags.")
plog(f"======================================================")

# [FIX-D] Cross-bag consistency check: load all bag means and compare
all_bag_means = []
all_bag_names = []

for file in npz_files:
    data = np.load(file)
    if "X_mean" in data:
        all_bag_means.append(data["X_mean"])
        all_bag_names.append(os.path.basename(file))

if len(all_bag_means) >= 2:
    plog("\n-- Cross-Bag EDA Consistency Check --")
    means_array = np.stack(all_bag_means)  # (n_bags, 14)
    cross_bag_mean = means_array.mean(axis=0)
    cross_bag_std = means_array.std(axis=0)
    channel_names = (
        [f"acc_{a}" for a in "xyz"]
        + [f"gyr_{a}" for a in "xyz"]
        + [f"quat_{a}" for a in "wxyz"]
        + [f"gps_v{a}" for a in "ned"]
        + ["mask"]
    )

    plog(
        f"  {'Channel':<18} {'Cross-Bag Mean':>16} {'Cross-Bag Std':>16}  Outlier Bags"
    )
    plog(f"  {'-' * 70}")
    for ch in range(N_CHANNELS):
        outliers = []
        for bag_i, bag_name in enumerate(all_bag_names):
            # Flag if this bag's mean is more than 3 std from cross-bag mean
            if cross_bag_std[ch] > 1e-6:
                z = abs(means_array[bag_i, ch] - cross_bag_mean[ch]) / cross_bag_std[ch]
                if z > 3.0:
                    outliers.append(f"{bag_name}(z={z:.1f})")
        outlier_str = ", ".join(outliers) if outliers else "—"
        plog(
            f"  {channel_names[ch]:<18} {cross_bag_mean[ch]:>16.4f} "
            f"{cross_bag_std[ch]:>16.4f}  {outlier_str}"
        )

    # Warn specifically for accelerometer channels — these are most likely to
    # have unit or gravity-subtraction issues across bags
    for ch in range(CH_ACC_START, CH_ACC_END):
        if cross_bag_std[ch] > 2.0:
            warn(
                f"High cross-bag std in {channel_names[ch]} "
                f"(std={cross_bag_std[ch]:.3f} m/s²). "
                f"Possible unit conversion inconsistency or gravity subtraction "
                f"failure in one or more bags."
            )

    # [DESIGN-FIX-1] Validate filtered gyro noise vs turn threshold.
    # If 2σ spikes in any bag's filtered gyro exceed 50% of TURN_GYRO_THRESH,
    # noise artefacts can still produce false turn classifications. Bags
    # processed with preprocess_vg.py < v5.3 will show "no data" here and
    # must be re-processed.
    plog(f"\n  Gyro Noise vs Turn Threshold ({TURN_GYRO_THRESH} rad/s):")
    plog(f"  {'Bag':<42} {'raw_max':>9} {'filt_max':>10} {'2σ_spike':>10}  Status")
    plog(f"  {'-' * 80}")
    # [P-3 FIX] Warn only when 2σ noise floor exceeds the actual threshold.
    # At threshold=0.5 rad/s, the old 0.5× factor fired false WARNs on 4/5 bags
    # (2σ max = 0.415 rad/s, which is below 0.5 but above 0.5×0.5=0.25).
    safe_limit = TURN_GYRO_THRESH
    all_noise_ok = True
    for file in npz_files:
        bname = os.path.basename(file)
        bdata = np.load(file)
        if "gyr_raw_std" in bdata and "gyr_filtered_std" in bdata:
            raw_s = float(bdata["gyr_raw_std"].max())
            filt_s = float(bdata["gyr_filtered_std"].max())
            spike = 2.0 * filt_s
            status = "OK ✓" if spike < safe_limit else "WARN ✗"
            if spike >= safe_limit:
                all_noise_ok = False
                warn(
                    f"{bname}: 2σ spike={spike:.4f} rad/s >= safe limit "
                    f"{safe_limit:.3f} rad/s. Turn windows may be inflated. "
                    f"Lower GYR_CUTOFF_HZ for this bag."
                )
            plog(
                f"  {bname:<42} {raw_s:>9.4f} {filt_s:>10.4f} {spike:>10.4f}  [{status}]"
            )
        else:
            plog(
                f"  {bname:<42} {'n/a':>9} {'n/a':>10} {'n/a':>10}  "
                f"[re-run preprocess v5.3 — gyr_filtered_std missing]"
            )
            all_noise_ok = False
    if all_noise_ok:
        plog(f"  All bags: 2σ filtered gyro spike < {safe_limit:.3f} rad/s. Clean.")

else:
    warn(
        "Fewer than 2 bags have EDA statistics saved. "
        "Cross-bag consistency check skipped. "
        "Re-run preprocess_vg.py v5.0 to generate per-bag X_mean/X_std."
    )

# =============================================================================
# STEP 2: Chronological 80/10/10 Split with Dead Zones
# =============================================================================
plog("\n-- Splitting Bags (80/10/10 with Dead Zones) --")

train_x_list, train_y_list, train_lengths = [], [], []
val_x_list, val_y_list, val_lengths = [], [], []
test_x_list, test_y_list, test_lengths = [], [], []
train_w_sample_list = []  # [D-2] per-window inverse-frequency weights
train_bag_names = []  # [D-2] for reporting

for file in npz_files:
    bag_name = os.path.basename(file)
    plog(f"\n  Processing: {bag_name}")

    data = np.load(file)
    X = data["X_train"]  # Shape: (N, 200, 14)
    Y = data["y_train"]  # Shape: (N, 3)
    N = len(X)

    # Verify channel count matches our frozen CHANNEL_MAP
    if X.shape[2] != N_CHANNELS:
        warn(
            f"  {bag_name} has {X.shape[2]} channels, expected {N_CHANNELS}. "
            f"Skipping — re-run preprocess_vg.py v5.0 for this bag."
        )
        continue

    # Calculate split sizes
    n_val_target = int(N * VAL_PCT)
    n_test_target = int(N * TEST_PCT)

    # Clamp to minimum required length
    n_val = max(n_val_target, MIN_WINDOWS_PER_SPLIT)
    n_test = max(n_test_target, MIN_WINDOWS_PER_SPLIT)

    # [FIX-B] Explicitly check n_train before slicing
    n_train = N - n_val - n_test - (DEAD_ZONE_WINDOWS * 2)

    if n_train < SEQ_LEN:
        # This bag is too short to produce at least one valid training sequence
        warn(
            f"  {bag_name}: n_train={n_train} < SEQ_LEN={SEQ_LEN}. "
            f"(N={N}, n_val={n_val}, n_test={n_test}). "
            f"This bag is too short for the current split configuration. "
            f"Options: (1) reduce SEQ_LEN, (2) reduce MIN_WINDOWS_PER_SPLIT, "
            f"(3) extend the cruise window in preprocess_vg.py for this bag. "
            f"SKIPPING this bag entirely."
        )
        continue

    if n_train < 0:
        # Redundant safety check after the above, but explicit is better
        die(
            f"{bag_name}: n_train is NEGATIVE ({n_train}). "
            f"N={N} is too small for the requested split configuration."
        )

    # Define split boundaries
    train_end = n_train
    val_start = train_end + DEAD_ZONE_WINDOWS
    val_end = val_start + n_val
    test_start = val_end + DEAD_ZONE_WINDOWS

    if test_start >= N:
        warn(
            f"  {bag_name}: test_start ({test_start}) >= N ({N}). "
            f"Dead zones eat into test split. Skipping."
        )
        continue

    # Extract splits
    X_tr = X[:train_end]
    Y_tr = Y[:train_end]
    X_va = X[val_start:val_end]
    Y_va = Y[val_start:val_end]
    X_te = X[test_start:]
    Y_te = Y[test_start:]

    # Warn if test split is very small
    if len(X_te) < MIN_WINDOWS_PER_SPLIT:
        warn(
            f"  {bag_name}: test split has only {len(X_te)} windows "
            f"(minimum recommended: {MIN_WINDOWS_PER_SPLIT})."
        )

    train_x_list.append(X_tr)
    train_y_list.append(Y_tr)
    train_lengths.append(len(X_tr))
    train_w_sample_list.append(
        len(X_tr)
    )  # [D-2] raw count; weights computed after all bags loaded
    train_bag_names.append(bag_name)
    val_x_list.append(X_va)
    val_y_list.append(Y_va)
    val_lengths.append(len(X_va))
    test_x_list.append(X_te)
    test_y_list.append(Y_te)
    test_lengths.append(len(X_te))

    plog(
        f"    Train: {len(X_tr):>5} | Val: {len(X_va):>5} | "
        f"Test: {len(X_te):>5} | Dead zones dropped: {DEAD_ZONE_WINDOWS * 2}"
    )
    # Report what % of the bag ended up in training
    pct_train = len(X_tr) / N * 100
    if pct_train < 50:
        warn(
            f"    Only {pct_train:.0f}% of {bag_name} is in train split "
            f"(val/test minimums consumed most of the data)."
        )

if not train_x_list:
    die(
        "No bags survived the split. All were too short or had errors. "
        "Reduce SEQ_LEN or MIN_WINDOWS_PER_SPLIT, or add more data."
    )

# Concatenate into master arrays
X_train_full = np.concatenate(train_x_list, axis=0)
Y_train_full = np.concatenate(train_y_list, axis=0)
X_val_full = np.concatenate(val_x_list, axis=0)
Y_val_full = np.concatenate(val_y_list, axis=0)
X_test_full = np.concatenate(test_x_list, axis=0)
Y_test_full = np.concatenate(test_y_list, axis=0)

plog(f"\n  MASTER TRAIN : X={X_train_full.shape} | Y={Y_train_full.shape}")
plog(f"  MASTER VAL   : X={X_val_full.shape}   | Y={Y_val_full.shape}")
plog(f"  MASTER TEST  : X={X_test_full.shape}  | Y={Y_test_full.shape}")

# =============================================================================
# [D-2] Per-Bag Inverse-Frequency Sample Weights
# =============================================================================
plog("\n-- Per-Bag Inverse-Frequency Sample Weights  [D-2] --")
plog("  WHY: gnss01 is 39% of windows but covers only 0-3.55 m/s (slow only).")
plog("  Without rebalancing, the slow-speed regime dominates every training epoch.")
plog("  Inverse-frequency weight for bag b = total_train / (n_bags × bag_windows).")
plog("  This makes each bag contribute equally in expectation per epoch,")
plog("  regardless of how many windows it produced.")

n_bags_train = len(train_w_sample_list)
total_train_windows = sum(train_w_sample_list)

# Build per-window weight array
W_sample_list = []
plog(f"\n  {'Bag':<42} {'windows':>8} {'share%':>8} {'weight':>8}")
plog(f"  {'-' * 70}")
for bag_nm, bag_n in zip(train_bag_names, train_w_sample_list):
    w = total_train_windows / (n_bags_train * bag_n)  # inverse frequency
    W_sample_list.append(np.full(bag_n, w, dtype=np.float32))
    share_pct = bag_n / total_train_windows * 100
    plog(f"  {bag_nm:<42} {bag_n:>8d} {share_pct:>7.1f}% {w:>8.4f}×")

W_sample = np.concatenate(W_sample_list)  # shape: (total_train_windows,)
plog(
    f"\n  W_sample after bag-level D-2: min={W_sample.min():.4f}  max={W_sample.max():.4f}  "
    f"mean={W_sample.mean():.4f}  shape={W_sample.shape}"
)

# =============================================================================
# [R7-S1] Within-Bag Turn Upsampling on W_sample
# =============================================================================
plog("\n-- Within-Bag Turn Upsampling for W_sample  [R7-S1] --")
plog("  WHY: D-2 gives equal exposure to each bag but within gnss01 the vast")
plog("  majority of windows are slow straight-flight. Adding a turn_factor")
plog("  = 1 + alpha * clip(window_gyro_mag / bag_p90, 0, 1) to W_sample")
plog("  upsamples high-gyro windows within each bag without changing the")
plog("  aggregate bag balance established by D-2.")
plog(f"  TURN_UPSAMPLE_ALPHA = {TURN_UPSAMPLE_ALPHA}")

# Compute per-window mean gyro magnitude (use centre sample, index 100)
# X_train_full shape: (N, 200, 14) — gyr channels CH_GYR_START:CH_GYR_END
gyro_centre = X_train_full[:, 100, CH_GYR_START:CH_GYR_END]  # (N, 3)
gyro_mag = np.sqrt((gyro_centre**2).sum(axis=1))  # (N,)

# Apply per-bag normalisation to avoid cross-bag threshold dependency
turn_factor = np.ones(len(X_train_full), dtype=np.float32)
offset = 0
plog(f"\n  {'Bag':<42} {'p90 gyro':>10} {'mean factor':>12} {'max factor':>12}")
plog(f"  {'-' * 78}")
for bag_nm, bag_n in zip(train_bag_names, train_w_sample_list):
    end = offset + bag_n
    bag_gyro = gyro_mag[offset:end]
    p90 = np.percentile(bag_gyro, 90)
    if p90 < 1e-6:
        p90 = 1e-6  # safety — avoid division by zero for perfectly still bags
    factor = 1.0 + TURN_UPSAMPLE_ALPHA * np.clip(bag_gyro / TURN_GYRO_THRESH, 0.0, 1.0)
    turn_factor[offset:end] = factor.astype(np.float32)
    plog(f"  {bag_nm:<42} {p90:>10.4f} {factor.mean():>12.4f} {factor.max():>12.4f}")
    offset = end

W_sample = W_sample * turn_factor

# Re-normalise so mean weight stays ≈1.0 (WeightedRandomSampler is scale-invariant
# but keeping mean≈1 makes the effective sample count easy to reason about)
W_sample = W_sample / W_sample.mean()
plog(
    f"\n  W_sample after turn upsampling: min={W_sample.min():.4f}  "
    f"max={W_sample.max():.4f}  mean={W_sample.mean():.4f}"
)

# =============================================================================
# STEP 3: Sequence Boundary Mapping
# =============================================================================
plog("\n-- Sequence Boundary Mapping --")


def get_valid_sequence_indices(lengths_list, seq_len, stride=1, bag_names_list=None):
    """
    Build an index array of all valid sequence starting positions.

    stride controls how densely sequences are sampled:
      stride=1  → every starting position (maximal overlap, near-duplicate sequences)
      stride=30 → one start per 3s step (train: genuinely distinct sequences)
      stride=10 → one start per 1s step  (val/test: finer evaluation coverage)

    [D-1] Default stride changed from 1 to caller-specified value.
    [R7-S2] bag_names_list: if provided, bags matching TURN_HEAVY_BAGS use
            TRAIN_STRIDE_TURN (15) instead of stride. This doubles sequence
            count for gnss03 and island_gnss03 without changing val/test strides.

    This function still prevents cross-bag sequence contamination: a valid
    start index i is only emitted if i + seq_len stays within one bag's
    contiguous block.
    """
    valid_idx = []
    current_offset = 0
    for bag_i, L in enumerate(lengths_list):
        # [R7-S2] Reduce stride for turn-heavy bags to get more sequences from them
        if bag_names_list is not None and bag_i < len(bag_names_list):
            bag_nm = bag_names_list[bag_i]
            is_turn_heavy = any(th in bag_nm for th in TURN_HEAVY_BAGS)
            effective_stride = TRAIN_STRIDE_TURN if is_turn_heavy else stride
        else:
            effective_stride = stride

        if L >= seq_len:
            valid_idx.extend(
                range(
                    current_offset, current_offset + L - seq_len + 1, effective_stride
                )
            )
        else:
            warn(
                f"Bag with {L} windows is shorter than SEQ_LEN={seq_len}. "
                f"No valid start indices from this bag."
            )
        current_offset += L
    return np.array(valid_idx, dtype=np.int32)


train_valid_idx = get_valid_sequence_indices(  # [D-1] [R7-S2]
    train_lengths,
    SEQ_LEN,
    stride=TRAIN_STRIDE_STD,
    bag_names_list=train_bag_names,  # turn-heavy bags use stride=15
)
val_valid_idx = get_valid_sequence_indices(val_lengths, SEQ_LEN, stride=10)  # [D-1]
test_valid_idx = get_valid_sequence_indices(test_lengths, SEQ_LEN, stride=10)  # [D-1]

plog(f"  Valid {SEQ_LEN}-window start indices:")
plog(
    f"    Train: {len(train_valid_idx)}  "
    f"(turn-heavy bags used stride={TRAIN_STRIDE_TURN}, others stride={TRAIN_STRIDE_STD})"
)
plog(f"    Val  : {len(val_valid_idx)}")
plog(f"    Test : {len(test_valid_idx)}")

if len(val_valid_idx) < 10:
    warn(
        "Fewer than 10 valid validation sequences. Evaluation will be noisy. "
        "Consider reducing SEQ_LEN or adding more bags."
    )

# =============================================================================
# STEP 4: Global Robust Scaling (Median & IQR, Train Set Only)
# =============================================================================
plog("\n-- Global Robust Scaling (Train Set Only) --")
plog("  WHY TRAIN ONLY: Computing scaling stats from val or test data would")
plog("  constitute data leakage — the network would indirectly see future")
plog("  flight statistics during training.")

# Flatten (N, 200, 14) -> (N*200, 14)
X_train_flat = X_train_full.reshape(-1, N_CHANNELS)

X_median = np.median(X_train_flat, axis=0)
X_iqr_ = iqr(X_train_flat, axis=0)

# [FIX-E] Warn on near-zero IQR channels before applying floor
for ch in range(N_CHANNELS):
    if ch in PASSTHROUGH_CHANNELS:
        continue
    if X_iqr_[ch] < 1e-4:
        warn(
            f"Channel {ch} has IQR={X_iqr_[ch]:.2e} (near zero). "
            f"This channel has almost no variance in the training set. "
            f"Possible causes: (1) constant sensor output, (2) wrong channel "
            f"index, (3) unit error. Investigate before training."
        )

# Apply floor to prevent division by near-zero
X_iqr_raw = X_iqr_.copy()
X_iqr_[X_iqr_ < 1e-4] = 1e-4

# [P-4] Apply X_GPS_IQR_FLOOR to GPS velocity input channels (vn, ve, vd).
# vD IQR is 0.016-0.032 m/s across bags. Without this floor the scaled vD
# channel has >300× the gradient magnitude of vE in the loss function.
plog(f"\n  [P-4] GPS input channel IQR floors (X_GPS_IQR_FLOOR={X_GPS_IQR_FLOOR} m/s):")
for ch in range(CH_GPS_START, CH_GPS_END):
    raw_iqr = X_iqr_raw[ch]
    if raw_iqr < X_GPS_IQR_FLOOR:
        X_iqr_[ch] = X_GPS_IQR_FLOOR
        plog(
            f"    ch{ch:02d} gps_v{'ned'[ch - CH_GPS_START]}: IQR floored {raw_iqr:.4f} → {X_GPS_IQR_FLOOR:.4f} m/s"
        )
    else:
        plog(
            f"    ch{ch:02d} gps_v{'ned'[ch - CH_GPS_START]}: IQR={raw_iqr:.4f} m/s (above floor, unchanged)"
        )

# [FIX-A] Protect quaternion channels AND mask channel from robust scaling.
# Quaternions: already bounded [-1,+1], unit-norm geometry must be preserved.
# Setting median=0 and iqr=1 passes them through unchanged.
for ch in PASSTHROUGH_CHANNELS:
    X_median[ch] = 0.0
    X_iqr_[ch] = 1.0

plog(f"\n  Channel Scaling Parameters (median | IQR):")
channel_names = (
    [f"acc_{a}" for a in "xyz"]
    + [f"gyr_{a}" for a in "xyz"]
    + [f"quat_{a}" for a in "wxyz"]
    + [f"gps_v{a}" for a in "ned"]
    + ["mask"]
)
for ch in range(N_CHANNELS):
    scaled = "PASSTHROUGH" if ch in PASSTHROUGH_CHANNELS else "scaled"
    plog(
        f"    ch{ch:02d} {channel_names[ch]:<10}: "
        f"median={X_median[ch]:>8.4f}  iqr={X_iqr_[ch]:>8.4f}  [{scaled}]"
    )

# Target velocity scaling
Y_median = np.median(Y_train_full, axis=0)
Y_iqr_ = iqr(Y_train_full, axis=0)

# [P-4] Apply Y_IQR_FLOOR to prevent vD gradient domination.
# vD IQR is 0.016-0.032 m/s. Y_IQR_FLOOR=0.1 caps the gradient ratio at ~80×.
Y_iqr_raw = Y_iqr_.copy()
Y_iqr_[Y_iqr_ < Y_IQR_FLOOR] = Y_IQR_FLOOR
plog(f"\n  [P-4] Target velocity IQR floors (Y_IQR_FLOOR={Y_IQR_FLOOR} m/s):")
for i, ax in enumerate(["v_N", "v_E", "v_D"]):
    raw, floored = Y_iqr_raw[i], Y_iqr_[i]
    if raw < Y_IQR_FLOOR:
        plog(f"    {ax}: IQR floored {raw:.4f} → {floored:.4f} m/s")
    else:
        plog(f"    {ax}: IQR={raw:.4f} m/s (above floor, unchanged)")

plog(f"\n  Target Velocity Medians [v_n, v_e, v_d]: {Y_median.round(4)}")
plog(f"  Target Velocity IQRs    [v_n, v_e, v_d]: {Y_iqr_.round(4)}")

# Sanity: after scaling, the training data should have approximately zero median
# and IQR of 1 for each scaled channel. Spot-check a few.
X_train_scaled_flat = (X_train_flat - X_median) / X_iqr_
for ch in [0, 3]:  # spot-check acc_x and gyr_x
    scaled_med = np.median(X_train_scaled_flat[:, ch])
    scaled_iqr = iqr(X_train_scaled_flat[:, ch])
    if abs(scaled_med) > 0.05 or abs(scaled_iqr - 1.0) > 0.05:
        warn(
            f"Scaling sanity FAILED for channel {ch}: "
            f"scaled_median={scaled_med:.4f} (expect ≈0), "
            f"scaled_IQR={scaled_iqr:.4f} (expect ≈1). "
            f"Investigate robust scaling computation."
        )
    else:
        plog(
            f"  Scaling sanity ch{ch} OK: median={scaled_med:.4f}, IQR={scaled_iqr:.4f}"
        )

# =============================================================================
# STEP 5: Turn-Weighting
# =============================================================================
plog("\n-- Turn-Weighting --")
plog("  Turn windows have elevated loss weight to force the model to learn")
plog("  the harder, non-straight-line dynamics more aggressively.")
plog(f"  Criterion: max absolute gyro magnitude > {TURN_GYRO_THRESH} rad/s")
plog(f"  Weight multiplier: {TURN_WEIGHT_MULTIPLIER}x for turn windows")

# [FIX-C] Use CH_GYR_START:CH_GYR_END constants, not magic numbers
# This operates on the pre-scaling raw data, which is in rad/s with bias removed.
# TURN_GYRO_THRESH = 0.3 rad/s is physically meaningful at this stage.
max_gyro_per_window = np.max(
    np.abs(X_train_full[:, :, CH_GYR_START:CH_GYR_END]), axis=(1, 2)
)

W_train = np.ones(len(X_train_full), dtype=np.float32)
W_train[max_gyro_per_window > TURN_GYRO_THRESH] = TURN_WEIGHT_MULTIPLIER

turn_count = int(np.sum(W_train == TURN_WEIGHT_MULTIPLIER))
straight_count = len(W_train) - turn_count
plog(f"  Turn windows    : {turn_count:>6} ({turn_count / len(W_train) * 100:.1f}%)")
plog(
    f"  Straight windows: {straight_count:>6} ({straight_count / len(W_train) * 100:.1f}%)"
)

# [DESIGN-FIX-1] Report per-axis max gyro contribution to turn classification.
# This lets you verify that the turn count is driven by real yaw/pitch/roll
# dynamics and not by noise on a single noisy axis.
plog(f"\n  Per-axis gyro contribution to turn windows:")
plog(f"  {'Axis':<8} {'windows > thresh':>18}  {'% of turns':>12}")
plog(f"  {'-' * 44}")
axis_names = ["gyr_x", "gyr_y", "gyr_z"]
for ax_i, ax_name in enumerate(axis_names):
    ax_max = np.max(np.abs(X_train_full[:, :, CH_GYR_START + ax_i]), axis=1)
    ax_count = int(np.sum(ax_max > TURN_GYRO_THRESH))
    pct_of_turns = ax_count / max(turn_count, 1) * 100
    plog(f"  {ax_name:<8} {ax_count:>18}  {pct_of_turns:>11.1f}%")
plog(f"  (A single dominant axis driving >90% of turns may indicate residual noise.)")

if turn_count == 0:
    warn(
        "No turn windows detected. Possible causes: (1) TURN_GYRO_THRESH is "
        "too high, (2) only straight-line flights in the dataset, (3) gyro "
        "channel indices are wrong (check CH_GYR_START/END)."
    )
if turn_count / len(W_train) > 0.6:
    warn(
        f"More than 60% of windows classified as turns. The turn-weighting "
        f"will dominate training. Consider raising TURN_GYRO_THRESH."
    )

# =============================================================================
# STEP 6: Export Master Dataset
# =============================================================================
plog(f"\n-- Exporting Master Dataset to {MASTER_OUT_FILE} --")

np.savez_compressed(
    MASTER_OUT_FILE,
    # Raw (unscaled) data — scaling is applied in the DataLoader
    X_train=X_train_full.astype(np.float32),
    Y_train=Y_train_full.astype(np.float32),
    W_train=W_train.astype(np.float32),
    W_sample=W_sample.astype(np.float32),  # [D-2] per-window bag balance weights
    X_val=X_val_full.astype(np.float32),
    Y_val=Y_val_full.astype(np.float32),
    X_test=X_test_full.astype(np.float32),
    Y_test=Y_test_full.astype(np.float32),
    # Sequence boundary indices for PyTorch DataLoader
    train_valid_idx=train_valid_idx,
    val_valid_idx=val_valid_idx,
    test_valid_idx=test_valid_idx,
    # Robust scaling statistics (computed on train only)
    X_median=X_median.astype(np.float32),
    X_iqr=X_iqr_.astype(np.float32),
    Y_median=Y_median.astype(np.float32),
    Y_iqr=Y_iqr_.astype(np.float32),
    # Architecture config frozen at dataset creation time
    seq_len=np.array([SEQ_LEN]),
    n_channels=np.array([N_CHANNELS]),
    ch_gyr_start=np.array([CH_GYR_START]),
    ch_gyr_end=np.array([CH_GYR_END]),
    ch_quat_start=np.array([CH_QUAT_START]),
    ch_quat_end=np.array([CH_QUAT_END]),
    ch_gps_start=np.array([CH_GPS_START]),
    ch_gps_end=np.array([CH_GPS_END]),
    ch_mask=np.array([CH_MASK]),
)

plog(f"\n  SUCCESS: Master dataset saved to {MASTER_OUT_FILE}")
plog(f"  Total training windows   : {len(X_train_full):,}")
plog(f"  Total validation windows : {len(X_val_full):,}")
plog(f"  Total test windows       : {len(X_test_full):,}")
plog(f"  Valid train sequences    : {len(train_valid_idx):,}  (stride=30)")
plog(f"  Valid val sequences      : {len(val_valid_idx):,}  (stride=10)")
plog(f"  Valid test sequences     : {len(test_valid_idx):,}  (stride=10)")
plog(
    f"  W_sample shape           : {W_sample.shape}  (min={W_sample.min():.3f}  max={W_sample.max():.3f})"
)
