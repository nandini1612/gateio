"""
=============================================================================
  MARS PYTORCH DATASET & DATALOADER  v2.0 (Data-Hardened)

  CHANGES FROM v1.0 → v2.0:

  [FIX-I]   Replace -100 outage sentinel with 0.
            WHY: After robust scaling, the normal feature range is ≈[-3, +3].
            Setting outage channels to -100 is 33 standard deviations out of
            distribution. Gradient signals are proportional to activation
            magnitude; this causes gradient explosions through residual
            connections and layer norms even when attention weights are near
            zero for those tokens. The correct approach is to zero the GNSS
            channels (neutral, in-distribution) and rely exclusively on the
            binary mask channel (index 13) to signal the outage. The mask
            is the hard, clean signal the network gates on — not the magnitude.

  [FIX-II]  Mask x_raw GNSS channels during the outage region.
            WHY: x_raw is kept "pristine for the physics loss". But the
            physics loss should use acc/gyro/quat from x_raw to compute
            the kinematic residual — it should NOT have access to the ground
            truth GPS velocity during the simulated outage, as that is the
            variable the network is supposed to predict. Previously, x_raw
            channels 10-12 contained true GPS velocity for the entire 300-window
            sequence, including the outage, creating a subtle data leakage path
            whenever x_raw was used in a loss computation.
            We now store a separate outage_mask tensor that the training loop
            can use to zero out GPS channels in x_raw when computing physics loss.

  [FIX-III] Deterministic, canonical outage patterns for val and test.
            WHY: Random outages on validation change the difficulty of the
            evaluation each epoch. Early stopping and model selection become
            unreliable because the validation loss oscillates due to outage
            variation, not model quality variation. Val and test now use three
            fixed, canonical outage patterns: a short (5s), medium (10s), and
            long (20s) outage at fixed positions within the 30s window.
            Each sequence in val/test always gets the same outage pattern,
            making model comparisons fair and reproducible.

  [FIX-IV]  Minimum post-outage anchor buffer enforced.
            WHY: The bidirectional integration used at inference time requires
            meaningful pre- AND post-outage GNSS data to anchor both ends
            of the velocity integration. Previously, an outage starting at
            window 90 with length 200 left only 10 windows (1 second) of
            post-outage GNSS data — nowhere near enough for backward integration.
            The minimum pre- and post-outage buffer is now enforced to be at
            least MIN_OUTAGE_BUFFER_WINDOWS on both sides.

  [FIX-V]   outage_prob is now ignored for val and test splits.
            Val and test always use canonical outages. This parameter now
            only controls whether training applies dynamic masking.

  Design fixes (v2.0 → v2.1):
  [DESIGN-FIX-2] apply_outage channel indices are now parameters, not constants.
            WHY: Hardcoded CH_GPS_START=10, CH_GPS_END=13, CH_MASK=13 inside
            apply_outage() duplicated the frozen channel map from combine_and_norm.py.
            If the channel order ever changes, the hardcoded values silently
            diverge while the npz-stored config is correct. Channel indices are
            now passed as keyword arguments (with safe defaults), and
            MARSDataset.__getitem__ always passes self.ch_gps_start / end / mask
            which are loaded from the npz file at construction time.

  [DESIGN-FIX-3] Per-pattern val/test breakdown + test split smoke test.
            WHY: The long canonical outage (200/300 windows = 66% of sequence)
            has substantially higher reconstruction difficulty than the short
            (50 windows) or medium (100 windows) patterns. Aggregating all three
            into a single val loss number obscures which difficulty tier the model
            is struggling on. The self-test now reports per-pattern buffer sizes,
            warns if any post-outage anchor buffer is < 3s (likely insufficient
            for backward integration), and adds a full smoke test for the test
            split (shapes, outage masking, canonical reproducibility) which was
            previously never exercised in the self-test.
=============================================================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# =============================================================================
# CANONICAL OUTAGE DEFINITIONS (for val and test — fixed and reproducible)
# =============================================================================
# Format: (outage_start_window, outage_length_windows)
# Applied to every sequence in val and test. Multiple patterns per sequence
# evaluate the model across a range of outage severities.
#
# For a 300-window (30s) sequence:
#   Short  outage: starts at window 50 (5s in), lasts 50 windows (5s)
#   Medium outage: starts at window 60 (6s in), lasts 100 windows (10s)
#   Long   outage: starts at window 50 (5s in), lasts 200 windows (20s)
#
# ⚠ These are applied ROUND-ROBIN across sequences in val/test.
# If you have 300 val sequences, 100 get short, 100 get medium, 100 get long.
CANONICAL_OUTAGES = [
    (50, 50),  # short:  5s outage,  5s pre, 20s post
    (60, 100),  # medium: 10s outage, 6s pre, 14s post
    (50, 200),  # long:   20s outage, 5s pre, 5s post
]

# Minimum GNSS windows required on each side of an outage
# to provide meaningful integration anchors
MIN_OUTAGE_BUFFER_WINDOWS = 20  # 2 seconds at 0.1s stride


def apply_outage(
    X_norm_seq,
    x_raw_seq,
    out_start,
    out_end,
    ch_gps_start=10,
    ch_gps_end=13,
    ch_mask=13,
):
    """
    Apply GNSS outage masking to a (seq_len, win_samples, n_channels) array.

    [FIX-I]   Zero the GNSS channels (not -100) in x_norm.
    [FIX-II]  Zero the GNSS channels in x_raw too, and return a boolean mask.
    [DESIGN-FIX-2] Channel indices are now parameters (not hardcoded constants).
                   This ensures correctness if the channel layout ever changes
                   and avoids silent divergence from the frozen channel map
                   saved in the .npz file.

    Args:
        X_norm_seq    : (seq_len, win_samples, n_ch) normalised features (modified in place)
        x_raw_seq     : (seq_len, win_samples, n_ch) raw features (modified in place)
        out_start     : int, first window index of outage (inclusive)
        out_end       : int, first window index AFTER outage (exclusive)
        ch_gps_start  : int, first GPS velocity channel index (default 10)
        ch_gps_end    : int, one-past-last GPS velocity channel index (default 13)
        ch_mask       : int, outage mask flag channel index (default 13)

    Returns:
        X_norm_seq, x_raw_seq (modified in place)
        outage_mask: (seq_len,) bool array, True where GNSS is denied
    """
    outage_mask = np.zeros(X_norm_seq.shape[0], dtype=bool)
    outage_mask[out_start:out_end] = True

    # [FIX-I] Zero GNSS velocity channels in normalised features (not -100).
    # Zero is in-distribution (≈ median after robust scaling); -100 is 33 σ
    # out of distribution and causes gradient explosions through residuals.
    X_norm_seq[out_start:out_end, :, ch_gps_start:ch_gps_end] = 0.0

    # Flip mask flag to 1.0 so the network can detect the outage via the mask channel
    X_norm_seq[out_start:out_end, :, ch_mask] = 1.0

    # [FIX-II] Zero GPS channels in x_raw as well.
    # The physics loss uses acc/gyro/quat from x_raw. If GPS channels in x_raw
    # were not zeroed, a naively written physics loss could leak ground-truth
    # velocities for the outage region — the variable the model should predict.
    x_raw_seq[out_start:out_end, :, ch_gps_start:ch_gps_end] = 0.0
    x_raw_seq[out_start:out_end, :, ch_mask] = 1.0

    return X_norm_seq, x_raw_seq, outage_mask


class MARSDataset(Dataset):
    """
    PyTorch Dataset for the MARS-LVIG Transformer.

    Sequence format (each __getitem__ call):
        x_norm    : (seq_len, win_samples, 14)  — normalised features with outage applied
        x_raw     : (seq_len, win_samples, 14)  — raw features; GPS channels zeroed during outage
        y_norm    : (seq_len, 3)                — normalised velocity targets
        y_raw     : (seq_len, 3)                — raw velocity targets (m/s) for evaluation
        weights   : (seq_len,)                  — per-window loss weights (turn weighting)
        outage_mask:(seq_len,)                  — True where GNSS is denied (bool)

    seq_len = 300 windows = 30 seconds
    win_samples = 200 IMU samples per window = 1 second at 200 Hz
    14 channels = [acc_xyz | gyr_xyz | quat_wxyz | gps_vned | mask]
    """

    def __init__(self, npz_path, split="train", outage_prob=0.8):
        """
        Args:
            npz_path    : Path to MARS_Master_Dataset.npz
            split       : 'train', 'val', or 'test'
            outage_prob : Probability of applying dynamic outage masking.
                          [FIX-V] Ignored for val and test — they always use
                          canonical deterministic outages for reproducibility.
        """
        super().__init__()
        data = np.load(npz_path)

        self.split = split
        self.outage_prob = outage_prob
        self.seq_len = int(data["seq_len"][0])

        # Load frozen channel layout (saved by combine_and_norm.py v2.0)
        # Fall back to hardcoded values if older dataset file is used
        self.ch_gps_start = (
            int(data["ch_gps_start"][0]) if "ch_gps_start" in data else 10
        )
        self.ch_gps_end = int(data["ch_gps_end"][0]) if "ch_gps_end" in data else 13
        self.ch_quat_start = (
            int(data["ch_quat_start"][0]) if "ch_quat_start" in data else 6
        )
        self.ch_quat_end = int(data["ch_quat_end"][0]) if "ch_quat_end" in data else 10
        self.ch_mask = int(data["ch_mask"][0]) if "ch_mask" in data else 13
        self.n_channels = int(data["n_channels"][0]) if "n_channels" in data else 14

        # Load split-specific arrays
        if split == "train":
            self.X = data["X_train"]
            self.Y = data["Y_train"]
            self.W = data["W_train"]
            self.valid_indices = data["train_valid_idx"]

        elif split == "val":
            self.X = data["X_val"]
            self.Y = data["Y_val"]
            self.W = np.ones(len(data["X_val"]), dtype=np.float32)
            self.valid_indices = data["val_valid_idx"]

        elif split == "test":
            self.X = data["X_test"]
            self.Y = data["Y_test"]
            self.W = np.ones(len(data["X_test"]), dtype=np.float32)
            self.valid_indices = data["test_valid_idx"]

        else:
            raise ValueError(f"split must be 'train', 'val', or 'test'. Got '{split}'.")

        # Load global robust scaling stats (computed on train set only)
        self.x_median = data["X_median"]  # shape (14,)
        self.x_iqr = data["X_iqr"]  # shape (14,)
        self.y_median = data["Y_median"]  # shape (3,)
        self.y_iqr = data["Y_iqr"]  # shape (3,)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # ─────────────────────────────────────────────────────────────────────
        # 1. Extract a valid 30-second continuous sequence
        # ─────────────────────────────────────────────────────────────────────
        start_idx = int(self.valid_indices[idx])
        end_idx = start_idx + self.seq_len

        # .copy() is essential — we will modify these arrays in place for masking
        X_raw = self.X[start_idx:end_idx].copy()  # (300, 200, 14)
        Y_raw = self.Y[start_idx:end_idx].copy()  # (300, 3)
        W = self.W[start_idx:end_idx].copy()  # (300,)

        # ─────────────────────────────────────────────────────────────────────
        # 2. Robust Normalisation (using train-set statistics)
        # Broadcasting: x_median shape (14,) broadcasts over (300, 200, 14)
        # ─────────────────────────────────────────────────────────────────────
        X_norm = (X_raw - self.x_median) / self.x_iqr  # (300, 200, 14)
        Y_norm = (Y_raw - self.y_median) / self.y_iqr  # (300, 3)

        # ─────────────────────────────────────────────────────────────────────
        # 3. Outage Simulation
        # ─────────────────────────────────────────────────────────────────────
        if self.split == "train":
            # Dynamic random masking for training — augments the model to handle
            # varying outage durations and positions
            if np.random.rand() < self.outage_prob:
                out_start, out_end = self._sample_random_outage()
                X_norm, X_raw, outage_mask = apply_outage(
                    X_norm,
                    X_raw,
                    out_start,
                    out_end,
                    self.ch_gps_start,
                    self.ch_gps_end,
                    self.ch_mask,  # [DESIGN-FIX-2]
                )
            else:
                outage_mask = np.zeros(self.seq_len, dtype=bool)

        else:
            # [FIX-III] Val and test use canonical deterministic outages
            pattern_idx = idx % len(CANONICAL_OUTAGES)
            out_start, out_len = CANONICAL_OUTAGES[pattern_idx]
            out_end = out_start + out_len

            # Safety: ensure outage fits within sequence
            if out_end > self.seq_len:
                out_end = self.seq_len
                out_start = max(0, out_end - out_len)

            X_norm, X_raw, outage_mask = apply_outage(
                X_norm,
                X_raw,
                out_start,
                out_end,
                self.ch_gps_start,
                self.ch_gps_end,
                self.ch_mask,  # [DESIGN-FIX-2]
            )

        # ─────────────────────────────────────────────────────────────────────
        # 4. Convert to PyTorch tensors
        # ─────────────────────────────────────────────────────────────────────
        return {
            # Primary model input: normalised, with outage applied
            "x_norm": torch.tensor(X_norm, dtype=torch.float32),
            # Physics loss input: raw (m/s²), GPS zeroed during outage
            # Use ONLY channels 0-9 (acc, gyr, quat) for physics constraint.
            # Channels 10-12 are zeroed during outage — do NOT use them in
            # the physics loss for the outage region.
            "x_raw": torch.tensor(X_raw, dtype=torch.float32),
            # Targets
            "y_norm": torch.tensor(Y_norm, dtype=torch.float32),
            "y_raw": torch.tensor(Y_raw, dtype=torch.float32),
            # Loss weighting (turn vs straight-line windows)
            "weights": torch.tensor(W, dtype=torch.float32),
            # Boolean mask: True = GNSS denied at this window
            # Training loop uses this to:
            # (a) know where to enforce physics constraints (outage region)
            # (b) know where data loss applies (non-outage region only)
            "outage_mask": torch.tensor(outage_mask, dtype=torch.bool),
        }

    def _sample_random_outage(self):
        """
        Sample a random outage (start, end) that respects the minimum
        pre- and post-outage buffer on both sides of the sequence.

        [FIX-IV] Enforces MIN_OUTAGE_BUFFER_WINDOWS on both sides.
                 This guarantees the bidirectional integration has at least
                 MIN_OUTAGE_BUFFER_WINDOWS of clean GNSS data to anchor
                 the forward and backward integration endpoints.
        """
        buf = MIN_OUTAGE_BUFFER_WINDOWS

        # Outage duration: between 5s (50 windows) and 20s (200 windows)
        max_outage_len = self.seq_len - 2 * buf  # can't consume the buffers
        min_outage_len = 50

        if max_outage_len < min_outage_len:
            # Sequence is too short for a meaningful outage + buffers
            # Return a minimal 1-window placeholder outage in the middle
            mid = self.seq_len // 2
            return mid, mid + 1

        outage_len = np.random.randint(min_outage_len, max_outage_len + 1)

        # Start position: must leave buf windows before AND (buf) windows after
        # out_end = out_start + outage_len <= seq_len - buf
        # → out_start <= seq_len - buf - outage_len
        max_start = self.seq_len - buf - outage_len
        out_start = np.random.randint(buf, max_start + 1)
        out_end = out_start + outage_len

        return out_start, out_end


# =============================================================================
# SELF-TEST
# =============================================================================
if __name__ == "__main__":
    DATA_FILE = r"./data/processed/MARS_Master_Dataset.npz"

    print("\n=== MARSDataset v2.0 Self-Test ===\n")

    # ── Train split ──────────────────────────────────────────────────────────
    train_ds = MARSDataset(DATA_FILE, split="train", outage_prob=1.0)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    batch = next(iter(train_loader))

    print("Train batch shapes:")
    for k, v in batch.items():
        print(f"  {k:<15}: {tuple(v.shape)}  dtype={v.dtype}")

    # Outage masking integrity
    mask_flag = batch["x_norm"][
        :, :, :, 13
    ]  # (B, seq, win)  ← scalar idx, shape is (B,seq,win)
    gnss_channels = batch["x_norm"][:, :, :, 10:13]
    outage_rows = batch["outage_mask"]  # (B, seq_len)

    print(f"\nMasking sanity (train):")
    print(
        f"  Outage mask flag range in x_norm: [{mask_flag.min():.1f}, {mask_flag.max():.1f}]"
        f"  (expect 0 and 1)"
    )
    print(
        f"  GNSS channels in x_norm during outage: "
        f"mean={gnss_channels[outage_rows].mean():.4f}  (expect 0.0)"
    )
    print(
        f"  GNSS channels in x_norm outside outage: "
        f"mean={gnss_channels[~outage_rows].mean():.4f}  (expect non-zero)"
    )

    # ── Val split: canonical outage reproducibility + per-pattern breakdown ──
    # [DESIGN-FIX-3] Evaluate all three canonical outage patterns separately
    # so that the long (20s) pattern's higher difficulty is visible in metrics
    # and does not silently inflate the aggregate val loss.
    print(f"\nVal canonical outage reproducibility:")
    val_ds = MARSDataset(DATA_FILE, split="val", outage_prob=0.0)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    val_loader2 = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)
    batch1 = next(iter(val_loader))
    batch2 = next(iter(val_loader2))
    masks_identical = torch.equal(batch1["outage_mask"], batch2["outage_mask"])
    print(
        f"  Same sequences → identical outage patterns: "
        f"{'PASS ✓' if masks_identical else 'FAIL ✗'}"
    )

    print(f"\nVal per-pattern breakdown (canonical outages):")
    print(
        f"  {'Pattern':<8} {'outage_s':>10} {'pre_buf_s':>10} {'post_buf_s':>10}"
        f"  {'seq_count':>10}  {'outage_%':>9}"
    )
    print(f"  {'-' * 65}")
    n_val_total = len(val_ds)
    for p_idx, (p_start, p_len) in enumerate(CANONICAL_OUTAGES):
        p_end = min(p_start + p_len, val_ds.seq_len)
        p_len_act = p_end - p_start
        pre_buf = p_start
        post_buf = val_ds.seq_len - p_end
        seq_count = sum(
            1 for i in range(n_val_total) if i % len(CANONICAL_OUTAGES) == p_idx
        )
        pct = p_len_act / val_ds.seq_len * 100
        label = ["short (5s)", "medium (10s)", "long (20s)"][p_idx]
        print(
            f"  {label:<12} {p_len_act * 0.1:>8.1f}s {pre_buf * 0.1:>9.1f}s "
            f"{post_buf * 0.1:>9.1f}s {seq_count:>11}  {pct:>8.1f}%"
        )

        # [DESIGN-FIX-3] Warn if post-outage buffer is dangerously short.
        # Bidirectional integration requires meaningful post-outage GNSS anchor.
        # Less than 3s (30 windows) is likely insufficient for backward integration.
        if post_buf * 0.1 < 3.0:
            print(
                f"    WARNING: {label} pattern has only {post_buf * 0.1:.1f}s post-outage "
                f"buffer. Backward integration anchor may be insufficient. "
                f"Consider shortening this canonical outage or extending SEQ_LEN."
            )

    # ── Test split smoke test ─────────────────────────────────────────────────
    # [DESIGN-FIX-3] Test split was previously never exercised in the self-test,
    # meaning shape errors or canonical outage assignment bugs could go unnoticed
    # until the final evaluation run. We now verify shapes, outage application,
    # and pattern reproducibility for the test split too.
    print(f"\nTest split smoke test:")
    test_ds = MARSDataset(DATA_FILE, split="test", outage_prob=0.0)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    test_batch = next(iter(test_loader))

    print(f"  Test batch shapes:")
    for k, v in test_batch.items():
        print(f"    {k:<15}: {tuple(v.shape)}  dtype={v.dtype}")

    # Verify test split sizes are as expected
    print(f"  Test dataset size : {len(test_ds)} sequences")
    print(f"  Val  dataset size : {len(val_ds)} sequences")
    if abs(len(test_ds) - len(val_ds)) > 5:
        print(
            f"  WARNING: Val ({len(val_ds)}) and test ({len(test_ds)}) sizes differ "
            f"by > 5. Check split ratios in combine_and_norm.py."
        )

    # Verify outage masking works on test split
    test_mask_flag = test_batch["x_norm"][:, :, :, 13]
    test_gnss = test_batch["x_norm"][:, :, :, 10:13]
    test_out_rows = test_batch["outage_mask"]
    gnss_during_ok = abs(test_gnss[test_out_rows].mean().item()) < 1e-4
    gnss_outside_ok = abs(test_gnss[~test_out_rows].mean().item()) > 1e-4
    flag_ok = test_mask_flag.max().item() == 1.0 and test_mask_flag.min().item() == 0.0
    print(f"  Outage masking on test split:")
    print(f"    GNSS zeroed during outage : {'PASS ✓' if gnss_during_ok else 'FAIL ✗'}")
    print(
        f"    GNSS non-zero outside     : {'PASS ✓' if gnss_outside_ok else 'FAIL ✗'}"
    )
    print(f"    Mask flag 0/1 range       : {'PASS ✓' if flag_ok else 'FAIL ✗'}")

    # Test reproducibility
    test_loader2 = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=0)
    test_batch2 = next(iter(test_loader2))
    test_repro = torch.equal(test_batch["outage_mask"], test_batch2["outage_mask"])
    print(f"  Canonical outage reproducibility: {'PASS ✓' if test_repro else 'FAIL ✗'}")

    # Overall result
    all_passed = (
        masks_identical
        and gnss_during_ok
        and gnss_outside_ok
        and flag_ok
        and test_repro
    )
    print(
        f"\n=== Self-Test {'PASSED ✓' if all_passed else 'FAILED ✗ — see warnings above'} ==="
    )
