"""
=============================================================================
  MARS-LVIG TRANSFORMER PREPROCESSING PIPELINE  v5.3
  Physics-Informed MAV Trajectory Prediction

  ╔══════════════════════════════════════════════════════════════════════╗
  ║  HOW TO RUN FOR EACH BAG — READ THIS FIRST                         ║
  ╠══════════════════════════════════════════════════════════════════════╣
  ║                                                                      ║
  ║  STEP A ── Set the bag name (ONE change per run):                   ║
  ║    Find the line near line 120:                                      ║
  ║       BAG_NAME = "gnss03"   ← change this string                    ║
  ║    Valid values: "gnss01" | "gnss02" | "gnss03"                     ║
  ║                  "island_gnss02" | "island_gnss03"                  ║
  ║                                                                      ║
  ║  STEP B ── Verify DATA_DIR for that bag in BAG_CONFIGS:             ║
  ║    Each bag entry has "DATA_DIR". Make sure the folder exists        ║
  ║    and contains livox-imu.csv and receiver_pvt.csv (or variants).   ║
  ║                                                                      ║
  ║  STEP C ── Run and watch for FAIL ✗ lines:                         ║
  ║    • acc_x mean should be |mean| ≈ 9.81 m/s² [v5.3: gravity ON]    ║
  ║    • acc_y, acc_z means should be |mean| < 2.0 m/s²               ║
  ║    • VQF line should say "VQFParams API — correct" (not dict warn)  ║
  ║    • No "LP gravity not converged" warning (raise CRUISE_START if)  ║
  ║                                                                      ║
  ║  OUTPUT FILES (in ./data/processed/):                               ║
  ║    transformer_ds_<bag>.npz   ← dataset for training               ║
  ║    report_transformer_<bag>.txt ← full log                         ║
  ║    plots_<bag>/               ← diagnostic plots                   ║
  ║                                                                      ║
  ╚══════════════════════════════════════════════════════════════════════╝

  CHANGELOG v5.2 → v5.3:

  [P-1] RAW SPECIFIC FORCE IN CHANNELS 0-2  (new — MUST re-run all bags)
    Previously Step 9 subtracted LP-estimated gravity and wrote gravity-free
    kinematic acc back into imu[acc_cols]. Now acc_gf is computed for QC only
    and NOT written to the IMU dataframe. Channels 0-2 store raw specific force.
    WHY: body-Y raw specific force at 20° bank = g·sin(20°) ≈ 3.36 m/s² —
    an 11× stronger tilt signal than the LP lag residual (~0.3 m/s²).
    Step 11 and T02 plot updated: body-X ≈ -9.81 m/s² is now CORRECT, not FAIL.

  [P-2] VQF API FIX — USE VQFParams OBJECT  (critical — must re-run all bags)
    All 5 bag reports showed "VQF params dict not supported" warning —
    every bag ran with default tauAcc=3s instead of 16s, corrupting quaternion
    features (channels 6-9). Now uses VQFParams() object (correct API).

  [P-5] ISLAND BAG CRUISE START +10s
    island_gnss02: 112.0 → 122.0s  |  island_gnss03: 118.0 → 128.0s
    LP filter needs ~9s to converge after attitude change at takeoff transition.

  [DESIGN-FIX-1] STORE LP-FILTERED GYRO IN WINDOWS  (carried from v5.2)
    LP-filtered gyro written into IMU dataframe before window extraction.
    Turn classifier in combine_and_norm.py operates on clean gyro signal.

  Prior fixes from v5.1 → v5.2:
  [FIX-7]  Gyro LP filter before VQF (zero-phase Butterworth, 20 Hz)
  [FIX-10] Gravity subtraction via accelerometer LP filter
  [FIX-9]  Per-bag STATIC_GYRO_STD_THRESH

  Prior fixes from v5.0 → v5.1:
  [FIX-1] Gravity subtraction DCM transpose fix (critical)
  [FIX-2] Gravity axis auto-detection
  [FIX-3] G-unit detection on gravity axis
  [FIX-4] Static window auto-detection via gyro variance scan
  [FIX-5] Acc bias auto-computation from static window
  [FIX-6] All-axis post-subtraction sanity check
=============================================================================
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt
from vqf import VQF
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
#  ██████╗  █████╗  ██████╗     ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗
#  ██╔══██╗██╔══██╗██╔════╝    ██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝
#  ██████╔╝███████║██║  ███╗   ██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗
#  ██╔══██╗██╔══██║██║   ██║   ██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║
#  ██████╔╝██║  ██║╚██████╔╝   ╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝
#  ╚═════╝ ╚═╝  ╚═╝ ╚═════╝     ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝
# =============================================================================
#
#  Per-bag configuration.
#
#  Keys you may need to adjust per bag:
#  ┌────────────────────────┬──────────────────────────────────────────────┐
#  │ Key                    │ How to set                                   │
#  ├────────────────────────┼──────────────────────────────────────────────┤
#  │ DATA_DIR               │ Path to folder with livox-imu.csv + pvt.csv  │
#  │ CRUISE_START_S         │ Seconds from overlap start where cruise      │
#  │                        │ begins (skip takeoff transient)              │
#  │ CRUISE_END_S           │ Seconds from overlap start where cruise ends │
#  │ GYRO_BIAS              │ Mean gyro reading from static pre-flight     │
#  │                        │ window (run diagnose.py → section [5])       │
#  │ ACC_BIAS               │ Leave [0,0,0] — auto-computed each run       │
#  │ GYR_CUTOFF_HZ          │ Gyro lowpass cutoff. Default 20 Hz works for │
#  │                        │ most bags. Lower to 10–15 Hz if sanity fails.│
#  │ STATIC_GYRO_STD_THRESH │ Max gyro std (rad/s) to qualify as static.   │
#  │                        │ Raise (e.g. 0.25) if your bag triggers the   │
#  │                        │ "Static window 0.0s" warning.                │
#  └────────────────────────┴──────────────────────────────────────────────┘
#
BAG_CONFIGS = {
    # ── CALIBRATION NOTES ──────────────────────────────────────────────────
    # GYRO_BIAS  : from diagnose.py section [5] — mean gyro over first 20s
    # ACC_BIAS   : auto-computed each run from static window; set to [0,0,0]
    # GRAVITY    : confirmed on body-X axis (nadir mounting) for all bags
    # VQF CONV   : NED convention — g_nav = [0, 0, +9.81] for all bags
    # ───────────────────────────────────────────────────────────────────────
    "gnss01": {
        "DATA_DIR": r"./data/raw/HKairport_GNSS01",
        "CRUISE_START_S": 115.0,
        "CRUISE_END_S": 742.0,
        "GYRO_BIAS": np.array([0.000728, -0.001390, -0.000936]),
        "ACC_BIAS": np.array([0.0, 0.0, 0.0]),
        "GYR_CUTOFF_HZ": 20.0,
        "STATIC_GYRO_STD_THRESH": 0.05,
        # Gravity LP: 0.3 Hz captures attitude changes (tilt rate < 0.3 Hz for MAV),
        # rejects kinematic accelerations. Raise slightly if hover mean still fails.
        "ACC_GRAVITY_LP_HZ": 0.3,
    },
    "gnss02": {
        "DATA_DIR": r"./data/raw/HKairport_GNSS02",
        "CRUISE_START_S": 95.0,
        "CRUISE_END_S": 408.0,
        "GYRO_BIAS": np.array([-0.000154, -0.001742, -0.000179]),
        "ACC_BIAS": np.array([0.0, 0.0, 0.0]),
        "GYR_CUTOFF_HZ": 20.0,
        "STATIC_GYRO_STD_THRESH": 0.05,
        "ACC_GRAVITY_LP_HZ": 0.3,
    },
    "gnss03": {
        "DATA_DIR": r"./data/raw/HKairport_GNSS03",
        "CRUISE_START_S": 110.0,
        "CRUISE_END_S": 330.0,
        "GYRO_BIAS": np.array([0.000309, -0.002491, 0.000499]),
        "ACC_BIAS": np.array([0.0, 0.0, 0.0]),
        "GYR_CUTOFF_HZ": 20.0,
        "STATIC_GYRO_STD_THRESH": 0.25,  # [FIX-9] raised: gyro std=0.19 rad/s at rest
        "ACC_GRAVITY_LP_HZ": 0.3,
    },
    "island_gnss02": {
        "DATA_DIR": r"./data/raw/HKisland_GNSS02",
        # [P-5] Raised from 112.0 → 122.0 (+10s).
        # The 0.3 Hz LP gravity filter needs ~9s to converge after attitude change
        # at takeoff. island_gnss02 showed LP gravity Y-offset of -0.71 m/s² and
        # Z-offset of +0.61 m/s² at the old cruise start (verified from report).
        # Skipping 10 additional seconds eliminates the LP transient windows and
        # removes the attitude transition from survey-approach to level cruise.
        "CRUISE_START_S": 122.0,
        "CRUISE_END_S": 383.0,
        "GYRO_BIAS": np.array([-0.001273, -0.002957, 0.004604]),
        "ACC_BIAS": np.array([0.0, 0.0, 0.0]),
        "GYR_CUTOFF_HZ": 20.0,
        "STATIC_GYRO_STD_THRESH": 0.05,
        "ACC_GRAVITY_LP_HZ": 0.3,
    },
    "island_gnss03": {
        "DATA_DIR": r"./data/raw/HKisland_GNSS03",
        # [P-5] Raised from 118.0 → 128.0 (+10s).
        # island_gnss03 showed worst LP transient of all bags: Y-offset -1.01 m/s²,
        # Z-offset +0.60 m/s² at cruise start. Same reasoning as island_gnss02.
        "CRUISE_START_S": 128.0,
        "CRUISE_END_S": 312.0,
        "GYRO_BIAS": np.array([-0.000446, -0.004060, 0.002459]),
        "ACC_BIAS": np.array([0.0, 0.0, 0.0]),
        "GYR_CUTOFF_HZ": 20.0,
        "STATIC_GYRO_STD_THRESH": 0.05,
        "ACC_GRAVITY_LP_HZ": 0.3,
    },
}

# =============================================================================
#  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
#  ► ONLY CHANGE THIS LINE BETWEEN RUNS ◄
BAG_NAME = "island_gnss03"
#  ▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄
# =============================================================================


# =============================================================================
# FIXED CONFIG  — do not change unless you change sensor hardware
# =============================================================================
OUT_DIR = r"./data/processed"
IMU_HZ = 200.0
DT = 1.0 / IMU_HZ
WINDOW_S = 1.0
STRIDE_S = 0.1
WIN_SAMPLES = int(WINDOW_S * IMU_HZ)
STRD_SAMPLES = int(STRIDE_S * IMU_HZ)
CRUISE_THRESH = 2.0  # m/s — below this is treated as hover
G_MPS2 = 9.81
DV_ARTIFACT_THRESH = 5.0  # m/s — GPS jump = multipath, skip window
GPS_MIN_FIX_TYPE = 3
GPS_MAX_H_ACC_M = 3.0
GPS_MIN_NUM_SV = 6
STATIC_MIN_DURATION_S = 10.0
IMU_VARIANTS = ["livox-imu.csv", "livox_imu.csv", "livox-IMU.csv", "imu.csv"]
PVT_VARIANTS = ["ublox_driver-receiver_pvt.csv", "receiver_pvt.csv", "pvt.csv"]
N_CHANNELS = 14


# =============================================================================
# VALIDATE BAG
# =============================================================================
if BAG_NAME not in BAG_CONFIGS:
    print(f"FATAL: '{BAG_NAME}' not in BAG_CONFIGS. Valid: {list(BAG_CONFIGS.keys())}")
    sys.exit(1)

cfg = BAG_CONFIGS[BAG_NAME]
DATA_DIR = cfg["DATA_DIR"]
CRUISE_START_S = cfg["CRUISE_START_S"]
CRUISE_END_S = cfg["CRUISE_END_S"]
GYRO_BIAS = cfg["GYRO_BIAS"]
ACC_BIAS_CFG = cfg["ACC_BIAS"]
GYR_CUTOFF_HZ = cfg.get("GYR_CUTOFF_HZ", 20.0)  # [FIX-7]
STATIC_GYRO_STD_THRESH = cfg.get("STATIC_GYRO_STD_THRESH", 0.05)  # [FIX-9]
ACC_GRAVITY_LP_HZ = cfg.get("ACC_GRAVITY_LP_HZ", 0.3)  # [FIX-10]


# =============================================================================
# LOGGING
# =============================================================================
_log = []


def log(m=""):
    print(m)
    _log.append(str(m))


def warn(m):
    s = f"  WARNING: {m}"
    print(s)
    _log.append(s)


def ok(m):
    s = f"  OK: {m}"
    print(s)
    _log.append(s)


def sub(t):
    log()
    log(f"-- {t} --")


def die(m):
    s = f"  FATAL: {m}"
    print(s)
    _log.append(s)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"report_transformer_{BAG_NAME}.txt"), "w") as f:
        f.write("\n".join(_log))
    sys.exit(1)


# =============================================================================
# HELPERS
# =============================================================================


def find_file(directory, variants):
    for name in variants:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            log(f"  Found: {name}")
            return p
    die(f"No file found in {directory}. Tried: {variants}")


def build_ts(df, sensor):
    if "header.stamp.secs" in df.columns and "header.stamp.nsecs" in df.columns:
        return (
            df["header.stamp.secs"].astype(np.float64)
            + df["header.stamp.nsecs"].astype(np.float64) * 1e-9
        )
    for col in ["Time", "time", "#time", "timestamp", "t_rel", "time_ns"]:
        if col in df.columns:
            ts = df[col].astype(np.float64)
            if ts.median() > 1e15:
                log(f"  [{sensor}] Nanosecond timestamp detected — converting.")
                ts *= 1e-9
            return ts
    die(f"[{sensor}] No timestamp column found. Columns: {list(df.columns)[:10]}")


def detect_fix_quality_cols(pvt_df):
    candidates = {
        "fix_type": ["fix_type", "fixType", "gnssFixOk"],
        "h_acc": ["h_acc", "hAcc", "horizontal_accuracy", "hacc"],
        "num_sv": ["num_sv", "numSV", "satellites_used"],
        "p_dop": ["p_dop", "pDOP", "pdop"],
    }
    return {
        k: next((n for n in v if n in pvt_df.columns), None)
        for k, v in candidates.items()
        if any(n in pvt_df.columns for n in v)
    }


def gate_pvt_quality(pvt_df, qc):
    bad = pd.Series(False, index=pvt_df.index)
    if "fix_type" in qc:
        b = pvt_df[qc["fix_type"]] < GPS_MIN_FIX_TYPE
        if b.sum():
            warn(f"fix_type<{GPS_MIN_FIX_TYPE}: {b.sum()} masked.")
            bad |= b
    if "h_acc" in qc:
        v = pvt_df[qc["h_acc"]].copy()
        if v.median() > 100:
            v /= 1000.0
            log("  h_acc mm→m.")
        b = v > GPS_MAX_H_ACC_M
        if b.sum():
            warn(f"h_acc>{GPS_MAX_H_ACC_M}m: {b.sum()} masked.")
            bad |= b
    if "num_sv" in qc:
        b = pvt_df[qc["num_sv"]] < GPS_MIN_NUM_SV
        if b.sum():
            warn(f"num_sv<{GPS_MIN_NUM_SV}: {b.sum()} masked.")
            bad |= b
    total = bad.sum()
    if total == 0:
        ok("All GNSS epochs passed quality gate.")
    else:
        warn(f"Masked {total}/{len(pvt_df)} ({total / len(pvt_df) * 100:.1f}%) epochs.")
    return pvt_df[~bad].copy()


def detect_static_window(imu_df, gyr_cols, std_thresh):
    """
    [FIX-4 / FIX-9] Find pre-flight static window via gyro variance scan.
    std_thresh is now per-bag (from STATIC_GYRO_STD_THRESH).
    """
    block_n = int(IMU_HZ)
    min_n = int(STATIC_MIN_DURATION_S * IMU_HZ)
    last_end = 0
    for k in range(0, len(imu_df) - block_n, block_n):
        stds = imu_df[gyr_cols].iloc[k : k + block_n].values.std(axis=0)
        if np.all(stds < std_thresh):
            last_end = k + block_n
        else:
            break
    if last_end < min_n:
        warn(
            f"Static window {last_end / IMU_HZ:.1f}s < min {STATIC_MIN_DURATION_S}s "
            f"(thresh={std_thresh} rad/s). Using fallback {STATIC_MIN_DURATION_S}s."
        )
        last_end = max(last_end, min(min_n, len(imu_df)))
    ok(f"Static pre-flight window: {last_end / IMU_HZ:.1f}s ({last_end} samples)")
    return 0, last_end


def detect_gravity_axis(imu_df, acc_cols, static_end):
    """[FIX-2] Find dominant gravity axis from static window mean."""
    g = imu_df[acc_cols].iloc[:static_end].values.mean(axis=0)
    axis = int(np.argmax(np.abs(g)))
    log(f"  Gravity body-frame vector (m/s²): {g.round(4)}")
    log(f"  Dominant axis: {'XYZ'[axis]}  value: {g[axis]:+.4f}")
    return axis, int(np.sign(g[axis])), g


def check_and_convert_units(imu_df, acc_cols, grav_axis):
    """[FIX-3] Detect g vs m/s² using the actual gravity axis."""
    val = abs(imu_df[acc_cols[grav_axis]].median())
    if 0.5 < val < 2.5:
        imu_df[acc_cols] *= G_MPS2
        ok(f"Converted g → m/s² (gravity-axis median = {val:.4f} g)")
        return True
    elif 4.0 < val < 15.0:
        ok(f"Already in m/s² (gravity-axis median = {val:.3f})")
        return False
    else:
        warn(f"Unexpected gravity-axis magnitude: {val:.4f}. Check units manually.")
        return False


def compute_acc_bias(imu_df, acc_cols, static_end, g_body_ms2):
    """[FIX-5] bias = measured_static_mean − pure_gravity_vector."""
    static_mean = imu_df[acc_cols].iloc[:static_end].values.mean(axis=0)
    g_pure = g_body_ms2 / np.linalg.norm(g_body_ms2) * G_MPS2
    return static_mean - g_pure


def lowpass_gyro(data, cutoff_hz, fs, order=4):
    """
    [FIX-7] Zero-phase Butterworth lowpass for gyroscope noise suppression.

    Uses filtfilt (forward + backward pass) → zero group delay → gyro and
    acc remain time-aligned inside VQF. Real MAV rotation dynamics are
    typically <10 Hz; the noise observed (std ≈ 0.19 rad/s) is broadband.
    A 20 Hz cutoff preserves all real manoeuvre content.
    """
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        warn(f"GYR_CUTOFF_HZ {cutoff_hz} Hz >= Nyquist {nyq} Hz. Skipping filter.")
        return data
    b, a = butter(order, cutoff_hz / nyq, btype="low")
    return filtfilt(b, a, data, axis=0)


def quat_to_dcm_batch(quats):
    """
    (N, 4) → (N, 3, 3) rotation matrices.
    VQF quat6D encodes C_n^b (nav→body). Returns C_n^b directly.
    DO NOT transpose before use — see FIX-1 comment in subtract_gravity().
    """
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    R = np.zeros((len(w), 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def subtract_gravity(acc_ms2, quats):
    """
    [FIX-1] Remove gravity from body-frame specific-force measurements.

    Physics:
      Accelerometer measures specific force:  f^b = a^b − g^b
      Kinematic acceleration:                 a^b = f^b + g^b
      Gravity in body frame:                  g^b = C_n^b @ g^n

    VQF quat6D encodes C_n^b.  quat_to_dcm_batch() returns C_n^b directly.

    FIX-1 (v5.0→v5.1): v5.0 transposed the DCM to get C_b^n and then
    computed C_b^n @ [0,0,9.81], projecting −9.81 onto body-X instead of
    +9.81 — doubling the gravity offset to −19.34 m/s².
    Fix: use quat_to_dcm_batch() output directly without any transpose.

    g^n = [0, 0, +9.81] m/s²  (NED — gravity points down = +Z).
    """
    g_nav = np.array([0.0, 0.0, G_MPS2])  # NED convention
    C_n_b = quat_to_dcm_batch(quats)  # C_n^b — no transpose
    g_body = np.einsum("nij,j->ni", C_n_b, g_nav)  # g^b = C_n^b @ g^n
    return acc_ms2 + g_body  # a^b = f^b + g^b


def log_eda_stats(data, label, ch_names):
    log(f"\n  [{label}] Per-channel statistics:")
    log(f"  {'Channel':<26} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    log(f"  {'-' * 68}")
    for i, name in enumerate(ch_names):
        col = data[:, i] if data.ndim == 2 else data[:, :, i].ravel()
        log(
            f"  {name:<26} {col.mean():>10.4f} {col.std():>10.4f} "
            f"{col.min():>10.4f} {col.max():>10.4f}"
        )


# =============================================================================
# EXECUTION
# =============================================================================
log(f"\n======================================================================")
log(f"  TRANSFORMER PREPROCESSING v5.3 | {BAG_NAME}")
log(f"======================================================================")
log(f"  GYRO_BIAS              : {GYRO_BIAS}")
log(f"  GYR_CUTOFF_HZ          : {GYR_CUTOFF_HZ} Hz")
log(f"  STATIC_GYRO_STD_THRESH : {STATIC_GYRO_STD_THRESH} rad/s")
log(f"  ACC_GRAVITY_LP_HZ      : {ACC_GRAVITY_LP_HZ} Hz")
log(f"  Cruise                 : {CRUISE_START_S}s → {CRUISE_END_S}s")
os.makedirs(OUT_DIR, exist_ok=True)
pdir = os.path.join(OUT_DIR, f"plots_{BAG_NAME}")
os.makedirs(pdir, exist_ok=True)


# ── STEP 1 ──────────────────────────────────────────────────────────────────
sub("STEP 1: File Discovery & Load")
imu = pd.read_csv(find_file(DATA_DIR, IMU_VARIANTS))
pvt = pd.read_csv(find_file(DATA_DIR, PVT_VARIANTS))
imu.columns = imu.columns.str.strip()
pvt.columns = [c.lstrip("#").strip() for c in pvt.columns]
ok(f"IMU ({len(imu)} rows)  PVT ({len(pvt)} rows).")


# ── STEP 2 ──────────────────────────────────────────────────────────────────
sub("STEP 2: Timestamp Alignment")
imu["t"] = build_ts(imu, "IMU")
pvt["t"] = build_ts(pvt, "PVT")
imu = imu.sort_values("t").reset_index(drop=True)
pvt = pvt.sort_values("t").reset_index(drop=True)
if (imu["t"].diff().dropna() <= 0).any():
    warn("Non-monotonic IMU timestamps detected.")
ov_start = max(imu.t.iloc[0], pvt.t.iloc[0])
ov_end = min(imu.t.iloc[-1], pvt.t.iloc[-1])
imu = imu[(imu.t >= ov_start) & (imu.t <= ov_end)].reset_index(drop=True)
pvt = pvt[(pvt.t >= ov_start) & (pvt.t <= ov_end)].reset_index(drop=True)
ok(f"Aligned {len(imu)} IMU, {len(pvt)} PVT. Overlap: {ov_end - ov_start:.1f}s")


# ── STEP 3 ──────────────────────────────────────────────────────────────────
sub("STEP 3: Column Detection & Ordering")
acc_cols = sorted(
    [c for c in imu.columns if "linear_acceleration" in c and "covariance" not in c],
    key=lambda c: c[-1],
)
gyr_cols = sorted(
    [c for c in imu.columns if "angular_velocity" in c and "covariance" not in c],
    key=lambda c: c[-1],
)
if len(acc_cols) != 3:
    die(f"Expected 3 acc cols, got: {acc_cols}")
if len(gyr_cols) != 3:
    die(f"Expected 3 gyr cols, got: {gyr_cols}")
for i, e in enumerate(["x", "y", "z"]):
    if not acc_cols[i].endswith(e):
        die(f"acc ordering wrong: {acc_cols}")
    if not gyr_cols[i].endswith(e):
        die(f"gyr ordering wrong: {gyr_cols}")
ok(f"Acc: {acc_cols}")
ok(f"Gyr: {gyr_cols}")


# ── STEP 4 ──────────────────────────────────────────────────────────────────
sub("STEP 4: Static Window Detection  [FIX-4 / FIX-9]")
_, static_end = detect_static_window(imu, gyr_cols, STATIC_GYRO_STD_THRESH)


# ── STEP 5 ──────────────────────────────────────────────────────────────────
sub("STEP 5: Unit Conversion & Bias Subtraction  [FIX-2 / FIX-3 / FIX-5]")
grav_idx, grav_sign, g_body_raw = detect_gravity_axis(imu, acc_cols, static_end)
check_and_convert_units(imu, acc_cols, grav_idx)
grav_idx, grav_sign, g_body_ms2 = detect_gravity_axis(imu, acc_cols, static_end)
if not (7.0 < abs(g_body_ms2[grav_idx]) < 12.0):
    warn(
        f"Gravity magnitude = {abs(g_body_ms2[grav_idx]):.3f} m/s². "
        f"Expected ≈9.81. Check units."
    )

acc_bias_computed = compute_acc_bias(imu, acc_cols, static_end, g_body_ms2)
log(f"\n  ┌─ AUTO-COMPUTED ACC_BIAS ────────────────────────────────")
log(f"  │  {acc_bias_computed.round(6)} m/s²")
log(f"  │  Copy to BAG_CONFIGS['{BAG_NAME}']['ACC_BIAS'] to persist.")
log(f"  └─────────────────────────────────────────────────────────")

ACC_BIAS = acc_bias_computed if np.all(ACC_BIAS_CFG == 0.0) else ACC_BIAS_CFG
if np.all(ACC_BIAS_CFG == 0.0):
    warn(f"ACC_BIAS placeholder active — auto-using: {ACC_BIAS.round(4)}")
else:
    ok(f"ACC_BIAS from config: {ACC_BIAS}")

imu[gyr_cols] -= GYRO_BIAS
ok("Subtracted GYRO_BIAS.")
if np.any(np.abs(ACC_BIAS) > 0.001):
    imu[acc_cols] -= ACC_BIAS
    ok(f"Subtracted ACC_BIAS: {ACC_BIAS.round(4)}")


# ── STEP 6 ──────────────────────────────────────────────────────────────────
sub("STEP 6: GPS Fix Quality Gating")
qc = detect_fix_quality_cols(pvt)
log(f"  GPS quality cols: {qc}")
if not qc:
    warn("No GPS quality columns found — skipping quality gate.")
else:
    pvt = gate_pvt_quality(pvt, qc)
    ok(f"{len(pvt)} clean PVT epochs.")


def _col(df, cands, name):
    for c in cands:
        if c in df.columns:
            return c
    die(f"Cannot find {name}. Tried: {cands}")


vn_col = _col(pvt, ["vel_n", "velocityNorth", "vn", "vel_n_m_s"], "vel_N")
ve_col = _col(pvt, ["vel_e", "velocityEast", "ve", "vel_e_m_s"], "vel_E")
vd_col = _col(pvt, ["vel_d", "velocityDown", "vd", "vel_d_m_s"], "vel_D")
ok(f"GPS vel cols: [{vn_col}, {ve_col}, {vd_col}]")


# ── STEP 7 ──────────────────────────────────────────────────────────────────
sub("STEP 7: GPS Interpolation (10 Hz → 200 Hz)")


def safe_interp(ts, vs, td):
    f = interp1d(
        ts,
        vs,
        kind="linear",
        bounds_error=False,
        fill_value=(float(vs.iloc[0]), float(vs.iloc[-1])),
    )
    return f(td)


imu["gps_vn"] = safe_interp(pvt.t, pvt[vn_col], imu.t)
imu["gps_ve"] = safe_interp(pvt.t, pvt[ve_col], imu.t)
imu["gps_vd"] = safe_interp(pvt.t, pvt[vd_col], imu.t)
imu["gps_speed_3d"] = np.sqrt(
    imu["gps_vn"] ** 2 + imu["gps_ve"] ** 2 + imu["gps_vd"] ** 2
)
ok(
    f"3D speed range: {imu['gps_speed_3d'].min():.2f} – "
    f"{imu['gps_speed_3d'].max():.2f} m/s"
)


# ── STEP 8 ──────────────────────────────────────────────────────────────────
sub("STEP 8: Gyro Lowpass Filter + VQF Orientation  [FIX-7 / FIX-8]")
log("  VQF is used here for quaternion features (channels 7-10) only.")
log("  Gravity subtraction is handled independently in Step 9 via LP filter.")

acc_vqf = np.ascontiguousarray(imu[acc_cols].values, dtype=np.float64)
gyr_raw = np.ascontiguousarray(imu[gyr_cols].values, dtype=np.float64)

# [FIX-7] Low-pass filter gyro before VQF.
gyr_filtered = np.ascontiguousarray(
    lowpass_gyro(gyr_raw, GYR_CUTOFF_HZ, IMU_HZ), dtype=np.float64
)
raw_std = gyr_raw.std(axis=0).round(4)
filtered_std = gyr_filtered.std(axis=0).round(4)
log(f"  Gyro std  raw     : {raw_std}")
log(f"  Gyro std  filtered: {filtered_std}  (cutoff={GYR_CUTOFF_HZ} Hz)")

# [FIX-8] Convergence via static-window prefix repetition.
N_REPEATS = 3
prefix_gyr = np.tile(gyr_filtered[:static_end], (N_REPEATS, 1))
prefix_acc = np.tile(acc_vqf[:static_end], (N_REPEATS, 1))
gyr_padded = np.ascontiguousarray(np.vstack([prefix_gyr, gyr_filtered]))
acc_padded = np.ascontiguousarray(np.vstack([prefix_acc, acc_vqf]))
pad_len = N_REPEATS * static_end
log(
    f"  Prepending {pad_len} static samples ({pad_len / IMU_HZ:.1f}s × {N_REPEATS} repeats) "
    f"for VQF convergence …"
)

# [P-2] Set VQF tauAcc = 16s using vqf.setTauAcc() — the correct API for
# vqf 2.1.1 (confirmed: VQF instance exposes setTauAcc as a method).
#
# WHY: tauAcc is the accelerometer time constant for VQF's tilt correction.
# At tauAcc=3s (default), centripetal acceleration during a 3s banked turn
# can corrupt the tilt estimate — VQF slowly mistakes sustained lateral
# force for the gravity direction. At tauAcc=16s the filter resists this
# contamination for the full duration of our longest outage (20s).
# ALL 5 preprocessing reports showed the old dict API was silently ignored,
# meaning every bag ran with tauAcc=3s and degraded quaternion features.
VQF_TAU_ACC = 16.0
vqf = VQF(DT)  # construct with default params first
vqf.setTauAcc(VQF_TAU_ACC)  # then override tauAcc — this is the correct call
log(f"  [P-2] VQF tauAcc = {VQF_TAU_ACC}s set via setTauAcc()  [FIXED]")
# Verify it took effect by reading back through .params
try:
    _actual = vqf.params.tauAcc
    if abs(_actual - VQF_TAU_ACC) < 0.01:
        log(f"  [P-2] Verified: vqf.params.tauAcc = {_actual}s  ✓")
    else:
        warn(
            f"[P-2] setTauAcc({VQF_TAU_ACC}) called but params.tauAcc reads {_actual}s"
        )
except AttributeError:
    log(f"  [P-2] params.tauAcc not readable — setTauAcc() call assumed correct.")

res_padded = vqf.updateBatch(gyr_padded, acc_padded)
res_quats = res_padded["quat6D"][pad_len:]  # discard warm-up prefix
log(f"  Convergence prefix discarded. Using last {len(res_quats)} samples.")

quats = res_quats
assert len(quats) == len(imu), (
    f"Quaternion length mismatch: {len(quats)} vs IMU {len(imu)}"
)

imu["quat_w"] = quats[:, 0]
imu["quat_x"] = quats[:, 1]
imu["quat_y"] = quats[:, 2]
imu["quat_z"] = quats[:, 3]

qn = np.sqrt((quats**2).sum(axis=1))
if abs(qn.mean() - 1.0) > 0.01:
    warn(f"Quaternion norms unhealthy: mean={qn.mean():.6f}")
else:
    ok(f"Quaternion norms: mean={qn.mean():.6f}")
q0 = quats[0]
log(f"  quat[0]: w={q0[0]:.4f} x={q0[1]:.4f} y={q0[2]:.4f} z={q0[3]:.4f}")
ok("VQF complete.")

# ── [DESIGN-FIX-1] Write LP-filtered gyro back into the IMU dataframe ───────
#
# WHY THIS MATTERS FOR TURN CLASSIFICATION:
#   The raw gyro for gnss03 has a noise floor of std ≈ 0.190 rad/s at rest.
#   A 2-sigma noise spike reaches ±0.38 rad/s — above the TURN_GYRO_THRESH
#   of 0.3 rad/s used by combine_and_norm.py. Without this fix, hover and
#   straight-line windows get falsely classified as "turn" windows and receive
#   3× loss weighting, biasing the model toward sensor noise artefacts rather
#   than real manoeuvre dynamics.
#
#   The LP-filtered gyro (cutoff=GYR_CUTOFF_HZ, default 20 Hz) keeps all
#   physically real rotation content (actual MAV rotation rates < 10 Hz) while
#   suppressing the broadband noise floor. Storing it in the windows ensures
#   the turn classifier in combine_and_norm.py operates on a clean signal.
#
#   The unfiltered gyro is still accessible via `gyr_raw` for diagnostics.
log(
    f"\n  [DESIGN-FIX-1] Writing LP-filtered gyro (cutoff={GYR_CUTOFF_HZ} Hz) "
    f"into IMU dataframe for window extraction."
)
log(f"  Raw gyro std     : {raw_std}  rad/s")
log(f"  Filtered gyro std: {filtered_std}  rad/s")
log(f"  Noise reduction  : {(1.0 - filtered_std / np.maximum(raw_std, 1e-9)).round(3)}")
for i, col in enumerate(gyr_cols):
    imu[col] = gyr_filtered[:, i]
ok("Filtered gyro written to IMU dataframe. Windows will contain clean gyro.")


# ── STEP 9 ──────────────────────────────────────────────────────────────────
sub("STEP 9: Gravity Subtraction  [FIX-10]")
log("  Method: body-frame gravity tracking via zero-phase accelerometer LP filter.")
log(f"  Cutoff: {ACC_GRAVITY_LP_HZ} Hz  —  captures MAV attitude dynamics,")
log("  rejects kinematic accelerations. filtfilt = zero phase, no time lag.")
log()
log("  Why not VQF-based removal:")
log("  VQF tilt estimate drifts during sustained banking because its acc aiding")
log("  (tauAcc=3s default) slowly adopts centripetal force as the new 'gravity'.")
log("  The LP filter approach is independent of orientation estimation and works")
log("  correctly as long as attitude changes are slower than ACC_GRAVITY_LP_HZ.")
log()
log("  Physics:")
log("    f^b (specific force, = raw acc reading) = a^b + (-g^b)")
log("    At rest: a^b = 0  →  f^b = -g^b  →  g^b = -f^b_static")
log("    LP filter of f^b over time ≈ -g^b (quasi-static gravity component)")
log("    Kinematic: a^b = f^b - LP(f^b)")

b_grav, a_grav = butter(2, ACC_GRAVITY_LP_HZ / (IMU_HZ / 2.0), btype="low")
acc_arr = imu[acc_cols].values.copy()

# LP filter extracts the slowly-varying gravity component in body frame.
g_est = filtfilt(b_grav, a_grav, acc_arr, axis=0)

# Normalize magnitude to exactly G_MPS2 on every sample.
# This prevents centripetal acceleration contamination from inflating or
# deflating the extracted gravity magnitude during banked turns.
g_norm = np.linalg.norm(g_est, axis=1, keepdims=True)
g_est_n = g_est * G_MPS2 / np.maximum(g_norm, 1.0)

# Gravity-free kinematic acceleration
acc_gf = acc_arr - g_est_n  # a^b = f^b - g^b_estimated

# [P-1] Raw specific force is retained in imu[acc_cols].
# The three lines below that previously overwrote acc_cols with gravity-free
# kinematic acceleration have been removed. Channels 0-2 in the output windows
# now contain raw specific force (with gravity included).
#
# WHY: During a 20° bank the gravity vector projects onto body-Y as
# g × sin(20°) ≈ 3.36 m/s². The old LP-filter gravity subtraction removed
# this signal — leaving only the LP lag residual (~0.3 m/s²), an 11× weaker
# tilt signal to the CNN. Raw specific force gives the CNN direct, high-SNR
# access to the drone's attitude at every 200 Hz sample.
#
# After IQR scaling in combine_and_norm.py, the gravity mean on body-X is
# subtracted by the robust scaler (median ≈ -9.754 m/s²), so the scaled
# representation remains centred at zero. The attitude-dependent modulation
# across all three body axes is preserved and visible to the encoder.
#
# acc_gf is kept as a local variable for the hover sanity check below.
# It is NOT written into the IMU dataframe — the physics loss recomputes
# gravity-free acceleration internally from the quaternion at inference time.
cruise_idx = (imu.t - ov_start >= CRUISE_START_S).idxmax()
log(
    f"\n  [P-1] Raw specific force retained in IMU channels (gravity NOT removed)."
    f"\n  acc_gf kept for hover sanity check only — NOT stored in windows."
    f"\n  Gravity estimate at cruise start (m/s²): {g_est_n[cruise_idx].round(4)}"
    f"\n  Expected near: gravity body-frame = {g_body_ms2.round(4)}"
)

hover_mask = imu["gps_speed_3d"] < 0.5
n_hov = hover_mask.sum()
hover_idx = hover_mask.values  # numpy bool array for indexing acc_gf

log(f"\n  Post-subtraction sanity on gravity-free acc ({n_hov} hover samples):")
log("  (Using acc_gf = acc_arr - g_est_n, NOT the stored window data)")
all_ok = True
for i, col in enumerate(acc_cols):
    if n_hov > 100:
        m = acc_gf[hover_idx, i].mean()
        s = acc_gf[hover_idx, i].std()
        passed = abs(m) < 1.0
        status = "OK ✓" if passed else "FAIL ✗"
        log(f"    {col}: grav-free hover mean={m:+.4f} m/s²  std={s:.4f}  [{status}]")
        if not passed:
            all_ok = False
            warn(
                f"{col} gravity-free hover mean {m:.3f} m/s² exceeds 1.0 threshold. "
                f"Try lowering ACC_GRAVITY_LP_HZ in BAG_CONFIGS['{BAG_NAME}'] "
                f"(current: {ACC_GRAVITY_LP_HZ} Hz). Smaller = slower gravity tracking."
            )
    else:
        log(f"    {col}: insufficient hover samples ({n_hov})")

if n_hov > 100:
    if all_ok:
        ok(
            "Gravity-free sanity check PASSED (acc_gf is clean — raw sf retained in windows)."
        )
    else:
        warn("Sanity FAILED on acc_gf. See actionable hints above.")
ok("Step 9 complete. Raw specific force stored in windows (gravity included).")


# ── STEP 10 ─────────────────────────────────────────────────────────────────
sub("STEP 10: Window Extraction & Artifact Rejection")
feature_cols = (
    acc_cols
    + gyr_cols
    + ["quat_w", "quat_x", "quat_y", "quat_z"]
    + ["gps_vn", "gps_ve", "gps_vd"]
)
assert len(feature_cols) == N_CHANNELS - 1, f"Expected {N_CHANNELS - 1} cols"

windows, labels, speeds = [], [], []
n_hskip = n_nskip = n_askip = 0

for k in range(0, len(imu) - WIN_SAMPLES + 1, STRD_SAMPLES):
    e = k + WIN_SAMPLES
    rel_t = imu.t.iloc[k] - ov_start
    if rel_t < CRUISE_START_S:
        continue
    if rel_t > CRUISE_END_S:
        continue

    seg_speed = imu["gps_speed_3d"].iloc[k:e]
    if seg_speed.median() <= CRUISE_THRESH:
        n_hskip += 1
        continue

    feats = imu[feature_cols].iloc[k:e].values
    if np.isnan(feats).any():
        n_nskip += 1
        continue

    dv = max(
        imu["gps_vn"].iloc[k:e].max() - imu["gps_vn"].iloc[k:e].min(),
        imu["gps_ve"].iloc[k:e].max() - imu["gps_ve"].iloc[k:e].min(),
        imu["gps_vd"].iloc[k:e].max() - imu["gps_vd"].iloc[k:e].min(),
    )
    if dv > DV_ARTIFACT_THRESH:
        n_askip += 1
        continue

    c = k + WIN_SAMPLES // 2
    tgt = imu[["gps_vn", "gps_ve", "gps_vd"]].iloc[c].values
    if np.isnan(tgt).any():
        n_nskip += 1
        continue

    windows.append(feats)
    labels.append(tgt)
    speeds.append(seg_speed.median())

if not windows:
    die("No valid windows extracted. Check CRUISE_START_S / CRUISE_END_S.")

X_data = np.array(windows, dtype=np.float32)
Y_data = np.array(labels, dtype=np.float32)
# Append placeholder mask channel (all zeros)
X_data = np.concatenate(
    [X_data, np.zeros((X_data.shape[0], X_data.shape[1], 1), dtype=np.float32)],
    axis=2,
)
assert X_data.shape[2] == N_CHANNELS

log(f"  Hover skipped   : {n_hskip}")
log(f"  NaN/bad skipped : {n_nskip}")
if n_askip:
    warn(f"Artifact (GPS jump) skipped: {n_askip}")
else:
    ok("No multipath artifacts detected.")
if n_hskip == 0:
    warn("Zero hover windows skipped — verify speed profile plot T03.")
ok(f"Dataset: X={X_data.shape}  Y={Y_data.shape}")
log(
    f"  vN[{Y_data[:, 0].min():.2f},{Y_data[:, 0].max():.2f}] "
    f"vE[{Y_data[:, 1].min():.2f},{Y_data[:, 1].max():.2f}] "
    f"vD[{Y_data[:, 2].min():.2f},{Y_data[:, 2].max():.2f}] m/s"
)


# ── STEP 11 ─────────────────────────────────────────────────────────────────
sub("STEP 11: EDA Statistics")
ch_names = [
    "acc_x",
    "acc_y",
    "acc_z",
    "gyr_x",
    "gyr_y",
    "gyr_z",
    "quat_w",
    "quat_x",
    "quat_y",
    "quat_z",
    "gps_vn",
    "gps_ve",
    "gps_vd",
    "mask",
]
X_flat = X_data.reshape(-1, N_CHANNELS)
log_eda_stats(X_flat, BAG_NAME, ch_names)

log(
    "\n  PASS CRITERIA — [P-1] acc channels now hold RAW SPECIFIC FORCE (gravity included):"
)
log("  acc_x (body-down) should have mean ≈ -9.81 m/s²  (gravity dominant).")
log("  acc_y and acc_z should have mean ≈ 0.0 m/s²  (kinematic, near-level cruise).")
for i in range(3):
    m = X_flat[:, i].mean()
    if i == 0:
        # Body-X is the nadir/gravity axis — expect mean near -G_MPS2
        passed = abs(abs(m) - G_MPS2) < 2.0
        status = "OK ✓" if passed else f"FAIL ✗ — expected ≈±{G_MPS2:.2f}, got {m:.3f}"
    else:
        # Body-Y and body-Z are lateral/forward — expect mean near 0 (kinematic mean is small)
        passed = abs(m) < 3.0
        status = (
            "OK ✓"
            if passed
            else f"FAIL ✗ — expected ≈0 for non-gravity axis, got {m:.3f}"
        )
    log(f"  acc_{'xyz'[i]}: mean={m:+.4f}  [{status}]")
for i, nm in enumerate(["v_N", "v_E", "v_D"]):
    log(
        f"  {nm}: mean={Y_data[:, i].mean():.3f}  std={Y_data[:, i].std():.3f}  "
        f"min={Y_data[:, i].min():.3f}  max={Y_data[:, i].max():.3f} m/s"
    )


# ── STEP 12 ─────────────────────────────────────────────────────────────────
sub("STEP 12: Plots")
DARK = "#262626"
YC = "#ffe600"
RC = "#ff6b6b"
TC = "#4ecdc4"
WC = "#f7fff7"

# T01 — Target velocity distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=DARK)
for i, nm in enumerate(["v_N", "v_E", "v_D"]):
    ax = axes[i]
    ax.set_facecolor(DARK)
    ax.hist(Y_data[:, i], bins=60, color=YC, alpha=0.85, edgecolor="none")
    ax.set_title(
        f"{nm} (m/s)\nμ={Y_data[:, i].mean():.2f} σ={Y_data[:, i].std():.2f}",
        color="white",
        fontsize=11,
    )
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#404040")
plt.suptitle(f"Target Velocities — {BAG_NAME}", color="white")
plt.tight_layout()
plt.savefig(os.path.join(pdir, "T01_targets.png"), dpi=150, facecolor=DARK)
plt.close()

# T02 — Raw specific force (10s cruise snippet)
# [P-1] After the P-1 change, acc channels contain raw specific force WITH gravity.
# Body-X (nadir-mounted) will show mean ≈ -9.81 m/s²; this is correct and expected.
# Body-Y and body-Z will show means near 0 m/s² (kinematic components).
# A separate dashed line shows the expected gravity on each axis for reference.
si = imu[imu.t - ov_start >= CRUISE_START_S].index[0]
pn = int(10 * IMU_HZ)
tp = np.linspace(0, 10, pn)
fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=DARK, sharex=True)
g_expected = g_body_ms2  # expected mean for body-frame raw specific force
for i, (col, clr) in enumerate(zip(acc_cols, [YC, RC, TC])):
    ax = axes[i]
    ax.set_facecolor(DARK)
    v = imu[col].iloc[si : si + pn]
    ax.plot(tp, v, color=clr, lw=0.9)
    ax.axhline(0, color="white", lw=0.5, linestyle="--", alpha=0.4, label="zero")
    ax.axhline(
        g_expected[i],
        color=WC,
        lw=0.8,
        linestyle=":",
        alpha=0.7,
        label=f"g_body={g_expected[i]:.2f}",
    )
    m = v.mean()
    # For gravity axis (body-X ≈ -9.81), check proximity to -G_MPS2.
    # For non-gravity axes, check proximity to 0.
    if i == 0:
        passed = abs(abs(m) - G_MPS2) < 2.0
        status = "OK ✓" if passed else "FAIL ✗"
    else:
        passed = abs(m) < 3.0
        status = "OK ✓" if passed else "FAIL ✗"
    ax.set_title(
        f"acc_{'xyz'[i]} (raw spec force, gravity IN) — cruise mean={m:+.3f} m/s² [{status}]",
        color="white",
        fontsize=10,
    )
    ax.legend(facecolor=DARK, edgecolor="#404040", labelcolor="white", fontsize=7)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#404040")
axes[-1].set_xlabel("Seconds from Cruise Start", color="white")
plt.suptitle(
    f"Raw Specific Force (WITH gravity) — {BAG_NAME}  "
    f"(body-X ≈ −9.81, body-Y/Z ≈ 0) [P-1]",
    color="white",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(os.path.join(pdir, "T02_acc_raw_sf.png"), dpi=150, facecolor=DARK)
plt.close()

# T03 — GPS 3D speed with cruise window markers
fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK)
ax.set_facecolor(DARK)
ax.plot(imu.t - ov_start, imu["gps_speed_3d"], color=YC, lw=0.7, alpha=0.8)
ax.axvline(CRUISE_START_S, color=RC, lw=2, linestyle="--", label="Cruise Start")
ax.axvline(CRUISE_END_S, color=TC, lw=2, linestyle="--", label="Cruise End")
ax.axhline(
    CRUISE_THRESH, color=WC, lw=1, linestyle=":", label=f"Hover ≤ {CRUISE_THRESH} m/s"
)
ax.set_title(f"GPS 3D Speed — {BAG_NAME}", color="white")
ax.set_xlabel("Time from overlap start (s)", color="white")
ax.set_ylabel("Speed (m/s)", color="white")
ax.legend(facecolor=DARK, edgecolor="#404040", labelcolor="white")
ax.tick_params(colors="white")
for s in ax.spines.values():
    s.set_color("#404040")
plt.tight_layout()
plt.savefig(os.path.join(pdir, "T03_speed.png"), dpi=150, facecolor=DARK)
plt.close()

# T04 — VQF quaternions (10s cruise snippet)
fig, ax = plt.subplots(figsize=(14, 4), facecolor=DARK)
ax.set_facecolor(DARK)
for col, clr in zip(["quat_w", "quat_x", "quat_y", "quat_z"], [YC, RC, TC, WC]):
    ax.plot(tp, imu[col].iloc[si : si + pn], color=clr, lw=1.2, alpha=0.85, label=col)
ax.set_title(f"VQF Quaternions — {BAG_NAME} (10s cruise snippet)", color="white")
ax.set_xlabel("Seconds", color="white")
ax.legend(facecolor=DARK, edgecolor="#404040", labelcolor="white")
ax.tick_params(colors="white")
for s in ax.spines.values():
    s.set_color("#404040")
plt.tight_layout()
plt.savefig(os.path.join(pdir, "T04_quat.png"), dpi=150, facecolor=DARK)
plt.close()

# T05 — Gyro noise: raw vs filtered (new diagnostic)
fig, axes = plt.subplots(3, 1, figsize=(14, 9), facecolor=DARK, sharex=True)
for i, (col, clr) in enumerate(zip(gyr_cols, [YC, RC, TC])):
    ax = axes[i]
    ax.set_facecolor(DARK)
    ax.plot(tp, gyr_raw[si : si + pn, i], color=WC, lw=0.6, alpha=0.5, label="raw")
    ax.plot(
        tp,
        gyr_filtered[si : si + pn, i],
        color=clr,
        lw=1.2,
        alpha=0.9,
        label="filtered",
    )
    ax.set_title(
        f"gyr_{'xyz'[i]} — raw std={raw_std[i]:.4f}  filtered std={filtered_std[i]:.4f} rad/s",
        color="white",
        fontsize=10,
    )
    ax.legend(facecolor=DARK, edgecolor="#404040", labelcolor="white", fontsize=8)
    ax.tick_params(colors="white")
    for s in ax.spines.values():
        s.set_color("#404040")
axes[-1].set_xlabel("Seconds from Cruise Start", color="white")
plt.suptitle(
    f"Gyro Raw vs Filtered ({GYR_CUTOFF_HZ} Hz LP) — {BAG_NAME}",
    color="white",
    fontsize=12,
)
plt.tight_layout()
plt.savefig(os.path.join(pdir, "T05_gyro_filter.png"), dpi=150, facecolor=DARK)
plt.close()

ok(f"Plots saved to {pdir}")


# ── STEP 13 ─────────────────────────────────────────────────────────────────
sub("STEP 13: Save")
fname = f"transformer_ds_{BAG_NAME}.npz"
X_save = X_data.reshape(-1, N_CHANNELS)
np.savez_compressed(
    os.path.join(OUT_DIR, fname),
    X_train=X_data,
    y_train=Y_data,
    speeds=np.array(speeds, dtype=np.float32),
    config=np.array([IMU_HZ, WINDOW_S, STRIDE_S]),
    X_mean=X_save.mean(axis=0).astype(np.float32),
    X_std=X_save.std(axis=0).astype(np.float32),
    acc_bias_used=ACC_BIAS.astype(np.float32),
    gyro_bias_used=GYRO_BIAS.astype(np.float32),
    gyr_cutoff_hz=np.float32(GYR_CUTOFF_HZ),
    acc_gravity_lp_hz=np.float32(ACC_GRAVITY_LP_HZ),
    # [DESIGN-FIX-1] Store both raw and filtered gyro std so combine_and_norm.py
    # can verify the turn classifier is operating on clean gyro data.
    gyr_raw_std=raw_std.astype(np.float32),
    gyr_filtered_std=filtered_std.astype(np.float32),
)
ok(f"Saved: {os.path.join(OUT_DIR, fname)}")

with open(
    os.path.join(OUT_DIR, f"report_transformer_{BAG_NAME}.txt"), "w", encoding="utf-8"
) as f:
    f.write("\n".join(_log))
ok("Report saved.")
log("\n=== PREPROCESSING COMPLETE ===")
