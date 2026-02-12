#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feature engineering for PPG-based drowsiness detection.
- Uses ULTRA-SHORT (10s) windows.
- Up-samples ~5 Hz PPG data to 200 Hz for HRV estimation.
- Forward-fills sparse labels.
- Includes window_end_elapsed column.
- Enforces strict column output order.
"""

from __future__ import annotations

import sys
import warnings
from typing import Any, List, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# Try importing NeuroKit2
try:
    import neurokit2 as nk
    _HAS_NEUROKIT = True
except ImportError:
    _HAS_NEUROKIT = False
    print("[WARN] NeuroKit2 not found. 'Raw Signal' mode will not work.")


# ====================== CONFIGURATION ======================
INPUT_CSV_PATH = r"src/drowsiness_detection_ros2/src/drowsiness_detection_pkg/drowsiness_detection/heart_rate/Datasets/physiotagger_full_2025-12-07T12-51-14.csv"
OUTPUT_CSV_PATH = r"src/drowsiness_detection_ros2/src/drowsiness_detection_pkg/drowsiness_detection/heart_rate/Datasets/engineered_features_with_hrv_hr_time.csv"

# Column Mapping (PhysioTagger -> Internal)
COL_MAP = {
    "Timestamp": "Timestamp",
    "Label": "Label",
    "PPG": "PPG",
    "Smooth_BPM": "Smooth_BPM",
    "Instant_BPM": "Instant_BPM",
    "Elapsed": "Elapsed(s)" # New mapping for elapsed time
}

# Settings
TARGET_SAMPLING_RATE = 200  # Hz after up-sampling
WINDOW_SECONDS = 10         # 10-second windows
STEP_SECONDS = 10           # non-overlapping

# ====================== TARGET COLUMN ORDER ======================
FINAL_COLUMN_ORDER = [
    # 1. Time Metadata
    "window_start_time",
    "window_end_time",
    "window_end_elapsed",  # <--- Added Here

    # 2. Smooth BPM Statistics
    "Smooth_BPM_mean",
    "Smooth_BPM_std",
    "Smooth_BPM_min",
    "Smooth_BPM_max",
    "Smooth_BPM",       # current / last Smooth_BPM in window
    "Instant_BPM",      # current / last Instant_BPM in window

    # 3. HRV Features
    "HRV_MeanNN", "HRV_SDNN", "HRV_SDANN1", "HRV_SDNNI1", "HRV_SDANN2", "HRV_SDNNI2",
    "HRV_SDANN5", "HRV_SDNNI5", "HRV_RMSSD", "HRV_SDSD", "HRV_CVNN", "HRV_CVSD",
    "HRV_MedianNN", "HRV_MadNN", "HRV_MCVNN", "HRV_IQRNN", "HRV_SDRMSSD", "HRV_Prc20NN",
    "HRV_Prc80NN", "HRV_pNN50", "HRV_pNN20", "HRV_MinNN", "HRV_MaxNN", "HRV_HTI",
    "HRV_TINN", "HRV_VLF", "HRV_LF", "HRV_HF", "HRV_VHF", "HRV_TP", "HRV_LFHF",
    "HRV_LFn", "HRV_HFn", "HRV_LnHF", "HRV_SD1", "HRV_SD2", "HRV_SD1SD2", "HRV_S",
    "HRV_CSI", "HRV_CVI", "HRV_CSI_Modified", "HRV_PIP", "HRV_IALS", "HRV_PSS",
    "HRV_PAS", "HRV_GI", "HRV_SI", "HRV_AI", "HRV_PI", "HRV_C1d", "HRV_C1a",
    "HRV_SD1d", "HRV_SD1a", "HRV_C2d", "HRV_C2a", "HRV_SD2d", "HRV_SD2a", "HRV_Cd",
    "HRV_Ca", "HRV_SDNNd", "HRV_SDNNa", "HRV_DFA_alpha1", "HRV_MFDFA_alpha1_Width",
    "HRV_MFDFA_alpha1_Peak", "HRV_MFDFA_alpha1_Mean", "HRV_MFDFA_alpha1_Max",
    "HRV_MFDFA_alpha1_Delta", "HRV_MFDFA_alpha1_Asymmetry", "HRV_MFDFA_alpha1_Fluctuation",
    "HRV_MFDFA_alpha1_Increment", "HRV_DFA_alpha2", "HRV_MFDFA_alpha2_Width",
    "HRV_MFDFA_alpha2_Peak", "HRV_MFDFA_alpha2_Mean", "HRV_MFDFA_alpha2_Max",
    "HRV_MFDFA_alpha2_Delta", "HRV_MFDFA_alpha2_Asymmetry", "HRV_MFDFA_alpha2_Fluctuation",
    "HRV_MFDFA_alpha2_Increment", "HRV_ApEn", "HRV_SampEn", "HRV_ShanEn", "HRV_FuzzyEn",
    "HRV_MSEn", "HRV_CMSEn", "HRV_RCMSEn", "HRV_CD", "HRV_HFD", "HRV_KFD", "HRV_LZC",

    # 4. Label
    "Label",
]
# =================================================================


def upsample_signal(signal: np.ndarray, original_fs: float, target_fs: float) -> np.ndarray:
    """Interpolates signal to higher frequency."""
    if original_fs >= target_fs:
        return signal

    duration = len(signal) / original_fs
    t_original = np.linspace(0, duration, len(signal))
    new_length = int(duration * target_fs)
    t_new = np.linspace(0, duration, new_length)

    f = interp1d(t_original, signal, kind="cubic", fill_value="extrapolate")
    return f(t_new)


def process_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[INFO] Starting Processing Pipeline...")

    # 1. Preprocessing: sort by Timestamp and estimate sampling rate
    if COL_MAP["Timestamp"] in df.columns:
        df[COL_MAP["Timestamp"]] = pd.to_datetime(df[COL_MAP["Timestamp"]])
        df = df.sort_values(by=COL_MAP["Timestamp"]).reset_index(drop=True)

        time_diffs = df[COL_MAP["Timestamp"]].diff().dropna().dt.total_seconds()
        avg_diff = time_diffs.mean()
        actual_fs = 1 / avg_diff if avg_diff > 0 else 5.0
        print(f"[INFO] Estimated Input Sampling Rate: {actual_fs:.2f} Hz")
    else:
        actual_fs = 5.0
        print("[WARN] No Timestamp found. Assuming 5 Hz.")

    # 2. Forward-fill labels (and backfill first entries)
    lbl_col = COL_MAP["Label"]
    if lbl_col in df.columns:
        print("[INFO] Forward-filling labels...")
        df[lbl_col] = df[lbl_col].ffill()
        df[lbl_col] = df[lbl_col].bfill()

    # 3. Prepare window parameters
    window_size_orig = int(actual_fs * WINDOW_SECONDS)
    step_size_orig = int(actual_fs * STEP_SECONDS)

    print(f"[INFO] Analysis Config: Window={WINDOW_SECONDS}s, Target Rate={TARGET_SAMPLING_RATE}Hz")

    if len(df) < window_size_orig:
        print(f"[ERROR] Dataset too short ({len(df)} rows). Need {window_size_orig} rows.")
        return pd.DataFrame()

    ppg_col = COL_MAP["PPG"]
    bpm_col = COL_MAP["Smooth_BPM"]
    inst_col = COL_MAP["Instant_BPM"]
    elapsed_col = COL_MAP["Elapsed"]

    ppg_data = df[ppg_col].to_numpy(dtype=float) if ppg_col in df.columns else None

    features_list: List[dict] = []
    processed_count = 0

    # 4. Sliding window
    for start in range(0, len(df) - window_size_orig + 1, step_size_orig):
        end = start + window_size_orig
        row: dict[str, Any] = {}

        # Time metadata
        if COL_MAP["Timestamp"] in df.columns:
            row["window_start_time"] = df[COL_MAP["Timestamp"]].iloc[start]
            row["window_end_time"] = df[COL_MAP["Timestamp"]].iloc[end - 1]
        
        # Elapsed time (End of window)
        if elapsed_col in df.columns:
            row["window_end_elapsed"] = df[elapsed_col].iloc[end - 1]

        # Label (mode within window)
        if lbl_col in df.columns:
            try:
                row["Label"] = df[lbl_col].iloc[start:end].mode()[0]
            except Exception:
                pass

        # Smooth BPM statistics and current value
        if bpm_col in df.columns:
            vals = df[bpm_col].iloc[start:end].values
            row["Smooth_BPM_mean"] = float(np.nanmean(vals))
            row["Smooth_BPM_std"] = float(np.nanstd(vals))
            row["Smooth_BPM_min"] = float(np.nanmin(vals))
            row["Smooth_BPM_max"] = float(np.nanmax(vals))
            row["Smooth_BPM"] = float(vals[-1])

        # Instant BPM as raw value (no stats, just last in window)
        if inst_col in df.columns:
            inst_vals = df[inst_col].iloc[start:end].values
            row["Instant_BPM"] = float(inst_vals[-1])

        # HRV with up-sampling
        if _HAS_NEUROKIT and ppg_data is not None:
            win_raw = ppg_data[start:end]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    if actual_fs < TARGET_SAMPLING_RATE:
                        win_proc = upsample_signal(win_raw, actual_fs, TARGET_SAMPLING_RATE)
                        proc_fs = TARGET_SAMPLING_RATE
                    else:
                        win_proc = win_raw
                        proc_fs = actual_fs

                    signals, info = nk.ppg_process(win_proc, sampling_rate=proc_fs)
                    hrv_df = nk.hrv(info, sampling_rate=proc_fs, show=False)

                    if not hrv_df.empty:
                        for col in hrv_df.columns:
                            row[col] = hrv_df.iloc[0][col]
            except Exception:
                # HRV failure for this window; keep BPM and label features only
                pass

        features_list.append(row)
        processed_count += 1
        if processed_count % 10 == 0:
            print(f"[INFO] Processed {processed_count} windows...")

    return pd.DataFrame(features_list)


def main():
    print(f"Loading: {INPUT_CSV_PATH}")

    try:
        df = pd.read_csv(INPUT_CSV_PATH)
    except Exception as e:
        print(f"[ERROR] Could not load CSV: {e}")
        sys.exit(1)

    out_df = process_features(df)

    if out_df.empty:
        print("[ERROR] No features generated.")
        sys.exit(1)

    # Ensure all desired columns exist, then order them
    for col in FINAL_COLUMN_ORDER:
        if col not in out_df.columns:
            out_df[col] = np.nan

    out_df = out_df[FINAL_COLUMN_ORDER]

    # Fill remaining NaNs sensibly
    out_df = out_df.fillna(method="ffill").fillna(method="bfill").fillna(0)

    try:
        Path(OUTPUT_CSV_PATH).parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print(f"\n[SUCCESS] Saved {len(out_df)} rows.")
        print(f"Location: {OUTPUT_CSV_PATH}")
    except Exception as e:
        print(f"[ERROR] Save failed: {e}")


if __name__ == "__main__":
    main()
