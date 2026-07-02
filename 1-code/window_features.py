"""
window_features.py — Window-based feature extraction for washing-machine cycles.

Motivation
----------
The original pipeline (extract_features.py) collapses each entire ~2-3 h wash
cycle into a SINGLE tsfresh feature vector, giving only ~108 labelled samples.
That is a very small, high-dimensional dataset.

This script slices every cycle's signals into short, non-overlapping time
windows and extracts the same tsfresh feature set PER WINDOW. One cycle then
yields many feature vectors, expanding the dataset by ~40-60x.

IMPORTANT (honesty / leakage)
-----------------------------
Windows from the same cycle are highly correlated and share the same label.
They are NOT independent samples. The output keeps a `cycle_id` column so the
model MUST be evaluated with GroupKFold grouped by cycle (never a plain shuffle
split) — otherwise near-duplicate windows leak between train and test and the
reported score becomes meaningless. Windowing multiplies training signal; it
does not create new independent fault *events* (there are still only ~20 of
those).

Sensors (matches WM_ExtractedFeatures.csv schema)
-------------------------------------------------
  ActP      — Active Power, from slow.csv (1 Hz)
  Current   — from fast.csv (~2 kHz, decimated on read)
  Vibration — from fast.csv (~2 kHz, decimated on read)

Data reality that shapes the design
-----------------------------------
slow.csv (ActP) spans the WHOLE ~3 h cycle at 1 Hz. fast.csv (Current,
Vibration) is a shorter high-frequency BURST — its length varies from ~30 s to
~3.5 h across cycles (median ~80 min) — and its timestamps are RELATIVE (start
at 0) with no absolute anchor, so the fast burst cannot be wall-clock aligned
to the slow signal.

Therefore we window the FAST burst (the discriminative Current/Vibration
signal) into fixed-length windows, and attach WHOLE-CYCLE ActP features to every
window of that cycle as low-frequency power context. Each output row is one fast
window: [Current/Vibration features for that window] + [cycle-level ActP
features] + cycle_id.

Output
------
  ../2-washing_machines/WM_WindowedFeatures.csv
  Columns: Id (window id = "<cycle>#w<k>"), cycle_id, ActP__*, Current__*,
  Vibration__* . Merge with metadata on cycle_id to attach labels.
"""

import os
import sys
import time
import glob

import numpy as np
import pandas as pd
from tsfresh import extract_features as tsfresh_extract
from tsfresh.utilities.dataframe_functions import impute

# ── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "2-washing_machines")
OUT_PATH = os.path.join(DATA_DIR, "WM_WindowedFeatures.csv")

WINDOW_SEC = 120          # window length in seconds (non-overlapping)
FAST_DECIMATE = 40        # keep every Nth fast-signal row (~2 kHz -> ~50 Hz)
MIN_WINDOW_FILL = 0.5     # drop trailing windows shorter than this fraction

# Same feature set as extract_features.py (25 stems x 3 sensors = 75 features).
FC_PARAMETERS = {
    "variance_larger_than_standard_deviation": None,
    "sum_values": None,
    "mean_abs_change": None,
    "mean_change": None,
    "median": None,
    "mean": None,
    "length": None,
    "standard_deviation": None,
    "variation_coefficient": None,
    "variance": None,
    "skewness": None,
    "kurtosis": None,
    "root_mean_square": None,
    "count_above_mean": None,
    "count_below_mean": None,
    "maximum": None,
    "absolute_maximum": None,
    "minimum": None,
    "number_peaks": [{"n": 5}],
    "fft_aggregated": [
        {"aggtype": "centroid"},
        {"aggtype": "variance"},
        {"aggtype": "skew"},
        {"aggtype": "kurtosis"},
    ],
    "fourier_entropy": [
        {"bins": 5},
        {"bins": 10},
    ],
}


def _extract(long_df, value_cols):
    """Run tsfresh on a long dataframe (Id, _t, <value_cols>), serial + quiet."""
    feats = tsfresh_extract(
        long_df[["Id", "_t"] + value_cols],
        column_id="Id",
        column_sort="_t",
        default_fc_parameters=FC_PARAMETERS,
        n_jobs=0,                 # serial: avoids Windows multiprocessing issues
        disable_progressbar=True,
    )
    return feats


def window_cycle(cycle_id):
    """Return a per-window feature dataframe for one cycle, or None if unusable.

    One row per FAST window (Current/Vibration features), with whole-cycle ActP
    features broadcast onto every window.
    """
    cdir = os.path.join(DATA_DIR, cycle_id)
    slow_path = os.path.join(cdir, "slow.csv")
    fast_path = os.path.join(cdir, "fast.csv")
    if not (os.path.exists(slow_path) and os.path.exists(fast_path)):
        return None

    # ── fast: Current + Vibration, decimated, windowed ─────────────────────
    fast = pd.read_csv(
        fast_path,
        usecols=["UnixTimestamp (us)", "Current", "Vibration"],
    )
    fast = fast.iloc[::FAST_DECIMATE].reset_index(drop=True)
    fast["_t"] = fast["UnixTimestamp (us)"] / 1e6
    fast["_t"] = fast["_t"] - fast["_t"].min()
    fast["win"] = (fast["_t"] // WINDOW_SEC).astype(int)

    # Drop a short trailing window that isn't sufficiently full.
    max_win = int(fast["win"].max())
    keep = list(range(max_win + 1))
    tail = fast.loc[fast["win"] == max_win, "_t"]
    if len(tail) and (tail.max() - tail.min()) < MIN_WINDOW_FILL * WINDOW_SEC:
        keep = keep[:-1]
    if not keep:
        return None
    fast = fast[fast["win"].isin(keep)].copy()
    fast["Id"] = cycle_id + "#w" + fast["win"].astype(str).str.zfill(3)

    f_fast = _extract(fast, ["Current", "Vibration"])   # index = window Id

    # ── slow: whole-cycle ActP features, broadcast to every window ─────────
    slow = pd.read_csv(slow_path, usecols=["Ts", "ActP"])
    slow["_t"] = slow["Ts"] - slow["Ts"].min()
    slow["Id"] = cycle_id                                # single group
    f_slow = _extract(slow, ["ActP"])                   # one row, index=cycle_id
    actp_row = f_slow.iloc[0]                            # Series of ActP__* values

    feats = f_fast.copy()
    for col, val in actp_row.items():
        feats[col] = val
    # Order columns ActP__*, Current__*, Vibration__* for schema parity.
    ordered = ([c for c in feats.columns if c.startswith("ActP__")]
               + [c for c in feats.columns if c.startswith("Current__")]
               + [c for c in feats.columns if c.startswith("Vibration__")])
    feats = feats[ordered]
    feats.insert(0, "cycle_id", cycle_id)
    feats.index.name = "Id"
    return feats.reset_index()


def main():
    cycle_dirs = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d))
    )
    limit = int(os.environ.get("CYCLE_LIMIT", "0"))
    if limit:
        cycle_dirs = cycle_dirs[:limit]

    print(f"Windowing {len(cycle_dirs)} cycles "
          f"(window={WINDOW_SEC}s, fast_decimate={FAST_DECIMATE})", flush=True)

    all_feats = []
    t0 = time.time()
    for i, cid in enumerate(cycle_dirs, 1):
        try:
            tc = time.time()
            feats = window_cycle(cid)
            if feats is None or feats.empty:
                print(f"[{i}/{len(cycle_dirs)}] {cid}: skipped", flush=True)
                continue
            all_feats.append(feats)
            print(f"[{i}/{len(cycle_dirs)}] {cid}: {len(feats)} windows "
                  f"({time.time()-tc:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad cycle shouldn't kill the run
            print(f"[{i}/{len(cycle_dirs)}] {cid}: ERROR {e}", flush=True)

    if not all_feats:
        print("No features produced.", flush=True)
        return

    out = pd.concat(all_feats, ignore_index=True)
    # tsfresh can emit NaN/inf for degenerate windows; impute to keep it model-ready.
    feat_cols = [c for c in out.columns if c not in ("Id", "cycle_id")]
    out[feat_cols] = impute(out[feat_cols])
    out.to_csv(OUT_PATH, index=False)

    print(f"\nDONE in {time.time()-t0:.1f}s", flush=True)
    print(f"  rows (windows): {len(out)}", flush=True)
    print(f"  cycles: {out['cycle_id'].nunique()}", flush=True)
    print(f"  feature cols: {len(feat_cols)}", flush=True)
    print(f"  -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
