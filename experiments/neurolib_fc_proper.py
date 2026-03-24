# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Proper neurolib FC baseline
#
# Previous experiment (neurolib_baseline.py) got r=0.20 with:
# - 60s simulation (too short, BOLD needs 5+ minutes)
# - Crude 1s averaging (no HRF convolution)
# - Coarse K sweep
#
# This version uses:
# - 300s simulation (matches Deco et al. 2018)
# - Balloon-Windkessel BOLD model (neurolib.models.bold)
# - Finer K sweep around the promising range
# - Proper bandpass filtering after HRF

"""Proper neurolib ALN → BOLD → FC correlation.

Usage:
    python experiments/neurolib_fc_proper.py
    python experiments/neurolib_fc_proper.py --coupling 2.0
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr

from neurolib.models.aln import ALNModel
from neurolib.models.bold import BOLDModel
from neurolib.utils.loadData import Dataset


def bandpass_bold(signal, fs, low=0.01, high=0.1):
    nyq = fs / 2
    if low / nyq >= 1.0 or high / nyq >= 1.0:
        return signal
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=1)


def fc_upper_triangle(fc1, fc2):
    n = fc1.shape[0]
    triu = np.triu_indices(n, k=1)
    r, p = pearsonr(fc1[triu], fc2[triu])
    return float(r), float(p), len(triu[0])


def run_one(ds, K, duration_ms, discard_ms=10000):
    """Run ALN at coupling K, return BOLD FC and wall time."""
    model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
    model.params["duration"] = duration_ms
    model.params["Ke_gl"] = K

    t0 = time.perf_counter()
    model.run()
    sim_time = time.perf_counter() - t0

    rates = model.rates_exc
    if rates is None or rates.shape[1] < 1000:
        return None, sim_time

    N = rates.shape[0]
    dt = model.params["dt"]

    # Balloon-Windkessel BOLD model
    bold_model = BOLDModel(N=N, dt=dt)
    bold_model.run(rates)

    bold = bold_model.BOLD
    t_bold = bold_model.t_BOLD

    if bold.shape[1] < 20:
        return None, sim_time

    # Discard initial transient
    discard_samples = max(0, int(discard_ms / (t_bold[1] - t_bold[0]))) if len(t_bold) > 1 else 0
    bold = bold[:, discard_samples:]

    if bold.shape[1] < 20:
        return None, sim_time

    # Compute TR from BOLD timestamps
    if len(t_bold) > 1:
        tr = (t_bold[-1] - t_bold[0]) / (len(t_bold) - 1) / 1000.0  # ms → s
    else:
        tr = 2.0
    fs = 1.0 / tr

    # Bandpass 0.01-0.1 Hz
    if bold.shape[1] > 30 and fs > 0.2:
        try:
            bold_filt = bandpass_bold(bold, fs)
        except Exception:
            bold_filt = bold
    else:
        bold_filt = bold

    sim_fc = np.corrcoef(bold_filt)
    np.fill_diagonal(sim_fc, 0.0)

    return {
        "sim_fc": sim_fc,
        "n_bold_timepoints": bold.shape[1],
        "tr_s": round(tr, 3),
        "fs_hz": round(fs, 3),
    }, sim_time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=300000, help="Duration in ms (default 300s)")
    parser.add_argument("--coupling", type=float, default=None, help="Single K value")
    parser.add_argument("--output", type=str, default="experiments/neurolib_fc_proper_results.json")
    args = parser.parse_args()

    print("Loading HCP data...")
    ds = Dataset("hcp")
    emp_fc = np.mean(ds.FCs, axis=0)
    np.fill_diagonal(emp_fc, 0.0)
    print(f"  SC: {ds.Cmat.shape}, FC: {emp_fc.shape}, subjects: {len(ds.FCs)}")

    if args.coupling is not None:
        K_values = [args.coupling]
    else:
        # Finer sweep around K=2.0 (best from previous experiment)
        K_values = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

    results = []
    for K in K_values:
        print(f"\nRunning ALN: K={K}, duration={args.duration}ms...")
        out, wall_time = run_one(ds, K, args.duration)

        if out is None:
            print(f"  FAILED: insufficient data")
            continue

        r, p, n_pairs = fc_upper_triangle(out["sim_fc"], emp_fc)
        entry = {
            "K": K,
            "fc_correlation": round(r, 4),
            "p_value": p,
            "n_pairs": n_pairs,
            "duration_ms": args.duration,
            "n_bold_timepoints": out["n_bold_timepoints"],
            "tr_s": out["tr_s"],
            "wall_time_s": round(wall_time, 1),
            "method": "Balloon-Windkessel BOLD + bandpass 0.01-0.1 Hz",
        }
        results.append(entry)
        print(f"  FC correlation: r={r:.4f} (p={p:.2e}), {out['n_bold_timepoints']} BOLD timepoints")

    if results:
        best = max(results, key=lambda r: r["fc_correlation"])
        print(f"\n{'='*60}")
        print(f"Best: K={best['K']}, r={best['fc_correlation']:.4f}")
        print(f"Literature target: r~0.72 (Deco et al. 2018)")
        gap = 0.72 - best["fc_correlation"]
        print(f"Gap: {gap:.4f}")
        print(f"{'='*60}")

    if args.output and results:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
