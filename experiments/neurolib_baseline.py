# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — neurolib ALN baseline experiment
#
# Establish the ground truth: neurolib's ALN model on HCP achieves ~72%
# FC correlation (Deco et al. 2018). We run it here to verify and then
# feed its output into SPO's supervision layer.

"""Run neurolib ALN on HCP, compute FC correlation with empirical data.

Usage:
    python experiments/neurolib_baseline.py
    python experiments/neurolib_baseline.py --duration 60000 --output results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from neurolib.models.aln import ALNModel
from neurolib.utils.loadData import Dataset
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr


def bandpass_bold(
    signal: np.ndarray, fs: float, low: float = 0.01, high: float = 0.1
) -> np.ndarray:
    """Bandpass filter to BOLD frequency range."""
    nyq = fs / 2
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal, axis=1)


def compute_fc(bold: np.ndarray) -> np.ndarray:
    """Pearson correlation FC matrix."""
    fc = np.corrcoef(bold)
    np.fill_diagonal(fc, 0.0)
    return fc


def compare_fc(sim_fc: np.ndarray, emp_fc: np.ndarray) -> dict:
    """Upper-triangle correlation between simulated and empirical FC."""
    n = sim_fc.shape[0]
    triu = np.triu_indices(n, k=1)
    r, p = pearsonr(sim_fc[triu], emp_fc[triu])
    return {
        "correlation": round(float(r), 4),
        "p_value": float(p),
        "n_pairs": len(triu[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duration", type=int, default=60000, help="Sim duration in ms"
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=None,
        help="Global coupling (default: auto-sweep)",
    )
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("Loading HCP data...")
    ds = Dataset("hcp")
    emp_fc = np.mean(ds.FCs, axis=0)
    np.fill_diagonal(emp_fc, 0.0)
    print(f"  SC: {ds.Cmat.shape}, FC: {emp_fc.shape}, subjects: {len(ds.FCs)}")

    if args.coupling is not None:
        K_values = [args.coupling]
    else:
        K_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]

    results = []
    for K in K_values:
        print(f"\nRunning ALN: K={K}, duration={args.duration}ms...")
        model = ALNModel(Cmat=ds.Cmat, Dmat=ds.Dmat)
        model.params["duration"] = args.duration
        model.params["Ke_gl"] = K

        t0 = time.perf_counter()
        model.run()
        elapsed = time.perf_counter() - t0
        print(f"  Simulation: {elapsed:.1f}s")

        # Extract BOLD-like signal (excitatory rate, downsampled)
        rates = model.rates_exc  # (80, T)
        if rates is None or rates.shape[1] < 100:
            print(f"  WARNING: rates too short ({rates.shape})")
            continue

        # Downsample to ~1 Hz (BOLD timescale)
        ds_factor = max(1, int(1000 / model.params["dt"]))  # 1s windows
        n_t = rates.shape[1]
        n_samples = n_t // ds_factor
        if n_samples < 10:
            print(f"  WARNING: too few samples after downsample ({n_samples})")
            continue
        bold_ds = (
            rates[:, : n_samples * ds_factor]
            .reshape(80, n_samples, ds_factor)
            .mean(axis=2)
        )

        # Bandpass filter if enough samples
        fs = 1.0  # 1 Hz after downsample
        if n_samples > 30:
            try:
                bold_filt = bandpass_bold(bold_ds, fs, low=0.01, high=0.1)
            except Exception:
                bold_filt = bold_ds
        else:
            bold_filt = bold_ds

        sim_fc = compute_fc(bold_filt)
        result = compare_fc(sim_fc, emp_fc)
        result["K"] = K
        result["duration_ms"] = args.duration
        result["wall_time_s"] = round(elapsed, 2)
        result["n_timepoints"] = n_samples
        results.append(result)

        print(f"  FC: r={result['correlation']:.4f} (p={result['p_value']:.2e})")

    if results:
        best = max(results, key=lambda r: r["correlation"])
        print(f"\n{'=' * 60}")
        print(f"Best: K={best['K']}, r={best['correlation']:.4f}")
        print("Target: r~0.72 (Deco et al. 2018)")
        print(f"{'=' * 60}")

    if args.output and results:
        with Path(args.output).open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
