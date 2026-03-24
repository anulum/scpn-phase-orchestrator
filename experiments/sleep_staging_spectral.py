# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep staging using Paper 2 spectral physics
#
# Paper 2 (L2 Neurochemical) specifies NT-frequency mapping:
#   Wake = gamma bandpass (30-80 Hz)
#   NREM (N2/N3) = delta lowpass (<4 Hz dominant)
#   REM = theta+gamma multiband
#   N2 = sigma/spindle (12-16 Hz)
#
# Previous attempt used cross-band Kuramoto R → 28% accuracy.
# This attempt uses band-specific spectral power ratios.

"""Sleep staging from real EEG using spectral power (Paper 2 physics).

Usage:
    python experiments/sleep_staging_spectral.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
from mne.datasets import sleep_physionet
from scipy.signal import welch

STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Sleep stage ?": "Unknown",
}


def band_power(epoch: np.ndarray, fs: float, lo: float, hi: float) -> float:
    """Relative power in frequency band via Welch PSD."""
    freqs, psd = welch(epoch, fs=fs, nperseg=min(len(epoch), int(4 * fs)))
    band_mask = (freqs >= lo) & (freqs <= hi)
    total_mask = freqs <= 45.0  # up to 45 Hz
    bp = np.trapz(psd[band_mask], freqs[band_mask])
    tp = np.trapz(psd[total_mask], freqs[total_mask])
    return float(bp / tp) if tp > 0 else 0.0


def classify_by_spectral_power(
    delta: float,
    theta: float,
    alpha: float,
    sigma: float,
    beta: float,
    gamma: float,
) -> str:
    """Classify sleep stage from relative band powers.

    Based on Paper 2 NT-frequency mapping (Ch 2.2-2.3):
    - N3: delta dominant (>50% relative power)
    - N2: sigma/spindle peak + moderate delta
    - N1: theta dominant, low delta
    - REM: theta + low delta + some gamma
    - Wake: alpha/beta/gamma dominant, low delta
    """
    # N3: delta dominates
    if delta > 0.50:
        return "N3"

    # N2: sigma spindles + moderate delta
    if delta > 0.30 and sigma > 0.08:
        return "N2"

    # Wake: alpha + beta + gamma > delta + theta
    fast = alpha + beta + gamma
    slow = delta + theta
    if fast > slow and alpha > 0.15:
        return "Wake"

    # REM: theta prominent, low delta, some gamma
    if theta > 0.20 and delta < 0.35:
        return "REM"

    # N1: theta moderate, transitional
    if theta > 0.15:
        return "N1"

    # Default to N2 (most common non-wake stage)
    return "N2"


def main() -> None:
    print("Loading PhysioNet Sleep-EDF...")
    paths = sleep_physionet.age.fetch_data(subjects=[0], recording=[1])
    raw = mne.io.read_raw_edf(paths[0][0], preload=True, verbose=False)
    annot = mne.read_annotations(paths[0][1])

    fs = raw.info["sfreq"]
    eeg = raw.get_data(picks=["EEG Fpz-Cz"])[0]
    print(f"  EEG: {len(eeg)} samples, {fs} Hz")

    epoch_len = int(30 * fs)
    n_epochs = len(eeg) // epoch_len

    # Expert hypnogram
    expert = ["Unknown"] * n_epochs
    for a in annot:
        stage = STAGE_MAP.get(a["description"], "Unknown")
        if stage == "Unknown":
            continue
        start_e = int(a["onset"] / 30)
        dur_e = max(1, int(a["duration"] / 30))
        for e in range(start_e, min(start_e + dur_e, n_epochs)):
            expert[e] = stage

    # Classify each epoch
    spo_stages = []
    powers = {
        "delta": [],
        "theta": [],
        "alpha": [],
        "sigma": [],
        "beta": [],
        "gamma": [],
    }

    for e in range(n_epochs):
        epoch = eeg[e * epoch_len : (e + 1) * epoch_len]
        d = band_power(epoch, fs, 0.5, 4.0)
        t = band_power(epoch, fs, 4.0, 8.0)
        a = band_power(epoch, fs, 8.0, 13.0)
        s = band_power(epoch, fs, 12.0, 16.0)
        b = band_power(epoch, fs, 16.0, 30.0)
        g = band_power(epoch, fs, 30.0, 45.0)

        powers["delta"].append(d)
        powers["theta"].append(t)
        powers["alpha"].append(a)
        powers["sigma"].append(s)
        powers["beta"].append(b)
        powers["gamma"].append(g)

        stage = classify_by_spectral_power(d, t, a, s, b, g)
        spo_stages.append(stage)

    # Score
    valid = [e != "Unknown" for e in expert]
    total_valid = sum(valid)
    correct = sum(1 for i in range(n_epochs) if valid[i] and spo_stages[i] == expert[i])
    accuracy = correct / total_valid if total_valid > 0 else 0.0

    per_stage = {}
    for stage in ["Wake", "N1", "N2", "N3", "REM"]:
        t_count = sum(1 for i in range(n_epochs) if valid[i] and expert[i] == stage)
        c_count = sum(
            1
            for i in range(n_epochs)
            if valid[i] and expert[i] == stage and spo_stages[i] == stage
        )
        per_stage[stage] = {
            "accuracy": round(c_count / t_count, 4) if t_count > 0 else 0.0,
            "correct": c_count,
            "total": t_count,
        }

    # Confusion matrix counts
    confusion = {}
    for true_s in ["Wake", "N1", "N2", "N3", "REM"]:
        confusion[true_s] = {}
        for pred_s in ["Wake", "N1", "N2", "N3", "REM"]:
            confusion[true_s][pred_s] = sum(
                1
                for i in range(n_epochs)
                if valid[i] and expert[i] == true_s and spo_stages[i] == pred_s
            )

    print(f"\n{'=' * 60}")
    print("Sleep Staging — Spectral Power (Paper 2 Physics)")
    print(f"{'=' * 60}")
    print(f"Total epochs: {n_epochs}, valid: {total_valid}")
    print(f"Overall accuracy: {accuracy:.1%} ({correct}/{total_valid})")
    print("\nPer-stage:")
    for stage in ["Wake", "N1", "N2", "N3", "REM"]:
        ps = per_stage[stage]
        print(f"  {stage:>5}: {ps['accuracy']:.1%} ({ps['correct']}/{ps['total']})")

    print("\nConfusion matrix:")
    print(f"{'True/Pred':>10}", end="")
    for s in ["Wake", "N1", "N2", "N3", "REM"]:
        print(f"{s:>6}", end="")
    print()
    for true_s in ["Wake", "N1", "N2", "N3", "REM"]:
        print(f"{true_s:>10}", end="")
        for pred_s in ["Wake", "N1", "N2", "N3", "REM"]:
            print(f"{confusion[true_s][pred_s]:>6}", end="")
        print()

    print("\nMean band powers:")
    for band in ["delta", "theta", "alpha", "sigma", "beta", "gamma"]:
        print(f"  {band:>6}: {np.mean(powers[band]):.4f}")

    results = {
        "method": "spectral_power_paper2",
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total_valid": total_valid,
        "per_stage": per_stage,
        "confusion": confusion,
        "mean_powers": {b: round(float(np.mean(v)), 4) for b, v in powers.items()},
    }

    with Path("experiments/sleep_staging_spectral_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/sleep_staging_spectral_results.json")
    print("\nPrevious (Kuramoto R): 28.0%")
    print(f"Current (Spectral power): {accuracy:.1%}")


if __name__ == "__main__":
    main()
