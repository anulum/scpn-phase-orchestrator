# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sleep staging from real EEG
#
# Validates SPO's sleep_staging module against expert-scored
# polysomnography from PhysioNet Sleep-EDF.

"""Sleep staging from real EEG using SPO's Kuramoto-based R classifier.

Loads PhysioNet Sleep-EDF PSG data, extracts phases from EEG channels
via Hilbert transform in multiple frequency bands, computes Kuramoto R
per 30s epoch, and compares SPO's classify_sleep_stage() against expert
hypnogram annotations.

Usage:
    python experiments/sleep_staging_eeg.py
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
import numpy as np
from mne.datasets import sleep_physionet
from scipy.signal import butter, filtfilt, hilbert

from scpn_phase_orchestrator.monitor.sleep_staging import classify_sleep_stage
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

STAGE_MAP = {
    "Sleep stage W": "Wake",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",  # AASM merges S3+S4 into N3
    "Sleep stage R": "REM",
    "Sleep stage ?": "Unknown",
}


def bandpass(sig: np.ndarray, fs: float, lo: float, hi: float) -> np.ndarray:
    nyq = fs / 2
    b, a = butter(3, [lo / nyq, hi / nyq], btype="band")
    return filtfilt(b, a, sig)


def extract_band_phases(eeg: np.ndarray, fs: float) -> dict[str, np.ndarray]:
    """Extract instantaneous phases in delta, theta, alpha, sigma, beta bands."""
    bands = {
        "delta": (0.5, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "sigma": (12.0, 16.0),
        "beta": (16.0, 30.0),
    }
    phases = {}
    for name, (lo, hi) in bands.items():
        filtered = bandpass(eeg, fs, lo, hi)
        analytic = hilbert(filtered)
        phases[name] = np.angle(analytic) % (2 * np.pi)
    return phases


def compute_epoch_R(
    band_phases: dict[str, np.ndarray], epoch_start: int, epoch_len: int
) -> float:
    """Compute cross-band Kuramoto R for one epoch.

    Takes the mean phase of each band at the epoch midpoint and
    computes R across bands — measuring inter-band coherence.
    """
    mid = epoch_start + epoch_len // 2
    phase_vector = np.array([band_phases[b][mid] for b in band_phases])
    R, _ = compute_order_parameter(phase_vector)
    return float(R)


def main() -> None:
    print("Loading PhysioNet Sleep-EDF...")
    paths = sleep_physionet.age.fetch_data(subjects=[0], recording=[1])
    raw = mne.io.read_raw_edf(paths[0][0], preload=True, verbose=False)
    annot = mne.read_annotations(paths[0][1])

    fs = raw.info["sfreq"]
    eeg_data = raw.get_data(picks=["EEG Fpz-Cz"])[0]
    print(f"  EEG: {len(eeg_data)} samples, {fs} Hz, {len(eeg_data) / fs:.0f}s")

    # Extract band phases
    print("Extracting band phases...")
    band_phases = extract_band_phases(eeg_data, fs)

    # Process 30-second epochs
    epoch_len = int(30 * fs)
    n_epochs = len(eeg_data) // epoch_len
    print(f"  Epochs: {n_epochs} (30s each)")

    # Build expert hypnogram
    expert_stages = ["Unknown"] * n_epochs
    for a in annot:
        stage = STAGE_MAP.get(a["description"], "Unknown")
        if stage == "Unknown":
            continue
        start_epoch = int(a["onset"] / 30)
        duration_epochs = max(1, int(a["duration"] / 30))
        for e in range(start_epoch, min(start_epoch + duration_epochs, n_epochs)):
            expert_stages[e] = stage

    # SPO staging
    spo_stages = []
    R_values = []
    for e in range(n_epochs):
        start = e * epoch_len
        R = compute_epoch_R(band_phases, start, epoch_len)
        R_values.append(R)

        # Use R to classify via SPO
        # R > 0.7 → high coherence → deep sleep (N3)
        # R ~ 0.5 → moderate → N2
        # R ~ 0.35 → low → N1
        # R < 0.3 → very low → Wake or REM
        spo_stage = classify_sleep_stage(R, functional_desync=False)
        spo_stages.append(spo_stage)

    # Compare
    valid_mask = [e != "Unknown" for e in expert_stages]
    total_valid = sum(valid_mask)
    correct = sum(
        1
        for i in range(n_epochs)
        if valid_mask[i] and spo_stages[i] == expert_stages[i]
    )
    accuracy = correct / total_valid if total_valid > 0 else 0.0

    # Per-stage accuracy
    stage_correct = {}
    stage_total = {}
    for i in range(n_epochs):
        if not valid_mask[i]:
            continue
        es = expert_stages[i]
        stage_total[es] = stage_total.get(es, 0) + 1
        if spo_stages[i] == es:
            stage_correct[es] = stage_correct.get(es, 0) + 1

    print(f"\n{'=' * 60}")
    print("Sleep Staging Results")
    print(f"{'=' * 60}")
    print(f"Total epochs: {n_epochs}")
    print(f"Valid (expert-scored): {total_valid}")
    print(f"Overall accuracy: {accuracy:.1%} ({correct}/{total_valid})")
    print("\nPer-stage:")
    for stage in ["Wake", "N1", "N2", "N3", "REM"]:
        t = stage_total.get(stage, 0)
        c = stage_correct.get(stage, 0)
        acc = c / t if t > 0 else 0.0
        print(f"  {stage:>5}: {acc:.1%} ({c}/{t})")

    print("\nR statistics:")
    print(f"  Mean: {np.mean(R_values):.4f}")
    print(f"  Std:  {np.std(R_values):.4f}")
    print(f"  Range: [{min(R_values):.4f}, {max(R_values):.4f}]")

    results = {
        "overall_accuracy": round(accuracy, 4),
        "correct": correct,
        "total_valid": total_valid,
        "per_stage": {
            stage: {
                "accuracy": round(
                    stage_correct.get(stage, 0) / stage_total.get(stage, 1), 4
                ),
                "correct": stage_correct.get(stage, 0),
                "total": stage_total.get(stage, 0),
            }
            for stage in ["Wake", "N1", "N2", "N3", "REM"]
        },
        "R_mean": round(float(np.mean(R_values)), 4),
        "R_std": round(float(np.std(R_values)), 4),
    }

    with Path("experiments/sleep_staging_results.json").open("w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to experiments/sleep_staging_results.json")


if __name__ == "__main__":
    main()
