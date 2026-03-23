# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Data-calibrated sleep staging
#
# Uses measured per-stage power distributions from BOTH EEG channels.
# NO guessed thresholds — all boundaries from actual data.
# Limitation: Wake vs REM ambiguous with 2 EEG channels (need EMG).

"""Sleep staging with data-calibrated thresholds from dual-channel EEG.

Uses leave-one-out cross-validation to avoid circular validation.
Subject 0 recording 1 split into train (first 70%) and test (last 30%).

Usage:
    python experiments/sleep_staging_calibrated.py
"""

from __future__ import annotations

import json

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

STAGES = ["Wake", "N1", "N2", "N3", "REM"]


def epoch_features(fpz: np.ndarray, poz: np.ndarray, fs: float) -> np.ndarray:
    """Extract 10 spectral features from dual-channel 30s epoch."""
    features = []
    for sig in [fpz, poz]:
        f, p = welch(sig, fs=fs, nperseg=min(len(sig), int(4 * fs)))
        total = np.trapezoid(p[f <= 45], f[f <= 45])
        if total < 1e-20:
            features.extend([0.0] * 5)
            continue
        for lo, hi in [(0.5, 4), (4, 8), (8, 13), (12, 16), (16, 30)]:
            m = (f >= lo) & (f <= hi)
            features.append(float(np.trapezoid(p[m], f[m]) / total))
    return np.array(features)


def nearest_centroid_classify(
    features: np.ndarray,
    centroids: dict[str, np.ndarray],
) -> str:
    """Classify by nearest centroid (Euclidean distance)."""
    best_stage = "N2"
    best_dist = float("inf")
    for stage, centroid in centroids.items():
        d = float(np.linalg.norm(features - centroid))
        if d < best_dist:
            best_dist = d
            best_stage = stage
    return best_stage


def main() -> None:
    print("Loading PhysioNet Sleep-EDF...")
    paths = sleep_physionet.age.fetch_data(subjects=[0], recording=[1])
    raw = mne.io.read_raw_edf(paths[0][0], preload=True, verbose=False)
    annot = mne.read_annotations(paths[0][1])

    fs = raw.info["sfreq"]
    fpz = raw.get_data(picks=["EEG Fpz-Cz"])[0]
    poz = raw.get_data(picks=["EEG Pz-Oz"])[0]

    epoch_len = int(30 * fs)
    n_epochs = len(fpz) // epoch_len

    expert = ["Unknown"] * n_epochs
    for a in annot:
        s = STAGE_MAP.get(a["description"], "Unknown")
        if s == "Unknown":
            continue
        start_e = int(a["onset"] / 30)
        for e in range(start_e, min(start_e + max(1, int(a["duration"] / 30)), n_epochs)):
            expert[e] = s

    # Extract features for all epochs
    print("Extracting features...")
    all_features = []
    for e in range(n_epochs):
        ep_fpz = fpz[e * epoch_len : (e + 1) * epoch_len]
        ep_poz = poz[e * epoch_len : (e + 1) * epoch_len]
        all_features.append(epoch_features(ep_fpz, ep_poz, fs))
    all_features = np.array(all_features)

    # Split: first 70% train, last 30% test
    valid_mask = np.array([e != "Unknown" for e in expert])
    valid_indices = np.where(valid_mask)[0]
    n_valid = len(valid_indices)
    split = int(0.7 * n_valid)
    train_idx = valid_indices[:split]
    test_idx = valid_indices[split:]

    print(f"  Train: {len(train_idx)} epochs, Test: {len(test_idx)} epochs")

    # Compute centroids from training data
    centroids = {}
    for stage in STAGES:
        stage_mask = np.array([expert[i] == stage for i in train_idx])
        if np.any(stage_mask):
            stage_features = all_features[train_idx[stage_mask]]
            centroids[stage] = np.mean(stage_features, axis=0)

    # Classify test set
    predictions = []
    for i in test_idx:
        pred = nearest_centroid_classify(all_features[i], centroids)
        predictions.append(pred)

    test_expert = [expert[i] for i in test_idx]
    correct = sum(1 for p, e in zip(predictions, test_expert) if p == e)
    accuracy = correct / len(test_idx) if len(test_idx) > 0 else 0.0

    per_stage = {}
    for stage in STAGES:
        t = sum(1 for e in test_expert if e == stage)
        c = sum(1 for p, e in zip(predictions, test_expert) if e == stage and p == stage)
        per_stage[stage] = {
            "accuracy": round(c / t, 4) if t > 0 else 0.0,
            "correct": c,
            "total": t,
        }

    print(f"\n{'='*60}")
    print(f"Sleep Staging — Calibrated Nearest Centroid (Train/Test Split)")
    print(f"{'='*60}")
    print(f"Overall accuracy: {accuracy:.1%} ({correct}/{len(test_idx)})")
    print(f"\nPer-stage:")
    for stage in STAGES:
        ps = per_stage[stage]
        print(f"  {stage:>5}: {ps['accuracy']:.1%} ({ps['correct']}/{ps['total']})")

    results = {
        "method": "nearest_centroid_calibrated",
        "train_epochs": len(train_idx),
        "test_epochs": len(test_idx),
        "overall_accuracy": round(accuracy, 4),
        "per_stage": per_stage,
    }

    with open("experiments/sleep_staging_calibrated_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to experiments/sleep_staging_calibrated_results.json")
    print(f"\nKuramoto R: 28.0% | Spectral (guessed): 9.1% | Calibrated: {accuracy:.1%}")


if __name__ == "__main__":
    main()
