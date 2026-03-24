# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Seizure detection on CHB-MIT EEG
#
# SPO regime FSM on real multi-channel EEG. If Kuramoto R changes
# significantly around seizure onset, SPO detects the transition.
# Dataset: CHB-MIT Scalp EEG (PhysioNet), 23 channels, 256 Hz.

"""Seizure detection experiment using SPO supervision on CHB-MIT EEG.

Downloads EEG records with known seizure annotations from PhysioNet,
computes multi-channel phase synchrony (Kuramoto R), and tests whether
SPO regime transitions correlate with seizure onset.

Usage:
    python experiments/seizure_detection_chbmit.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import mne
import numpy as np
from scipy.signal import hilbert

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.events import EventBus
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

PHYSIONET_BASE = "https://physionet.org/files/chbmit/1.0.0"
# Window: 2s at 256 Hz = 512 samples
WINDOW_S = 2.0
FS = 256


def parse_summary(subject: str) -> list[dict]:
    """Parse seizure annotations from CHB-MIT summary file."""
    url = f"{PHYSIONET_BASE}/{subject}/{subject}-summary.txt"
    text = urllib.request.urlopen(url, timeout=30).read().decode()

    seizures = []
    current_file = None
    for line in text.splitlines():
        m = re.match(r"File Name:\s+(\S+)", line)
        if m:
            current_file = m.group(1)
        m = re.match(r"Seizure\s+\d*\s*Start Time:\s+(\d+)", line)
        if not m:
            m = re.match(r"Seizure Start Time:\s+(\d+)", line)
        if m and current_file:
            start = int(m.group(1))
            continue
        m = re.match(r"Seizure\s+\d*\s*End Time:\s+(\d+)", line)
        if not m:
            m = re.match(r"Seizure End Time:\s+(\d+)", line)
        if m and current_file:
            end = int(m.group(1))
            seizures.append(
                {
                    "file": current_file,
                    "start_s": start,
                    "end_s": end,
                }
            )
    return seizures


def download_edf(subject: str, filename: str, cache_dir: Path) -> Path:
    """Download EDF file if not cached."""
    local = cache_dir / filename
    if local.exists():
        return local
    local.parent.mkdir(parents=True, exist_ok=True)
    url = f"{PHYSIONET_BASE}/{subject}/{filename}"
    print(f"  Downloading {filename}...")
    urllib.request.urlretrieve(url, local)
    return local


def load_eeg(edf_path: Path) -> np.ndarray:
    """Load EEG from EDF, return (n_channels, n_samples)."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    # Bandpass 1-70 Hz to remove DC and high-freq noise
    raw.filter(1.0, 70.0, verbose=False)
    return raw.get_data()


def compute_windowed_R(
    eeg: np.ndarray, fs: float, window_s: float
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Kuramoto R in sliding windows.

    Returns (R_values, window_center_times).
    """
    n_ch, n_t = eeg.shape
    window = int(window_s * fs)
    step = window // 2  # 50% overlap
    n_windows = (n_t - window) // step + 1

    R_vals = np.zeros(n_windows)
    times = np.zeros(n_windows)

    for w in range(n_windows):
        start = w * step
        end = start + window
        segment = eeg[:, start:end]

        # Extract phases via Hilbert per channel
        phases_at_end = np.zeros(n_ch)
        for ch in range(n_ch):
            analytic = hilbert(segment[ch])
            phases_at_end[ch] = np.angle(analytic[-1]) % (2 * np.pi)

        R, _ = compute_order_parameter(phases_at_end)
        R_vals[w] = R
        times[w] = (start + end) / 2.0 / fs

    return R_vals, times


def detect_transitions(R_vals: np.ndarray, times: np.ndarray) -> list[dict]:
    """Run SPO regime FSM on R trajectory, return transitions."""
    event_bus = EventBus()
    regime_mgr = RegimeManager(event_bus=event_bus)

    transitions = []
    for i, R in enumerate(R_vals):
        layer_state = LayerState(R=float(R), psi=0.0)
        upde_state = UPDEState(
            layers=[layer_state],
            cross_layer_alignment=np.eye(1),
            stability_proxy=float(R),
            regime_id=regime_mgr.current_regime.value,
        )
        proposed = regime_mgr.evaluate(upde_state, BoundaryState())
        old = regime_mgr.current_regime
        regime_mgr.transition(proposed)
        if regime_mgr.current_regime != old:
            transitions.append(
                {
                    "time_s": round(float(times[i]), 2),
                    "from": old.name,
                    "to": regime_mgr.current_regime.name,
                    "R": round(float(R), 4),
                }
            )

    return transitions


def evaluate_detection(
    transitions: list[dict],
    seizure_start: float,
    seizure_end: float,
    pre_ictal_window: float = 30.0,
) -> dict:
    """Check if any regime transition occurred in the pre-ictal window."""
    pre_start = seizure_start - pre_ictal_window
    pre_transitions = [
        t for t in transitions if pre_start <= t["time_s"] <= seizure_start
    ]
    ictal_transitions = [
        t for t in transitions if seizure_start <= t["time_s"] <= seizure_end
    ]
    return {
        "pre_ictal_transitions": len(pre_transitions),
        "ictal_transitions": len(ictal_transitions),
        "detected": len(pre_transitions) > 0 or len(ictal_transitions) > 0,
        "earliest_detection_s": (
            min(t["time_s"] for t in pre_transitions) - seizure_start
            if pre_transitions
            else None
        ),
    }


def main():
    subject = "chb01"
    cache_dir = Path("experiments/.eeg_cache")

    print(f"Parsing seizure annotations for {subject}...")
    seizures = parse_summary(subject)
    print(f"  Found {len(seizures)} seizures")

    results = []
    for sz in seizures[:4]:  # First 4 seizures for speed
        fname = sz["file"]
        print(f"\nProcessing {fname} (seizure {sz['start_s']}-{sz['end_s']}s)...")

        edf_path = download_edf(subject, fname, cache_dir)
        eeg = load_eeg(edf_path)
        n_ch, n_t = eeg.shape
        duration_s = n_t / FS
        print(f"  Channels: {n_ch}, Duration: {duration_s:.0f}s")

        R_vals, times = compute_windowed_R(eeg, FS, WINDOW_S)
        print(
            f"  R: mean={R_vals.mean():.4f}, "
            f"std={R_vals.std():.4f}, "
            f"range=[{R_vals.min():.4f}, {R_vals.max():.4f}]"
        )

        # R around seizure
        sz_mask = (times >= sz["start_s"]) & (times <= sz["end_s"])
        pre_mask = (times >= sz["start_s"] - 30) & (times < sz["start_s"])
        baseline_mask = times < sz["start_s"] - 60

        R_ictal = R_vals[sz_mask].mean() if sz_mask.any() else 0
        R_pre = R_vals[pre_mask].mean() if pre_mask.any() else 0
        R_baseline = R_vals[baseline_mask].mean() if baseline_mask.any() else 0

        transitions = detect_transitions(R_vals, times)
        detection = evaluate_detection(transitions, sz["start_s"], sz["end_s"])

        entry = {
            "file": fname,
            "seizure_start_s": sz["start_s"],
            "seizure_end_s": sz["end_s"],
            "duration_s": round(duration_s, 1),
            "n_channels": n_ch,
            "R_baseline": round(float(R_baseline), 4),
            "R_pre_ictal": round(float(R_pre), 4),
            "R_ictal": round(float(R_ictal), 4),
            "R_change_pre": round(float(R_pre - R_baseline), 4),
            "R_change_ictal": round(float(R_ictal - R_baseline), 4),
            "total_transitions": len(transitions),
            **detection,
        }
        results.append(entry)

        print(
            f"  R baseline={R_baseline:.4f}, pre-ictal={R_pre:.4f}, ictal={R_ictal:.4f}"
        )
        print(
            f"  Detection: {detection['detected']} "
            f"({detection['pre_ictal_transitions']} pre-ictal, "
            f"{detection['ictal_transitions']} ictal transitions)"
        )

    # Summary
    detected = sum(1 for r in results if r["detected"])
    print(f"\n{'=' * 60}")
    print(f"Seizure Detection Results ({subject})")
    print(f"{'=' * 60}")
    print(f"Seizures tested: {len(results)}")
    print(f"Detected: {detected}/{len(results)}")
    sensitivity = detected / len(results) if results else 0
    print(f"Sensitivity: {sensitivity:.1%}")

    mean_R_change = np.mean([r["R_change_ictal"] for r in results])
    print(f"Mean R change (ictal vs baseline): {mean_R_change:+.4f}")

    output = {
        "subject": subject,
        "n_seizures": len(results),
        "detected": detected,
        "sensitivity": round(sensitivity, 4),
        "mean_R_change_ictal": round(float(mean_R_change), 4),
        "seizures": results,
    }

    out_path = Path("experiments/seizure_detection_results.json")
    with out_path.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
