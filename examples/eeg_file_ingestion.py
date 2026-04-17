#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: EEG File Ingestion
#
# Shows how to go from a real EEG data file to SPO phase dynamics.
# Generates synthetic EEG-like data (alpha band + noise) as a stand-in
# for real .edf/.csv files, then extracts phases via Hilbert transform
# and runs the UPDE engine.
#
# For real EEG: replace the synthetic signal with mne.io.read_raw_edf()
# or numpy.loadtxt() from your recording system.
#
# Usage: python examples/eeg_file_ingestion.py
# Requires: pip install scpn-phase-orchestrator scipy

from __future__ import annotations

import numpy as np
from scipy.signal import hilbert

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def generate_synthetic_eeg(
    n_channels: int = 8,
    duration_s: float = 2.0,
    sample_rate: float = 256.0,
    alpha_freq: float = 10.0,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Generate synthetic EEG-like signals (alpha band + pink noise).

    Returns (n_samples, n_channels) array and sample rate.
    Replace this with your real EEG loader.
    """
    rng = np.random.default_rng(seed)
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples) / sample_rate

    signals = np.zeros((n_samples, n_channels))
    for ch in range(n_channels):
        freq = alpha_freq + rng.normal(0, 0.5)
        phase_offset = rng.uniform(0, TWO_PI)
        alpha = np.sin(TWO_PI * freq * t + phase_offset)
        noise = rng.standard_normal(n_samples) * 0.3
        signals[:, ch] = alpha + noise

    return signals, sample_rate


def extract_phases_hilbert(signals: np.ndarray) -> np.ndarray:
    """Extract instantaneous phase from each channel via Hilbert transform.

    This is the standard method for narrowband EEG signals.
    For broadband: bandpass filter first (e.g. 8-13 Hz for alpha).
    """
    analytic = hilbert(signals, axis=0)
    return np.angle(analytic) % TWO_PI


def main() -> None:
    print("EEG File Ingestion → SPO Phase Dynamics")
    print("=" * 50)

    # Step 1: Load data (synthetic here; replace with real loader)
    print("\n1. Loading EEG data...")
    signals, sr = generate_synthetic_eeg(n_channels=8, duration_s=2.0)
    print(f"   {signals.shape[1]} channels, {signals.shape[0]} samples, {sr} Hz")

    # Step 2: Extract phases via Hilbert transform
    print("\n2. Extracting phases (Hilbert transform)...")
    phases_all = extract_phases_hilbert(signals)
    print(f"   Phase array: {phases_all.shape}")

    # Step 3: Build coupling matrix from electrode distances
    n = signals.shape[1]
    np.random.default_rng(0)
    dist = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    knm = 1.5 * np.exp(-0.5 * dist)
    np.fill_diagonal(knm, 0.0)

    # Step 4: Run SPO engine on extracted phases
    print("\n3. Running UPDE engine...")
    engine = UPDEEngine(n, dt=1.0 / sr)
    alpha = np.zeros((n, n))
    omegas = np.full(n, TWO_PI * 10.0)

    # Use extracted phases as initial conditions
    phases = phases_all[-1, :]

    # Simulate forward 500 steps
    for epoch in range(5):
        for _ in range(100):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        R, _ = compute_order_parameter(phases)
        npe = compute_npe(phases)
        chimera = detect_chimera(phases, knm)
        t = (epoch + 1) * 100 / sr
        print(
            f"   t={t:.2f}s: R={R:.3f}, NPE={npe:.3f}, "
            f"chimera={chimera.chimera_index:.3f}"
        )

    print("\nDone. In production, replace generate_synthetic_eeg()")
    print("with your real data loader (mne, numpy, pandas, etc.)")


if __name__ == "__main__":
    main()
