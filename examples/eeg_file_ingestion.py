#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Example: EEG File Ingestion


"""Demonstrate synthetic oscillator phases through the public UPDE and monitor.

The chimera monitor uses positive non-self adjacency and unweighted local
coherence, with strict 0.7/0.3 thresholds. Its index is the boundary fraction;
one snapshot does not demonstrate a persistent chimera or classify clinical EEG.
Run with ``python examples/eeg_file_ingestion.py``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


class HilbertTransform(Protocol):
    """Describe the float64 SciPy Hilbert call used by this example."""

    def __call__(self, x: NDArray[np.float64], *, axis: int) -> NDArray[np.complex128]:
        """Return the complex128 analytic signal along the given sample axis."""
        ...


hilbert = cast(HilbertTransform, import_module("scipy.signal").hilbert)

TWO_PI = 2.0 * np.pi


def generate_synthetic_eeg(
    n_channels: int = 8,
    duration_s: float = 2.0,
    sample_rate: float = 256.0,
    alpha_freq: float = 10.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], float]:
    """Generate an alpha sinusoid with independent white Gaussian noise.

    Parameters
    ----------
    n_channels : int
        Number of simulated channels.
    duration_s : float
        Duration in seconds.
    sample_rate : float
        Samples per second.
    alpha_freq : float
        Mean oscillator frequency in hertz.
    seed : int
        NumPy random seed.

    Returns
    -------
    tuple[numpy.ndarray, float]
        Float64 samples of shape (n_samples, n_channels) and sample rate.
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


def extract_phases_hilbert(signals: NDArray[np.float64]) -> NDArray[np.float64]:
    """Extract phases from the Hilbert analytic signal along the sample axis.

    Parameters
    ----------
    signals : numpy.ndarray
        Float64 array with samples in rows and channels in columns. A recording
        needs application-appropriate filtering before interpreting its phase.

    Returns
    -------
    numpy.ndarray
        Float64 phase angles in [0, 2*pi), with the same shape as signals.
    """
    analytic = hilbert(signals, axis=0)
    return np.asarray(np.angle(analytic) % TWO_PI, dtype=np.float64)


def main() -> None:
    """Run the original synthetic example through real public consumers."""
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
            f"boundary_fraction={chimera.chimera_index:.3f}"
        )

    print("\nDone. In production, replace generate_synthetic_eeg()")
    print("with your real data loader (mne, numpy, pandas, etc.)")


if __name__ == "__main__":
    main()
