#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: EEG Phase Synchronization
#
# Simulates an 8-electrode EEG network with distance-dependent coupling.
# Detects chimera states (coexistent coherent/incoherent clusters)
# and monitors the Normalized Persistent Entropy (NPE).
#
# Usage: python examples/neuroscience_eeg.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.chimera import detect_chimera
from scpn_phase_orchestrator.monitor.npe import compute_npe
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 8  # electrodes: Fp1, Fp2, F3, F4, C3, C4, P3, P4
    labels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4"]
    rng = np.random.default_rng(42)

    # Alpha-band natural frequencies (8-13 Hz) with individual variation
    omegas = TWO_PI * (10.0 + rng.normal(0, 0.5, n))

    # Distance-dependent coupling (nearby electrodes couple more strongly)
    positions = np.array(
        [
            [0.0, 1.0],
            [1.0, 1.0],  # Fp1, Fp2 (frontal)
            [0.0, 0.5],
            [1.0, 0.5],  # F3, F4
            [0.0, 0.0],
            [1.0, 0.0],  # C3, C4 (central)
            [0.0, -0.5],
            [1.0, -0.5],  # P3, P4 (parietal)
        ]
    )
    dist = np.sqrt(np.sum((positions[:, None] - positions[None, :]) ** 2, axis=2))
    knm = 2.0 * np.exp(-dist)
    np.fill_diagonal(knm, 0.0)

    engine = UPDEEngine(n, dt=0.001)
    alpha = np.zeros((n, n))
    phases = rng.uniform(0, TWO_PI, n)

    # Run 2000 steps (2 seconds at 1kHz)
    print("EEG Phase Synchronization (8 electrodes, alpha band)")
    print("-" * 55)
    for epoch in range(4):
        for _ in range(500):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)

        R, psi = compute_order_parameter(phases)
        npe = compute_npe(phases)
        chimera = detect_chimera(phases, knm)

        t = (epoch + 1) * 0.5
        ci = chimera.chimera_index
        print(f"t={t:.1f}s: R={R:.3f}, NPE={npe:.3f}, chi={ci:.3f}")
        if chimera.coherent_indices:
            coh = [labels[i] for i in chimera.coherent_indices]
            print(f"  Coherent cluster: {', '.join(coh)}")

    print("\nFinal phase angles:")
    for i, label in enumerate(labels):
        print(f"  {label}: θ={phases[i]:.2f} rad ({np.degrees(phases[i]):.0f}°)")


if __name__ == "__main__":
    main()
