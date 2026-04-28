#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Cardiac Rhythm Synchronization
#
# Models the SA node as a pacemaker driving 5 cardiac oscillators.
# Demonstrates external drive (zeta) pulling the network into sync.
#
# Usage: python examples/cardiac_rhythm.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 5

    # Natural frequencies: SA node sets the pace (~72 bpm = 1.2 Hz)
    omegas = TWO_PI * np.array([1.2, 1.15, 1.18, 1.10, 1.12])

    # Coupling: SA → AV strong, AV → His, His → ventricles
    knm = np.zeros((n, n))
    knm[1, 0] = 3.0  # SA → AV
    knm[2, 1] = 2.5  # AV → His
    knm[3, 2] = 2.0  # His → LV
    knm[4, 2] = 2.0  # His → RV
    knm[3, 4] = 1.0  # LV ↔ RV (biventricular coupling)
    knm[4, 3] = 1.0

    engine = UPDEEngine(n, dt=0.001)
    alpha = np.zeros((n, n))
    phases = np.zeros(n)

    print("Cardiac Rhythm Synchronization")
    print("=" * 45)

    # Normal sinus rhythm
    print("\n--- Normal Sinus Rhythm ---")
    for epoch in range(4):
        for _ in range(500):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        print(f"  t={0.5 * (epoch + 1):.1f}s: R={R:.3f}")

    # Simulate AV block: break AV → His coupling
    print("\n--- AV Block (His bundle decoupled) ---")
    knm_block = knm.copy()
    knm_block[2, 1] = 0.0
    phases_block = np.zeros(n)
    for epoch in range(4):
        for _ in range(500):
            phases_block = engine.step(phases_block, omegas, knm_block, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases_block)
        print(f"  t={0.5 * (epoch + 1):.1f}s: R={R:.3f}")
    if R < 0.5:
        print("  ALERT: Desynchronization detected — arrhythmia risk")

    # External pacemaker drive
    print("\n--- External Pacemaker (ζ=2.0, Ψ=SA frequency) ---")
    psi = 0.0
    phases_paced = phases_block.copy()
    for epoch in range(4):
        for step in range(500):
            psi = (TWO_PI * 1.2 * step * 0.001) % TWO_PI
            phases_paced = engine.step(phases_paced, omegas, knm_block, 2.0, psi, alpha)
        R, _ = compute_order_parameter(phases_paced)
        print(f"  t={0.5 * (epoch + 1):.1f}s: R={R:.3f}")


if __name__ == "__main__":
    main()
