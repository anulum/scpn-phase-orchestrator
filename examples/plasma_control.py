#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Tokamak Plasma Mode Locking
#
# Models toroidal MHD mode coupling in a tokamak. When modes lock
# (R→1), a disruption is imminent. The supervisor detects degradation
# and increases coupling damping as a pre-emptive response.
#
# Usage: python examples/plasma_control.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.lyapunov import LyapunovGuard
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 6  # MHD modes: m/n = 2/1, 3/2, 4/3, 1/1, 3/1, 5/2
    mode_labels = ["2/1", "3/2", "4/3", "1/1", "3/1", "5/2"]

    # Mode frequencies scale with safety factor q
    omegas = TWO_PI * np.array([1.0, 1.5, 2.0, 0.5, 3.0, 2.5])

    # Coupling: nearest rational surfaces couple strongly
    rng = np.random.default_rng(0)
    knm = rng.uniform(0.5, 1.5, (n, n))
    knm = 0.5 * (knm + knm.T)
    np.fill_diagonal(knm, 0.0)

    engine = UPDEEngine(n, dt=0.001)
    alpha = np.zeros((n, n))
    phases = rng.uniform(0, TWO_PI, n)
    guard = LyapunovGuard()

    print("Tokamak MHD Mode Coupling")
    print("=" * 50)

    # Phase 1: Normal operation (weak coupling)
    print("\n--- Normal Operation (K_scale=1.0) ---")
    K_scale = 1.0
    for epoch in range(4):
        for _ in range(500):
            phases = engine.step(phases, omegas, knm * K_scale, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        lstate = guard.evaluate(phases, knm * K_scale)
        t = 0.5 * (epoch + 1)
        ib = lstate.in_basin
        print(f"  t={t:.1f}s: R={R:.3f}, V={lstate.V:.3f}, basin={ib}")

    # Phase 2: Mode locking onset (increase coupling — disruption precursor)
    print("\n--- Mode Locking Onset (K_scale=5.0) ---")
    K_scale = 5.0
    guard.reset()
    for epoch in range(4):
        for _ in range(500):
            phases = engine.step(phases, omegas, knm * K_scale, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        lstate = guard.evaluate(phases, knm * K_scale)
        t = 0.5 * (epoch + 1)
        ib = lstate.in_basin
        print(f"  t={t:.1f}s: R={R:.3f}, V={lstate.V:.3f}, basin={ib}")
        if R > 0.85:
            print("  DISRUPTION WARNING: mode locking detected")

    print("\nFinal mode phases:")
    for i, label in enumerate(mode_labels):
        print(f"  m/n={label}: θ={np.degrees(phases[i]):.0f}°")


if __name__ == "__main__":
    main()
