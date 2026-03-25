#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Power Grid Stability Analysis
#
# Simulates a 4-bus power grid, then tests what happens when a
# generator trips (sudden loss of generation).
#
# Usage: python examples/power_grid_stability.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.inertial import InertialKuramotoEngine


def main() -> None:
    N = 4
    engine = InertialKuramotoEngine(N, dt=0.01)

    # 4-bus grid: 2 generators (P>0), 2 loads (P<0)
    theta = np.zeros(N)
    omega_dot = np.zeros(N)
    power = np.array([1.0, 1.0, -1.0, -1.0])
    knm = np.ones((N, N)) * 2.0
    np.fill_diagonal(knm, 0.0)
    inertia = np.ones(N) * 5.0
    damping = np.ones(N) * 1.0

    # Baseline: balanced grid
    ft, fo, _, _ = engine.run(theta, omega_dot, power, knm, inertia, damping, 500)
    R_base = engine.coherence(ft)
    dev_base = engine.frequency_deviation(fo)
    print(f"Balanced grid: R={R_base:.3f}, freq deviation={dev_base:.4f} Hz")

    # Scenario: Generator 0 trips (loses all power output)
    power_trip = power.copy()
    power_trip[0] = 0.0
    ft2, fo2, _, omega_traj = engine.run(
        theta, omega_dot, power_trip, knm, inertia, damping, 500
    )
    R_trip = engine.coherence(ft2)
    dev_trip = engine.frequency_deviation(fo2)
    max_dev = np.max(np.abs(omega_traj)) / (2 * np.pi)
    print(f"Generator trip: R={R_trip:.3f}, freq deviation={dev_trip:.4f} Hz")
    print(f"  Max transient deviation: {max_dev:.4f} Hz")

    # Scenario: Weak transmission lines
    theta_perturbed = np.array([0.0, 0.5, 1.0, 1.5])
    knm_weak = knm * 0.01
    ft3, fo3, _, _ = engine.run(
        theta_perturbed, omega_dot, power, knm_weak, inertia, damping, 500
    )
    R_weak = engine.coherence(ft3)
    dev_weak = engine.frequency_deviation(fo3)
    print(f"Weak lines:    R={R_weak:.3f}, freq deviation={dev_weak:.4f} Hz")

    if R_weak < 0.5:
        print("  WARNING: Grid desynchronized — cascading failure risk")


if __name__ == "__main__":
    main()
