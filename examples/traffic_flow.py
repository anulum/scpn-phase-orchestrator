#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Traffic Signal Synchronization
#
# Models a corridor of 8 traffic lights as coupled phase oscillators.
# "Green wave" = full synchronization (R≈1). Detects desynchronization
# from signal timing drift and shows re-synchronization via coupling boost.
#
# Usage: python examples/traffic_flow.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 8  # traffic lights on a corridor

    # All lights aim for 90-second cycle (0.011 Hz) with slight drift
    rng = np.random.default_rng(42)
    omegas = TWO_PI * (1.0 / 90.0 + rng.normal(0, 0.0005, n))

    # Chain coupling: each light couples to its neighbours
    knm = np.zeros((n, n))
    for i in range(n - 1):
        knm[i, i + 1] = 0.5
        knm[i + 1, i] = 0.5

    engine = UPDEEngine(n, dt=0.1)
    alpha = np.zeros((n, n))

    # Start with staggered phases (progressive offset for green wave)
    phases = np.linspace(0, np.pi, n)

    print("Traffic Signal Green Wave Synchronization")
    print("=" * 50)

    # Normal operation
    print("\n--- Normal Coupling (K=0.5) ---")
    for epoch in range(5):
        for _ in range(100):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        t = 10.0 * (epoch + 1)
        status = "GREEN WAVE" if R > 0.7 else "DEGRADED" if R > 0.4 else "DESYNC"
        print(f"  t={t:.0f}s: R={R:.3f} [{status}]")

    # Coupling failure: light 3 disconnects
    print("\n--- Light 3 Disconnected ---")
    knm_broken = knm.copy()
    knm_broken[2, 3] = 0.0
    knm_broken[3, 2] = 0.0
    knm_broken[3, 4] = 0.0
    knm_broken[4, 3] = 0.0
    for epoch in range(5):
        for _ in range(100):
            phases = engine.step(phases, omegas, knm_broken, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        t = 10.0 * (epoch + 1)
        print(f"  t={t:.0f}s: R={R:.3f}")

    # Recovery: boost coupling on remaining links
    print("\n--- Recovery (K boosted to 2.0) ---")
    knm_boosted = knm_broken * 4.0
    for epoch in range(5):
        for _ in range(100):
            phases = engine.step(phases, omegas, knm_boosted, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(phases)
        t = 10.0 * (epoch + 1)
        print(f"  t={t:.0f}s: R={R:.3f}")


if __name__ == "__main__":
    main()
