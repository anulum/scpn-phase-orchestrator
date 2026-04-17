#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Epidemic Synchronization (SIR)
#
# Models 6 regions as coupled oscillators where phase represents
# epidemic wave position. Synchronization = waves arriving simultaneously
# (worst case for healthcare). Desync = staggered peaks (manageable).
#
# Usage: python examples/epidemic_sir.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.transfer_entropy import transfer_entropy_matrix
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 6
    labels = ["Region_A", "Region_B", "Region_C", "Region_D", "Region_E", "Region_F"]
    rng = np.random.default_rng(0)

    # Epidemic wave frequencies: ~3-month cycles with regional variation
    omegas = TWO_PI * (1.0 / 90.0 + rng.normal(0, 0.001, n))

    # Mobility coupling: population flow between regions
    knm = (
        np.array(
            [
                [0, 2, 1, 0.5, 0.2, 0.1],
                [2, 0, 1.5, 1, 0.3, 0.2],
                [1, 1.5, 0, 2, 1, 0.5],
                [0.5, 1, 2, 0, 1.5, 1],
                [0.2, 0.3, 1, 1.5, 0, 2],
                [0.1, 0.2, 0.5, 1, 2, 0],
            ],
            dtype=float,
        )
        * 0.3
    )

    engine = UPDEEngine(n, dt=0.1)
    alpha = np.zeros((n, n))
    phases = rng.uniform(0, TWO_PI, n)

    print("Epidemic Wave Synchronization (6 Regions)")
    print("=" * 50)

    # Simulate and collect trajectory for TE analysis
    trajectory = []
    for epoch in range(10):
        for _ in range(100):
            phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
            trajectory.append(phases.copy())
        R, _ = compute_order_parameter(phases)
        t = 10.0 * (epoch + 1)
        risk = "HIGH" if R > 0.7 else "MODERATE" if R > 0.4 else "LOW"
        print(f"  t={t:.0f}s: R={R:.3f} — simultaneous peak risk: {risk}")

    # Transfer entropy: which regions drive which?
    traj = np.array(trajectory[-200:]).T  # (n, T)
    te = transfer_entropy_matrix(traj)
    print("\nTransfer Entropy (causal influence):")
    max_te_idx = np.unravel_index(np.argmax(te), te.shape)
    print(
        f"  Strongest: {labels[max_te_idx[0]]} → {labels[max_te_idx[1]]} "
        f"(TE={te[max_te_idx]:.3f})"
    )
    print(f"  Implication: {labels[max_te_idx[0]]} drives epidemic timing")


if __name__ == "__main__":
    main()
