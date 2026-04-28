#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Hebbian Plasticity + TE Adaptation
#
# Coupling that learns: oscillators that synchronise together strengthen
# their connection (Hebbian), while transfer entropy steers coupling
# along causal information flow. The coupling matrix evolves over time.
#
# Usage: python examples/plasticity_learning.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)
from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 6
    rng = np.random.default_rng(42)
    omegas = rng.uniform(-1, 1, n)

    # Start with weak uniform coupling
    knm = np.ones((n, n)) * 0.3
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    eng = UPDEEngine(n, dt=0.01)
    phases = rng.uniform(0, TWO_PI, n)

    print("Hebbian Plasticity + TE-Adaptive Coupling")
    print("=" * 50)
    print(f"Initial K mean: {knm[knm > 0].mean():.3f}")

    trajectory = []

    for epoch in range(10):
        # Run 100 steps, collect trajectory
        epoch_traj = []
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            epoch_traj.append(phases.copy())
        trajectory.extend(epoch_traj)

        R, _ = compute_order_parameter(phases)

        # Hebbian update: oscillators in phase strengthen coupling
        elig = compute_eligibility(phases)
        knm = three_factor_update(knm, elig, modulator=R, phase_gate=True, lr=0.005)
        knm = np.maximum(knm, 0.0)
        np.fill_diagonal(knm, 0.0)

        # TE-adaptive: steer coupling along causal flow
        if len(trajectory) >= 50:
            recent = np.array(trajectory[-50:]).T
            knm = te_adapt_coupling(knm, recent, lr=0.002, decay=0.001)

        k_mean = knm[knm > 0].mean()
        k_max = knm.max()
        print(
            f"  Epoch {epoch + 1:>2d}: R={R:.3f}, "
            f"K_mean={k_mean:.3f}, K_max={k_max:.3f}"
        )

    print("\nFinal coupling matrix (top 3 strongest links):")
    flat = [(knm[i, j], i, j) for i in range(n) for j in range(n) if i != j]
    flat.sort(reverse=True)
    for val, i, j in flat[:3]:
        print(f"  {i} → {j}: K={val:.3f}")

    print("\nCoupling learned from dynamics — not prescribed.")


if __name__ == "__main__":
    main()
