#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Multi-Engine Comparison
#
# Same initial conditions, same coupling, three different engines.
# Shows how engine choice affects dynamics: standard Kuramoto,
# geometric (torus-preserving), and Strang splitting.
#
# Usage: python examples/multi_engine_comparison.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.splitting import SplittingEngine

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 8
    rng = np.random.default_rng(0)
    omegas = rng.uniform(-1, 1, n)
    knm = np.ones((n, n)) * 1.5
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    phases_init = rng.uniform(0, TWO_PI, n)

    engines = {
        "UPDE Euler": UPDEEngine(n, dt=0.01, method="euler"),
        "UPDE RK4": UPDEEngine(n, dt=0.01, method="rk4"),
        "Torus (geometric)": TorusEngine(n, dt=0.01),
        "Strang splitting": SplittingEngine(n, dt=0.01),
    }

    n_steps = 300

    print("Multi-Engine Comparison")
    print("=" * 60)
    print(f"N={n}, K=1.5, dt=0.01, {n_steps} steps")
    print(f"\n{'Engine':<22s}", end="")
    for step in [50, 100, 200, 300]:
        print(f"  R@{step:<4d}", end="")
    print()
    print("-" * 60)

    for name, eng in engines.items():
        phases = phases_init.copy()
        checkpoints = {}
        for step in range(1, n_steps + 1):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            if step in {50, 100, 200, 300}:
                R, _ = compute_order_parameter(phases)
                checkpoints[step] = R

        print(f"{name:<22s}", end="")
        for step in [50, 100, 200, 300]:
            print(f"  {checkpoints[step]:.3f}", end="")
        print()

    print("\nAll engines converge to the same sync state.")
    print("Torus engine avoids mod-2pi discontinuities.")
    print("Strang splitting separates rotation from coupling.")
    print("RK4 is most accurate; Euler is fastest.")


if __name__ == "__main__":
    main()
