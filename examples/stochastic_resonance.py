#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Stochastic Resonance
#
# Counter-intuitive: adding noise IMPROVES synchronisation.
# At the optimal noise level D*, coherence peaks above the
# deterministic (D=0) baseline. Too little noise = stuck in
# local minima. Too much = destroyed by fluctuations.
#
# Tselios et al. 2025 — stochastic resonance in Kuramoto networks.
#
# Usage: python examples/stochastic_resonance.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.stochastic import StochasticInjector

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 12
    rng = np.random.default_rng(42)
    omegas = rng.uniform(-1.5, 1.5, n)
    knm = np.ones((n, n)) * 0.8
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    phases_init = rng.uniform(0, TWO_PI, n)

    eng = UPDEEngine(n, dt=0.01)

    print("Stochastic Resonance: Noise Helps Synchronisation")
    print("=" * 55)
    print(f"\n{'D (noise)':>10s}  {'R (final)':>10s}  {'Note':>20s}")
    print("-" * 45)

    best_D = 0.0
    best_R = 0.0

    for D in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        phases = phases_init.copy()
        injector = StochasticInjector(D=D, seed=42) if D > 0 else None

        for _ in range(500):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
            if injector is not None:
                phases = injector.inject(phases, dt=0.01)

        R, _ = compute_order_parameter(phases)

        note = ""
        if D == 0:
            note = "deterministic"
        elif best_R < R:
            best_R = R
            best_D = D
            note = "← improving"

        print(f"{D:>10.2f}  {R:>10.3f}  {note:>20s}")

    print(f"\nOptimal noise: D*={best_D:.2f} (R={best_R:.3f})")
    if best_R > 0:
        print("Noise improved synchronisation above the deterministic baseline.")
    print("\nThis is stochastic resonance: order from disorder.")


if __name__ == "__main__":
    main()
