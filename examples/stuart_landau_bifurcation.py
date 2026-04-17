#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Stuart-Landau Bifurcation
#
# Demonstrates the supercritical Hopf bifurcation: as μ crosses zero,
# oscillators transition from fixed point (r→0) to limit cycle (r→√μ).
# This is the canonical amplitude dynamics engine in SPO.
#
# Pikovsky et al. 2001, "Synchronization: A Universal Concept".
#
# Usage: python examples/stuart_landau_bifurcation.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi


def main() -> None:
    n = 4
    rng = np.random.default_rng(42)

    omegas = TWO_PI * np.array([1.0, 1.2, 0.8, 1.1])
    knm = np.zeros((n, n))  # uncoupled: observe individual bifurcation
    knm_r = np.zeros((n, n))
    alpha = np.zeros((n, n))

    print("Stuart-Landau Hopf Bifurcation")
    print("=" * 55)
    print(f"{'μ':>6s}  {'r_mean':>8s}  {'√μ (theory)':>12s}  {'|error|':>8s}  Status")
    print("-" * 55)

    for mu_val in [-1.0, -0.5, 0.0, 0.1, 0.5, 1.0, 2.0, 4.0]:
        mu = np.full(n, mu_val)
        engine = StuartLandauEngine(n, dt=0.001)

        # State: [θ₀, θ₁, ..., θ_{n-1}, r₀, r₁, ..., r_{n-1}]
        phases = rng.uniform(0, TWO_PI, n)
        amplitudes = np.full(n, 0.5)
        state = np.concatenate([phases, amplitudes])

        for _ in range(5000):
            state = engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        r_final = state[n:]  # amplitudes
        r_mean = float(np.mean(r_final))

        if mu_val > 0:
            r_theory = np.sqrt(mu_val)
            error = abs(r_mean - r_theory)
            status = "limit cycle"
            print(f"{mu_val:6.1f}  {r_mean:8.4f}  {r_theory:12.4f}  {error:8.4f}")
        else:
            status = "fixed point" if mu_val < 0 else "bifurcation"
            print(f"{mu_val:6.1f}  {r_mean:8.4f}  {'—':>12s}  {'—':>8s}  {status}")

    print("\nAnalytical: r → √μ for μ > 0 (Hopf bifurcation theorem)")


if __name__ == "__main__":
    main()
