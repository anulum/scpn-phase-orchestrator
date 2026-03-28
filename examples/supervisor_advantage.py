#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Supervisor Advantage
#
# The key differentiator: SPO has closed-loop supervisory control.
# This example compares open-loop Kuramoto (what every other library
# does) against SPO's supervised loop (detect degradation → boost
# coupling → recover). Quantifies the advantage.
#
# Usage: python examples/supervisor_advantage.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def run_open_loop(
    n: int, omegas: np.ndarray, knm: np.ndarray, phases: np.ndarray, n_steps: int
) -> list[float]:
    """Passive Kuramoto: no intervention, no monitoring."""
    eng = UPDEEngine(n, dt=0.01)
    alpha = np.zeros((n, n))
    p = phases.copy()
    history = []
    for _ in range(n_steps):
        p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(p)
        history.append(R)
    return history


def run_supervised(
    n: int, omegas: np.ndarray, knm: np.ndarray, phases: np.ndarray, n_steps: int
) -> list[float]:
    """SPO supervised: monitor R, boost coupling when degraded."""
    eng = UPDEEngine(n, dt=0.01)
    alpha = np.zeros((n, n))
    p = phases.copy()
    K_scale = 1.0
    history = []
    for _step in range(n_steps):
        p = eng.step(p, omegas, knm * K_scale, 0.0, 0.0, alpha)
        R, _ = compute_order_parameter(p)
        history.append(R)

        # Supervisor logic: detect degradation, act pre-emptively
        if R < 0.4:
            K_scale = min(K_scale * 1.1, 5.0)  # boost coupling
        elif R > 0.8:
            K_scale = max(K_scale * 0.99, 1.0)  # relax toward baseline
    return history


def main() -> None:
    n = 8
    rng = np.random.default_rng(42)
    omegas = rng.uniform(-2, 2, n)
    knm = np.ones((n, n)) * 0.3
    np.fill_diagonal(knm, 0.0)
    phases = rng.uniform(0, TWO_PI, n)

    n_steps = 500

    print("Supervisor Advantage: Open-Loop vs Closed-Loop")
    print("=" * 55)

    r_open = run_open_loop(n, omegas, knm, phases, n_steps)
    r_supervised = run_supervised(n, omegas, knm, phases, n_steps)

    # Compare at key checkpoints
    hdr = f"{'Step':>6s}  {'Open-Loop':>10s}  {'Supervised':>11s}  {'Adv':>8s}"
    print(f"\n{hdr}")
    print("-" * 50)
    for step in [50, 100, 200, 300, 500]:
        ro = r_open[step - 1]
        rs = r_supervised[step - 1]
        adv = (rs - ro) / max(ro, 0.001) * 100
        print(f"{step:>6d}  {ro:>12.3f}  {rs:>13.3f}  {adv:>+9.1f}%")

    # Summary
    r_open_final = r_open[-1]
    r_sup_final = r_supervised[-1]
    print(f"\nFinal: open-loop R={r_open_final:.3f}, supervised R={r_sup_final:.3f}")
    if r_sup_final > r_open_final:
        pct = (r_sup_final - r_open_final) / max(r_open_final, 0.001) * 100
        print(f"Supervisor advantage: +{pct:.1f}% coherence improvement")
    print("\nThis is why SPO exists: closed-loop control, not just simulation.")


if __name__ == "__main__":
    main()
