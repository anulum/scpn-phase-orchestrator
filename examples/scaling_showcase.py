#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# SCPN Phase Orchestrator — Example: Scaling from N=4 to N=1000
#
# Shows that SPO works from tiny toy systems to production scale.
# Same API, same engine, same supervisor — only N changes.
#
# Usage: python examples/scaling_showcase.py
# Requires: pip install scpn-phase-orchestrator

from __future__ import annotations

import time

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi


def benchmark_n(n: int, n_steps: int = 100) -> tuple[float, float, float]:
    """Run N oscillators for n_steps, return (R, wall_time_ms, us_per_step)."""
    rng = np.random.default_rng(42)
    omegas = np.zeros(n)
    knm = np.ones((n, n)) * (2.0 / n)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    phases = rng.uniform(0, TWO_PI, n)

    eng = UPDEEngine(n, dt=0.01)

    start = time.perf_counter()
    for _ in range(n_steps):
        phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
    elapsed = time.perf_counter() - start

    R, _ = compute_order_parameter(phases)
    wall_ms = elapsed * 1000
    us_per_step = elapsed / n_steps * 1e6
    return R, wall_ms, us_per_step


def main() -> None:
    print("Scaling Showcase: N=4 to N=1000")
    print("=" * 60)
    hdr = f"{'N':>6s}  {'Steps':>6s}  {'R':>6s}  {'ms':>8s}  {'us/step':>8s}"
    print(f"\n{hdr}")
    print("-" * 50)

    for n in [4, 8, 16, 32, 64, 128, 256, 512, 1000]:
        steps = 100 if n <= 256 else 50
        R, wall_ms, us_step = benchmark_n(n, steps)
        print(f"{n:>6d}  {steps:>6d}  {R:>6.3f}  {wall_ms:>7.1f}  {us_step:>7.0f}")

    print("\nSame API at every scale. Rust kernel accelerates large N.")
    print("For N>1000 with GPU: pip install scpn-phase-orchestrator[nn]")


if __name__ == "__main__":
    main()
