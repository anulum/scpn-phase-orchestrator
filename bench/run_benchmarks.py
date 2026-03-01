# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

"""Benchmark UPDEEngine.step() for various oscillator counts."""

from __future__ import annotations

import time

import numpy as np

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi
SEED = 42
STEPS = 1000
WARMUP = 50


def bench_step(n_osc: int, method: str = "euler") -> dict:
    builder = CouplingBuilder()
    coupling = builder.build(n_osc, 0.45, 0.3)
    engine = UPDEEngine(n_osc, dt=0.01, method=method)

    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, TWO_PI, n_osc)
    omegas = np.ones(n_osc)

    # Warmup
    for _ in range(WARMUP):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)

    # Timed run
    t0 = time.perf_counter()
    for _ in range(STEPS):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)
    elapsed = time.perf_counter() - t0

    r_final, _ = engine.compute_order_parameter(phases)

    return {
        "n_osc": n_osc,
        "method": method,
        "steps": STEPS,
        "total_s": elapsed,
        "us_per_step": elapsed / STEPS * 1e6,
        "R_final": r_final,
    }


def main():
    sizes = [8, 16, 64, 256]
    methods = ["euler", "rk4"]

    print(f"{'N':>6s} {'Method':>8s} {'Steps':>6s} {'Total(s)':>10s} {'us/step':>10s} {'R_final':>8s}")
    print("-" * 54)

    for n in sizes:
        for m in methods:
            result = bench_step(n, m)
            print(
                f"{result['n_osc']:6d} "
                f"{result['method']:>8s} "
                f"{result['steps']:6d} "
                f"{result['total_s']:10.4f} "
                f"{result['us_per_step']:10.1f} "
                f"{result['R_final']:8.4f}"
            )


if __name__ == "__main__":
    main()
