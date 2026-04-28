# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Internal engine comparison benchmark

"""Compare SPO engine variants on synchronization speed and accuracy.

Run: python -m benchmarks.engine_comparison
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from scpn_phase_orchestrator.upde.delay import DelayedEngine
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.geometric import TorusEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter
from scpn_phase_orchestrator.upde.simplicial import SimplicialEngine
from scpn_phase_orchestrator.upde.splitting import SplittingEngine


@dataclass
class BenchResult:
    engine: str
    n_oscillators: int
    n_steps: int
    wall_time_ms: float
    final_R: float
    steps_per_sec: float


def run_benchmark(
    n: int = 16, n_steps: int = 1000, K: float = 0.5, dt: float = 0.01
) -> list[BenchResult]:
    rng = np.random.default_rng(42)
    phases0 = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = np.full((n, n), K)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engines = {
        "UPDEEngine(euler)": UPDEEngine(n, dt, method="euler"),
        "UPDEEngine(rk4)": UPDEEngine(n, dt, method="rk4"),
        "UPDEEngine(rk45)": UPDEEngine(n, dt, method="rk45"),
        "SplittingEngine": SplittingEngine(n, dt),
        "TorusEngine": TorusEngine(n, dt),
        "SimplicialEngine(σ₂=0)": SimplicialEngine(n, dt, sigma2=0.0),
        "SimplicialEngine(σ₂=0.1)": SimplicialEngine(n, dt, sigma2=0.1),
        "DelayedEngine(τ=3)": DelayedEngine(n, dt, delay_steps=3),
    }

    results = []
    for name, eng in engines.items():
        phases = phases0.copy()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        elapsed = (time.perf_counter() - t0) * 1000
        R, _ = compute_order_parameter(phases)
        results.append(
            BenchResult(
                engine=name,
                n_oscillators=n,
                n_steps=n_steps,
                wall_time_ms=round(elapsed, 2),
                final_R=round(float(R), 4),
                steps_per_sec=round(n_steps / (elapsed / 1000), 0),
            )
        )
    return results


if __name__ == "__main__":
    print(f"{'Engine':<30} {'Time(ms)':>10} {'R':>8} {'Steps/s':>10}")
    print("-" * 62)
    for r in run_benchmark():
        print(
            f"{r.engine:<20} {r.wall_time_ms:>8.1f}"
            f" {r.final_R:>7.4f} {r.steps_per_sec:>9.0f}"
        )
