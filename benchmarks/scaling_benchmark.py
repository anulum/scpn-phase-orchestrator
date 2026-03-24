# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Scaling benchmark

"""Measure performance scaling from N=16 to N=4000.

Usage:
    python benchmarks/scaling_benchmark.py
    python benchmarks/scaling_benchmark.py --output benchmarks/scaling_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def bench_one(n: int, n_steps: int = 100, dt: float = 0.01) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = np.full((n, n), 0.5 / n)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))

    engine = UPDEEngine(n, dt=dt)

    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
    elapsed = time.perf_counter() - t0

    R, _ = compute_order_parameter(phases)
    mem_mb = (knm.nbytes + alpha.nbytes + phases.nbytes) / 1e6

    return {
        "N": n,
        "n_steps": n_steps,
        "wall_time_s": round(elapsed, 4),
        "ms_per_step": round(elapsed / n_steps * 1000, 3),
        "steps_per_sec": round(n_steps / elapsed, 1),
        "final_R": round(float(R), 4),
        "memory_MB": round(mem_mb, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 64, 256, 1000],
    )
    args = parser.parse_args()

    print(f"{'N':>6} {'ms/step':>10} {'steps/s':>10} {'R':>8} {'mem MB':>8}")
    print("-" * 50)

    results = []
    for n in args.sizes:
        r = bench_one(n)
        results.append(r)
        print(
            f"{r['N']:>6} {r['ms_per_step']:>10.3f} "
            f"{r['steps_per_sec']:>10.1f} {r['final_R']:>8.4f} "
            f"{r['memory_MB']:>8.2f}"
        )

    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
