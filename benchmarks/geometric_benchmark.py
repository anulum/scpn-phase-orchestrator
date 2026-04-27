# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus integrator multi-backend benchmark

"""Per-backend wall-clock benchmark for
``upde.geometric.TorusEngine.run`` (symplectic Euler on T^N)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import geometric as g_mod
from scpn_phase_orchestrator.upde.geometric import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    TorusEngine,
)


def _bench(
    backend: str, theta, omegas, knm, alpha, n: int, n_steps: int, calls: int
) -> float:
    saved = g_mod.ACTIVE_BACKEND
    try:
        g_mod.ACTIVE_BACKEND = backend
        eng = TorusEngine(n, 0.01)
        eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=1)  # warm
        t0 = time.perf_counter()
        for _ in range(calls):
            eng.run(theta, omegas, knm, 0.0, 0.0, alpha, n_steps=n_steps)
        return time.perf_counter() - t0
    finally:
        g_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, n_steps: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    theta = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = rng.uniform(0, 0.5 / n, (n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n))
    row: dict = {
        "N": n,
        "n_steps": n_steps,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, theta, omegas, knm, alpha, n, n_steps, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--n-steps", type=int, default=50)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'nsteps':>7} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.n_steps, args.calls)
        results.append(row)
        line = f"{n:>4} {args.n_steps:>7} {args.calls:>6}"
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_ms_per_call']:>12.4f}"
        print(line)
    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
