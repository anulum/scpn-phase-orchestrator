# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability multi-backend benchmark

"""Per-backend wall-clock benchmark for
``upde.basin_stability.steady_state_r`` (single-trial kernel)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import basin_stability as b_mod
from scpn_phase_orchestrator.upde.basin_stability import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    steady_state_r,
)


def _all_to_all(n: int, strength: float = 2.0) -> np.ndarray:
    k = np.ones((n, n)) * strength / n
    np.fill_diagonal(k, 0.0)
    return k


def _bench(backend: str, phases, omegas, knm, n_transient, n_measure,
           calls: int) -> float:
    saved = b_mod.ACTIVE_BACKEND
    try:
        b_mod.ACTIVE_BACKEND = backend
        steady_state_r(
            phases, omegas, knm, dt=0.01,
            n_transient=n_transient, n_measure=n_measure,
        )
        t0 = time.perf_counter()
        for _ in range(calls):
            steady_state_r(
                phases, omegas, knm, dt=0.01,
                n_transient=n_transient, n_measure=n_measure,
            )
        return time.perf_counter() - t0
    finally:
        b_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, n_transient: int, n_measure: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, n)
    omegas = np.ones(n)
    knm = _all_to_all(n, strength=2.0)
    row: dict = {
        "N": n, "n_transient": n_transient, "n_measure": n_measure,
        "calls": calls, "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, phases, omegas, knm,
                   n_transient, n_measure, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument("--n-transient", type=int, default=200)
    parser.add_argument("--n-measure", type=int, default=100)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'nt':>5} {'nm':>5} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.n_transient, args.n_measure, args.calls)
        results.append(row)
        line = (f"{n:>4} {args.n_transient:>5} {args.n_measure:>5} "
                f"{args.calls:>6}")
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
