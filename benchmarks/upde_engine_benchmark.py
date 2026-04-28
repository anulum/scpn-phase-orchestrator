# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE engine multi-backend benchmark

"""Per-backend wall-clock benchmark for ``upde.engine.upde_run``.

Runs the three integrators (Euler, RK4, Dormand-Prince RK45) across
the Rust → Mojo → Julia → Go → Python fallback chain at a range of
network sizes / step counts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import engine as eng_mod
from scpn_phase_orchestrator.upde.engine import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    upde_run,
)

TWO_PI = 2.0 * np.pi


def _bench(
    backend: str,
    phases: np.ndarray,
    omegas: np.ndarray,
    knm: np.ndarray,
    alpha: np.ndarray,
    method: str,
    n_steps: int,
    calls: int,
) -> float:
    saved = eng_mod.ACTIVE_BACKEND
    try:
        eng_mod.ACTIVE_BACKEND = backend
        # Warm-up amortises juliacall init, rayon thread pool, subprocess
        # fork, etc.
        upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            dt=0.01,
            n_steps=n_steps,
            method=method,
        )
        t0 = time.perf_counter()
        for _ in range(calls):
            upde_run(
                phases,
                omegas,
                knm,
                alpha,
                zeta=0.0,
                psi=0.0,
                dt=0.01,
                n_steps=n_steps,
                method=method,
            )
        return time.perf_counter() - t0
    finally:
        eng_mod.ACTIVE_BACKEND = saved


def _problem(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 1.0, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.1, 0.1, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    return phases, omegas, knm, alpha


def bench_at(n: int, method: str, n_steps: int, calls: int) -> dict:
    phases, omegas, knm, alpha = _problem(n)
    row: dict = {
        "n": n,
        "method": method,
        "n_steps": n_steps,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(
            backend,
            phases,
            omegas,
            knm,
            alpha,
            method,
            n_steps,
            calls,
        )
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 64])
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["euler", "rk4", "rk45"],
    )
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'method':>6} {'steps':>6} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        for method in args.methods:
            row = bench_at(n, method, args.n_steps, args.calls)
            results.append(row)
            line = f"{n:>4} {method:>6} {args.n_steps:>6} {args.calls:>6}"
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
