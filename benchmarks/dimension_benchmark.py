# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Fractal-dimension multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.dimension.correlation_integral`` (full-pairs mode)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.monitor import dimension as dim_mod
from scpn_phase_orchestrator.monitor.dimension import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    correlation_integral,
)


def _bench(
    backend: str, traj: np.ndarray, eps: np.ndarray, calls: int,
) -> float:
    saved = dim_mod.ACTIVE_BACKEND
    try:
        dim_mod.ACTIVE_BACKEND = backend
        correlation_integral(traj, eps, max_pairs=1_000_000)
        t0 = time.perf_counter()
        for _ in range(calls):
            correlation_integral(traj, eps, max_pairs=1_000_000)
        return time.perf_counter() - t0
    finally:
        dim_mod.ACTIVE_BACKEND = saved


def bench_at(t: int, d: int, n_k: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    traj = rng.normal(0, 1, (t, d))
    eps = np.logspace(-1, 0.5, n_k)
    row: dict = {
        "T": t, "d": d, "n_k": n_k, "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        tt = _bench(backend, traj, eps, calls)
        row[f"{backend}_ms_per_call"] = (tt / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--T-list", type=int, nargs="+", default=[50, 150, 400]
    )
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--n-k", type=int, default=20)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'T':>5} {'d':>3} {'K':>3} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for t in args.T_list:
        row = bench_at(t, args.d, args.n_k, args.calls)
        results.append(row)
        line = f"{t:>5} {args.d:>3} {args.n_k:>3} {args.calls:>6}"
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
