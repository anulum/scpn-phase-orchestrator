# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase winding multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.winding.winding_numbers``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.monitor import winding as w_mod
from scpn_phase_orchestrator.monitor.winding import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    winding_numbers,
)

TWO_PI = 2.0 * np.pi


def _bench(backend: str, traj, calls: int) -> float:
    saved = w_mod.ACTIVE_BACKEND
    try:
        w_mod.ACTIVE_BACKEND = backend
        winding_numbers(traj)
        t0 = time.perf_counter()
        for _ in range(calls):
            winding_numbers(traj)
        return time.perf_counter() - t0
    finally:
        w_mod.ACTIVE_BACKEND = saved


def bench_at(t: int, n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    omegas = rng.normal(0, 0.5, n)
    dt = 0.05
    hist = np.zeros((t, n))
    hist[0] = rng.uniform(0, TWO_PI, n)
    for i in range(1, t):
        hist[i] = (hist[i - 1] + omegas * dt) % TWO_PI
    row: dict = {"T": t, "N": n, "calls": calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        tm = _bench(backend, hist, calls)
        row[f"{backend}_ms_per_call"] = (tm / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--T-list", type=int, nargs="+", default=[500, 2000, 10000])
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'T':>6} {'N':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for t in args.T_list:
        row = bench_at(t, args.N, args.calls)
        results.append(row)
        line = f"{t:>6} {args.N:>4} {args.calls:>6}"
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
