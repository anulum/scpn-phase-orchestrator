# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Recurrence matrix multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.recurrence.recurrence_matrix``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.monitor import recurrence as r_mod
from scpn_phase_orchestrator.monitor.recurrence import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    recurrence_matrix,
)


def _bench(backend: str, traj, epsilon, calls: int) -> float:
    saved = r_mod.ACTIVE_BACKEND
    try:
        r_mod.ACTIVE_BACKEND = backend
        recurrence_matrix(traj, epsilon)
        t0 = time.perf_counter()
        for _ in range(calls):
            recurrence_matrix(traj, epsilon)
        return time.perf_counter() - t0
    finally:
        r_mod.ACTIVE_BACKEND = saved


def bench_at(t: int, d: int, epsilon: float, calls: int) -> dict:
    rng = np.random.default_rng(42)
    traj = rng.normal(0, 1, (t, d))
    row: dict = {
        "T": t, "d": d, "epsilon": epsilon, "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        elapsed = _bench(backend, traj, epsilon, calls)
        row[f"{backend}_ms_per_call"] = (elapsed / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--T-list", type=int, nargs="+", default=[30, 100, 300]
    )
    parser.add_argument("--d", type=int, default=3)
    parser.add_argument("--epsilon", type=float, default=0.8)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'T':>5} {'d':>3} {'eps':>5} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for t in args.T_list:
        row = bench_at(t, args.d, args.epsilon, args.calls)
        results.append(row)
        line = f"{t:>5} {args.d:>3} {args.epsilon:>5.2f} {args.calls:>6}"
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
