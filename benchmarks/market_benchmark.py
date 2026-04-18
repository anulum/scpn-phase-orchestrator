# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Market PLV / R(t) multi-backend benchmark

"""Per-backend wall-clock benchmarks for the two market kernels:

* ``market_order_parameter`` — ``R(t)`` per timestep (O(T·N)).
* ``market_plv`` — rolling PLV matrix (O((T−W+1)·N²·W)).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import market as m_mod
from scpn_phase_orchestrator.upde.market import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    market_order_parameter,
    market_plv,
)


def _bench_op(backend: str, phases, calls: int) -> float:
    saved = m_mod.ACTIVE_BACKEND
    try:
        m_mod.ACTIVE_BACKEND = backend
        market_order_parameter(phases)  # warm
        t0 = time.perf_counter()
        for _ in range(calls):
            market_order_parameter(phases)
        return time.perf_counter() - t0
    finally:
        m_mod.ACTIVE_BACKEND = saved


def _bench_plv(backend: str, phases, window: int, calls: int) -> float:
    saved = m_mod.ACTIVE_BACKEND
    try:
        m_mod.ACTIVE_BACKEND = backend
        market_plv(phases, window=window)  # warm
        t0 = time.perf_counter()
        for _ in range(calls):
            market_plv(phases, window=window)
        return time.perf_counter() - t0
    finally:
        m_mod.ACTIVE_BACKEND = saved


def bench_at(t: int, n: int, window: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, (t, n))
    row: dict = {
        "T": t, "N": n, "window": window, "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        tt = _bench_op(backend, phases, calls)
        row[f"{backend}_op_ms"] = (tt / calls) * 1000.0
    for backend in AVAILABLE_BACKENDS:
        tt = _bench_plv(backend, phases, window, calls)
        row[f"{backend}_plv_ms"] = (tt / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--configs", nargs="+", default=["100x8x20", "500x16x50", "2000x8x50"],
        help="Each config is T×N×window, e.g. 500x16x50.",
    )
    parser.add_argument("--calls", type=int, default=3)
    args = parser.parse_args()

    configs = []
    for c in args.configs:
        t, n, w = (int(x) for x in c.split("x"))
        configs.append((t, n, w))

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")

    # Order parameter table
    print("market_order_parameter  (ms per call):")
    header = f"{'T':>6} {'N':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b:>10}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for (t, n, w) in configs:
        row = bench_at(t, n, w, args.calls)
        results.append(row)
        line = f"{t:>6} {n:>4} {args.calls:>6}"
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_op_ms']:>10.4f}"
        print(line)

    # PLV table
    print("\nmarket_plv  (ms per call):")
    header = f"{'T':>6} {'N':>4} {'W':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b:>10}"
    print(header)
    print("-" * len(header))
    for row in results:
        line = (f"{row['T']:>6} {row['N']:>4} {row['window']:>4} "
                f"{args.calls:>6}")
        for b in AVAILABLE_BACKENDS:
            line += f" {row[f'{b}_plv_ms']:>10.4f}"
        print(line)

    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
