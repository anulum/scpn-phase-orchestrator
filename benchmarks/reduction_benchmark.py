# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen multi-backend benchmark

"""Per-backend wall-clock benchmark for
``upde.reduction.OttAntonsenReduction.run`` — scalar-complex RK4
on the Ott-Antonsen mean-field ODE."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from scpn_phase_orchestrator.upde import reduction as r_mod
from scpn_phase_orchestrator.upde.reduction import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    OttAntonsenReduction,
)


def _bench(backend: str, n_steps: int, calls: int) -> float:
    saved = r_mod.ACTIVE_BACKEND
    try:
        r_mod.ACTIVE_BACKEND = backend
        red = OttAntonsenReduction(
            omega_0=0.5, delta=0.1, K=1.0, dt=0.01,
        )
        red.run(complex(0.2, 0.1), n_steps=1)  # warm
        t0 = time.perf_counter()
        for _ in range(calls):
            red.run(complex(0.2, 0.1), n_steps=n_steps)
        return time.perf_counter() - t0
    finally:
        r_mod.ACTIVE_BACKEND = saved


def bench_at(n_steps: int, calls: int) -> dict:
    row: dict = {
        "n_steps": n_steps, "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, n_steps, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--n-steps", type=int, nargs="+", default=[500, 5000, 50000],
    )
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'n_steps':>8} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for ns in args.n_steps:
        row = bench_at(ns, args.calls)
        results.append(row)
        line = f"{ns:>8} {args.calls:>6}"
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
