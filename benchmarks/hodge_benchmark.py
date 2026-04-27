# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Hodge decomposition multi-backend benchmark

"""Per-backend wall-clock benchmark for
``coupling.hodge.hodge_decomposition``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.coupling import hodge as h_mod
from scpn_phase_orchestrator.coupling.hodge import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    hodge_decomposition,
)

TWO_PI = 2.0 * np.pi


def _bench(backend: str, knm, phases, calls: int) -> float:
    saved = h_mod.ACTIVE_BACKEND
    try:
        h_mod.ACTIVE_BACKEND = backend
        hodge_decomposition(knm, phases)
        t0 = time.perf_counter()
        for _ in range(calls):
            hodge_decomposition(knm, phases)
        return time.perf_counter() - t0
    finally:
        h_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    knm = rng.uniform(-1, 1, (n, n))
    np.fill_diagonal(knm, 0.0)
    phases = rng.uniform(0, TWO_PI, n)
    row: dict = {
        "N": n,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, knm, phases, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>5} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>5} {args.calls:>6}"
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
