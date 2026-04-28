# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Swarmalator multi-backend benchmark

"""Per-backend wall-clock benchmark for
``upde.swarmalator.SwarmalatorEngine.step``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import swarmalator as sw_mod
from scpn_phase_orchestrator.upde.swarmalator import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    SwarmalatorEngine,
)

TWO_PI = 2.0 * np.pi


def _bench(
    backend: str,
    pos,
    phases,
    omegas,
    n: int,
    dim: int,
    calls: int,
) -> float:
    saved = sw_mod.ACTIVE_BACKEND
    try:
        sw_mod.ACTIVE_BACKEND = backend
        eng = SwarmalatorEngine(n, dim, 0.01)
        eng.step(pos, phases, omegas)
        t0 = time.perf_counter()
        for _ in range(calls):
            eng.step(pos, phases, omegas)
        return time.perf_counter() - t0
    finally:
        sw_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, dim: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    pos = rng.uniform(-1, 1, (n, dim))
    phases = rng.uniform(0, TWO_PI, n)
    omegas = rng.normal(0.5, 0.2, n)
    row: dict = {
        "N": n,
        "dim": dim,
        "calls": calls,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, pos, phases, omegas, n, dim, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'dim':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.dim, args.calls)
        results.append(row)
        line = f"{n:>4} {args.dim:>4} {args.calls:>6}"
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
