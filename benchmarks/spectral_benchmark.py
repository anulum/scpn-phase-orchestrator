# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral eigendecomposition benchmark

"""Per-backend wall-clock benchmark for
``coupling.spectral.fiedler_value`` + ``fiedler_vector``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.coupling import spectral as s_mod
from scpn_phase_orchestrator.coupling.spectral import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    fiedler_value,
    fiedler_vector,
)


def _bench(backend: str, W, calls: int) -> float:
    prev = s_mod.ACTIVE_BACKEND
    s_mod.ACTIVE_BACKEND = backend
    s_mod._PRIM_CACHE = None
    s_mod._RUST_CACHE = None
    try:
        fiedler_value(W)
        fiedler_vector(W)  # warm
        t0 = time.perf_counter()
        for _ in range(calls):
            fiedler_value(W)
            fiedler_vector(W)
        return time.perf_counter() - t0
    finally:
        s_mod.ACTIVE_BACKEND = prev
        s_mod._PRIM_CACHE = None
        s_mod._RUST_CACHE = None


def bench_at(n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    W = rng.uniform(0, 1, (n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0.0)
    row: dict = {"N": n, "calls": calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, W, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 64, 128])
    parser.add_argument("--calls", type=int, default=5)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4} {'calls':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>4} {args.calls:>6}"
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
