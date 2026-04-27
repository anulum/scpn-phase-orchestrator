# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Transfer entropy multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.transfer_entropy.phase_transfer_entropy``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.monitor import transfer_entropy as te_mod
from scpn_phase_orchestrator.monitor.transfer_entropy import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    phase_transfer_entropy,
)

TWO_PI = 2.0 * np.pi


def _bench(backend: str, src: np.ndarray, tgt: np.ndarray, calls: int) -> float:
    saved = te_mod.ACTIVE_BACKEND
    try:
        te_mod.ACTIVE_BACKEND = backend
        t0 = time.perf_counter()
        for _ in range(calls):
            phase_transfer_entropy(src, tgt, 16)
        return time.perf_counter() - t0
    finally:
        te_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, calls: int) -> dict:
    rng = np.random.default_rng(42)
    src = rng.uniform(0.0, TWO_PI, size=n)
    tgt = rng.uniform(0.0, TWO_PI, size=n)
    row: dict = {"n": n, "calls": calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        t = _bench(backend, src, tgt, calls)
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--sizes", type=int, nargs="+", default=[200, 1000, 5000])
    parser.add_argument("--calls", type=int, default=50)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>6}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>6}"
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
