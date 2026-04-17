# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Order parameters multi-backend benchmark

"""Per-backend wall-clock benchmark for ``upde/order_params.py``.

Measures ``compute_order_parameter`` across every loaded backend for
a range of N. Matches the AttnRes benchmark template.

Usage::

    python benchmarks/order_params_benchmark.py
    python benchmarks/order_params_benchmark.py \\
        --output benchmarks/order_params_results.json --sizes 100 1000 10000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.upde import order_params as op_mod
from scpn_phase_orchestrator.upde.order_params import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    compute_order_parameter,
)

TWO_PI = 2.0 * np.pi


def _bench_one(backend: str, phases: np.ndarray, n_calls: int) -> float:
    saved = op_mod.ACTIVE_BACKEND
    try:
        op_mod.ACTIVE_BACKEND = backend
        t0 = time.perf_counter()
        for _ in range(n_calls):
            compute_order_parameter(phases)
        return time.perf_counter() - t0
    finally:
        op_mod.ACTIVE_BACKEND = saved


def bench_at(n: int, n_calls: int) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    row: dict = {"n": n, "n_calls": n_calls, "available": AVAILABLE_BACKENDS}
    for backend in AVAILABLE_BACKENDS:
        t = _bench_one(backend, phases, n_calls)
        row[f"{backend}_ms_per_call"] = (t / n_calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=None, help="JSON results file."
    )
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[16, 256, 4096, 65536]
    )
    parser.add_argument("--calls", type=int, default=500)
    args = parser.parse_args()

    print(f"Active backend: {ACTIVE_BACKEND}")
    print(f"Available (fastest first): {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>7}"
    for backend in AVAILABLE_BACKENDS:
        header += f" {backend + '_ms':>12}"
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(n, args.calls)
        results.append(row)
        line = f"{n:>7}"
        for backend in AVAILABLE_BACKENDS:
            line += f" {row[f'{backend}_ms_per_call']:>12.4f}"
        print(line)

    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
