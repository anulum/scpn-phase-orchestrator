# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ethical-cost backend comparison

"""Measure both real ethical-cost paths on identical finite fixtures.

Run with the repository virtual environment and a freshly built spo-kernel.
The selected backend flag is restored after each size. No numerical kernel
is substituted; timings include the public Python entry-point validation.
"""

from __future__ import annotations

import argparse
import json
from statistics import median
from time import perf_counter_ns

import numpy as np

from scpn_phase_orchestrator.ssgf import ethical


def benchmark_size(n: int, calls: int, batches: int) -> dict[str, float | int]:
    """Return timings and parity for one fixed-seed oscillator count."""
    if n < 1 or calls < 1 or batches < 1:
        raise ValueError("sizes, calls and batches must be positive")
    if not ethical._HAS_RUST:
        raise RuntimeError("build spo-kernel before the two-backend comparison")
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    knm = rng.uniform(0.0, 0.5, (n, n))
    np.fill_diagonal(knm, 0.0)
    original_backend = ethical._HAS_RUST
    row: dict[str, float | int] = {"n": n}
    reference: ethical.EthicalCost | None = None
    try:
        for backend, enabled in (("python", False), ("rust", True)):
            ethical._HAS_RUST = enabled
            result = ethical.compute_ethical_cost(phases, knm)
            if reference is None:
                reference = result
            elif (
                not np.allclose(
                    [result.J_sec, result.phi_ethics, result.c15_sec],
                    [reference.J_sec, reference.phi_ethics, reference.c15_sec],
                    rtol=1e-9,
                    atol=1e-9,
                )
                or result.constraints_violated != reference.constraints_violated
            ):
                raise RuntimeError(f"ethical backend parity failed for N={n}")
            samples: list[float] = []
            for _ in range(batches):
                start = perf_counter_ns()
                for _ in range(calls):
                    ethical.compute_ethical_cost(phases, knm)
                samples.append((perf_counter_ns() - start) / calls / 1000.0)
            row[f"{backend}_us"] = median(samples)
    finally:
        ethical._HAS_RUST = original_backend
    row["python_over_rust"] = row["python_us"] / row["rust_us"]
    return row


def main() -> None:
    """Run the reproducible public-entry-point comparison and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--batches", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "fixture": (
                    "default_rng(42); phases U(0,2pi); coupling U(0,.5); zero diagonal"
                ),
                "calls": args.calls,
                "batches": args.batches,
                "parity_passed": True,
                "results": [
                    benchmark_size(n, args.calls, args.batches) for n in args.sizes
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
