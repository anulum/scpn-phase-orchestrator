# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes modulation overhead benchmark

"""Measure the per-step overhead of ``attnres_modulate`` vs a static K_nm.

The research doc ``research_attention_residuals_2026-04-06.md §5`` sets
a <10% overhead budget for the state-dependent coupling modulation.
Both the baseline coupling computation and the AttnRes modulation are
O(N²) per step, so the overhead should be a small multiplicative
constant rather than an asymptotic slowdown.

Usage::

    python benchmarks/attnres_modulation_benchmark.py
    python benchmarks/attnres_modulation_benchmark.py \\
        --output benchmarks/attnres_results.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.coupling import attention_residuals as attnres_mod
from scpn_phase_orchestrator.coupling.attention_residuals import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    attnres_modulate,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi


def _symmetric_knm(n: int, strength: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    half = rng.uniform(0.0, 2.0 * strength, size=(n, n))
    knm = 0.5 * (half + half.T)
    np.fill_diagonal(knm, 0.0)
    return knm.astype(np.float64)


def _bench_with_backend(
    backend: str, n: int, n_steps: int, engine: UPDEEngine,
    phases0: np.ndarray, omegas: np.ndarray, knm: np.ndarray, alpha: np.ndarray,
) -> float:
    """Return wall-clock seconds for `n_steps` of AttnRes-modulated integration
    using the named backend."""
    saved = attnres_mod.ACTIVE_BACKEND
    try:
        attnres_mod.ACTIVE_BACKEND = backend
        phases = phases0.copy()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            knm_mod = attnres_modulate(
                knm, phases, block_size=4, lambda_=0.5
            )
            phases = engine.step(phases, omegas, knm_mod, 0.0, 0.0, alpha)
        return time.perf_counter() - t0
    finally:
        attnres_mod.ACTIVE_BACKEND = saved


def bench_one(n: int, n_steps: int = 200, dt: float = 0.01) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    omegas = (rng.standard_normal(n) * 1.5).astype(np.float64)
    knm = _symmetric_knm(n, strength=0.5 / n, seed=42)
    alpha = np.zeros((n, n), dtype=np.float64)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")

    # Baseline — static K_nm reused every step (no modulation overhead).
    phases_base = phases.copy()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases_base = engine.step(phases_base, omegas, knm, 0.0, 0.0, alpha)
    t_base = time.perf_counter() - t0

    # AttnRes — run once per available backend.
    row: dict = {"n": n, "n_steps": n_steps, "available": AVAILABLE_BACKENDS}
    row["baseline_ms_per_step"] = (t_base / n_steps) * 1000.0
    for backend in AVAILABLE_BACKENDS:
        t_attn = _bench_with_backend(
            backend, n, n_steps, engine, phases, omegas, knm, alpha
        )
        row[f"{backend}_ms_per_step"] = (t_attn / n_steps) * 1000.0
        row[f"{backend}_overhead_pct"] = (
            100.0 * (t_attn - t_base) / t_base if t_base > 0 else 0.0
        )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON file to write the results to.",
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[16, 64, 256, 512],
        help="N values to benchmark.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Number of integration steps per run.",
    )
    args = parser.parse_args()

    print(f"Active backend: {ACTIVE_BACKEND}")
    print(f"Available backends (fastest first): {AVAILABLE_BACKENDS}\n")

    header = f"{'N':>6} {'base_ms':>10}"
    for backend in AVAILABLE_BACKENDS:
        header += f" {backend + '_ms':>12} {backend + '_%':>8}"
    print(header)
    print("-" * len(header))

    results: list[dict] = []
    for n in args.sizes:
        r = bench_one(n, n_steps=args.steps)
        results.append(r)
        row = f"{r['n']:>6} {r['baseline_ms_per_step']:>10.3f}"
        for backend in AVAILABLE_BACKENDS:
            row += (
                f" {r[f'{backend}_ms_per_step']:>12.3f}"
                f" {r[f'{backend}_overhead_pct']:>7.1f}%"
            )
        print(row)

    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
