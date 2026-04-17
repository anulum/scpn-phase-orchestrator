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

from scpn_phase_orchestrator.coupling.attention_residuals import (
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


def bench_one(n: int, n_steps: int = 200, dt: float = 0.01) -> dict:
    rng = np.random.default_rng(42)
    phases = rng.uniform(0.0, TWO_PI, size=n).astype(np.float64)
    omegas = (rng.standard_normal(n) * 1.5).astype(np.float64)
    knm = _symmetric_knm(n, strength=0.5 / n, seed=42)
    alpha = np.zeros((n, n), dtype=np.float64)

    engine = UPDEEngine(n_oscillators=n, dt=dt, method="euler")

    # Baseline — static K_nm reused every step.
    phases_base = phases.copy()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        phases_base = engine.step(phases_base, omegas, knm, 0.0, 0.0, alpha)
    t_base = time.perf_counter() - t0

    # AttnRes — modulate K_nm every step from current phases.
    phases_attn = phases.copy()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        knm_mod = attnres_modulate(
            knm, phases_attn, block_size=4, lambda_=0.5
        )
        phases_attn = engine.step(phases_attn, omegas, knm_mod, 0.0, 0.0, alpha)
    t_attn = time.perf_counter() - t0

    overhead_pct = 100.0 * (t_attn - t_base) / t_base if t_base > 0 else 0.0

    return {
        "n": n,
        "n_steps": n_steps,
        "baseline_ms_per_step": (t_base / n_steps) * 1000.0,
        "attnres_ms_per_step": (t_attn / n_steps) * 1000.0,
        "overhead_pct": overhead_pct,
    }


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

    print(f"{'N':>6} {'base_ms':>10} {'attnres_ms':>12} {'overhead':>10}")
    print("-" * 44)
    results: list[dict] = []
    for n in args.sizes:
        r = bench_one(n, n_steps=args.steps)
        results.append(r)
        print(
            f"{r['n']:>6} {r['baseline_ms_per_step']:>10.3f} "
            f"{r['attnres_ms_per_step']:>12.3f} "
            f"{r['overhead_pct']:>9.1f}%"
        )

    if args.output:
        args.output.write_text(
            json.dumps({"results": results}, indent=2), encoding="utf-8"
        )
        print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
