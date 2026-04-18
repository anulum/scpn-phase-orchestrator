# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov spectrum multi-backend benchmark

"""Per-backend wall-clock benchmark for
``monitor.lyapunov.lyapunov_spectrum``.

Runs the Benettin 1980 / Shimada-Nagashima 1979 algorithm across the
Rust → Mojo → Julia → Go → Python fallback chain at increasing network
sizes so the cost profile of each backend is visible on the same axis.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.monitor import lyapunov as ly_mod
from scpn_phase_orchestrator.monitor.lyapunov import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    lyapunov_spectrum,
)

TWO_PI = 2.0 * np.pi


def _bench(
    backend: str,
    phases: np.ndarray,
    omegas: np.ndarray,
    knm: np.ndarray,
    alpha: np.ndarray,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
    calls: int,
) -> float:
    saved = ly_mod.ACTIVE_BACKEND
    try:
        ly_mod.ACTIVE_BACKEND = backend
        # Warm-up — first call covers JIT / library init / FFI cache.
        lyapunov_spectrum(
            phases, omegas, knm, alpha,
            n_steps=n_steps, qr_interval=qr_interval,
            zeta=zeta, psi=psi,
        )
        t0 = time.perf_counter()
        for _ in range(calls):
            lyapunov_spectrum(
                phases, omegas, knm, alpha,
                n_steps=n_steps, qr_interval=qr_interval,
                zeta=zeta, psi=psi,
            )
        return time.perf_counter() - t0
    finally:
        ly_mod.ACTIVE_BACKEND = saved


def _problem(n: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0.0, TWO_PI, size=n)
    omegas = rng.normal(0.0, 0.2, size=n)
    knm = rng.uniform(0.3, 1.2, size=(n, n))
    np.fill_diagonal(knm, 0.0)
    alpha = rng.uniform(-0.15, 0.15, size=(n, n))
    np.fill_diagonal(alpha, 0.0)
    return phases, omegas, knm, alpha


def bench_at(
    n: int,
    n_steps: int,
    qr_interval: int,
    calls: int,
    zeta: float,
    psi: float,
) -> dict:
    phases, omegas, knm, alpha = _problem(n)
    row: dict = {
        "n": n,
        "n_steps": n_steps,
        "qr_interval": qr_interval,
        "calls": calls,
        "zeta": zeta,
        "psi": psi,
        "available": AVAILABLE_BACKENDS,
    }
    for backend in AVAILABLE_BACKENDS:
        t = _bench(
            backend, phases, omegas, knm, alpha,
            n_steps, qr_interval, zeta, psi, calls,
        )
        row[f"{backend}_ms_per_call"] = (t / calls) * 1000.0
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sizes", type=int, nargs="+", default=[4, 8, 16]
    )
    parser.add_argument("--n-steps", type=int, default=500)
    parser.add_argument("--qr-interval", type=int, default=10)
    parser.add_argument("--calls", type=int, default=3)
    parser.add_argument("--zeta", type=float, default=0.0)
    parser.add_argument("--psi", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Active: {ACTIVE_BACKEND}  Available: {AVAILABLE_BACKENDS}\n")
    header = f"{'N':>4}{'steps':>7}{'calls':>7}"
    for b in AVAILABLE_BACKENDS:
        header += f" {b + '_ms':>12}"
    print(header)
    print("-" * len(header))
    results: list[dict] = []
    for n in args.sizes:
        row = bench_at(
            n, args.n_steps, args.qr_interval, args.calls,
            args.zeta, args.psi,
        )
        results.append(row)
        line = f"{n:>4}{args.n_steps:>7}{args.calls:>7}"
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
