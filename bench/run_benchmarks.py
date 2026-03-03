# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

"""Benchmark UPDEEngine.step() for various oscillator counts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import scipy

import scpn_phase_orchestrator._compat as _compat
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi
SEED = 42
STEPS = 1000
WARMUP = 50


def bench_step(n_osc: int, method: str = "euler", force_python: bool = False) -> dict:
    saved = _compat.HAS_RUST
    if force_python:
        _compat.HAS_RUST = False
    try:
        builder = CouplingBuilder()
        coupling = builder.build(n_osc, 0.45, 0.3)
        engine = UPDEEngine(n_osc, dt=0.01, method=method)

        rng = np.random.default_rng(SEED)
        phases = rng.uniform(0, TWO_PI, n_osc)
        omegas = np.ones(n_osc)

        for _ in range(WARMUP):
            phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)

        t0 = time.perf_counter()
        for _ in range(STEPS):
            phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)
        elapsed = time.perf_counter() - t0

        r_final, _ = engine.compute_order_parameter(phases)
    finally:
        _compat.HAS_RUST = saved

    return {
        "n_osc": n_osc,
        "method": method,
        "backend": "python" if force_python else ("rust" if saved else "python"),
        "steps": STEPS,
        "total_s": round(elapsed, 6),
        "us_per_step": round(elapsed / STEPS * 1e6, 1),
        "R_final": round(float(r_final), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="UPDEEngine benchmarks")
    parser.add_argument("--json", action="store_true", help="Output JSON to stdout")
    args = parser.parse_args()

    sizes = [8, 16, 64, 256, 1024]
    methods = ["euler", "rk4", "rk45"]
    results = []

    for n in sizes:
        for m in methods:
            results.append(bench_step(n, m))

    # Python-vs-Rust comparison (rk45 is Python-only, skip it)
    if _compat.HAS_RUST:
        for n in sizes:
            for m in ("euler", "rk4"):
                results.append(bench_step(n, m, force_python=True))

    if args.json:
        output = {
            "meta": {
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "scipy_version": scipy.__version__,
                "platform": platform.platform(),
            },
            "results": results,
        }
        json.dump(output, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    cols = ["N", "Method", "Backend", "Steps", "Total(s)", "us/step", "R_final"]
    widths = [6, 8, 8, 6, 10, 10, 8]
    hdr = " ".join(f"{c:>{w}s}" for c, w in zip(cols, widths, strict=True))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['n_osc']:6d} "
            f"{r['method']:>8s} "
            f"{r['backend']:>8s} "
            f"{r['steps']:6d} "
            f"{r['total_s']:10.4f} "
            f"{r['us_per_step']:10.1f} "
            f"{r['R_final']:8.4f}"
        )

    if _compat.HAS_RUST:
        print("\n--- Rust speedup ---")
        rust_by_key = {}
        py_by_key = {}
        for r in results:
            key = (r["n_osc"], r["method"])
            if r["backend"] == "rust":
                rust_by_key[key] = r["us_per_step"]
            elif r["backend"] == "python":
                py_by_key[key] = r["us_per_step"]
        for key in sorted(rust_by_key):
            if key in py_by_key and rust_by_key[key] > 0:
                ratio = py_by_key[key] / rust_by_key[key]
                print(f"  N={key[0]:4d} {key[1]:>5s}: {ratio:.1f}x")


if __name__ == "__main__":
    main()
