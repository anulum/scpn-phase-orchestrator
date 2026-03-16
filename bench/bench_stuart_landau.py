# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau benchmarks

"""Benchmark StuartLandauEngine.step() for various oscillator counts."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time

import numpy as np
import scipy

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

TWO_PI = 2.0 * np.pi
SEED = 42
STEPS = 500
WARMUP = 50


def bench_sl(n_osc: int, method: str = "euler") -> dict:
    builder = CouplingBuilder()
    coupling = builder.build(n_osc, 0.45, 0.3)
    engine = StuartLandauEngine(n_osc, dt=0.01, method=method)

    rng = np.random.default_rng(SEED)
    state = np.empty(2 * n_osc)
    state[:n_osc] = rng.uniform(0, TWO_PI, n_osc)
    state[n_osc:] = rng.uniform(0.5, 1.5, n_osc)

    omegas = np.ones(n_osc)
    mu = np.ones(n_osc)
    knm_r = coupling.knm * 0.5
    alpha = coupling.alpha

    for _ in range(WARMUP):
        state = engine.step(state, omegas, mu, coupling.knm, knm_r, 0.0, 0.0, alpha)

    t0 = time.perf_counter()
    for _ in range(STEPS):
        state = engine.step(state, omegas, mu, coupling.knm, knm_r, 0.0, 0.0, alpha)
    elapsed = time.perf_counter() - t0

    r_final, _ = engine.compute_order_parameter(state)
    r_mean = engine.compute_mean_amplitude(state)

    return {
        "n_osc": n_osc,
        "method": method,
        "steps": STEPS,
        "total_s": round(elapsed, 6),
        "us_per_step": round(elapsed / STEPS * 1e6, 1),
        "R_final": round(float(r_final), 6),
        "r_mean_amplitude": round(float(r_mean), 6),
    }


def main():
    parser = argparse.ArgumentParser(description="StuartLandauEngine benchmarks")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sizes = [8, 16, 64, 256]
    methods = ["euler", "rk4", "rk45"]
    results = []

    for n in sizes:
        for m in methods:
            results.append(bench_sl(n, m))

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

    cols = ["N", "Method", "Steps", "Total(s)", "us/step", "R_final", "r_mean"]
    widths = [6, 8, 6, 10, 10, 8, 8]
    hdr = " ".join(f"{c:>{w}s}" for c, w in zip(cols, widths, strict=True))
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(
            f"{r['n_osc']:6d} "
            f"{r['method']:>8s} "
            f"{r['steps']:6d} "
            f"{r['total_s']:10.4f} "
            f"{r['us_per_step']:10.1f} "
            f"{r['R_final']:8.4f} "
            f"{r['r_mean_amplitude']:8.4f}"
        )


if __name__ == "__main__":
    main()
