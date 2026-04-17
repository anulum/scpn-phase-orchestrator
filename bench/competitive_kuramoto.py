# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Competitive Kuramoto benchmarks

"""Competitive benchmark: SPO supervisor recovery vs raw SciPy Kuramoto.

Measures time-to-recovery (steps until R > 0.7) after a phase perturbation.
SPO supervisor actively adjusts coupling K to restore coherence, while
raw SciPy relies only on the natural Kuramoto dynamics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np
from scipy.integrate import solve_ivp

from scpn_phase_orchestrator._compat import HAS_RUST
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

TWO_PI = 2.0 * np.pi
SEED = 42
R_THRESHOLD = 0.7
MAX_STEPS = 5000
DT = 0.01


def _kuramoto_rhs(n, omegas, knm):
    """Return Kuramoto ODE rhs for scipy solve_ivp."""

    def rhs(_t, theta):
        diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        coupling = np.sum(knm * np.sin(diff), axis=1)
        return omegas + coupling

    return rhs


def run_scipy_recovery(n, omegas, knm, perturbed_phases):
    """Step raw SciPy Kuramoto until R > threshold or MAX_STEPS exceeded."""
    rhs = _kuramoto_rhs(n, omegas, knm)
    phases = perturbed_phases.copy()
    steps = 0
    t0 = time.perf_counter()

    for _ in range(MAX_STEPS):
        sol = solve_ivp(rhs, [0, DT], phases, method="RK45", max_step=DT)
        phases = sol.y[:, -1] % TWO_PI
        steps += 1
        r, _ = compute_order_parameter(phases)
        if r >= R_THRESHOLD:
            break

    elapsed = time.perf_counter() - t0
    r_final, _ = compute_order_parameter(phases)
    return {
        "method": "scipy_rk45",
        "steps_to_recovery": steps,
        "recovered": r_final >= R_THRESHOLD,
        "R_final": round(float(r_final), 6),
        "wall_s": round(elapsed, 6),
    }


def run_spo_recovery(n, omegas, knm_base, alpha, perturbed_phases, use_supervisor):
    """Step SPO engine until R > threshold or MAX_STEPS exceeded.

    If use_supervisor=True, boost coupling K by 1.5x when R < 0.5 (mimics
    supervisor regime escalation).
    """
    engine = UPDEEngine(n, dt=DT, method="euler")
    phases = perturbed_phases.copy()
    knm = knm_base.copy()
    steps = 0
    t0 = time.perf_counter()

    for _ in range(MAX_STEPS):
        if use_supervisor:
            r_now, _ = compute_order_parameter(phases)
            if r_now < 0.5:
                knm = knm_base * 1.5
            elif r_now < R_THRESHOLD:
                knm = knm_base * 1.2
            else:
                knm = knm_base
        phases = engine.step(phases, omegas, knm, 0.0, 0.0, alpha)
        steps += 1
        r, _ = compute_order_parameter(phases)
        if r >= R_THRESHOLD:
            break

    elapsed = time.perf_counter() - t0
    r_final, _ = compute_order_parameter(phases)
    backend = "rust" if HAS_RUST else "python"
    label = f"spo_supervisor_{backend}" if use_supervisor else f"spo_passive_{backend}"
    return {
        "method": label,
        "steps_to_recovery": steps,
        "recovered": r_final >= R_THRESHOLD,
        "R_final": round(float(r_final), 6),
        "wall_s": round(elapsed, 6),
    }


def run_benchmark(n_osc):
    rng = np.random.default_rng(SEED)

    builder = CouplingBuilder()
    coupling = builder.build(n_osc, 0.45, 0.3)
    omegas = np.ones(n_osc) + rng.normal(0, 0.1, n_osc)

    # Start synchronized, then perturb
    synced = rng.uniform(0, 0.1, n_osc)
    perturbed = synced + rng.uniform(-np.pi, np.pi, n_osc)
    perturbed %= TWO_PI

    r0, _ = compute_order_parameter(perturbed)

    results = {
        "n_osc": n_osc,
        "R_initial": round(float(r0), 6),
        "R_threshold": R_THRESHOLD,
        "max_steps": MAX_STEPS,
        "dt": DT,
        "runs": [],
    }

    results["runs"].append(
        run_spo_recovery(
            n_osc, omegas, coupling.knm, coupling.alpha, perturbed, use_supervisor=False
        )
    )
    results["runs"].append(
        run_spo_recovery(
            n_osc, omegas, coupling.knm, coupling.alpha, perturbed, use_supervisor=True
        )
    )
    results["runs"].append(run_scipy_recovery(n_osc, omegas, coupling.knm, perturbed))
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Competitive Kuramoto recovery benchmark"
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    sizes = [8, 16, 64]
    all_results = [run_benchmark(n) for n in sizes]

    if args.json:
        json.dump(all_results, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return

    for res in all_results:
        print(f"\n=== N={res['n_osc']}  R_initial={res['R_initial']:.3f} ===")
        for run in res["runs"]:
            ok = "YES" if run["recovered"] else "NO"
            print(
                f"  {run['method']:>25s}: "
                f"steps={run['steps_to_recovery']:5d}  "
                f"R={run['R_final']:.4f}  "
                f"recovered={ok}  "
                f"wall={run['wall_s']:.3f}s"
            )


if __name__ == "__main__":
    main()
