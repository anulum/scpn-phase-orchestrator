# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

"""Ablation study: measure R convergence with components on/off."""

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi
SEED = 42


def run_trial(n_osc: int, steps: int, base_k: float, zeta: float) -> list[float]:
    """Run UPDE for given parameters, return R at each step."""
    builder = CouplingBuilder()
    coupling = builder.build(n_osc, base_k, 0.3)
    engine = UPDEEngine(n_osc, dt=0.01)

    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, TWO_PI, n_osc)
    omegas = np.ones(n_osc)

    r_trace = []
    for _ in range(steps):
        phases = engine.step(phases, omegas, coupling.knm, zeta, 0.0, coupling.alpha)
        r, _ = engine.compute_order_parameter(phases)
        r_trace.append(r)
    return r_trace


def ablation_suite(n_osc: int = 16, steps: int = 200):
    """Run ablation variants and print results."""
    configs = {
        "full": {"base_k": 0.45, "zeta": 0.1},
        "no_coupling": {"base_k": 0.0, "zeta": 0.1},
        "no_driver": {"base_k": 0.45, "zeta": 0.0},
        "no_coupling_no_driver": {"base_k": 0.0, "zeta": 0.0},
    }

    print(f"Ablation study: N={n_osc}, steps={steps}")
    print(f"{'Variant':<28s} {'R_final':>8s} {'R_mean':>8s} {'R_max':>8s}")
    print("-" * 56)

    for name, params in configs.items():
        r_trace = run_trial(n_osc, steps, **params)
        r_final = r_trace[-1]
        r_mean = float(np.mean(r_trace))
        r_max = float(np.max(r_trace))
        print(f"{name:<28s} {r_final:8.4f} {r_mean:8.4f} {r_max:8.4f}")


if __name__ == "__main__":
    ablation_suite()
