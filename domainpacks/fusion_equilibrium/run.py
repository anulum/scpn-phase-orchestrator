# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.adapters.fusion_core_bridge import FusionCoreBridge
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
N_STEPS = 200
SEED = 42
SAWTOOTH_PERIOD = 50


def main():
    spec = load_binding_spec(SPEC_PATH)
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)

    builder = CouplingBuilder()
    coupling = builder.build(
        n_osc,
        spec.coupling.base_strength,
        spec.coupling.decay_alpha,
    )
    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)

    bridge = FusionCoreBridge(n_layers=6)

    rng = np.random.default_rng(SEED)
    omegas = np.array([0.3, 0.3, 0.5, 0.5, 1.0, 1.0, 5.0, 5.0, 0.2, 0.2, 0.1, 0.1])
    phases = rng.uniform(0, TWO_PI, n_osc)

    saw_count = 0
    elm_count = 0

    for step in range(N_STEPS):
        # Periodic sawtooth crash
        if step > 0 and step % SAWTOOTH_PERIOD == 0:
            saw_count += 1

        phases = engine.step(phases, omegas, coupling.knm, 0.05, 0.0, coupling.alpha)
        r, _ = engine.compute_order_parameter(phases)

        if step % 40 == 0:
            snapshot = {
                "q_profile": 1.5 + 0.5 * np.sin(step * 0.05),
                "q_min": 1.0 + 0.3 * r,
                "q_max": 4.5,
                "beta_n": 1.2 + 0.8 * (1 - r),
                "tau_e": 1.5 + r,
                "sawtooth_count": saw_count,
                "elm_count": elm_count,
                "mhd_amplitude": 0.3 * (1 - r),
            }
            obs_phases = bridge.observables_to_phases(snapshot)
            feedback = bridge.phases_to_feedback(obs_phases, omegas[:6])
            violations = bridge.check_stability(snapshot)

            print(
                f"step={step:4d}  R_upde={r:.4f}  R_obs={feedback['R_global']:.4f}  "
                f"saw={saw_count}  violations={len(violations)}"
            )

    r_final, _ = engine.compute_order_parameter(phases)
    print(f"\nFinal R={r_final:.4f}")


if __name__ == "__main__":
    main()
