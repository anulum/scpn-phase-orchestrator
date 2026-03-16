# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Plasma control example

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.adapters.plasma_control_bridge import PlasmaControlBridge
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
N_STEPS = 200
SEED = 42


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

    bridge = PlasmaControlBridge(n_layers=8)
    omegas = bridge.import_plasma_omega(n_osc_per_layer=2)

    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, TWO_PI, n_osc)

    for step in range(N_STEPS):
        phases = engine.step(phases, omegas, coupling.knm, 0.05, 0.0, coupling.alpha)
        r, _ = engine.compute_order_parameter(phases)

        if step % 40 == 0:
            snapshot = {
                "phases": phases.tolist(),
                "regime": "NOMINAL" if r > 0.4 else "DEGRADED",
                "layer_sizes": [2] * 8,
                "stability": float(r),
            }
            state = bridge.import_snapshot(snapshot)

            violations = bridge.check_physics_invariants(
                {
                    "q_min": 1.2 - 0.5 * (1 - r),
                    "beta_n": 1.5 + 1.5 * (1 - r),
                    "greenwald": 0.8,
                }
            )

            print(
                f"step={step:4d}  R={r:.4f}  regime={state.regime_id}  "
                f"violations={len(violations)}"
            )

    r_final, _ = engine.compute_order_parameter(phases)
    print(f"\nFinal R={r_final:.4f}")


if __name__ == "__main__":
    main()
