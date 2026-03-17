# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Quantum simulation example

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.adapters.quantum_control_bridge import QuantumControlBridge
from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
from scpn_phase_orchestrator.upde.engine import UPDEEngine

TWO_PI = 2.0 * np.pi
SPEC_PATH = Path(__file__).parent / "binding_spec.yaml"
N_STEPS = 100
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

    omegas = np.ones(n_osc) * 1.0
    phases = extract_initial_phases(spec, omegas)

    bridge = QuantumControlBridge(n_oscillators=n_osc)

    for step in range(N_STEPS):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)
        r, _ = engine.compute_order_parameter(phases)

        if step % 20 == 0:
            artifact = {
                "phases": phases.tolist(),
                "fidelity": float(r),
                "layer_assignments": [
                    list(range(4)),
                    list(range(4, 8)),
                ],
            }
            state = bridge.import_artifact(artifact)
            exported = bridge.export_artifact(state)
            print(
                f"step={step:4d}  R={r:.4f}  "
                f"qubit_R={state.layers[0].R:.4f}  "
                f"logical_R={state.layers[1].R:.4f}  "
                f"fidelity={exported['fidelity']:.4f}"
            )

    r_final, _ = engine.compute_order_parameter(phases)
    print(f"\nFinal R={r_final:.4f}")


if __name__ == "__main__":
    main()
