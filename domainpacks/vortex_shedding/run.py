# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anylum.li
# SCPN Phase Orchestrator — Vortex Shedding

"""Run vortex_shedding simulation."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from scpn_phase_orchestrator.binding.loader import load_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.oscillators.init_phases import extract_initial_phases
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def main() -> None:
    spec_path = Path(__file__).parent / "binding_spec.yaml"
    spec = load_binding_spec(spec_path)
    n_osc = sum(len(ly.oscillator_ids) for ly in spec.layers)
    omegas = np.array(spec.get_omegas(), dtype=np.float64)
    coupling = CouplingBuilder().build(
        n_osc, spec.coupling.base_strength, spec.coupling.decay_alpha
    )
    phases = extract_initial_phases(spec, omegas)

    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)
    for step in range(200):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)
        if step % 50 == 0:
            R, _ = compute_order_parameter(phases)
            print(f"step={step:4d}  R={R:.4f}")

    R_final, _ = compute_order_parameter(phases)
    print(f"\nFinal R={R_final:.4f}")


if __name__ == "__main__":
    main()
