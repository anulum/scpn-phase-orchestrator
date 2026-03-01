# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from pathlib import Path

import numpy as np

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
    coupling = builder.build(n_osc, spec.coupling.base_strength, spec.coupling.decay_alpha)
    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)

    rng = np.random.default_rng(SEED)
    phases = rng.uniform(0, TWO_PI, n_osc)
    omegas = np.array([1.0, 1.1, 0.9, 0.5, 0.2, 0.15])

    # Layer masks
    bad_ids = {0, 1, 2}   # micro layer oscillator indices
    good_ids = {4, 5}     # macro layer oscillator indices

    for step in range(N_STEPS):
        phases = engine.step(phases, omegas, coupling.knm, 0.0, 0.0, coupling.alpha)

        bad_phases = phases[list(bad_ids)]
        good_phases = phases[list(good_ids)]
        r_bad, _ = engine.compute_order_parameter(bad_phases)
        r_good, _ = engine.compute_order_parameter(good_phases)

        if step % 20 == 0:
            print(f"step={step:4d}  R_good={r_good:.4f}  R_bad={r_bad:.4f}")

    r_bad_final, _ = engine.compute_order_parameter(phases[list(bad_ids)])
    r_good_final, _ = engine.compute_order_parameter(phases[list(good_ids)])
    print(f"\nFinal  R_good={r_good_final:.4f}  R_bad={r_bad_final:.4f}")


if __name__ == "__main__":
    main()
