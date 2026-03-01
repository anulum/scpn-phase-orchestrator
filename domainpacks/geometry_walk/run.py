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
N_STEPS = 100
SEED = 42


def main():
    spec = load_binding_spec(SPEC_PATH)
    n_osc = sum(len(layer.oscillator_ids) for layer in spec.layers)

    builder = CouplingBuilder()
    coupling = builder.build(n_osc, spec.coupling.base_strength, spec.coupling.decay_alpha)
    engine = UPDEEngine(n_osc, dt=spec.sample_period_s)

    rng = np.random.default_rng(SEED)
    # Initialise phases as if walkers start at random graph nodes
    n_states = 16
    initial_nodes = rng.integers(0, n_states, size=n_osc)
    phases = TWO_PI * initial_nodes / n_states
    omegas = rng.uniform(0.5, 1.5, n_osc)

    for step in range(N_STEPS):
        phases = engine.step(phases, omegas, coupling.knm, 0.05, 0.0, coupling.alpha)
        r, _ = engine.compute_order_parameter(phases)
        if step % 10 == 0:
            print(f"step={step:4d}  R={r:.4f}")

    r_final, _ = engine.compute_order_parameter(phases)
    print(f"\nFinal R={r_final:.4f}")


if __name__ == "__main__":
    main()
