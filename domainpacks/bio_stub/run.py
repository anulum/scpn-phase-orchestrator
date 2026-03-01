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

    # Natural frequencies spanning biological timescales (normalised)
    omegas = np.array([
        10.0, 8.0, 0.5, 0.01,    # cellular: fast
        1.0, 0.8, 0.3, 5.0,      # tissue
        1.2, 0.25, 0.05, 0.04,   # organ
        0.001, 0.01, 0.5, 0.002, # systemic: slow
    ])

    for step in range(N_STEPS):
        phases = engine.step(phases, omegas, coupling.knm, 0.1, 0.0, coupling.alpha)

        if step % 20 == 0:
            # Per-layer R
            for layer in spec.layers:
                n_layer = len(layer.oscillator_ids)
                start = layer.index * 4  # 4 oscillators per layer
                end = start + n_layer
                r, _ = engine.compute_order_parameter(phases[start:end])
                print(f"step={step:4d}  {layer.name:10s}  R={r:.4f}")
            print()

    r_final, _ = engine.compute_order_parameter(phases)
    print(f"Final global R={r_final:.4f}")


if __name__ == "__main__":
    main()
