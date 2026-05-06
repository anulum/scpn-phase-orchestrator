# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid causal attribution demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
from scpn_phase_orchestrator.supervisor import CausalInterventionEngine

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")


def power_grid_disturbance_state() -> tuple[FloatArray, FloatArray]:
    """Return deterministic phasor state for a load-step synchronism disturbance."""
    phases = np.array(
        [0.00, 0.05, 0.11, 0.18, 0.25, 1.60, 2.80, 3.70, 4.50, 5.10, 5.70, 6.00],
        dtype=np.float64,
    )
    omegas = np.array(
        [1.00, 1.01, 0.99, 0.98, 1.02, 0.60, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10],
        dtype=np.float64,
    )
    return phases, omegas


def run_demo() -> dict[str, object]:
    """Run no-action vs governor-coupling attribution for power-grid control."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid power_grid binding spec: {errors}")

    phases, omegas = power_grid_disturbance_state()
    coupling = CouplingBuilder().build(
        n_layers=phases.size,
        base_strength=spec.coupling.base_strength,
        decay_alpha=spec.coupling.decay_alpha,
    )
    action = ControlAction(
        knob="K",
        scope="global",
        value=0.08,
        ttl_s=spec.control_period_s * 20.0,
        justification="power-grid attribution demo: governor droop candidate",
    )
    rollout = CausalInterventionEngine(
        n_oscillators=phases.size,
        dt=spec.sample_period_s,
        horizon=40,
    ).evaluate_actions(
        phases,
        omegas,
        coupling.knm,
        coupling.alpha,
        zeta=spec.drivers.physical.get("zeta", 0.0),
        psi=spec.drivers.physical.get("psi", 0.0),
        actions=(action,),
    )
    attribution = rollout.attribute(threshold=1e-5)
    return {
        "domainpack": spec.name,
        "scenario": "load_step_governor_counterfactual",
        "counterfactual": rollout.to_audit_record(),
        "attribution": attribution.to_audit_record(),
    }


def main() -> None:
    """Print the attribution demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
