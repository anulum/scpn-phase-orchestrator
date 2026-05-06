# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid FEP hierarchy demo

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor import FEPPredictiveSupervisor
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")


def _state(r_value: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r_value, psi=0.0)],
        cross_layer_alignment=np.eye(1),
        stability_proxy=r_value,
        regime_id="nominal",
    )


def power_grid_hierarchy_state() -> dict[str, tuple[FloatArray, FloatArray]]:
    """Return deterministic child states for a two-region power-grid proof."""
    return {
        "generation_area": (
            np.array([0.0, 0.05, 0.10, 0.16, 0.21], dtype=np.float64),
            np.array([1.0, 1.02, 0.98, 0.50, 0.50], dtype=np.float64),
        ),
        "demand_renewable_area": (
            np.array([0.0, 2.30, 4.10, 1.20, 3.60], dtype=np.float64),
            np.array([0.30, 0.30, 0.10, 0.15, 0.08], dtype=np.float64),
        ),
    }


def _child_record(
    name: str,
    phases: FloatArray,
    omegas: FloatArray,
    *,
    dt: float,
    target_R: float,
) -> dict[str, object]:
    supervisor = FEPPredictiveSupervisor(
        n_oscillators=phases.size,
        dt=dt,
        target_R=target_R,
        free_energy_threshold=0.0,
        drive_gain=0.08,
    )
    assessment = supervisor.assess(phases, omegas)
    actions = supervisor.decide(
        phases,
        omegas,
        _state(assessment.observed_R),
        BoundaryState(),
    )
    return {
        "name": name,
        "assessment": assessment.to_audit_record(),
        "actions": [
            {
                "knob": action.knob,
                "scope": action.scope,
                "value": action.value,
                "ttl_s": action.ttl_s,
                "justification": action.justification,
            }
            for action in actions
        ],
    }


def _observed_r(record: dict[str, object]) -> float:
    assessment = cast(Mapping[str, object], record["assessment"])
    value = assessment["observed_R"]
    if not isinstance(value, (int, float)):
        raise TypeError("child assessment observed_R must be numeric")
    return float(value)


def run_demo() -> dict[str, object]:
    """Run a two-child plus parent FEP hierarchy proof for power-grid control."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid power_grid binding spec: {errors}")

    child_records = [
        _child_record(name, phases, omegas, dt=spec.sample_period_s, target_R=0.82)
        for name, (phases, omegas) in power_grid_hierarchy_state().items()
    ]
    child_rs = np.array(
        [_observed_r(record) for record in child_records],
        dtype=np.float64,
    )
    parent_phases = np.arccos(np.clip(2.0 * child_rs - 1.0, -1.0, 1.0))
    parent_omegas = np.full(parent_phases.shape, 1.0, dtype=np.float64)
    parent = FEPPredictiveSupervisor(
        n_oscillators=parent_phases.size,
        dt=spec.control_period_s,
        target_R=0.8,
        free_energy_threshold=0.0,
        drive_gain=0.05,
    )
    parent_assessment = parent.assess(parent_phases, parent_omegas)
    parent_actions = parent.decide(
        parent_phases,
        parent_omegas,
        _state(parent_assessment.observed_R),
        BoundaryState(),
    )
    return {
        "domainpack": spec.name,
        "hierarchy": "two_child_regions_to_parent_fep_supervisor",
        "children": child_records,
        "parent": {
            "assessment": parent_assessment.to_audit_record(),
            "actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in parent_actions
            ],
        },
    }


def main() -> None:
    """Print the hierarchy proof payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
