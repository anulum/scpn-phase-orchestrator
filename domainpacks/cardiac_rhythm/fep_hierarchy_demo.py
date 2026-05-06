# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac FEP hierarchy demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.supervisor import assess_fep_hierarchy

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")


def cardiac_hierarchy_state() -> dict[str, tuple[FloatArray, FloatArray]]:
    """Return deterministic child states for cardiac conduction hierarchy proof."""
    return {
        "pacemaker_atrial_axis": (
            np.array([0.00, 0.04, 0.08, 0.54, 0.58], dtype=np.float64),
            np.array([1.17, 0.83, 0.58, 0.75, 0.67], dtype=np.float64),
        ),
        "ventricular_recovery_axis": (
            np.array([2.70, 3.05, 3.40, 3.72, 4.05], dtype=np.float64),
            np.array([0.50, 0.42, 0.33, 0.33, 0.33], dtype=np.float64),
        ),
    }


def run_demo() -> dict[str, object]:
    """Run a two-child plus parent FEP hierarchy proof for cardiac rhythm."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid cardiac_rhythm binding spec: {errors}")

    hierarchy = assess_fep_hierarchy(
        cardiac_hierarchy_state(),
        dt=spec.sample_period_s,
        parent_dt=spec.control_period_s,
        child_target_R=0.78,
        parent_target_R=0.76,
        free_energy_threshold=0.0,
        child_drive_gain=0.06,
        parent_drive_gain=0.04,
        hierarchy="cardiac_child_axes_to_parent_fep_supervisor",
    )
    payload = hierarchy.to_audit_record()
    payload["domainpack"] = spec.name
    return {
        "domainpack": spec.name,
        **payload,
    }


def main() -> None:
    """Print the hierarchy proof payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
