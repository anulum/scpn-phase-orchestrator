# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid sheaf obstruction demo

from __future__ import annotations

import json
from pathlib import Path
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.binding import load_binding_spec, validate_binding_spec
from scpn_phase_orchestrator.supervisor import (
    SheafCoherenceSupervisor,
    build_sheaf_obstruction_summary,
)

FloatArray: TypeAlias = NDArray[np.float64]

SPEC_PATH = Path(__file__).with_name("binding_spec.yaml")
CHANNELS = (
    "RotorAngle",
    "FrequencyDeviation",
    "TieLineFlow",
    "LoadDemand",
    "RenewableRamp",
)
NODES = (
    "generation_area",
    "tie_line_corridor",
    "load_area",
    "renewable_area",
)


def nominal_power_grid_sheaf_state() -> FloatArray:
    """Return a nominal four-region grid section."""
    return np.array(
        [
            [0.04, 0.01, 0.22, 0.58, 0.31],
            [0.05, 0.02, 0.24, 0.60, 0.34],
            [0.06, 0.02, 0.23, 0.62, 0.35],
            [0.07, 0.01, 0.21, 0.59, 0.38],
        ],
        dtype=np.float64,
    )


def line_fault_power_grid_sheaf_state() -> FloatArray:
    """Return a line-fault section with tie-flow and demand disagreement."""
    return np.array(
        [
            [0.04, 0.01, 0.22, 0.58, 0.31],
            [0.42, -0.18, 0.91, 0.88, 0.40],
            [0.25, -0.12, 0.52, 0.95, 0.37],
            [0.10, 0.04, 0.29, 0.63, 0.82],
        ],
        dtype=np.float64,
    )


def power_grid_restriction_maps() -> FloatArray:
    """Return directed channel maps for grid-area consistency checks."""
    n_nodes = len(NODES)
    n_channels = len(CHANNELS)
    maps = np.zeros((n_nodes, n_nodes, n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)
    for target, source in (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (1, 3),
        (3, 1),
        (0, 2),
        (2, 0),
    ):
        restriction = identity.copy()
        if target == 0 and source == 1:
            restriction[1, 2] = -0.12
        if target == 1 and source == 0:
            restriction[2, 0] = 0.18
        if target == 2 and source == 1:
            restriction[3, 2] = 0.20
        if target == 1 and source == 2:
            restriction[2, 3] = -0.10
        if target == 1 and source == 3:
            restriction[2, 4] = 0.16
        if target == 3 and source == 1:
            restriction[4, 2] = 0.12
        maps[target, source] = restriction
    return maps


def run_demo() -> dict[str, object]:
    """Compare nominal and line-fault sheaf obstruction for power-grid replay."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid power_grid binding spec: {errors}")

    supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
    maps = power_grid_restriction_maps()
    nominal = supervisor.assess(nominal_power_grid_sheaf_state(), maps)
    line_fault = supervisor.assess(line_fault_power_grid_sheaf_state(), maps)
    nominal_summary = build_sheaf_obstruction_summary(
        nominal,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    line_fault_summary = build_sheaf_obstruction_summary(
        line_fault,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    return {
        "domainpack": spec.name,
        "scenario": "line_fault_grid_sheaf_obstruction",
        "nodes": list(NODES),
        "channels": list(CHANNELS),
        "nominal": nominal.to_audit_record(),
        "line_fault": line_fault.to_audit_record(),
        "nominal_summary": nominal_summary.to_audit_record(),
        "line_fault_summary": line_fault_summary.to_audit_record(),
        "obstruction_delta": line_fault.obstruction_score - nominal.obstruction_score,
        "actuating": False,
    }


def main() -> None:
    """Print the sheaf obstruction demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
