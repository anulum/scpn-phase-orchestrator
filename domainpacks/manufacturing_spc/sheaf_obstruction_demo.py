# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Manufacturing SPC sheaf obstruction demo

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
    "vibration",
    "temperature",
    "pressure",
    "flow_rate",
    "oee",
    "cycle_time",
    "throughput",
    "yield_rate",
    "defect_class",
)
NODES = (
    "sensor",
    "machine",
    "line",
)


def manufacturing_nominal_sheaf_state() -> FloatArray:
    """Return a stable manufacturing process sheaf section."""
    return np.array(
        [
            [0.32, 0.44, 0.35, 0.58, 0.82, 0.73, 0.76, 0.88, 0.12],
            [0.31, 0.43, 0.34, 0.60, 0.84, 0.72, 0.77, 0.87, 0.11],
            [0.33, 0.45, 0.33, 0.59, 0.83, 0.74, 0.75, 0.89, 0.13],
        ],
        dtype=np.float64,
    )


def manufacturing_process_drift_sheaf_state() -> FloatArray:
    """Return an out-of-control process section with aligned drift and defects."""
    return np.array(
        [
            [0.43, 0.58, 0.69, 0.84, 0.64, 0.51, 0.60, 0.85, 0.28],
            [0.56, 0.77, 0.48, 0.83, 0.56, 0.45, 0.42, 0.64, 0.45],
            [0.56, 0.64, 0.52, 0.65, 0.31, 0.27, 0.29, 0.62, 0.61],
        ],
        dtype=np.float64,
    )


def manufacturing_restriction_maps() -> FloatArray:
    """Build directed heterogeneous channel restrictions for manufacturing coupling."""
    n_nodes = len(NODES)
    n_channels = len(CHANNELS)
    maps = np.zeros((n_nodes, n_nodes, n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)

    for target, source in (
        (1, 0),
        (0, 1),
        (1, 2),
        (2, 1),
        (2, 0),
        (0, 2),
    ):
        restriction = identity.copy()
        if target == 1 and source == 0:
            restriction[4, 2] = 0.24
            restriction[5, 0] = 0.19
            restriction[6, 3] = 0.15
        elif target == 0 and source == 1:
            restriction[0, 4] = -0.17
            restriction[1, 5] = -0.12
            restriction[2, 6] = -0.10
        elif target == 1 and source == 2:
            restriction[4, 7] = 0.22
            restriction[5, 8] = -0.18
            restriction[6, 7] = 0.16
        elif target == 2 and source == 1:
            restriction[7, 4] = 0.20
            restriction[8, 6] = 0.18
            restriction[7, 8] = -0.09
        elif target == 2 and source == 0:
            restriction[7, 1] = 0.11
            restriction[8, 2] = 0.14
            restriction[8, 3] = -0.08
        elif target == 0 and source == 2:
            restriction[2, 8] = -0.11
            restriction[3, 7] = -0.07
            restriction[1, 6] = 0.10
        maps[target, source] = restriction

    return maps


def run_demo() -> dict[str, object]:
    """Compare nominal and out-of-control manufacturing obstructions."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid manufacturing_spc binding spec: {errors}")

    supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
    maps = manufacturing_restriction_maps()
    nominal = supervisor.assess(manufacturing_nominal_sheaf_state(), maps)
    incident = supervisor.assess(manufacturing_process_drift_sheaf_state(), maps)

    nominal_summary = build_sheaf_obstruction_summary(
        nominal,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    incident_summary = build_sheaf_obstruction_summary(
        incident,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )

    return {
        "domainpack": spec.name,
        "scenario": "manufacturing_process_drift_obstruction",
        "nodes": list(NODES),
        "channels": list(CHANNELS),
        "nominal": nominal.to_audit_record(),
        "incident": incident.to_audit_record(),
        "nominal_summary": nominal_summary.to_audit_record(),
        "incident_summary": incident_summary.to_audit_record(),
        "obstruction_delta": incident.obstruction_score - nominal.obstruction_score,
        "actuating": False,
    }


def main() -> None:
    """Print the sheaf obstruction demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
