# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Traffic-flow sheaf obstruction demo

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
CHANNELS = ("P", "I", "S", "Weather", "Equity")
NODES = (
    "intersection",
    "corridor",
    "network",
    "demand",
    "weather",
    "equity_pressure",
)


def nominal_traffic_sheaf_state() -> FloatArray:
    """Return a balanced nominal traffic-flow sheaf section."""
    return np.array(
        [
            [0.12, 0.22, 0.55, 0.18, 0.12],
            [0.14, 0.24, 0.57, 0.20, 0.14],
            [0.16, 0.26, 0.59, 0.22, 0.16],
            [0.31, 0.34, 0.43, 0.25, 0.18],
            [0.22, 0.15, 0.33, 0.67, 0.24],
            [0.24, 0.19, 0.44, 0.40, 0.41],
        ],
        dtype=np.float64,
    )


def incident_traffic_sheaf_state() -> FloatArray:
    """Return a congested/spillback traffic-flow sheaf section."""
    return np.array(
        [
            [0.21, 0.72, 0.94, 0.34, 0.48],
            [0.43, 0.84, 0.96, 0.49, 0.56],
            [0.55, 0.66, 0.74, 0.63, 0.72],
            [0.88, 0.95, 0.49, 0.62, 0.94],
            [0.35, 0.30, 0.57, 0.91, 0.76],
            [0.41, 0.33, 0.50, 0.74, 0.98],
        ],
        dtype=np.float64,
    )


def traffic_restriction_maps() -> FloatArray:
    """Build directed heterogeneous channel-restriction maps."""
    n_nodes = len(NODES)
    n_channels = len(CHANNELS)
    maps = np.zeros((n_nodes, n_nodes, n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)

    for target, source in (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (3, 1),
        (1, 3),
        (3, 2),
        (2, 3),
        (4, 1),
        (1, 4),
        (5, 2),
        (2, 5),
    ):
        restriction = identity.copy()
        if target == 0 and source == 1:
            restriction[1, 2] = 0.14
            restriction[4, 3] = 0.08
            restriction[3, 4] = -0.05
        elif target == 1 and source == 0:
            restriction[2, 1] = 0.10
            restriction[0, 2] = 0.06
        elif target == 1 and source == 2:
            restriction[0, 1] = 0.07
            restriction[4, 2] = -0.04
        elif target == 2 and source == 1:
            restriction[3, 1] = 0.09
            restriction[2, 0] = 0.05
        elif target == 3 and source == 1:
            restriction[1, 0] = 0.04
            restriction[1, 4] = 0.06
        elif target == 1 and source == 3:
            restriction[2, 1] = 0.08
            restriction[0, 3] = 0.03
        elif target == 3 and source == 2:
            restriction[1, 3] = 0.12
            restriction[4, 2] = -0.03
        elif target == 2 and source == 3:
            restriction[0, 2] = 0.06
            restriction[2, 4] = 0.04
        elif target == 4 and source == 1:
            restriction[3, 4] = -0.02
            restriction[2, 1] = 0.05
        elif target == 1 and source == 4:
            restriction[2, 4] = 0.09
            restriction[0, 4] = 0.05
        elif target == 5 and source == 2:
            restriction[4, 4] = 0.70
            restriction[3, 0] = 0.08
        elif target == 2 and source == 5:
            restriction[1, 4] = 0.11
            restriction[2, 4] = 0.03
        maps[target, source] = restriction

    return maps


def run_demo() -> dict[str, object]:
    """Compare nominal and spillback traffic sheaf coherence."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid traffic_flow binding spec: {errors}")

    supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
    maps = traffic_restriction_maps()
    nominal = supervisor.assess(nominal_traffic_sheaf_state(), maps)
    incident = supervisor.assess(incident_traffic_sheaf_state(), maps)
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
        "scenario": "congested_spillback_traffic_sheaf_obstruction",
        "nodes": list(NODES),
        "channels": list(CHANNELS),
        "nominal": nominal.to_audit_record(),
        "incident": incident.to_audit_record(),
        "nominal_summary": nominal_summary.to_audit_record(),
        "incident_summary": incident_summary.to_audit_record(),
        "obstruction_delta": float(
            incident.obstruction_score - nominal.obstruction_score
        ),
        "actuating": False,
    }


def main() -> None:
    """Print the sheaf obstruction demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
