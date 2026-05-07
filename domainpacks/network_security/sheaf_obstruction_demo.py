# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network-security sheaf obstruction demo

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
CHANNELS = ("TrafficRate", "ThreatLevel", "DefensePhase", "TrustScore")
NODES = ("normal_traffic", "attack_vector", "defense_response")


def nominal_network_security_sheaf_state() -> FloatArray:
    """Return a nominal traffic/attack/defence sheaf section."""
    return np.array(
        [
            [0.75, 0.20, 0.70, 0.86],
            [0.73, 0.22, 0.69, 0.84],
            [0.74, 0.21, 0.71, 0.85],
        ],
        dtype=np.float64,
    )


def lateral_movement_sheaf_state() -> FloatArray:
    """Return a lateral-movement section with threat/defence disagreement."""
    return np.array(
        [
            [0.82, 0.12, 0.20, 0.90],
            [0.68, 0.96, 0.55, 0.18],
            [0.50, 0.52, 0.40, 0.48],
        ],
        dtype=np.float64,
    )


def network_security_restriction_maps() -> FloatArray:
    """Return directed channel maps for security-cohort consistency checks."""
    n_nodes = len(NODES)
    n_channels = len(CHANNELS)
    maps = np.zeros((n_nodes, n_nodes, n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)
    for target, source in (
        (0, 1),
        (1, 0),
        (1, 2),
        (2, 1),
        (0, 2),
        (2, 0),
    ):
        restriction = identity.copy()
        if target == 2 and source == 1:
            restriction[2, 1] = 0.05
            restriction[3, 1] = -0.03
        if target == 1 and source == 2:
            restriction[1, 2] = 0.04
            restriction[1, 3] = -0.03
        if target == 0 and source == 2:
            restriction[0, 2] = 0.03
            restriction[3, 2] = 0.02
        maps[target, source] = restriction
    return maps


def run_demo() -> dict[str, object]:
    """Compare nominal and lateral-movement sheaf obstruction for security."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid network_security binding spec: {errors}")

    supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
    maps = network_security_restriction_maps()
    nominal = supervisor.assess(nominal_network_security_sheaf_state(), maps)
    lateral = supervisor.assess(lateral_movement_sheaf_state(), maps)
    nominal_summary = build_sheaf_obstruction_summary(
        nominal,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    lateral_summary = build_sheaf_obstruction_summary(
        lateral,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    return {
        "domainpack": spec.name,
        "scenario": "lateral_movement_sheaf_obstruction",
        "nodes": list(NODES),
        "channels": list(CHANNELS),
        "nominal": nominal.to_audit_record(),
        "lateral_movement": lateral.to_audit_record(),
        "nominal_summary": nominal_summary.to_audit_record(),
        "lateral_movement_summary": lateral_summary.to_audit_record(),
        "obstruction_delta": lateral.obstruction_score - nominal.obstruction_score,
        "actuating": False,
    }


def main() -> None:
    """Print the sheaf obstruction demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
