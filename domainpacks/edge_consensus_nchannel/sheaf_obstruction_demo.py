# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Edge-consensus sheaf obstruction demo

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
CHANNELS = ("P", "I", "S", "Load", "Trust", "ConsensusHealth")
NODES = ("leaf_cluster", "regional_gateway", "parent_supervisor")


def nominal_edge_consensus_state() -> FloatArray:
    """Return a nominal heterogeneous edge-consensus sheaf section."""
    return np.array(
        [
            [0.82, 0.88, 0.25, 0.35, 0.91, 0.87],
            [0.80, 0.86, 0.25, 0.38, 0.88, 0.85],
            [0.78, 0.84, 0.25, 0.42, 0.86, 0.83],
        ],
        dtype=np.float64,
    )


def stressed_edge_consensus_state() -> FloatArray:
    """Return a gateway-stressed section with load/trust disagreement."""
    return np.array(
        [
            [0.82, 0.88, 0.25, 0.35, 0.91, 0.87],
            [0.62, 0.58, 0.50, 0.82, 0.45, 0.55],
            [0.78, 0.84, 0.25, 0.42, 0.86, 0.83],
        ],
        dtype=np.float64,
    )


def edge_consensus_restriction_maps() -> FloatArray:
    """Return directed heterogeneous channel restriction maps."""
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
            restriction[5, 0] = 0.20
            restriction[5, 1] = 0.20
            restriction[5, 4] = 0.20
            restriction[5, 5] = 0.40
        if target == 1 and source == 2:
            restriction[3, 3] = 0.80
        maps[target, source] = restriction
    return maps


def run_demo() -> dict[str, object]:
    """Compare nominal and stressed sheaf obstruction for edge consensus."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid edge_consensus_nchannel binding spec: {errors}")

    supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
    maps = edge_consensus_restriction_maps()
    nominal = supervisor.assess(nominal_edge_consensus_state(), maps)
    stressed = supervisor.assess(stressed_edge_consensus_state(), maps)
    nominal_summary = build_sheaf_obstruction_summary(
        nominal,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    stressed_summary = build_sheaf_obstruction_summary(
        stressed,
        warning_threshold=0.08,
        critical_threshold=0.35,
        top_k=3,
    )
    return {
        "domainpack": spec.name,
        "scenario": "heterogeneous_edge_gateway_obstruction",
        "nodes": list(NODES),
        "channels": list(CHANNELS),
        "nominal": nominal.to_audit_record(),
        "stressed": stressed.to_audit_record(),
        "nominal_summary": nominal_summary.to_audit_record(),
        "stressed_summary": stressed_summary.to_audit_record(),
        "obstruction_delta": (stressed.obstruction_score - nominal.obstruction_score),
        "actuating": False,
    }


def main() -> None:
    """Print the sheaf obstruction demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
