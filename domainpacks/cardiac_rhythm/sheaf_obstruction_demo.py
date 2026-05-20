# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cardiac rhythm sheaf obstruction demo

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
CHANNELS = ("PWave", "PRInterval", "QRSAmplitude", "QTInterval", "RRDispersion")
NODES = ("sa_node", "atrial_conduction", "ventricular_depolarization", "repolarization")


def nominal_cardiac_rhythm_sheaf_state() -> FloatArray:
    """Return deterministic nominal conduction-repolarisation coherence."""
    return np.array(
        [
            [1.000, 0.170, 0.250, 0.390, 0.520],
            [0.980, 0.182, 0.258, 0.372, 0.506],
            [0.960, 0.204, 0.268, 0.412, 0.468],
            [0.930, 0.235, 0.246, 0.430, 0.438],
        ],
        dtype=np.float64,
    )


def incident_cardiac_rhythm_sheaf_state() -> FloatArray:
    """Return deterministic arrhythmic stress with phase drift and QT stretch."""
    return np.array(
        [
            [0.990, 0.250, 0.240, 0.460, 0.520],
            [0.540, 0.660, 0.470, 0.360, 0.920],
            [0.280, 0.790, 0.720, 0.840, 0.290],
            [0.240, 0.650, 0.820, 1.040, 0.640],
        ],
        dtype=np.float64,
    )


def cardiac_restriction_maps() -> FloatArray:
    """Build directed heterogeneous inter-node restriction maps."""
    n_nodes = len(NODES)
    n_channels = len(CHANNELS)
    maps = np.zeros((n_nodes, n_nodes, n_channels, n_channels), dtype=np.float64)
    identity = np.eye(n_channels, dtype=np.float64)

    for target, source in (
        (1, 0),
        (0, 1),
        (2, 1),
        (1, 2),
        (3, 2),
        (2, 3),
        (0, 2),
        (2, 0),
        (3, 0),
        (0, 3),
    ):
        restriction = identity.copy()
        if target == 1 and source == 0:
            restriction[1, 0] = 0.012
            restriction[4, 1] = 0.010
        elif target == 0 and source == 1:
            restriction[0, 1] = -0.010
            restriction[4, 4] = 0.985
        elif target == 2 and source == 1:
            restriction[2, 1] = 0.020
            restriction[3, 2] = 0.024
        elif target == 1 and source == 2:
            restriction[1, 2] = 0.015
            restriction[4, 3] = -0.012
        elif target == 3 and source == 2:
            restriction[3, 2] = 0.030
            restriction[4, 3] = 0.015
        elif target == 2 and source == 3:
            restriction[2, 3] = -0.018
            restriction[4, 2] = 0.012
        elif target == 0 and source == 2:
            restriction[0, 2] = -0.012
            restriction[4, 4] = 0.980
        elif target == 2 and source == 0:
            restriction[2, 0] = 0.015
            restriction[1, 3] = 0.010
        elif target == 3 and source == 0:
            restriction[3, 1] = 0.012
            restriction[4, 0] = 0.015
        elif target == 0 and source == 3:
            restriction[0, 3] = -0.010
            restriction[4, 3] = 0.018
        maps[target, source] = restriction

    return maps


def run_demo() -> dict[str, object]:
    """Compare nominal coherence against an arrhythmic incident state."""
    spec = load_binding_spec(SPEC_PATH)
    errors = validate_binding_spec(spec)
    if errors:
        raise ValueError(f"invalid cardiac_rhythm binding spec: {errors}")

    supervisor = SheafCoherenceSupervisor(tolerance=1e-8)
    maps = cardiac_restriction_maps()
    nominal = supervisor.assess(nominal_cardiac_rhythm_sheaf_state(), maps)
    incident = supervisor.assess(incident_cardiac_rhythm_sheaf_state(), maps)

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
        "scenario": "arrhythmic_desynchronization_sheaf_obstruction",
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
    """Print the sheaf-obstruction demo payload as stable JSON."""
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
