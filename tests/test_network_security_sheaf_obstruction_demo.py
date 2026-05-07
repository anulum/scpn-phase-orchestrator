# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Network-security sheaf obstruction demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.network_security.sheaf_obstruction_demo import (
    CHANNELS,
    NODES,
    lateral_movement_sheaf_state,
    network_security_restriction_maps,
    nominal_network_security_sheaf_state,
    run_demo,
)


def test_network_security_sheaf_demo_reports_lateral_obstruction_growth() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "network_security"
    assert payload["scenario"] == "lateral_movement_sheaf_obstruction"
    assert payload["actuating"] is False
    assert payload["nodes"] == list(NODES)
    assert payload["channels"] == list(CHANNELS)

    nominal = payload["nominal"]
    lateral = payload["lateral_movement"]
    assert nominal["method"] == "directed_cellular_sheaf_laplacian"
    assert lateral["method"] == "directed_cellular_sheaf_laplacian"
    assert nominal["laplacian_shape"] == [12, 12]
    assert lateral["residual_shape"] == [3, 3, 4]
    assert lateral["obstruction_score"] > nominal["obstruction_score"] * 20.0
    assert payload["obstruction_delta"] > 0.5

    assert payload["nominal_summary"]["severity"] == "nominal"
    summary = payload["lateral_movement_summary"]
    assert summary["severity"] == "critical"
    assert len(summary["top_residual_edges"]) == 3
    assert (
        summary["top_residual_edges"][0]["norm"]
        >= (summary["top_residual_edges"][1]["norm"])
    )
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "network_security"
    )


def test_network_security_sheaf_inputs_capture_lateral_movement_channels() -> None:
    nominal = nominal_network_security_sheaf_state()
    lateral = lateral_movement_sheaf_state()
    maps = network_security_restriction_maps()

    assert nominal.shape == (3, 4)
    assert lateral.shape == (3, 4)
    assert maps.shape == (3, 3, 4, 4)
    assert np.all(np.isfinite(nominal))
    assert np.all(np.isfinite(lateral))
    assert np.all(np.isfinite(maps))
    assert lateral[1, CHANNELS.index("ThreatLevel")] > (
        nominal[1, CHANNELS.index("ThreatLevel")] * 4.0
    )
    assert (
        lateral[1, CHANNELS.index("TrustScore")]
        < (nominal[1, CHANNELS.index("TrustScore")])
    )
    assert (
        lateral[2, CHANNELS.index("DefensePhase")]
        < (nominal[2, CHANNELS.index("DefensePhase")])
    )
    assert np.linalg.norm(maps[2, 1]) > 0.0
