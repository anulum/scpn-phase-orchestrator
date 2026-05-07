# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Edge-consensus sheaf obstruction demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.edge_consensus_nchannel.sheaf_obstruction_demo import (
    CHANNELS,
    NODES,
    edge_consensus_restriction_maps,
    nominal_edge_consensus_state,
    run_demo,
    stressed_edge_consensus_state,
)


def test_edge_consensus_sheaf_demo_reports_stressed_obstruction_growth() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "edge_consensus_nchannel"
    assert payload["scenario"] == "heterogeneous_edge_gateway_obstruction"
    assert payload["actuating"] is False
    assert payload["nodes"] == list(NODES)
    assert payload["channels"] == list(CHANNELS)

    nominal = payload["nominal"]
    stressed = payload["stressed"]
    assert nominal["method"] == "directed_cellular_sheaf_laplacian"
    assert stressed["method"] == "directed_cellular_sheaf_laplacian"
    assert nominal["laplacian_shape"] == [18, 18]
    assert stressed["residual_shape"] == [3, 3, 6]
    assert stressed["obstruction_score"] > nominal["obstruction_score"] * 5.0
    assert payload["obstruction_delta"] > 0.5
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == (
        "edge_consensus_nchannel"
    )


def test_edge_consensus_sheaf_inputs_are_heterogeneous_and_finite() -> None:
    nominal = nominal_edge_consensus_state()
    stressed = stressed_edge_consensus_state()
    maps = edge_consensus_restriction_maps()

    assert nominal.shape == (3, 6)
    assert stressed.shape == (3, 6)
    assert maps.shape == (3, 3, 6, 6)
    assert np.all(np.isfinite(nominal))
    assert np.all(np.isfinite(stressed))
    assert np.all(np.isfinite(maps))
    assert stressed[1, CHANNELS.index("Load")] > nominal[1, CHANNELS.index("Load")]
    assert stressed[1, CHANNELS.index("Trust")] < nominal[1, CHANNELS.index("Trust")]
    assert np.linalg.norm(maps[2, 1]) > 0.0
