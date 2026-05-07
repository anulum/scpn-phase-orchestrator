# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Power-grid sheaf obstruction demo tests

from __future__ import annotations

import json

import numpy as np

from domainpacks.power_grid.sheaf_obstruction_demo import (
    CHANNELS,
    NODES,
    line_fault_power_grid_sheaf_state,
    nominal_power_grid_sheaf_state,
    power_grid_restriction_maps,
    run_demo,
)


def test_power_grid_sheaf_demo_reports_line_fault_obstruction_growth() -> None:
    payload = run_demo()

    assert payload["domainpack"] == "power_grid"
    assert payload["scenario"] == "line_fault_grid_sheaf_obstruction"
    assert payload["actuating"] is False
    assert payload["nodes"] == list(NODES)
    assert payload["channels"] == list(CHANNELS)

    nominal = payload["nominal"]
    line_fault = payload["line_fault"]
    assert nominal["method"] == "directed_cellular_sheaf_laplacian"
    assert line_fault["method"] == "directed_cellular_sheaf_laplacian"
    assert nominal["laplacian_shape"] == [20, 20]
    assert line_fault["residual_shape"] == [4, 4, 5]
    assert line_fault["obstruction_score"] > nominal["obstruction_score"] * 3.0
    assert payload["obstruction_delta"] > 0.5

    summary = payload["line_fault_summary"]
    assert summary["severity"] == "critical"
    assert len(summary["top_residual_edges"]) == 3
    assert (
        summary["top_residual_edges"][0]["norm"]
        >= (summary["top_residual_edges"][1]["norm"])
    )
    assert json.loads(json.dumps(payload, sort_keys=True))["domainpack"] == "power_grid"


def test_power_grid_sheaf_demo_inputs_capture_grid_fault_channels() -> None:
    nominal = nominal_power_grid_sheaf_state()
    line_fault = line_fault_power_grid_sheaf_state()
    maps = power_grid_restriction_maps()

    assert nominal.shape == (4, 5)
    assert line_fault.shape == (4, 5)
    assert maps.shape == (4, 4, 5, 5)
    assert np.all(np.isfinite(nominal))
    assert np.all(np.isfinite(line_fault))
    assert np.all(np.isfinite(maps))
    assert line_fault[1, CHANNELS.index("TieLineFlow")] > (
        nominal[1, CHANNELS.index("TieLineFlow")] * 3.0
    )
    assert (
        line_fault[2, CHANNELS.index("LoadDemand")]
        > (nominal[2, CHANNELS.index("LoadDemand")])
    )
    assert line_fault[3, CHANNELS.index("RenewableRamp")] > (
        nominal[3, CHANNELS.index("RenewableRamp")] * 2.0
    )
    assert np.linalg.norm(maps[2, 1]) > 0.0
