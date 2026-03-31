# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Visualization tests

from __future__ import annotations

import json

import numpy as np

from scpn_phase_orchestrator.visualization.network import (
    coupling_heatmap_json,
    network_graph_json,
)
from scpn_phase_orchestrator.visualization.torus import (
    phase_wheel_json,
    torus_points_json,
)


class TestNetworkGraph:
    def test_valid_json(self):
        knm = np.full((4, 4), 0.5)
        np.fill_diagonal(knm, 0.0)
        data = json.loads(network_graph_json(knm))
        assert "nodes" in data
        assert "links" in data

    def test_node_count(self):
        knm = np.eye(3) * 0
        data = json.loads(network_graph_json(knm))
        assert len(data["nodes"]) == 3

    def test_link_threshold(self):
        knm = np.array([[0, 0.5, 0.001], [0.5, 0, 0.001], [0.001, 0.001, 0]])
        data = json.loads(network_graph_json(knm, threshold=0.01))
        assert len(data["links"]) == 1

    def test_custom_names(self):
        knm = np.full((2, 2), 0.5)
        np.fill_diagonal(knm, 0.0)
        data = json.loads(network_graph_json(knm, layer_names=["Alpha", "Beta"]))
        assert data["nodes"][0]["name"] == "Alpha"

    def test_R_values(self):
        knm = np.full((2, 2), 0.5)
        np.fill_diagonal(knm, 0.0)
        data = json.loads(network_graph_json(knm, R_values=[0.8, 0.9]))
        assert data["nodes"][0]["R"] == 0.8


class TestCouplingHeatmap:
    def test_valid_json(self):
        knm = np.full((3, 3), 0.5)
        np.fill_diagonal(knm, 0.0)
        data = json.loads(coupling_heatmap_json(knm))
        assert "labels" in data
        assert "matrix" in data
        assert len(data["matrix"]) == 3

    def test_min_max(self):
        knm = np.array([[0, 1], [0.5, 0]])
        data = json.loads(coupling_heatmap_json(knm))
        assert data["min"] == 0.0
        assert data["max"] == 1.0


class TestTorusPoints:
    def test_valid_json(self):
        phases = np.array([0.0, 1.0, 2.0, 3.0])
        data = json.loads(torus_points_json(phases))
        assert "points" in data
        assert len(data["points"]) == 4

    def test_point_fields(self):
        phases = np.array([0.0, np.pi])
        data = json.loads(torus_points_json(phases))
        p = data["points"][0]
        assert "x" in p and "y" in p and "z" in p
        assert "phase" in p and "R" in p

    def test_custom_radii(self):
        phases = np.array([0.0])
        data = json.loads(torus_points_json(phases, major_radius=5.0, minor_radius=1.0))
        p = data["points"][0]
        assert abs(p["x"] - 6.0) < 0.01  # (5+1)·cos(0)

    def test_R_values(self):
        phases = np.array([0.0, 1.0])
        data = json.loads(torus_points_json(phases, R_values=[0.5, 0.9]))
        assert data["points"][0]["R"] == 0.5


class TestPhaseWheel:
    def test_valid_json(self):
        phases = np.array([0.0, np.pi / 2, np.pi])
        data = json.loads(phase_wheel_json(phases))
        assert "oscillators" in data
        assert len(data["oscillators"]) == 3

    def test_unit_circle(self):
        phases = np.array([0.0])
        data = json.loads(phase_wheel_json(phases))
        osc = data["oscillators"][0]
        assert abs(osc["x"] - 1.0) < 1e-4
        assert abs(osc["y"] - 0.0) < 1e-4

    def test_custom_names(self):
        phases = np.array([0.0, 1.0])
        data = json.loads(phase_wheel_json(phases, layer_names=["A", "B"]))
        assert data["oscillators"][0]["name"] == "A"


class TestVisualizationPipelineWiring:
    """Pipeline: engine → phases/K_nm → visualization JSON."""

    def test_engine_state_to_visualization(self):
        """Engine phases + K_nm → network_graph_json + phase_wheel_json."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.5 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, np.zeros((n, n)))

        graph = json.loads(network_graph_json(knm))
        assert len(graph["nodes"]) == n
        assert len(graph["links"]) > 0

        wheel = json.loads(phase_wheel_json(phases))
        assert len(wheel["oscillators"]) == n
