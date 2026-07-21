# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Neuromorphic IR export tests

"""Behaviour and edge-case tests for the NIR-structural graph export."""

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.adapters.neuromorphic_ir_export import (
    NIR_STRUCTURAL_FORMAT,
    UNMODELLED_NIR_LIF_PARAMETERS,
    NeuromorphicIRGraph,
    to_nir_graph,
)

POPULATIONS = [
    {"name": "layer_0", "estimated_rate_hz": 12.5, "lava_process": "LIF"},
    {"name": "layer_1", "estimated_rate_hz": 0.0, "lava_process": "LIF"},
]
PROJECTIONS = [
    {"source": "layer_0", "target": "layer_1", "weight": 0.4, "delay_ms": 1.0},
]


def _graph() -> NeuromorphicIRGraph:
    return to_nir_graph(
        POPULATIONS,
        PROJECTIONS,
        tau_membrane_ms=20.0,
        tau_refractory_ms=2.0,
    )


class TestGraphConstruction:
    """A well-formed schedule compiles into an honest NIR-structural graph."""

    def test_nodes_carry_only_modelled_lif_parameters(self):
        graph = _graph()
        assert [n["id"] for n in graph.nodes] == ["layer_0", "layer_1"]
        node = graph.nodes[0]
        assert node["type"] == "LIF"
        assert node["tau_membrane_ms"] == 20.0
        assert node["tau_refractory_ms"] == 2.0
        assert node["v_threshold_normalised"] == 1.0
        assert node["estimated_rate_hz"] == 12.5
        # No fabricated NIR physical parameters leak into the node.
        for unmodelled in UNMODELLED_NIR_LIF_PARAMETERS:
            assert unmodelled not in node

    def test_edges_are_weighted_linear_connections(self):
        graph = _graph()
        assert len(graph.edges) == 1
        edge = graph.edges[0]
        assert edge["type"] == "Linear"
        assert edge["source"] == "layer_0"
        assert edge["target"] == "layer_1"
        assert edge["weight"] == 0.4
        assert edge["delay_ms"] == 1.0

    def test_metadata_declares_honest_conformance_posture(self):
        meta = _graph().metadata
        assert meta["format"] == NIR_STRUCTURAL_FORMAT
        assert meta["conformance"] == "structural_subset"
        assert meta["reference_spec"] == "neuromorphs/NIR"
        assert meta["v_threshold_unit"] == "normalised"
        assert meta["unmodelled_nir_lif_parameters"] == list(
            UNMODELLED_NIR_LIF_PARAMETERS
        )
        assert meta["node_count"] == 2
        assert meta["edge_count"] == 1

    def test_empty_schedule_yields_empty_graph(self):
        graph = to_nir_graph([], [], tau_membrane_ms=10.0, tau_refractory_ms=1.0)
        assert graph.nodes == ()
        assert graph.edges == ()
        assert graph.metadata["node_count"] == 0

    def test_missing_estimated_rate_defaults_to_zero(self):
        graph = to_nir_graph(
            [{"name": "only"}], [], tau_membrane_ms=10.0, tau_refractory_ms=1.0
        )
        assert graph.nodes[0]["estimated_rate_hz"] == 0.0

    def test_missing_delay_defaults_to_zero(self):
        graph = to_nir_graph(
            [{"name": "a"}, {"name": "b"}],
            [{"source": "a", "target": "b", "weight": 0.1}],
            tau_membrane_ms=10.0,
            tau_refractory_ms=1.0,
        )
        assert graph.edges[0]["delay_ms"] == 0.0


class TestDeterminismAndHashing:
    """The graph and its digest are deterministic and JSON-safe."""

    def test_identical_input_yields_identical_sha256(self):
        assert _graph().sha256 == _graph().sha256
        assert len(_graph().sha256) == 64

    def test_weight_change_changes_the_digest(self):
        other = to_nir_graph(
            POPULATIONS,
            [{"source": "layer_0", "target": "layer_1", "weight": 0.9}],
            tau_membrane_ms=20.0,
            tau_refractory_ms=2.0,
        )
        assert other.sha256 != _graph().sha256

    def test_canonical_json_is_sorted_and_reloads(self):
        graph = _graph()
        payload = graph.canonical_json()
        assert json.loads(payload) == graph.to_record()
        # Canonical form is sorted and compact.
        assert " " not in payload

    def test_to_record_is_a_deep_copy(self):
        graph = _graph()
        record = graph.to_record()
        record["nodes"][0]["id"] = "mutated"
        assert graph.nodes[0]["id"] == "layer_0"


class TestValidation:
    """Malformed records and dangling edges are rejected."""

    def test_edge_referencing_undeclared_node_is_rejected(self):
        with pytest.raises(ValueError, match="references an undeclared node"):
            to_nir_graph(
                [{"name": "a"}],
                [{"source": "a", "target": "ghost", "weight": 0.1}],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_non_mapping_population_is_rejected(self):
        with pytest.raises(ValueError, match="population must be a mapping"):
            to_nir_graph(
                ["not-a-dict"],
                [],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_non_mapping_projection_is_rejected(self):
        with pytest.raises(ValueError, match="projection must be a mapping"):
            to_nir_graph(
                [{"name": "a"}],
                ["not-a-dict"],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_missing_population_name_is_rejected(self):
        with pytest.raises(ValueError, match="population.name"):
            to_nir_graph(
                [{"estimated_rate_hz": 1.0}],
                [],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_missing_projection_endpoint_is_rejected(self):
        with pytest.raises(ValueError, match="projection.target"):
            to_nir_graph(
                [{"name": "a"}],
                [{"source": "a", "weight": 0.1}],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_boolean_weight_is_rejected(self):
        with pytest.raises(ValueError, match="projection.weight must be a real number"):
            to_nir_graph(
                [{"name": "a"}, {"name": "b"}],
                [{"source": "a", "target": "b", "weight": True}],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_negative_weight_is_rejected(self):
        with pytest.raises(ValueError, match="projection.weight must be >= 0"):
            to_nir_graph(
                [{"name": "a"}, {"name": "b"}],
                [{"source": "a", "target": "b", "weight": -0.5}],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_non_finite_rate_is_rejected(self):
        with pytest.raises(ValueError, match="estimated_rate_hz must be finite"):
            to_nir_graph(
                [{"name": "a", "estimated_rate_hz": float("nan")}],
                [],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
            )

    def test_infinite_tau_is_rejected(self):
        with pytest.raises(ValueError, match="tau_membrane_ms must be finite"):
            to_nir_graph(
                [{"name": "a"}],
                [],
                tau_membrane_ms=float("inf"),
                tau_refractory_ms=1.0,
            )

    def test_negative_tau_refractory_is_rejected(self):
        with pytest.raises(ValueError, match="tau_refractory_ms must be >= 0"):
            to_nir_graph(
                [{"name": "a"}],
                [],
                tau_membrane_ms=10.0,
                tau_refractory_ms=-1.0,
            )

    def test_non_real_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="v_threshold_normalised must be a real"):
            to_nir_graph(
                [{"name": "a"}],
                [],
                tau_membrane_ms=10.0,
                tau_refractory_ms=1.0,
                v_threshold_normalised="high",
            )
