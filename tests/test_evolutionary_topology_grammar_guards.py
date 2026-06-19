# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topology mutation grammar validation guards

from __future__ import annotations

from typing import Any

import pytest

from scpn_phase_orchestrator.supervisor.evolutionary_topology_grammar import (
    run_offline_evolutionary_topology_mutation_search as _run_search,
)

_NODES = [
    {"node_id": 0, "community": "alpha"},
    {"node_id": 1, "community": "alpha"},
    {"node_id": 2, "community": "beta"},
]
_EDGES = [{"nodes": [0, 1], "weight": 0.28}]


def _run(
    nodes: Any = None,
    edges: Any = None,
    **config: Any,
) -> Any:
    return _run_search(
        _NODES if nodes is None else nodes,
        _EDGES if edges is None else edges,
        **config,
    )


class TestConfigGuards:
    @pytest.mark.parametrize(
        ("config", "match"),
        [
            (
                {"min_edge_weight": 5.0, "max_edge_weight": 1.0},
                "min_edge_weight must not exceed max_edge_weight",
            ),
            (
                {"max_add_candidates": 0},
                "max_add_candidates must be a positive integer",
            ),
            ({"generation_count": 1.5}, "generation_count must be a positive integer"),
            ({"generation_count": 0}, "generation_count must be a positive integer"),
            ({"mutation_step": 0.0}, "mutation_step must be a positive finite number"),
            (
                {"min_edge_weight": -1.0},
                "min_edge_weight must be finite and non-negative",
            ),
            ({"mutation_step": "fast"}, "mutation_step must be a finite real number"),
            (
                {"mutation_step": float("inf")},
                "mutation_step must be a finite real number",
            ),
        ],
    )
    def test_rejects_invalid_config(self, config: dict[str, Any], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _run(**config)


class TestNodeRecordGuards:
    def test_rejects_non_sequence_node_records(self) -> None:
        with pytest.raises(ValueError, match="non-empty sequence of mappings"):
            _run(nodes="not-a-sequence", edges=[])

    def test_rejects_node_without_id(self) -> None:
        with pytest.raises(ValueError, match="requires node_id"):
            _run(nodes=[{"community": "a"}], edges=[])

    def test_rejects_duplicate_node_id(self) -> None:
        with pytest.raises(ValueError, match="duplicate node_id"):
            _run(nodes=[{"node_id": 0}, {"node_id": 0}], edges=[])

    def test_rejects_non_integer_node_id(self) -> None:
        with pytest.raises(ValueError, match="node_id must be an integer"):
            _run(nodes=[{"node_id": "x"}], edges=[])


class TestEdgeRecordGuards:
    def test_rejects_non_sequence_edge_records(self) -> None:
        with pytest.raises(ValueError, match="edge_records must be a sequence"):
            _run(edges="not-a-sequence")

    def test_rejects_edge_without_endpoints(self) -> None:
        with pytest.raises(ValueError, match="must define nodes or source/target"):
            _run(edges=[{}])

    def test_rejects_edge_nodes_not_a_sequence(self) -> None:
        with pytest.raises(ValueError, match="nodes must be a sequence"):
            _run(edges=[{"nodes": 5}])

    def test_rejects_self_loop_edge(self) -> None:
        with pytest.raises(ValueError, match="two distinct nodes"):
            _run(edges=[{"nodes": [0, 0]}])

    def test_rejects_duplicate_edge(self) -> None:
        with pytest.raises(ValueError, match="duplicate edge"):
            _run(edges=[{"nodes": [0, 1]}, {"nodes": [1, 0]}])

    def test_rejects_negative_edge_weight(self) -> None:
        with pytest.raises(ValueError, match="weight must be non-negative"):
            _run(edges=[{"nodes": [0, 1], "weight": -1.0}])


class TestMutationAxisGeneration:
    def test_single_node_topology_has_no_mutation_axes(self) -> None:
        with pytest.raises(ValueError, match="do not enable topology mutation axis"):
            _run(nodes=[{"node_id": 0}], edges=[])

    def test_single_community_yields_no_bridge_axes(self) -> None:
        report = _run(
            nodes=[{"node_id": 0, "community": "a"}, {"node_id": 1, "community": "a"}],
            edges=[{"nodes": [0, 1], "weight": 0.5}],
            generation_count=1,
            population_size=2,
        )
        assert report.candidate_count == 2

    def test_saturated_cross_community_pair_skips_bridge(self) -> None:
        report = _run(
            nodes=[{"node_id": 0, "community": "a"}, {"node_id": 1, "community": "b"}],
            edges=[{"nodes": [0, 1], "weight": 0.5}],
            generation_count=1,
            population_size=2,
        )
        assert report.candidate_count == 2

    def test_edge_add_candidates_are_capped(self) -> None:
        # Five fully disconnected nodes leave many addable pairs; the cap of one
        # stops axis generation after the first edge-add candidate.
        report = _run(
            nodes=[{"node_id": index} for index in range(5)],
            edges=[{"nodes": [0, 1], "weight": 0.5}],
            max_add_candidates=1,
            generation_count=1,
            population_size=2,
        )
        assert report.candidate_count == 2


class TestBlockedReweightAndAdd:
    def test_blocks_reweight_and_add_below_minimum_weight(self) -> None:
        report = _run(min_edge_weight=0.5, generation_count=2, population_size=6)

        blocked = {
            reason
            for candidate in report.candidates
            for reason in candidate.blocked_reasons
        }
        assert "edge_reweight_below_min_weight" in blocked
        assert "edge_add_below_min_weight" in blocked

    def test_blocks_reweight_and_add_above_maximum_weight(self) -> None:
        report = _run(
            edges=[{"nodes": [0, 1], "weight": 0.29}],
            max_edge_weight=0.3,
            edge_add_base_weight=0.4,
            mutation_step=1.0,
            generation_count=3,
            population_size=8,
        )

        blocked = {
            reason
            for candidate in report.candidates
            for reason in candidate.blocked_reasons
        }
        assert "edge_reweight_above_max_weight" in blocked
        assert "edge_add_above_max_weight" in blocked

    def test_blocks_removal_of_zero_weight_edge(self) -> None:
        report = _run(
            edges=[{"nodes": [0, 1], "weight": 0.0}],
            generation_count=2,
            population_size=6,
        )

        blocked = {
            reason
            for candidate in report.candidates
            for reason in candidate.blocked_reasons
        }
        assert "edge_remove_from_zero_weight" in blocked
