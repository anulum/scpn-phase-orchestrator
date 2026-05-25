# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Topology supervisor tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.supervisor import (
    HigherOrderTopologySupervisor,
    TopologyMutationPolicy,
    TopologyMutationResult,
)
from scpn_phase_orchestrator.upde.hypergraph import Hyperedge, HypergraphEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter


def _zero_knm(n: int) -> np.ndarray:
    return np.zeros((n, n), dtype=np.float64)


class TestTopologyMutationPolicy:
    def test_rejects_invalid_mutation_rate(self) -> None:
        with pytest.raises(ValueError, match="mutation_rate"):
            TopologyMutationPolicy(mutation_rate=1.5)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"mutation_rate": True}, "mutation_rate"),
            ({"coherence_floor": "0.75"}, "coherence_floor"),
            ({"pairwise_threshold": True}, "pairwise_threshold"),
            ({"simplex_threshold": "0.9"}, "simplex_threshold"),
            ({"max_pairwise_delta": True}, "max_pairwise_delta"),
            ({"max_simplex_strength": "0.2"}, "max_simplex_strength"),
            ({"prune_threshold": True}, "prune_threshold"),
            (
                {"simplex_pairwise_support_floor": "0.0"},
                "simplex_pairwise_support_floor",
            ),
            ({"max_coupling": True}, "max_coupling"),
            ({"max_coupling": 0.0}, "max_coupling"),
            ({"max_new_simplices": True}, "max_new_simplices"),
            ({"max_new_simplices": 1.5}, "max_new_simplices"),
        ],
    )
    def test_rejects_non_real_or_non_integral_policy_bounds(
        self,
        kwargs: dict[str, object],
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            TopologyMutationPolicy(**kwargs)

    def test_rejects_negative_simplex_budget(self) -> None:
        with pytest.raises(ValueError, match="max_new_simplices"):
            TopologyMutationPolicy(max_new_simplices=-1)

    def test_rejects_negative_simplex_support_floor(self) -> None:
        with pytest.raises(ValueError, match="simplex_pairwise_support_floor"):
            TopologyMutationPolicy(simplex_pairwise_support_floor=-0.1)


class TestHigherOrderTopologySupervisor:
    def test_zero_mutation_rate_freezes_topology(self) -> None:
        phases = np.array([0.0, 0.1, 2.4, 4.7])
        knm = _zero_knm(4)
        edges = (Hyperedge((0, 1, 2), strength=0.2),)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(mutation_rate=0.0)
        )

        result = supervisor.mutate(phases, knm, edges)

        np.testing.assert_allclose(result.knm, knm)
        assert result.hyperedges == edges
        assert result.added_simplices == ()
        assert result.pruned_simplices == ()
        assert result.pairwise_delta_norm == 0.0

    def test_adds_coherent_triad_when_global_coherence_is_low(self) -> None:
        phases = np.array([0.0, 0.03, 0.06, np.pi, np.pi + 0.02, np.pi + 0.04])
        knm = _zero_knm(6)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.9,
                simplex_threshold=0.99,
                max_new_simplices=2,
                max_simplex_strength=0.4,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert 0 < len(result.added_simplices) <= 2
        assert all(edge.order == 3 for edge in result.added_simplices)
        assert all(
            edge.strength == pytest.approx(0.2) for edge in result.added_simplices
        )
        assert result.global_coherence < 0.9
        assert result.to_audit_record()["hyperedge_count"] == len(result.hyperedges)

    def test_pairwise_update_is_bounded_and_keeps_zero_diagonal(self) -> None:
        phases = np.array([0.0, 0.02, 2.0, 4.0])
        knm = 0.1 * np.ones((4, 4), dtype=np.float64)
        np.fill_diagonal(knm, 0.0)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=1.0,
                pairwise_threshold=0.95,
                prune_threshold=0.1,
                max_pairwise_delta=0.2,
                max_coupling=0.25,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert np.all(result.knm >= 0.0)
        assert np.all(result.knm <= 0.25)
        np.testing.assert_allclose(np.diag(result.knm), 0.0)
        assert result.knm[0, 1] > knm[0, 1]
        assert result.pairwise_delta_norm > 0.0

    def test_prunes_incoherent_existing_simplex(self) -> None:
        phases = np.array([0.0, 2.0, 4.0, 0.1])
        edge = Hyperedge((0, 1, 2), strength=0.3)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(mutation_rate=0.5, prune_threshold=0.4)
        )

        result = supervisor.mutate(phases, _zero_knm(4), (edge,))

        assert edge in result.pruned_simplices
        assert edge not in result.hyperedges

    def test_canonicalises_existing_simplex_nodes(self) -> None:
        phases = np.array([0.0, 0.01, 0.02, np.pi])
        edge = Hyperedge((2, 0, 1), strength=0.3)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.95,
                simplex_threshold=0.99,
                max_new_simplices=4,
            )
        )

        result = supervisor.mutate(phases, _zero_knm(4), (edge,))

        assert Hyperedge((0, 1, 2), strength=0.3) in result.hyperedges
        assert result.added_simplices == ()

    def test_rejects_invalid_inputs(self) -> None:
        supervisor = HigherOrderTopologySupervisor()
        with pytest.raises(ValueError, match="at least one oscillator"):
            supervisor.mutate(np.array([]), np.zeros((0, 0)))
        with pytest.raises(ValueError, match="one-dimensional"):
            supervisor.mutate(np.zeros((2, 2)), _zero_knm(2))
        with pytest.raises(ValueError, match="phases must be finite"):
            supervisor.mutate(np.array([0.0, np.nan]), _zero_knm(2))
        with pytest.raises(ValueError, match="shape"):
            supervisor.mutate(np.zeros(3), _zero_knm(2))
        invalid_knm = _zero_knm(3)
        invalid_knm[0, 1] = np.inf
        with pytest.raises(ValueError, match="knm must be finite"):
            supervisor.mutate(np.zeros(3), invalid_knm)
        negative_knm = _zero_knm(3)
        negative_knm[0, 1] = -0.1
        with pytest.raises(ValueError, match="knm must be non-negative"):
            supervisor.mutate(np.zeros(3), negative_knm)
        with pytest.raises(ValueError, match="at least 3 nodes"):
            supervisor.mutate(np.zeros(3), _zero_knm(3), (Hyperedge((0, 1)),))
        with pytest.raises(ValueError, match="nodes must be unique"):
            supervisor.mutate(np.zeros(3), _zero_knm(3), (Hyperedge((0, 0, 1)),))
        with pytest.raises(ValueError, match="out of range"):
            supervisor.mutate(np.zeros(3), _zero_knm(3), (Hyperedge((0, 1, 4)),))
        with pytest.raises(ValueError, match="strength"):
            supervisor.mutate(
                np.zeros(3),
                _zero_knm(3),
                (Hyperedge((0, 1, 2), strength=np.nan),),
            )
        with pytest.raises(ValueError, match="hyperedges must be Hyperedge"):
            supervisor.mutate(np.zeros(3), _zero_knm(3), (object(),))
        with pytest.raises(ValueError, match="integer"):
            supervisor.mutate(
                np.zeros(4),
                _zero_knm(4),
                (Hyperedge((True, 2, 3), strength=0.1),),
            )
        with pytest.raises(ValueError, match="strength"):
            supervisor.mutate(
                np.zeros(3),
                _zero_knm(3),
                (Hyperedge((0, 1, 2), strength=True),),
            )

    @pytest.mark.parametrize(
        ("phases", "knm", "match"),
        [
            ([False, True, False], _zero_knm(3), "phases"),
            (
                np.zeros(3),
                [[False, True, False], [True, False, True], [False, True, False]],
                "knm",
            ),
        ],
    )
    def test_rejects_boolean_phase_or_coupling_aliases(
        self,
        phases: object,
        knm: object,
        match: str,
    ) -> None:
        supervisor = HigherOrderTopologySupervisor()

        with pytest.raises(ValueError, match=match):
            supervisor.mutate(phases, knm)

    def test_max_new_simplices_limits_candidate_additions(self) -> None:
        phases = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi], dtype=np.float64)
        knm = _zero_knm(6)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=1.0,
                coherence_floor=0.15,
                simplex_threshold=0.99,
                max_new_simplices=1,
                max_simplex_strength=0.5,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert len(result.added_simplices) == 1
        assert result.added_simplices[0].strength == pytest.approx(0.5)
        assert result.added_simplices[0].nodes in {(0, 1, 2), (3, 4, 5)}
        assert len(result.hyperedges) == len(result.added_simplices)

    def test_max_new_simplices_zero_disables_additions_even_when_coherent(self) -> None:
        phases = np.array([0.0, 0.0, 0.0, np.pi, np.pi, np.pi], dtype=np.float64)
        knm = _zero_knm(6)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=1.0,
                coherence_floor=0.15,
                simplex_threshold=0.99,
                max_new_simplices=0,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert result.added_simplices == ()
        assert result.hyperedges == ()

    def test_zero_strength_simplex_is_pruned(self) -> None:
        phases = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float64)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(coherence_floor=0.99)
        )
        existing = (Hyperedge((0, 1, 2), strength=0.0),)

        result = supervisor.mutate(phases, _zero_knm(4), existing)

        assert existing[0] in result.pruned_simplices
        assert result.hyperedges == ()

    def test_to_audit_record_is_deterministically_repeatable(self) -> None:
        phases = np.array([0.0, 0.03, 0.06, np.pi], dtype=np.float64)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.9,
                simplex_threshold=0.99,
                max_simplex_strength=0.2,
            )
        )

        result = supervisor.mutate(phases, _zero_knm(4))
        first = json.dumps(result.to_audit_record(), sort_keys=True)
        second = json.dumps(result.to_audit_record(), sort_keys=True)

        assert first == second

    def test_mutate_does_not_mutate_inputs(self) -> None:
        phases = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        knm = np.full((3, 3), 0.1, dtype=np.float64)
        np.fill_diagonal(knm, 0.0)
        original_knm = knm.copy()
        original_phases = phases.copy()
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.2,
                max_pairwise_delta=0.05,
                max_coupling=1.0,
            )
        )

        _ = supervisor.mutate(phases, knm)

        np.testing.assert_allclose(phases, original_phases)
        np.testing.assert_allclose(knm, original_knm)

    def test_to_audit_record_is_json_serialisable(self) -> None:
        phases = np.array([0.0, 0.03, 0.06, np.pi, np.pi + 0.02, np.pi + 0.04])
        result = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.95,
                simplex_threshold=0.99,
                max_simplex_strength=0.2,
            )
        ).mutate(phases, _zero_knm(6))

        payload = result.to_audit_record()
        payload_text = json.dumps(payload)

        assert payload["global_coherence"] == pytest.approx(result.global_coherence)
        assert payload["pairwise_delta_norm"] == pytest.approx(
            result.pairwise_delta_norm
        )
        assert payload["hyperedge_count"] == len(result.hyperedges)
        assert isinstance(payload_text, str)

    def test_does_not_add_simplices_when_global_coherence_is_already_high(self) -> None:
        phases = np.array([0.0, 0.01, 0.02, 0.03])
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=1.0,
                coherence_floor=0.8,
                simplex_threshold=0.99,
                max_new_simplices=4,
            )
        )

        result = supervisor.mutate(phases, _zero_knm(4))

        assert result.global_coherence >= 0.8
        assert result.added_simplices == ()
        assert result.hyperedges == ()

    def test_pairwise_support_floor_blocks_unsupported_candidate_simplex(self) -> None:
        phases = np.array([0.0, 0.02, 0.04, np.pi, np.pi + 0.02, np.pi + 0.04])
        knm = _zero_knm(6)
        knm[0, 1] = knm[1, 0] = 1.0
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.95,
                simplex_threshold=0.99,
                simplex_pairwise_support_floor=0.5,
                max_new_simplices=4,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert result.global_coherence < 0.95
        assert result.added_simplices == ()

    def test_pairwise_support_floor_requires_bidirectional_support(self) -> None:
        phases = np.array([0.0, 0.02, 0.04, np.pi, np.pi + 0.02, np.pi + 0.04])
        knm = _zero_knm(6)
        knm[0, 1] = knm[0, 2] = knm[1, 2] = 1.0
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.95,
                simplex_threshold=0.99,
                simplex_pairwise_support_floor=0.5,
                max_new_simplices=4,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert result.global_coherence < 0.95
        assert result.added_simplices == ()

    def test_result_feeds_hypergraph_engine(self) -> None:
        phases = np.array([0.0, 0.02, 0.04, np.pi, np.pi + 0.01, np.pi + 0.03])
        omegas = np.zeros(6, dtype=np.float64)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=0.5,
                coherence_floor=0.95,
                simplex_threshold=0.99,
                max_new_simplices=1,
            )
        )

        result = supervisor.mutate(phases, _zero_knm(6))
        engine = HypergraphEngine(6, dt=0.01, hyperedges=list(result.hyperedges))
        updated = engine.step(phases, omegas, pairwise_knm=result.knm)
        r_updated, _ = compute_order_parameter(updated)

        assert isinstance(result, TopologyMutationResult)
        assert updated.shape == phases.shape
        assert np.all(np.isfinite(updated))
        assert 0.0 <= r_updated <= 1.0

    def test_small_system_never_adds_simplices(self) -> None:
        phases = np.array([0.0, 1.25], dtype=np.float64)
        knm = np.array([[0.0, 0.25], [0.5, 0.0]], dtype=np.float64)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=1.0,
                coherence_floor=0.5,
                simplex_threshold=0.2,
                max_new_simplices=4,
            )
        )

        result = supervisor.mutate(phases, knm)

        assert result.added_simplices == ()
        assert result.pruned_simplices == ()
        assert result.hyperedges == ()

    def test_tied_candidate_simplex_ordering_is_deterministic(self) -> None:
        phases = np.array([0.0, 0.0, np.pi, np.pi], dtype=np.float64)
        knm = np.full((4, 4), 0.4, dtype=np.float64)
        np.fill_diagonal(knm, 0.0)
        supervisor = HigherOrderTopologySupervisor(
            TopologyMutationPolicy(
                mutation_rate=1.0,
                coherence_floor=0.1,
                simplex_threshold=0.33,
                max_new_simplices=2,
                max_simplex_strength=0.2,
            )
        )

        result = supervisor.mutate(phases, knm)
        expected_nodes = sorted(
            [
                (0, 1, 2),
                (0, 1, 3),
                (0, 2, 3),
                (1, 2, 3),
            ],
            reverse=True,
        )

        assert len(result.added_simplices) == 2
        assert tuple(edge.nodes for edge in result.added_simplices) == tuple(
            expected_nodes[:2]
        )
        assert all(
            edge.strength == pytest.approx(0.2) for edge in result.added_simplices
        )
