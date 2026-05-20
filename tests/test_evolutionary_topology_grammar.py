# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for offline topology mutation grammar

from __future__ import annotations

import json

import pytest

from scpn_phase_orchestrator.supervisor.evolutionary_topology_grammar import (
    TopologyMutationCandidate,
    run_offline_evolutionary_topology_mutation_search,
)


def _node_records() -> list[dict[str, object]]:
    return [
        {"node_id": 0, "community": "alpha"},
        {"node_id": 1, "community": "alpha"},
        {"node_id": 2, "community": "beta"},
    ]


def _edge_records() -> list[dict[str, object]]:
    return [
        {"nodes": [0, 1], "weight": 0.28},
    ]


def test_search_is_deterministic_for_identical_inputs() -> None:
    first = run_offline_evolutionary_topology_mutation_search(
        _node_records(),
        _edge_records(),
        generation_count=2,
        population_size=4,
        mutation_step=0.03,
    )
    second = run_offline_evolutionary_topology_mutation_search(
        _node_records(),
        _edge_records(),
        generation_count=2,
        population_size=4,
        mutation_step=0.03,
    )

    assert first == second
    assert first.report_hash == second.report_hash
    assert first.candidate_count == 8
    assert len(first.candidates) == 8
    assert first.report_hash


def test_search_generates_expected_topology_operations() -> None:
    report = run_offline_evolutionary_topology_mutation_search(
        _node_records(),
        _edge_records(),
        generation_count=1,
        population_size=8,
        mutation_step=0.02,
        max_add_candidates=4,
    )

    operations = {candidate.plan.operation for candidate in report.candidates}
    assert "edge_reweight" in operations
    assert "edge_remove" in operations
    assert "edge_add" in operations
    assert "community_bridge" in operations

    assert all(
        isinstance(candidate, TopologyMutationCandidate)
        for candidate in report.candidates
    )
    assert all(candidate.operator_review_required for candidate in report.candidates)
    assert all(candidate.execution_disabled for candidate in report.candidates)
    assert all(
        candidate.live_merge_permitted is False for candidate in report.candidates
    )
    assert all(
        candidate.hot_patch_permitted is False for candidate in report.candidates
    )
    assert all(
        candidate.actuation_permitted is False for candidate in report.candidates
    )


def test_report_flags_mark_no_live_actuation() -> None:
    report = run_offline_evolutionary_topology_mutation_search(
        _node_records(),
        _edge_records(),
        generation_count=1,
        population_size=2,
        mutation_step=0.01,
    )

    assert report.operator_review_required is True
    assert report.non_actuating is True
    assert report.execution_disabled is True
    assert report.live_merge_permitted is False
    assert report.hot_patch_permitted is False
    assert report.actuation_permitted is False


def test_audit_records_are_json_safe() -> None:
    report = run_offline_evolutionary_topology_mutation_search(
        _node_records(),
        _edge_records(),
        generation_count=1,
        population_size=2,
        mutation_step=0.01,
    )

    encoded = json.dumps(report.to_audit_record(), allow_nan=False)
    decoded = json.loads(encoded)

    assert isinstance(decoded["schema_name"], str)
    assert isinstance(decoded["candidates"], list)
    for candidate in report.candidates:
        candidate_encoded = json.dumps(candidate.to_audit_record(), allow_nan=False)
        assert isinstance(json.loads(candidate_encoded), dict)


def test_inputs_fail_closed_for_malformed_records() -> None:
    with pytest.raises(ValueError, match="node_records must be a non-empty sequence"):
        run_offline_evolutionary_topology_mutation_search(
            [],
            _edge_records(),
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match=r"node_records\[0\] must be a mapping"):
        run_offline_evolutionary_topology_mutation_search(
            ["bad-node"],
            _edge_records(),
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match=r"edge_records\[0\] must be a mapping"):
        run_offline_evolutionary_topology_mutation_search(
            _node_records(),
            ["bad-edge"],
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(
        ValueError, match=r"edge_records\[0\] nodes must be a two-element sequence"
    ):
        run_offline_evolutionary_topology_mutation_search(
            _node_records(),
            [{"nodes": [1], "weight": 0.2}],
            generation_count=1,
            population_size=1,
        )

    with pytest.raises(ValueError, match=r"edge_records\[0\] references unknown node"):
        run_offline_evolutionary_topology_mutation_search(
            _node_records(),
            [{"source": 0, "target": 99, "weight": 0.3}],
            generation_count=1,
            population_size=1,
        )
