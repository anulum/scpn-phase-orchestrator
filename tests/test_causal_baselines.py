# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — causal baseline numeric contract tests

"""Real trace tests for causal baseline scaling and fail-closed arithmetic."""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import (
    CausalInterventionEngine,
    CounterfactualRollout,
    build_temporal_causal_hypergraph_experiment,
    learn_causal_graph,
)
from scpn_phase_orchestrator.supervisor._causal_baselines import (
    _causal_baseline_family,
)


def _trace(scale: float) -> dict[str, list[float]]:
    return {
        "driver": [scale * value for value in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)],
        "response": [scale * value for value in (0.0, 0.0, 1.0, 2.0, 3.0, 4.0)],
    }


def _candidate() -> list[dict[str, object]]:
    return [
        {
            "sources": ["driver"],
            "target": "response",
            "time_offsets": [-1],
            "score": 0.5,
        }
    ]


def _baseline_records(report: dict[str, object]) -> list[dict[str, float | int | str]]:
    baseline = report["baseline"]
    assert isinstance(baseline, dict)
    records = baseline["baseline_family"]
    assert isinstance(records, list)
    assert all(isinstance(record, dict) for record in records)
    return cast(list[dict[str, float | int | str]], records)


def test_large_finite_traces_preserve_public_causal_baseline_verdict() -> None:
    """Public baseline scores and graph records survive finite trace rescaling."""
    reference = build_temporal_causal_hypergraph_experiment(_trace(1.0), _candidate())
    high_magnitude_trace = _trace(1e155)
    scaled = build_temporal_causal_hypergraph_experiment(
        high_magnitude_trace, _candidate()
    )

    expected = {
        record["name"]: record["score"] for record in _baseline_records(reference)
    }
    actual = {record["name"]: record["score"] for record in _baseline_records(scaled)}
    assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12)
    assert scaled["baseline_beaten"] is False
    assert scaled["accepted_hyperedge_count"] == 0

    graph = learn_causal_graph(high_magnitude_trace)
    baseline = scaled["baseline"]
    assert isinstance(baseline, dict)
    assert graph.to_audit_record()["edges"] == baseline["edges"]
    assert _causal_baseline_family(
        high_magnitude_trace, lag=1, min_abs_weight=1e-6, graph=graph
    ) == _baseline_records(scaled)


def test_public_causal_entries_refuse_nested_trace_samples() -> None:
    """A sample vector cannot be replaced by a same-length matrix of JSON rows."""
    trace = json.loads('{"driver": [[0.0], [1.0], [2.0]], "response": [0.0, 1.0, 2.0]}')
    with pytest.raises(
        ValueError, match="trace signal 'driver' must be one-dimensional"
    ):
        learn_causal_graph(trace)
    with pytest.raises(
        ValueError, match="trace signal 'driver' must be one-dimensional"
    ):
        build_temporal_causal_hypergraph_experiment(trace, _candidate())


def test_unrepresentable_influence_refuses_public_causal_evidence() -> None:
    """Finite subnormal drivers cannot yield an infinite influence edge or report."""
    trace = {
        "driver": [0.0, 1e-320, 2e-320, 3e-320, 4e-320],
        "response": [0.0, 0.0, 1.0, 3.0, 6.0],
    }
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="causal trace influence must be finite"):
            learn_causal_graph(trace)
        with pytest.raises(ValueError, match="causal trace influence must be finite"):
            build_temporal_causal_hypergraph_experiment(trace, _candidate())


def test_baseline_centring_overflow_refuses_research_report() -> None:
    """A valid lagged graph cannot bypass overflow in future-sample correlation."""
    trace = {
        "driver": [0.0, 1.0, 2.0, 3.0],
        "response": [0.0, 0.0, 9e307, 9e307],
    }
    with np.errstate(over="ignore", invalid="ignore"):
        graph = learn_causal_graph(trace)
        assert not graph.edges
        with pytest.raises(
            ValueError, match="causal baseline centred samples must be finite"
        ):
            build_temporal_causal_hypergraph_experiment(trace, _candidate())


def test_unrepresentable_trace_delta_refuses_public_causal_estimate() -> None:
    trace = {
        "driver": [0.0, 1e308, -1e308, 0.0],
        "response": [0.0, 0.0, 1e308, -1e308],
    }
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="centred samples must be finite"):
            learn_causal_graph(trace)
        with pytest.raises(ValueError, match="centred samples must be finite"):
            build_temporal_causal_hypergraph_experiment(trace, _candidate())


@pytest.mark.parametrize(
    ("delta", "action_value", "message"),
    [
        (float("nan"), 0.1, "causal effects must be finite"),
        (float("inf"), 0.1, "causal effects must be finite"),
        (1.0, float("nan"), "action values must be finite"),
        (1.0, 1e-320, "causal influence must be finite"),
    ],
)
def test_invalid_rollout_cannot_become_confident_causal_edge(
    delta: float, action_value: float, message: str
) -> None:
    rollout = CounterfactualRollout(
        baseline_R=[0.5],
        intervention_R=[0.5],
        baseline_psi=[0.0],
        intervention_psi=[0.0],
        delta_R_final=delta,
        delta_R_mean=delta,
        delta_psi_final=0.0,
        actions=(ControlAction("K", "global", action_value, 1.0, "review"),),
    )
    with pytest.raises(ValueError, match=message):
        learn_causal_graph(_trace(1.0), (rollout,))


def test_zero_information_trace_has_no_causal_baseline_edges() -> None:
    trace = {"driver": [0.0] * 6, "response": [0.0] * 6}
    graph = learn_causal_graph(trace)
    assert not graph.edges
    experiment = build_temporal_causal_hypergraph_experiment(trace, _candidate())
    for baseline in _baseline_records(experiment):
        assert baseline["score"] == 0.0
        assert baseline["edge_count"] == 0


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ('{"sources": ["driver"]}', "must be a non-empty sequence"),
        ("[42]", "must be a mapping"),
        ("[{}]", "sources must be a list"),
        ('[{"sources": []}]', "sources must be a list"),
        ('[{"sources": [42]}]', "sources must be non-empty strings"),
        ('[{"sources": [""]}]', "sources must be non-empty strings"),
        ('[{"sources": ["driver"]}]', "target is required"),
        (
            '[{"sources": ["driver"], "target": "response"}]',
            "time_offsets must be a list",
        ),
        (
            '[{"sources": ["driver"], "target": "response", "time_offsets": []}]',
            "time_offsets must be a list",
        ),
        (
            '[{"sources": ["driver"], "target": "response", "time_offsets": [true]}]',
            "time_offsets must be integers",
        ),
        (
            '[{"sources": ["driver"], "target": "response", "time_offsets": [0.5]}]',
            "time_offsets must be integers",
        ),
    ],
)
def test_temporal_candidate_json_refusal_preserves_research_inputs(
    payload: str, message: str
) -> None:
    """Malformed candidates cannot mutate evidence or admit a research report."""
    candidates = json.loads(payload)
    trace = _trace(1.0)
    original_trace = json.dumps(trace, sort_keys=True)
    original_candidates = json.dumps(candidates, sort_keys=True)
    with pytest.raises(ValueError, match=message):
        build_temporal_causal_hypergraph_experiment(trace, candidates)
    assert json.dumps(trace, sort_keys=True) == original_trace
    assert json.dumps(candidates, sort_keys=True) == original_candidates

    recovered = build_temporal_causal_hypergraph_experiment(trace, _candidate())
    assert recovered["research_only"] is True
    assert recovered["production_claim_permitted"] is False
    assert recovered["hot_patch_permitted"] is False
    assert recovered["actuation_permitted"] is False
    assert recovered["candidate_hyperedge_count"] == 1
    assert recovered["accepted_hyperedge_count"] == 0


@pytest.mark.parametrize("parameter", ["knm", "alpha"])
def test_public_intervention_matrix_shape_refusal_preserves_parameters(
    parameter: str,
) -> None:
    """Wrong-sized action matrices fail before altering caller-owned parameters."""
    engine = CausalInterventionEngine(2, 0.01, horizon=2)
    knm = np.zeros((1, 2)) if parameter == "knm" else np.zeros((2, 2))
    alpha = np.zeros((1, 2)) if parameter == "alpha" else np.zeros((2, 2))
    original_knm = knm.copy()
    original_alpha = alpha.copy()
    action = ControlAction("K", "global", 0.25, 1.0, "shape admission")
    with pytest.raises(ValueError, match=parameter + r"\.shape"):
        engine.apply_actions(knm, alpha, 0.0, 0.0, (action,))
    np.testing.assert_array_equal(knm, original_knm)
    np.testing.assert_array_equal(alpha, original_alpha)

    valid_knm = np.zeros((2, 2))
    valid_alpha = np.zeros((2, 2))
    admitted = engine.apply_actions(valid_knm, valid_alpha, 0.0, 0.0, (action,))
    np.testing.assert_array_equal(admitted.knm, [[0.0, 0.25], [0.25, 0.0]])
    np.testing.assert_array_equal(admitted.alpha, valid_alpha)
    np.testing.assert_array_equal(valid_knm, np.zeros((2, 2)))
