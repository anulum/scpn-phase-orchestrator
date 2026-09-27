# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — causal baseline numeric contract tests

"""Real trace tests for causal baseline scaling and fail-closed arithmetic."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import (
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
    assert _causal_baseline_family(
        high_magnitude_trace, lag=1, min_abs_weight=1e-6, graph=graph
    ) == _baseline_records(scaled)


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
