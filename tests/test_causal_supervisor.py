# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Causal supervisor tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.supervisor import (
    CausalGraphEstimate,
    CausalInfluenceEdge,
    CausalInterventionEngine,
    CounterfactualRollout,
    build_temporal_causal_hypergraph_experiment,
    learn_causal_graph,
)


def _system(n: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(11)
    phases = rng.uniform(0.0, 2.0 * np.pi, n)
    omegas = rng.normal(1.0, 0.2, n)
    knm = np.full((n, n), 0.05, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((n, n), dtype=np.float64)
    return phases, omegas, knm, alpha


def test_global_k_intervention_changes_counterfactual_trajectory() -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01, horizon=12)
    rollout = engine.evaluate_actions(
        phases,
        omegas,
        knm,
        alpha,
        0.0,
        0.0,
        [
            ControlAction(
                knob="K",
                scope="global",
                value=0.4,
                ttl_s=5.0,
                justification="test K intervention",
            )
        ],
    )
    assert isinstance(rollout, CounterfactualRollout)
    assert len(rollout.baseline_R) == 13
    assert len(rollout.intervention_R) == 13
    assert any(
        abs(a - b) > 1e-8
        for a, b in zip(rollout.baseline_R, rollout.intervention_R, strict=True)
    )
    assert -1.0 <= rollout.delta_R_final <= 1.0


def test_no_action_counterfactual_matches_baseline() -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01, horizon=8)
    rollout = engine.evaluate_actions(phases, omegas, knm, alpha, 0.0, 0.0, [])
    np.testing.assert_allclose(rollout.baseline_R, rollout.intervention_R)
    np.testing.assert_allclose(rollout.baseline_psi, rollout.intervention_psi)
    assert rollout.delta_R_final == pytest.approx(0.0)
    assert rollout.delta_R_mean == pytest.approx(0.0)


def test_counterfactual_attribution_thresholds_neutral_effect() -> None:
    rollout = CounterfactualRollout(
        baseline_R=[0.3, 0.31, 0.32],
        intervention_R=[0.3, 0.3101, 0.3202],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.1, 0.2],
        delta_R_final=0.0002,
        delta_R_mean=0.0001,
        delta_psi_final=0.0,
        actions=(),
    )

    attribution = rollout.attribute(threshold=0.001)

    assert attribution.effect == "neutral"
    assert attribution.score == pytest.approx(0.00015)
    assert 0.0 < attribution.confidence < 1.0
    assert attribution.to_audit_record() == {
        "effect": "neutral",
        "confidence": attribution.confidence,
        "score": attribution.score,
        "delta_R_final": 0.0002,
        "delta_R_mean": 0.0001,
        "threshold": 0.001,
    }


def test_counterfactual_attribution_classifies_signed_effects() -> None:
    stabilising = CounterfactualRollout(
        baseline_R=[0.2, 0.25, 0.3],
        intervention_R=[0.2, 0.33, 0.46],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.12, 0.25],
        delta_R_final=0.16,
        delta_R_mean=0.08,
        delta_psi_final=0.05,
        actions=(),
    )
    destabilising = CounterfactualRollout(
        baseline_R=[0.6, 0.58, 0.56],
        intervention_R=[0.6, 0.5, 0.43],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.08, 0.15],
        delta_R_final=-0.13,
        delta_R_mean=-0.07,
        delta_psi_final=-0.05,
        actions=(),
    )

    assert stabilising.attribute(threshold=0.01).effect == "stabilising"
    assert stabilising.attribute(threshold=0.01).confidence == pytest.approx(1.0)
    assert destabilising.attribute(threshold=0.01).effect == "destabilising"
    with pytest.raises(ValueError, match="threshold must be finite and non-negative"):
        stabilising.attribute(threshold=-0.1)


def test_audit_record_is_serialisable_shape() -> None:
    phases, omegas, knm, alpha = _system(4)
    engine = CausalInterventionEngine(4, dt=0.01, horizon=3)
    action = ControlAction(
        knob="zeta",
        scope="global",
        value=0.1,
        ttl_s=2.0,
        justification="test drive",
    )
    record = engine.evaluate_actions(
        phases, omegas, knm, alpha, 0.0, 0.0, [action]
    ).to_audit_record()
    assert set(record) == {
        "baseline_R",
        "intervention_R",
        "baseline_psi",
        "intervention_psi",
        "delta_R_final",
        "delta_R_mean",
        "delta_psi_final",
        "actions",
    }
    assert record["actions"] == [
        {
            "knob": "zeta",
            "scope": "global",
            "value": 0.1,
            "ttl_s": 2.0,
            "justification": "test drive",
        }
    ]


def test_unsupported_scope_raises() -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01)
    with pytest.raises(ValueError, match="layer-scoped"):
        engine.evaluate_actions(
            phases,
            omegas,
            knm,
            alpha,
            0.0,
            0.0,
            [
                ControlAction(
                    knob="K",
                    scope="layer_0",
                    value=0.1,
                    ttl_s=1.0,
                    justification="needs layer membership",
                )
            ],
        )

    with pytest.raises(ValueError, match="unsupported causal intervention scope"):
        engine.evaluate_actions(
            phases,
            omegas,
            knm,
            alpha,
            0.0,
            0.0,
            [
                ControlAction(
                    knob="K",
                    scope="ring_0",
                    value=0.1,
                    ttl_s=1.0,
                    justification="unknown scope must fail closed",
                )
            ],
        )


def test_alpha_oscillator_and_global_phase_interventions_apply_bounds() -> None:
    _, _, knm, alpha = _system(4)
    engine = CausalInterventionEngine(4, dt=0.01)

    params = engine.apply_actions(
        knm,
        alpha,
        zeta=0.2,
        psi=2.0 * np.pi - 0.05,
        actions=(
            ControlAction(
                knob="alpha",
                scope="oscillator_1",
                value=0.25,
                ttl_s=1.0,
                justification="shift oscillator phase lag row and column",
            ),
            ControlAction(
                knob="zeta",
                scope="global",
                value=-0.1,
                ttl_s=1.0,
                justification="shift global drive",
            ),
            ControlAction(
                knob="Psi",
                scope="global",
                value=0.2,
                ttl_s=1.0,
                justification="wrap global phase target",
            ),
        ),
    )

    assert params.zeta == pytest.approx(0.1)
    assert params.psi == pytest.approx(0.15)
    np.testing.assert_allclose(np.diag(params.alpha), 0.0)
    assert params.alpha[1, 0] == pytest.approx(0.25)
    assert params.alpha[0, 1] == pytest.approx(0.25)
    assert params.alpha[2, 3] == pytest.approx(0.0)


def test_unsupported_intervention_knob_raises() -> None:
    _, _, knm, alpha = _system(4)
    engine = CausalInterventionEngine(4, dt=0.01)

    with pytest.raises(ValueError, match="unsupported causal intervention knob"):
        engine.apply_actions(
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            actions=(
                ControlAction(
                    knob="gain",
                    scope="global",
                    value=0.1,
                    ttl_s=1.0,
                    justification="unsupported knob must fail closed",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("knob", "value"),
    [
        ("K", np.inf),
        ("alpha", "0.1"),
        ("zeta", True),
        ("Psi", np.nan),
    ],
)
def test_invalid_causal_action_values_fail_closed(knob: str, value: object) -> None:
    _, _, knm, alpha = _system(4)
    engine = CausalInterventionEngine(4, dt=0.01)

    with pytest.raises(ValueError, match="action.value must be finite"):
        engine.apply_actions(
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            actions=(
                ControlAction(
                    knob=knob,
                    scope="global",
                    value=value,
                    ttl_s=1.0,
                    justification="invalid action value must fail closed",
                ),
            ),
        )


@pytest.mark.parametrize("scope", ["oscillator_bad", "oscillator_99"])
def test_invalid_oscillator_scope_fails_closed(scope: str) -> None:
    _, _, knm, alpha = _system(4)
    engine = CausalInterventionEngine(4, dt=0.01)

    with pytest.raises(ValueError, match="oscillator-scoped causal intervention"):
        engine.apply_actions(
            knm,
            alpha,
            zeta=0.0,
            psi=0.0,
            actions=(
                ControlAction(
                    knob="K",
                    scope=scope,
                    value=0.1,
                    ttl_s=1.0,
                    justification="invalid oscillator scope must fail closed",
                ),
            ),
        )


def test_input_shape_validation() -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01)
    with pytest.raises(ValueError, match="phases.shape"):
        engine.evaluate_actions(phases[:-1], omegas, knm, alpha, 0.0, 0.0, [])

    invalid_knm = knm.copy()
    invalid_knm[0, 1] = np.nan
    with pytest.raises(ValueError, match="knm contains NaN/Inf"):
        engine.evaluate_actions(phases, omegas, invalid_knm, alpha, 0.0, 0.0, [])

    with pytest.raises(ValueError, match="zeta must be finite"):
        engine.evaluate_actions(phases, omegas, knm, alpha, np.inf, 0.0, [])


@pytest.mark.parametrize(
    ("field", "payload", "match"),
    [
        ("phases", [0.0, 1.0, True, 2.0], "phases must not contain boolean"),
        (
            "omegas",
            np.asarray([1.0, complex(1.1, 0.0), 0.9, 1.2], dtype=object),
            "omegas must contain real-valued",
        ),
        (
            "knm",
            np.asarray(
                [
                    [0.0, complex(0.1, 0.0), 0.1, 0.1],
                    [0.1, 0.0, 0.1, 0.1],
                    [0.1, 0.1, 0.0, 0.1],
                    [0.1, 0.1, 0.1, 0.0],
                ],
                dtype=object,
            ),
            "knm must contain real-valued",
        ),
        (
            "alpha",
            [
                [0.0, False, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            "alpha must not contain boolean",
        ),
    ],
)
def test_evaluate_actions_rejects_boolean_and_complex_array_aliases(
    field: str,
    payload: object,
    match: str,
) -> None:
    phases, omegas, knm, alpha = _system(4)
    values: dict[str, object] = {
        "phases": phases,
        "omegas": omegas,
        "knm": knm,
        "alpha": alpha,
    }
    values[field] = payload
    engine = CausalInterventionEngine(4, dt=0.01, horizon=2)

    with pytest.raises(ValueError, match=match):
        engine.evaluate_actions(
            values["phases"],
            values["omegas"],
            values["knm"],
            values["alpha"],
            0.0,
            0.0,
            [],
        )


def test_apply_actions_rejects_non_real_matrix_parameters() -> None:
    _, _, knm, alpha = _system(3)
    engine = CausalInterventionEngine(3, dt=0.01)
    complex_knm = np.asarray(
        [
            [0.0, complex(0.1, 0.0), 0.1],
            [0.1, 0.0, 0.1],
            [0.1, 0.1, 0.0],
        ],
        dtype=object,
    )
    boolean_alpha = [[0.0, True, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    with pytest.raises(ValueError, match="knm must contain real-valued"):
        engine.apply_actions(complex_knm, alpha, 0.0, 0.0, ())

    with pytest.raises(ValueError, match="alpha must not contain boolean"):
        engine.apply_actions(knm, boolean_alpha, 0.0, 0.0, ())


def test_evaluate_actions_accepts_array_like_inputs_after_validation() -> None:
    phases, omegas, knm, alpha = _system(4)
    engine = CausalInterventionEngine(4, dt=0.01, horizon=2)

    rollout = engine.evaluate_actions(
        phases.tolist(),
        omegas.tolist(),
        knm.tolist(),
        alpha.tolist(),
        zeta=0.0,
        psi=0.0,
        actions=[],
    )

    assert len(rollout.baseline_R) == 3
    assert len(rollout.intervention_R) == 3


@pytest.mark.parametrize(
    ("zeta", "psi", "message"),
    [
        (True, 0.0, "zeta must be finite"),
        (0.0, True, "psi must be finite"),
        ("0.0", 0.0, "zeta must be finite"),
        (0.0, "0.0", "psi must be finite"),
    ],
)
def test_evaluate_actions_rejects_non_real_drive_scalars(
    zeta: object,
    psi: object,
    message: str,
) -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01)

    with pytest.raises(ValueError, match=message):
        engine.evaluate_actions(phases, omegas, knm, alpha, zeta, psi, [])


@pytest.mark.parametrize(
    ("n_oscillators", "dt", "horizon", "message"),
    [
        (0, 0.01, 20, "n_oscillators must be >= 1"),
        (True, 0.01, 20, "n_oscillators must be an integer"),
        (4, 0.0, 20, "dt must be finite and > 0"),
        (4, np.inf, 20, "dt must be finite and > 0"),
        (4, True, 20, "dt must be finite and > 0"),
        (4, "0.01", 20, "dt must be finite and > 0"),
        (4, 0.01, 0, "horizon must be >= 1"),
        (4, 0.01, True, "horizon must be a positive integer"),
        (4, 0.01, 1.5, "horizon must be a positive integer"),
    ],
)
def test_constructor_validation(
    n_oscillators: int | bool,
    dt: object,
    horizon: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CausalInterventionEngine(n_oscillators, dt=dt, horizon=horizon)


def test_exported_from_supervisor_package() -> None:
    import scpn_phase_orchestrator.supervisor as supervisor

    assert supervisor.CausalInterventionEngine is CausalInterventionEngine
    assert supervisor.learn_causal_graph is learn_causal_graph


def test_learn_causal_graph_estimates_directed_lagged_influence() -> None:
    driver = [0.0, 1.0, 2.0, 3.0, 4.0]
    response = [0.0, 0.0, 2.0, 6.0, 12.0]
    distractor = [1.0, 1.0, 1.0, 1.0, 1.0]

    graph = learn_causal_graph(
        {"driver": driver, "response": response, "distractor": distractor},
        lag=1,
        min_abs_weight=0.1,
    )
    records = [edge.to_audit_record() for edge in graph.edges]

    assert isinstance(graph, CausalGraphEstimate)
    assert graph.nodes == ("driver", "response", "distractor", "R")
    assert (
        CausalInfluenceEdge(
            source="driver",
            target="response",
            weight=2.0,
            confidence=1.0,
            lag=1,
            evidence="lagged_trace",
        )
        in graph.edges
    )
    assert all(record["source"] != "distractor" for record in records)
    assert graph.to_audit_record()["edges"][0]["evidence"] == "lagged_trace"


def test_learn_causal_graph_adds_counterfactual_do_edges() -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01, horizon=6)
    rollout = engine.evaluate_actions(
        phases,
        omegas,
        knm,
        alpha,
        0.0,
        0.0,
        [
            ControlAction(
                knob="K",
                scope="global",
                value=0.4,
                ttl_s=5.0,
                justification="graph intervention",
            )
        ],
    )

    graph = learn_causal_graph(
        {"R_baseline": rollout.baseline_R, "R_intervention": rollout.intervention_R},
        [rollout],
        min_abs_weight=1e-12,
    )
    do_edges = [
        edge for edge in graph.edges if edge.evidence == "counterfactual_rollout"
    ]

    assert graph.nodes[-2:] == ("do(K:global)", "R")
    assert len(do_edges) == 1
    assert do_edges[0].source == "do(K:global)"
    assert do_edges[0].target == "R"
    assert do_edges[0].weight == pytest.approx(rollout.delta_R_mean / 0.4)
    assert 0.0 <= do_edges[0].confidence <= 1.0


def test_learn_causal_graph_filters_tiny_counterfactual_do_edges() -> None:
    rollout = CounterfactualRollout(
        baseline_R=[0.4, 0.41, 0.42],
        intervention_R=[0.4, 0.41001, 0.42002],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.1, 0.2],
        delta_R_final=0.00002,
        delta_R_mean=0.00001,
        delta_psi_final=0.0,
        actions=(
            ControlAction(
                knob="K",
                scope="global",
                value=0.0,
                ttl_s=1.0,
                justification="zero-valued do intervention records audit node",
            ),
        ),
    )

    graph = learn_causal_graph(
        {"baseline": rollout.baseline_R, "intervention": rollout.intervention_R},
        [rollout],
        min_abs_weight=0.001,
    )

    assert "do(K:global)" in graph.nodes
    assert all(edge.evidence != "counterfactual_rollout" for edge in graph.edges)


def test_learn_causal_graph_rejects_invalid_trace_inputs() -> None:
    with pytest.raises(ValueError, match="at least two signals"):
        learn_causal_graph({"R": [0.1, 0.2]})

    with pytest.raises(ValueError, match="equal length"):
        learn_causal_graph({"a": [0.1, 0.2], "b": [0.1]})

    with pytest.raises(ValueError, match="greater than lag"):
        learn_causal_graph({"a": [0.1], "b": [0.2]}, lag=1)

    with pytest.raises(ValueError, match="positive integer"):
        learn_causal_graph({"a": [0.1, 0.2], "b": [0.2, 0.3]}, lag=0)

    with pytest.raises(ValueError, match="finite and non-negative"):
        learn_causal_graph(
            {"a": [0.1, 0.2], "b": [0.2, 0.3]},
            min_abs_weight=np.nan,
        )

    with pytest.raises(ValueError, match="trace signal 'a' contains NaN/Inf"):
        learn_causal_graph({"a": [0.1, np.inf], "b": [0.2, 0.3]})

    with pytest.raises(ValueError, match="trace signal 'a' must not contain boolean"):
        learn_causal_graph({"a": [0.1, True, 0.3], "b": [0.2, 0.3, 0.4]})

    with pytest.raises(ValueError, match="trace signal 'a' must contain real-valued"):
        learn_causal_graph(
            {
                "a": np.asarray([0.1, complex(0.2, 0.0), 0.3], dtype=object),
                "b": [0.2, 0.3, 0.4],
            }
        )


def test_learn_causal_graph_rejects_empty_trace_evidence() -> None:
    with pytest.raises(ValueError, match="at least two signals"):
        learn_causal_graph({})


def test_temporal_causal_hypergraph_experiment_beats_baseline_research_only() -> None:
    trace = {
        "driver": [0.0, 1.0, 2.0, 3.0, 4.0],
        "response": [0.0, 0.0, 2.0, 6.0, 12.0],
        "distractor": [1.0, 1.0, 1.0, 1.0, 1.0],
    }
    candidate_hyperedges = [
        {
            "sources": ["driver", "response"],
            "target": "response",
            "time_offsets": [-1, 0],
            "score": 2.6,
        },
        {
            "sources": ["distractor", "driver"],
            "target": "response",
            "time_offsets": [-1, 1],
            "score": 0.1,
        },
    ]

    first = build_temporal_causal_hypergraph_experiment(
        trace,
        candidate_hyperedges,
        lag=1,
        min_abs_weight=0.1,
        required_baseline_margin=0.1,
    )
    second = build_temporal_causal_hypergraph_experiment(
        trace,
        candidate_hyperedges,
        lag=1,
        min_abs_weight=0.1,
        required_baseline_margin=0.1,
    )

    assert first == second
    assert first["schema"] == "scpn_temporal_causal_hypergraph_experiment_v1"
    assert first["research_only"] is True
    assert first["production_claim_permitted"] is False
    assert first["actuation_permitted"] is False
    assert first["baseline_beaten"] is True
    assert first["accepted_hyperedge_count"] == 1
    assert len(str(first["experiment_sha256"])) == 64
    assert first["accepted_hyperedges"][0]["target"] == "response"
    assert first["baseline"]["edge_count"] >= 1


def test_temporal_causal_hypergraph_experiment_blocks_without_baseline_win() -> None:
    trace = {
        "driver": [0.0, 1.0, 2.0, 3.0],
        "response": [0.0, 0.0, 1.0, 2.0],
    }
    manifest = build_temporal_causal_hypergraph_experiment(
        trace,
        [
            {
                "sources": ["driver", "response"],
                "target": "response",
                "time_offsets": [-1, 0],
                "score": 0.01,
            }
        ],
        lag=1,
        min_abs_weight=0.1,
        required_baseline_margin=0.1,
    )

    assert manifest["baseline_beaten"] is False
    assert manifest["accepted_hyperedge_count"] == 0
    assert manifest["blocked_reasons"] == ["conventional_causal_baseline_not_beaten"]
    assert manifest["production_claim_permitted"] is False

    with pytest.raises(ValueError, match="candidate_hyperedges"):
        build_temporal_causal_hypergraph_experiment(trace, [])

    with pytest.raises(ValueError, match="time_offsets"):
        build_temporal_causal_hypergraph_experiment(
            trace,
            [{"sources": ["driver"], "target": "response", "time_offsets": []}],
        )


def test_temporal_causal_hypergraph_experiment_rejects_negative_baseline_margin() -> (
    None
):
    trace = {
        "driver": [0.0, 1.0, 2.0, 3.0],
        "response": [0.0, 0.0, 2.0, 6.0],
    }

    with pytest.raises(ValueError, match="required_baseline_margin must be finite"):
        build_temporal_causal_hypergraph_experiment(
            trace,
            [
                {
                    "sources": ["driver"],
                    "target": "response",
                    "time_offsets": [0],
                    "score": 0.3,
                }
            ],
            required_baseline_margin=-0.1,
        )


def test_temporal_causal_hypergraph_experiment_rejects_non_finite_candidate_score() -> (
    None
):
    trace = {
        "driver": [0.0, 1.0, 2.0, 3.0],
        "response": [0.0, 0.0, 2.0, 6.0],
    }

    with pytest.raises(ValueError, match="score must be finite"):
        build_temporal_causal_hypergraph_experiment(
            trace,
            [
                {
                    "sources": ["driver"],
                    "target": "response",
                    "time_offsets": [0],
                    "score": float("inf"),
                }
            ],
        )


def test_counterfactual_audit_record_is_deterministic_and_ordered() -> None:
    rollout = CounterfactualRollout(
        baseline_R=[0.35, 0.40, 0.45],
        intervention_R=[0.35, 0.39, 0.43],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.11, 0.19],
        delta_R_final=0.08,
        delta_R_mean=0.02,
        delta_psi_final=0.12,
        actions=(
            ControlAction(
                knob="K",
                scope="global",
                value=0.4,
                ttl_s=5.0,
                justification="first",
            ),
            ControlAction(
                knob="zeta",
                scope="global",
                value=-0.1,
                ttl_s=1.0,
                justification="second",
            ),
        ),
    )

    record = rollout.to_audit_record()
    assert list(record) == [
        "baseline_R",
        "intervention_R",
        "baseline_psi",
        "intervention_psi",
        "delta_R_final",
        "delta_R_mean",
        "delta_psi_final",
        "actions",
    ]
    assert record["actions"] == [
        {
            "knob": "K",
            "scope": "global",
            "value": 0.4,
            "ttl_s": 5.0,
            "justification": "first",
        },
        {
            "knob": "zeta",
            "scope": "global",
            "value": -0.1,
            "ttl_s": 1.0,
            "justification": "second",
        },
    ]
    assert rollout.to_audit_record() == record


def test_rollout_attribution_boundary_thresholding() -> None:
    stabilising_tie = CounterfactualRollout(
        baseline_R=[0.2, 0.2, 0.3],
        intervention_R=[0.2, 0.2, 0.4],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.1, 0.2],
        delta_R_final=0.10,
        delta_R_mean=0.10,
        delta_psi_final=0.0,
        actions=(),
    )

    tied = stabilising_tie.attribute(threshold=0.10)
    assert tied.effect == "neutral"
    assert tied.confidence == pytest.approx(1.0)

    zero = stabilising_tie.attribute(threshold=0.0)
    assert zero.effect == "stabilising"
    assert zero.confidence == 0.0
    with pytest.raises(ValueError, match="non-negative"):
        stabilising_tie.attribute(threshold=float("nan"))


def test_graph_edges_are_ranked_cycle_safe_and_intervention_nodes_deduplicate() -> None:
    trace = {
        "A": [0.0, 1.0, 0.0, 1.0, 0.0],
        "B": [1.0, 0.0, 1.0, 0.0, 1.0],
        "C": [2.0, 2.0, 2.0, 2.0, 2.0],
    }
    rollout_one = CounterfactualRollout(
        baseline_R=[0.2, 0.21, 0.22],
        intervention_R=[0.2, 0.31, 0.45],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.1, 0.2],
        delta_R_final=0.23,
        delta_R_mean=0.13,
        delta_psi_final=0.0,
        actions=(
            ControlAction(
                knob="K",
                scope="global",
                value=0.2,
                ttl_s=4.0,
                justification="shared intervention source",
            ),
        ),
    )
    rollout_two = CounterfactualRollout(
        baseline_R=[0.2, 0.21, 0.22],
        intervention_R=[0.2, 0.31, 0.44],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.1, 0.2],
        delta_R_final=0.22,
        delta_R_mean=0.12,
        delta_psi_final=0.0,
        actions=(
            ControlAction(
                knob="K",
                scope="global",
                value=0.2,
                ttl_s=4.0,
                justification="shared intervention source",
            ),
        ),
    )
    rollout_three = CounterfactualRollout(
        baseline_R=[0.2, 0.21, 0.22],
        intervention_R=[0.2, 0.20, 0.24],
        baseline_psi=[0.0, 0.1, 0.2],
        intervention_psi=[0.0, 0.1, 0.2],
        delta_R_final=0.02,
        delta_R_mean=0.01,
        delta_psi_final=0.0,
        actions=(
            ControlAction(
                knob="zeta",
                scope="global",
                value=0.03,
                ttl_s=1.0,
                justification="small intervention",
            ),
        ),
    )

    graph = learn_causal_graph(
        trace,
        [rollout_one, rollout_two, rollout_three],
        lag=1,
        min_abs_weight=1e-3,
    )

    edge_keys = [(edge.source, edge.target, edge.evidence) for edge in graph.edges]
    assert edge_keys == sorted(edge_keys)
    assert ("A", "B", "lagged_trace") in edge_keys
    assert ("B", "A", "lagged_trace") in edge_keys
    assert graph.nodes.count("do(K:global)") == 1
    assert graph.nodes[-1] == "R"

    do_edge_weights = [
        edge.weight
        for edge in graph.edges
        if edge.evidence == "counterfactual_rollout" and edge.source == "do(K:global)"
    ]
    assert len(do_edge_weights) >= 2
    assert do_edge_weights[0] == pytest.approx(0.65)
    assert do_edge_weights[1] == pytest.approx(0.6)
    assert do_edge_weights[0] != do_edge_weights[1]

    assert graph.to_audit_record()["nodes"] == list(graph.nodes)
