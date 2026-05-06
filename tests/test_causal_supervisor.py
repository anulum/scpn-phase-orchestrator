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
    CausalInterventionEngine,
    CounterfactualRollout,
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


def test_input_shape_validation() -> None:
    phases, omegas, knm, alpha = _system()
    engine = CausalInterventionEngine(6, dt=0.01)
    with pytest.raises(ValueError, match="phases.shape"):
        engine.evaluate_actions(phases[:-1], omegas, knm, alpha, 0.0, 0.0, [])


@pytest.mark.parametrize(
    ("n_oscillators", "dt", "message"),
    [
        (0, 0.01, "n_oscillators must be >= 1"),
        (True, 0.01, "n_oscillators must be an integer"),
        (4, 0.0, "dt must be finite and > 0"),
        (4, np.inf, "dt must be finite and > 0"),
    ],
)
def test_constructor_validation(
    n_oscillators: int | bool,
    dt: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        CausalInterventionEngine(n_oscillators, dt=dt)


def test_exported_from_supervisor_package() -> None:
    import scpn_phase_orchestrator.supervisor as supervisor

    assert supervisor.CausalInterventionEngine is CausalInterventionEngine
