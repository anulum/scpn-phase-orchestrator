# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman MPC controller tests

"""Behavioural tests for the review-only Koopman model-predictive controller.

The suite proves the controller on closed-loop behaviour (it damps an
oscillatory system and tracks a non-zero set-point), on its constraint handling
(actuator bounds and move limits), on the reproducibility of its content-hashed
decision, and across the configuration and call validation surface. A
composition test confirms the proposed input flows into the foundation-model
safety governor.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation import koopman_mpc as mpc_mod
from scpn_phase_orchestrator.actuation._qp_admm import QPSolution
from scpn_phase_orchestrator.actuation.foundation_model_governor import (
    FoundationModelGovernor,
)
from scpn_phase_orchestrator.actuation.koopman_mpc import (
    KoopmanMPCConfig,
    KoopmanMPCController,
    KoopmanMPCDecision,
)
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    KoopmanDictionary,
    KoopmanPredictor,
    fit_koopman_predictor,
)

# A controllable, lightly damped oscillatory plant: rotation with slow decay.
_PLANT_A = np.array([[0.98, 0.30], [-0.30, 0.98]])
_PLANT_B = np.array([[0.0], [1.0]])


def _fit_predictor() -> KoopmanPredictor:
    rng = np.random.default_rng(0)
    samples = 600
    states = rng.standard_normal((samples, 2))
    inputs = rng.standard_normal((samples, 1))
    next_states = states @ _PLANT_A.T + inputs @ _PLANT_B.T
    return fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(
            kind="identity", state_dim=2, include_constant=False
        ),
        regularisation=1.0e-10,
    )


def _controller(**config: object) -> KoopmanMPCController:
    base: dict[str, object] = {
        "horizon": 20,
        "output_weight": 1.0,
        "input_weight": 0.01,
        "input_lower": -3.0,
        "input_upper": 3.0,
    }
    base.update(config)
    return KoopmanMPCController(
        predictor=_fit_predictor(),
        config=KoopmanMPCConfig(**base),  # type: ignore[arg-type]
    )


def _closed_loop(
    controller: KoopmanMPCController | None, steps: int = 40
) -> np.ndarray:
    state = np.array([1.0, 1.0])
    trajectory = [state.copy()]
    for _ in range(steps):
        control = (
            controller.solve(state).proposed_input
            if controller is not None
            else np.zeros(1)
        )
        state = _PLANT_A @ state + _PLANT_B @ control
        trajectory.append(state.copy())
    return np.array(trajectory)


# --------------------------------------------------------------------------- #
# Closed-loop behaviour                                                       #
# --------------------------------------------------------------------------- #
def test_controller_damps_the_oscillation() -> None:
    open_loop = _closed_loop(None)
    controlled = _closed_loop(_controller())
    assert np.linalg.norm(controlled[-1]) < 0.05 * np.linalg.norm(open_loop[-1])


def test_controller_drives_toward_a_non_zero_set_point() -> None:
    # The basic (non-integral) controller has a steady-state offset for a
    # non-equilibrium-input set-point, but a small effort weight makes it drive
    # the state substantially toward a reachable equilibrium x* = (I−A)⁻¹ B u*.
    controller = _controller(horizon=30, input_weight=1.0e-4)
    reference = np.linalg.solve(np.eye(2) - _PLANT_A, _PLANT_B.ravel() * 0.1)
    state = np.zeros(2)
    for _ in range(120):
        control = controller.solve(state, reference=reference).proposed_input
        state = _PLANT_A @ state + _PLANT_B @ control
    # The remaining error is well under half the set-point distance.
    assert np.linalg.norm(state - reference) < 0.4 * np.linalg.norm(reference)


# --------------------------------------------------------------------------- #
# Condensed prediction ground truth                                           #
# --------------------------------------------------------------------------- #
def test_condensed_prediction_matches_forward_roll() -> None:
    # Ground-truth check of the condensed matrices Ψ, Θ against an independent
    # forward roll of z_{k+1}=A z_k + B u_k, y=C z_k. Uses a lift dimension N > n
    # and a non-identity C so both the free response (C A^{i+1}) and every forced
    # block (C A^{i-j} B, diagonal C B) are exercised — the previous code applied
    # one extra factor of A to Θ, which this comparison catches exactly.
    rng = np.random.default_rng(3)
    lift_dim, n_input, n_state, horizon = 5, 2, 3, 4
    state_matrix = 0.5 * rng.standard_normal((lift_dim, lift_dim))
    input_matrix = rng.standard_normal((lift_dim, n_input))
    output_matrix = rng.standard_normal((n_state, lift_dim))
    predictor = KoopmanPredictor(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
        dictionary=KoopmanDictionary(
            kind="identity", state_dim=n_state, include_constant=False
        ),
        fit_residual=0.0,
    )
    psi, theta = mpc_mod._condensed_prediction(predictor, horizon)

    z0 = rng.standard_normal(lift_dim)
    inputs = rng.standard_normal((horizon, n_input))
    predicted = (psi @ z0 + theta @ inputs.ravel()).reshape(horizon, n_state)

    z = z0.copy()
    expected = np.empty((horizon, n_state), dtype=np.float64)
    for step in range(horizon):
        z = state_matrix @ z + input_matrix @ inputs[step]
        expected[step] = output_matrix @ z
    assert np.allclose(predicted, expected, atol=1e-12)

    # The diagonal forced block is the direct feed-through C B (not C A B).
    diag_block = theta[:n_state, :n_input]
    assert np.allclose(diag_block, output_matrix @ input_matrix, atol=1e-12)


def test_condensed_prediction_agrees_with_predictor_predict() -> None:
    # The integration path: Ψ ψ(x_0) + Θ U must equal the predictor's own forward
    # roll of the physical output over the same inputs.
    predictor = _fit_predictor()
    horizon = 6
    psi, theta = mpc_mod._condensed_prediction(predictor, horizon)
    rng = np.random.default_rng(11)
    x0 = rng.standard_normal(predictor.state_dim)
    inputs = rng.standard_normal((horizon, predictor.input_dim))
    condensed = (psi @ predictor.lift(x0) + theta @ inputs.ravel()).reshape(
        horizon, predictor.state_dim
    )
    rolled = predictor.predict(x0, inputs)[1:]
    assert np.allclose(condensed, rolled, atol=1e-9)


# --------------------------------------------------------------------------- #
# Decision contract                                                           #
# --------------------------------------------------------------------------- #
def test_decision_shapes_and_reproducible_hash() -> None:
    controller = _controller(horizon=12)
    first = controller.solve(np.array([1.0, 1.0]))
    second = controller.solve(np.array([1.0, 1.0]))
    assert isinstance(first, KoopmanMPCDecision)
    assert first.input_plan.shape == (12, 1)
    assert first.predicted_outputs.shape == (12, 2)
    assert first.proposed_input.shape == (1,)
    assert first.status == "OPTIMAL"
    assert first.content_hash == second.content_hash
    assert len(first.content_hash) == 64


def test_tight_bounds_activate_and_clip_the_input() -> None:
    controller = _controller(input_lower=-0.1, input_upper=0.1)
    decision = controller.solve(np.array([2.0, 2.0]))
    assert decision.active_bounds is True
    assert -0.1 - 1.0e-9 <= decision.proposed_input[0] <= 0.1 + 1.0e-9


def test_max_iter_status_is_reported(monkeypatch) -> None:
    def _unconverged(*_args: object, **_kwargs: object) -> QPSolution:
        return QPSolution(
            x=np.zeros(20),
            objective=0.0,
            iterations=1,
            primal_residual=1.0,
            dual_residual=1.0,
            converged=False,
        )

    monkeypatch.setattr(mpc_mod, "solve_qp", _unconverged)
    decision = _controller().solve(np.array([1.0, 1.0]))
    assert decision.status == "MAX_ITER"


# --------------------------------------------------------------------------- #
# Move limits                                                                 #
# --------------------------------------------------------------------------- #
def test_move_limit_bounds_the_first_step() -> None:
    controller = _controller(move_limit=0.2)
    decision = controller.solve(np.array([2.0, 2.0]), previous_input=np.array([0.0]))
    assert abs(decision.proposed_input[0]) <= 0.2 + 1.0e-6


def test_move_limit_requires_previous_input() -> None:
    controller = _controller(move_limit=0.2)
    with pytest.raises(ValueError, match="move limit requires previous_input"):
        controller.solve(np.array([1.0, 1.0]))


# --------------------------------------------------------------------------- #
# Composition with the safety governor                                        #
# --------------------------------------------------------------------------- #
def test_proposed_input_feeds_the_safety_governor() -> None:
    decision = _controller().solve(np.array([1.0, 1.0]))
    governor = FoundationModelGovernor(control_lo=-1.0, control_hi=1.0, max_rate=0.5)
    verdict = governor.govern(
        float(decision.proposed_input[0]),
        np.array([1.0, 1.0]),
        np.zeros(2),
    )
    assert verdict.status in {"admitted", "constrained", "rejected"}


# --------------------------------------------------------------------------- #
# Configuration validation                                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("config", "match"),
    [
        ({"horizon": 0}, "horizon must be at least 1"),
        ({"horizon": 1.5}, "horizon must be an integer"),
        ({"terminal_weight": -1.0}, "terminal_weight must be finite and non-negative"),
        ({"terminal_weight": "x"}, "terminal_weight must be a real number"),
        ({"move_limit": 0.0}, "move_limit must be finite and positive"),
        ({"move_limit": "x"}, "move_limit must be a real number"),
    ],
)
def test_config_rejects_invalid_values(config, match) -> None:
    base: dict[str, object] = {"horizon": 10}
    base.update(config)
    with pytest.raises((TypeError, ValueError), match=match):
        KoopmanMPCConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Call validation                                                             #
# --------------------------------------------------------------------------- #
def test_solve_rejects_inverted_input_bounds() -> None:
    controller = _controller(input_lower=1.0, input_upper=-1.0)
    with pytest.raises(ValueError, match="input_upper must not be below"):
        controller.solve(np.array([1.0, 1.0]))


def test_solve_rejects_a_negative_output_weight() -> None:
    controller = _controller(output_weight=-1.0)
    with pytest.raises(ValueError, match="non-negative"):
        controller.solve(np.array([1.0, 1.0]))


def test_solve_rejects_a_negative_weight_in_a_vector() -> None:
    controller = _controller(output_weight=np.array([-1.0, 1.0]))
    with pytest.raises(ValueError, match="finite non-negative weights"):
        controller.solve(np.array([1.0, 1.0]))


def test_solve_rejects_a_wrong_length_weight_vector() -> None:
    controller = _controller(output_weight=np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="scalar or a length-2 vector"):
        controller.solve(np.array([1.0, 1.0]))


def test_solve_rejects_a_reference_of_the_wrong_size() -> None:
    controller = _controller()
    with pytest.raises(ValueError, match="scalar or a length-2 vector"):
        controller.solve(np.array([1.0, 1.0]), reference=np.array([1.0, 2.0, 3.0]))


def test_solve_accepts_vector_weights_and_bounds() -> None:
    controller = _controller(
        output_weight=np.array([1.0, 0.5]),
        input_weight=np.array([0.02]),
        input_lower=np.array([-2.0]),
        input_upper=np.array([2.0]),
    )
    decision = controller.solve(np.array([1.0, 1.0]))
    assert decision.status == "OPTIMAL"


def test_solve_rejects_nan_input_bounds() -> None:
    controller = _controller(input_lower=np.array([np.nan]))
    with pytest.raises(ValueError, match="must not contain NaN values"):
        controller.solve(np.array([1.0, 1.0]))
