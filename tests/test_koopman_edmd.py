# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Koopman EDMD-with-control predictor tests

"""Behavioural tests for ``monitor.koopman_edmd``.

The suite proves the EDMD-with-control predictor on three fronts: exact
recovery of a known linear controlled system (the identity dictionary makes the
lift trivially closed), accurate prediction of a genuinely nonlinear system
whose dynamics are linear on a Koopman-invariant polynomial subspace
(Brunton-Tu slow-manifold form), and the full input-validation surface of the
dictionaries, the fit, and the predictor roll-out.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import scpn_phase_orchestrator.monitor.koopman_edmd as koopman_module
from scpn_phase_orchestrator.monitor.koopman_edmd import (
    ACTIVE_BACKEND,
    AVAILABLE_BACKENDS,
    KoopmanDictionary,
    KoopmanPredictor,
    fit_koopman_predictor,
    lift_states,
)


def _linear_system() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    rng = np.random.default_rng(20260622)
    n, m, samples = 3, 2, 500
    state_matrix = np.array(
        [[0.90, 0.10, 0.00], [0.00, 0.80, 0.20], [0.10, 0.00, 0.85]]
    )
    input_matrix = np.array([[0.5, 0.0], [0.0, 0.3], [0.2, 0.1]])
    states = rng.standard_normal((samples, n))
    inputs = rng.standard_normal((samples, m))
    next_states = states @ state_matrix.T + inputs @ input_matrix.T
    return states, next_states, inputs, state_matrix, input_matrix


def _brunton_system(
    samples: int = 800,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Discrete slow-manifold system, linear on the span of ``[x1, x2, x1^2]``."""
    rng = np.random.default_rng(7)
    lam, mu, coupling, gain = 0.95, 0.6, 0.4, 0.5
    states = rng.uniform(-1.0, 1.0, size=(samples, 2))
    inputs = rng.standard_normal((samples, 1))
    x1, x2 = states[:, 0], states[:, 1]
    next_states = np.column_stack(
        (lam * x1, mu * x2 + coupling * x1**2 + gain * inputs[:, 0])
    )
    return states, next_states, inputs


# --------------------------------------------------------------------------- #
# Backend chain                                                               #
# --------------------------------------------------------------------------- #
def test_python_backend_is_always_the_floor() -> None:
    assert "python" in AVAILABLE_BACKENDS
    assert AVAILABLE_BACKENDS[-1] == "python"
    assert ACTIVE_BACKEND in AVAILABLE_BACKENDS


def test_julia_loader_returns_the_runtime_callable(monkeypatch) -> None:
    module_name = (
        "scpn_phase_orchestrator.experimental.accelerators.monitor._koopman_edmd_julia"
    )
    fake_module = types.ModuleType(module_name)

    def fake_solver(*args):
        return args

    fake_module.koopman_edmd_solve_julia = fake_solver
    monkeypatch.setitem(sys.modules, module_name, fake_module)
    monkeypatch.setattr(koopman_module, "require_juliacall_main", lambda: object())

    assert koopman_module._load_julia_fns() == {"edmd_solve": fake_solver}


# --------------------------------------------------------------------------- #
# Exact recovery of a linear controlled system                                #
# --------------------------------------------------------------------------- #
def test_identity_dictionary_recovers_a_linear_system() -> None:
    states, next_states, inputs, state_matrix, input_matrix = _linear_system()
    dictionary = KoopmanDictionary(kind="identity", state_dim=3, include_constant=False)
    predictor = fit_koopman_predictor(
        states, next_states, inputs, dictionary=dictionary, regularisation=1.0e-12
    )
    np.testing.assert_allclose(predictor.state_matrix, state_matrix, atol=1.0e-9)
    np.testing.assert_allclose(predictor.input_matrix, input_matrix, atol=1.0e-9)
    np.testing.assert_allclose(predictor.output_matrix, np.eye(3), atol=1.0e-9)
    assert predictor.fit_residual < 1.0e-9
    assert predictor.lift_dim == 3
    assert predictor.input_dim == 2
    assert predictor.state_dim == 3


def test_predict_matches_a_fresh_linear_trajectory() -> None:
    states, next_states, inputs, state_matrix, input_matrix = _linear_system()
    dictionary = KoopmanDictionary(kind="identity", state_dim=3, include_constant=False)
    predictor = fit_koopman_predictor(
        states, next_states, inputs, dictionary=dictionary, regularisation=1.0e-12
    )
    rng = np.random.default_rng(99)
    initial = rng.standard_normal(3)
    sequence = rng.standard_normal((15, 2))
    truth = [initial.copy()]
    current = initial.copy()
    for control in sequence:
        current = state_matrix @ current + input_matrix @ control
        truth.append(current.copy())
    predicted = predictor.predict(initial, sequence)
    assert predicted.shape == (16, 3)
    np.testing.assert_allclose(predicted, np.array(truth), atol=1.0e-8)


# --------------------------------------------------------------------------- #
# Nonlinear system on a Koopman-invariant subspace                            #
# --------------------------------------------------------------------------- #
def _brunton_rollout(initial: np.ndarray, controls: np.ndarray) -> np.ndarray:
    lam, mu, coupling, gain = 0.95, 0.6, 0.4, 0.5
    state = initial.astype(np.float64).copy()
    trajectory = [state.copy()]
    for control in controls:
        state = np.array(
            [
                lam * state[0],
                mu * state[1] + coupling * state[0] ** 2 + gain * control[0],
            ]
        )
        trajectory.append(state.copy())
    return np.array(trajectory)


def _multistep_error(predictor: KoopmanPredictor) -> float:
    initial = np.array([0.4, -0.3])
    controls = np.zeros((25, 1))
    truth = _brunton_rollout(initial, controls)
    predicted = predictor.predict(initial, controls)
    return float(np.max(np.abs(predicted - truth)))


def test_polynomial_lift_beats_identity_on_a_nonlinear_system() -> None:
    states, next_states, inputs = _brunton_system()
    identity = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=2),
    )
    polynomial = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="polynomial", state_dim=2, degree=2),
    )
    # [x1, x2, x1^2] is a Koopman-invariant subspace of this system, so the
    # quadratic dictionary predicts the original states accurately over a long
    # horizon while the purely linear dictionary cannot capture the x1^2 channel.
    polynomial_error = _multistep_error(polynomial)
    identity_error = _multistep_error(identity)
    assert polynomial_error < 1.0e-6
    assert polynomial_error < 0.01 * identity_error


def _onestep_state_error(
    predictor: KoopmanPredictor,
    states: np.ndarray,
    next_states: np.ndarray,
    inputs: np.ndarray,
) -> float:
    lifted = predictor.dictionary.lift(states)
    lifted_next = lifted @ predictor.state_matrix.T + inputs @ predictor.input_matrix.T
    predicted = lifted_next @ predictor.output_matrix.T
    return float(np.sqrt(np.mean((predicted - next_states) ** 2)))


def test_rbf_lift_reduces_nonlinear_prediction_error() -> None:
    states, next_states, inputs = _brunton_system()
    rng = np.random.default_rng(3)
    centres = rng.uniform(-1.0, 1.0, size=(60, 2))
    rbf = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(
            kind="rbf", state_dim=2, centres=centres, width=0.6
        ),
    )
    identity = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=2),
    )
    # The radial-basis lift captures the x1^2 nonlinearity for the next step,
    # so the one-step state-prediction error drops below the linear dictionary.
    rbf_error = _onestep_state_error(rbf, states, next_states, inputs)
    identity_error = _onestep_state_error(identity, states, next_states, inputs)
    assert rbf_error < identity_error


# --------------------------------------------------------------------------- #
# Dictionaries                                                                 #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kind", "kwargs", "expected_dim"),
    [
        ("identity", {}, 1 + 3),
        ("polynomial", {"degree": 2}, 1 + 3 + 6),
        ("polynomial", {"degree": 3}, 1 + 3 + 6 + 10),
        ("phase", {}, 1 + 3 + 3 + 3 + 2),
    ],
)
def test_dictionary_output_dimensions(kind, kwargs, expected_dim) -> None:
    dictionary = KoopmanDictionary(kind=kind, state_dim=3, **kwargs)
    assert dictionary.output_dim == expected_dim
    lifted = lift_states(dictionary, np.zeros((5, 3)))
    assert lifted.shape == (5, expected_dim)


def test_rbf_dictionary_output_dimension() -> None:
    centres = np.zeros((7, 3))
    dictionary = KoopmanDictionary(kind="rbf", state_dim=3, centres=centres, width=1.0)
    assert dictionary.output_dim == 1 + 3 + 7
    assert dictionary.lift(np.ones((4, 3))).shape == (4, 11)


def test_constant_observable_can_be_disabled() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=3, include_constant=False)
    assert dictionary.output_dim == 3
    lifted = dictionary.lift(np.arange(6.0).reshape(2, 3))
    np.testing.assert_allclose(lifted, np.arange(6.0).reshape(2, 3))


def test_phase_dictionary_encodes_the_order_parameter() -> None:
    dictionary = KoopmanDictionary(kind="phase", state_dim=4, include_constant=False)
    phases = np.array([[0.0, np.pi / 2, np.pi, 3 * np.pi / 2]])
    lifted = dictionary.lift(phases)
    # Layout: [theta(4), cos(4), sin(4), R cos Psi, R sin Psi].
    order = np.mean(np.exp(1j * phases[0]))
    assert lifted[0, -2] == pytest.approx(order.real, abs=1.0e-12)
    assert lifted[0, -1] == pytest.approx(order.imag, abs=1.0e-12)
    np.testing.assert_allclose(lifted[0, 4:8], np.cos(phases[0]), atol=1.0e-12)
    np.testing.assert_allclose(lifted[0, 8:12], np.sin(phases[0]), atol=1.0e-12)


# --------------------------------------------------------------------------- #
# Predictor roll-out                                                          #
# --------------------------------------------------------------------------- #
def test_predict_first_row_is_the_reconstruction() -> None:
    states, next_states, inputs, _, _ = _linear_system()
    predictor = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=3),
    )
    initial = np.array([0.3, -0.2, 0.5])
    predicted = predictor.predict(initial, np.zeros((0, 2)))
    assert predicted.shape == (1, 3)
    np.testing.assert_allclose(predicted[0], initial, atol=1.0e-8)


def test_lift_single_state_matches_batch_lift() -> None:
    dictionary = KoopmanDictionary(kind="polynomial", state_dim=3, degree=2)
    predictor = KoopmanPredictor(
        state_matrix=np.eye(dictionary.output_dim),
        input_matrix=np.zeros((dictionary.output_dim, 1)),
        output_matrix=np.zeros((3, dictionary.output_dim)),
        dictionary=dictionary,
        fit_residual=0.0,
    )
    state = np.array([0.1, -0.4, 0.7])
    np.testing.assert_allclose(
        predictor.lift(state), dictionary.lift(state[None, :])[0]
    )


def test_fit_is_deterministic() -> None:
    states, next_states, inputs, _, _ = _linear_system()
    dictionary = KoopmanDictionary(kind="polynomial", state_dim=3, degree=2)
    first = fit_koopman_predictor(states, next_states, inputs, dictionary=dictionary)
    second = fit_koopman_predictor(states, next_states, inputs, dictionary=dictionary)
    np.testing.assert_array_equal(first.state_matrix, second.state_matrix)
    np.testing.assert_array_equal(first.input_matrix, second.input_matrix)


# --------------------------------------------------------------------------- #
# Dictionary validation                                                        #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "bogus", "state_dim": 3}, "kind must be one of"),
        ({"kind": "identity", "state_dim": 0}, "state_dim must be at least 1"),
        ({"kind": "identity", "state_dim": 1.5}, "state_dim must be an integer"),
        (
            {"kind": "polynomial", "state_dim": 3, "degree": 0},
            "degree must be at least",
        ),
        ({"kind": "rbf", "state_dim": 3}, "rbf dictionary requires centres"),
        (
            {"kind": "rbf", "state_dim": 3, "centres": np.zeros((4, 2)), "width": 1.0},
            "centres must have 3 columns",
        ),
        (
            {"kind": "rbf", "state_dim": 3, "centres": np.zeros((4, 3)), "width": 0.0},
            "positive width",
        ),
        ({"kind": "identity", "state_dim": 3, "width": -1.0}, "width must be finite"),
    ],
)
def test_dictionary_rejects_invalid_configuration(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        KoopmanDictionary(**kwargs)


def test_dictionary_rejects_non_string_kind() -> None:
    with pytest.raises(TypeError, match="kind must be a string"):
        KoopmanDictionary(kind=1, state_dim=2)  # type: ignore[arg-type]


def test_lift_rejects_state_dimension_mismatch() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=3)
    with pytest.raises(ValueError, match="states must have 3 columns"):
        dictionary.lift(np.zeros((4, 2)))


# --------------------------------------------------------------------------- #
# Fit validation                                                               #
# --------------------------------------------------------------------------- #
def test_fit_rejects_mismatched_state_shapes() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=3)
    with pytest.raises(ValueError, match="must have the same shape"):
        fit_koopman_predictor(
            np.zeros((10, 3)),
            np.zeros((9, 3)),
            np.zeros((10, 2)),
            dictionary=dictionary,
        )


def test_fit_rejects_input_row_mismatch() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=3)
    with pytest.raises(ValueError, match="inputs must have 10 rows"):
        fit_koopman_predictor(
            np.zeros((10, 3)),
            np.zeros((10, 3)),
            np.zeros((9, 2)),
            dictionary=dictionary,
        )


def test_fit_rejects_dictionary_dimension_mismatch() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=4)
    with pytest.raises(ValueError, match="to match the dictionary"):
        fit_koopman_predictor(
            np.zeros((10, 3)),
            np.zeros((10, 3)),
            np.zeros((10, 2)),
            dictionary=dictionary,
        )


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.full((10, 3), np.inf), "only finite values"),
        (np.ones((10, 3), dtype=bool), "must not contain boolean"),
        (np.zeros((10, 3), dtype=complex) + 1j, "must be real-valued"),
        (np.zeros((10, 3, 1)), "must be a 2-D array"),
        (np.zeros((0, 3)), "must be non-empty"),
    ],
)
def test_fit_rejects_invalid_state_arrays(bad, match) -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=3)
    with pytest.raises(ValueError, match=match):
        fit_koopman_predictor(
            bad, np.zeros_like(bad), np.zeros((bad.shape[0], 2)), dictionary=dictionary
        )


def test_fit_rejects_negative_regularisation() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=3)
    with pytest.raises(ValueError, match="regularisation must be finite"):
        fit_koopman_predictor(
            np.zeros((10, 3)),
            np.zeros((10, 3)),
            np.zeros((10, 2)),
            dictionary=dictionary,
            regularisation=-1.0,
        )


def test_predict_rejects_wrong_input_width() -> None:
    states, next_states, inputs, _, _ = _linear_system()
    predictor = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=3),
    )
    with pytest.raises(ValueError, match="input_sequence must have 2 columns"):
        predictor.predict(np.zeros(3), np.zeros((5, 1)))


def test_lift_rejects_wrong_state_length() -> None:
    states, next_states, inputs, _, _ = _linear_system()
    predictor = fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=3),
    )
    with pytest.raises(ValueError, match="state must have 3 entries"):
        predictor.lift(np.zeros(2))


def _identity_predictor() -> KoopmanPredictor:
    states, next_states, inputs, _, _ = _linear_system()
    return fit_koopman_predictor(
        states,
        next_states,
        inputs,
        dictionary=KoopmanDictionary(kind="identity", state_dim=3),
    )


@pytest.mark.parametrize(
    ("bad", "match"),
    [
        (np.ones(3, dtype=bool), "must not contain boolean"),
        (np.array([1j, 2j, 3j]), "must be real-valued"),
        (np.array([np.inf, 0.0, 0.0]), "only finite values"),
        (np.array([]), "must be non-empty"),
    ],
)
def test_lift_rejects_invalid_state_vectors(bad, match) -> None:
    with pytest.raises(ValueError, match=match):
        _identity_predictor().lift(bad)


def test_dictionary_rejects_non_real_width() -> None:
    with pytest.raises(TypeError, match="width must be a real number"):
        KoopmanDictionary(kind="identity", state_dim=3, width="wide")  # type: ignore[arg-type]


def test_non_rbf_dictionary_validates_supplied_centres() -> None:
    dictionary = KoopmanDictionary(
        kind="identity", state_dim=3, centres=np.zeros((2, 3))
    )
    assert dictionary.centres is not None
    assert dictionary.output_dim == 1 + 3


def test_fit_rejects_a_non_numeric_state_matrix() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=2)
    bad = np.array([["a", "b"], ["c", "d"]])
    with pytest.raises(ValueError, match="must contain only real numbers"):
        fit_koopman_predictor(bad, bad, np.zeros((2, 1)), dictionary=dictionary)


def test_lift_rejects_a_non_numeric_state_vector() -> None:
    with pytest.raises(ValueError, match="must contain only real numbers"):
        _identity_predictor().lift(np.array(["a", "b", "c"]))


@pytest.mark.parametrize(
    "field",
    ["states", "next_states", "inputs", "centres", "lift", "predict"],
)
def test_public_matrix_boundaries_reject_numeric_string_aliases(field: str) -> None:
    """Numeric text must not become physical EDMD evidence by coercion."""
    dictionary = KoopmanDictionary(kind="identity", state_dim=2)
    text_matrix = np.full((3, 2), "0.0")

    with pytest.raises(ValueError, match="must contain only real numbers"):
        if field == "states":
            fit_koopman_predictor(
                text_matrix,
                np.zeros((3, 2)),
                np.zeros((3, 1)),
                dictionary=dictionary,
            )
        elif field == "next_states":
            fit_koopman_predictor(
                np.zeros((3, 2)),
                text_matrix,
                np.zeros((3, 1)),
                dictionary=dictionary,
            )
        elif field == "inputs":
            fit_koopman_predictor(
                np.zeros((3, 2)),
                np.zeros((3, 2)),
                np.full((3, 1), "0.0"),
                dictionary=dictionary,
            )
        elif field == "centres":
            KoopmanDictionary(kind="rbf", state_dim=2, centres=text_matrix, width=1.0)
        elif field == "lift":
            dictionary.lift(text_matrix)
        else:
            predictor = KoopmanPredictor(
                state_matrix=np.eye(3),
                input_matrix=np.zeros((3, 1)),
                output_matrix=np.column_stack((np.zeros(2), np.eye(2))),
                dictionary=dictionary,
                fit_residual=0.0,
            )
            predictor.predict(np.zeros(2), np.full((3, 1), "0.0"))


def test_public_vector_boundary_rejects_numeric_string_aliases() -> None:
    with pytest.raises(ValueError, match="state must contain only real numbers"):
        _identity_predictor().lift(np.array(["0.0", "1.0", "2.0"]))


class _BrokenArrayProtocol:
    """Fixture whose array conversion fails at every public probe."""

    def __array__(self, dtype=None, copy=None):
        raise TypeError("array protocol failed")


def test_public_matrix_boundary_wraps_broken_array_protocol() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=2)
    with pytest.raises(ValueError, match="states must contain only real numbers"):
        fit_koopman_predictor(
            _BrokenArrayProtocol(),
            np.zeros((2, 2)),
            np.zeros((2, 1)),
            dictionary=dictionary,
        )


def test_public_matrix_boundary_preserves_real_numeric_object_arrays() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=2)
    states = np.array([[0.0, 1], [2, 3.0]], dtype=object)
    predictor = fit_koopman_predictor(
        states,
        states,
        np.zeros((2, 1), dtype=object),
        dictionary=dictionary,
        regularisation=1.0e-8,
    )
    assert predictor.state_dim == 2


def test_public_matrix_boundary_rejects_object_complex_aliases() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=2)
    states = np.array([[0.0, 1.0], [2.0, 3.0 + 1.0j]], dtype=object)
    with pytest.raises(ValueError, match="states must be real-valued"):
        fit_koopman_predictor(
            states,
            np.zeros((2, 2)),
            np.zeros((2, 1)),
            dictionary=dictionary,
        )


def test_public_matrix_boundary_rejects_object_numeric_string_aliases() -> None:
    dictionary = KoopmanDictionary(kind="identity", state_dim=2)
    states = np.full((2, 2), "0.0", dtype=object)
    with pytest.raises(ValueError, match="states must contain only real numbers"):
        fit_koopman_predictor(
            states,
            np.zeros((2, 2)),
            np.zeros((2, 1)),
            dictionary=dictionary,
        )


@pytest.mark.parametrize("include_constant", [0, 1, np.bool_(True), "yes"])
def test_dictionary_requires_a_canonical_boolean_include_constant(
    include_constant: object,
) -> None:
    with pytest.raises(TypeError, match="include_constant must be a boolean"):
        KoopmanDictionary(
            kind="identity",
            state_dim=2,
            include_constant=include_constant,  # type: ignore[arg-type]
        )


def _direct_predictor(**overrides: object) -> KoopmanPredictor:
    payload: dict[str, object] = {
        "state_matrix": np.eye(2),
        "input_matrix": np.zeros((2, 1)),
        "output_matrix": np.eye(2),
        "dictionary": KoopmanDictionary(
            kind="identity", state_dim=2, include_constant=False
        ),
        "fit_residual": 0.0,
    }
    payload.update(overrides)
    return KoopmanPredictor(**payload)  # type: ignore[arg-type]


class _InvalidDictionary:
    """Configurable malformed observable-map fixture."""

    state_dim = 2
    lift: object = None


class _BrokenLiftDictionary:
    """Observable map whose lift cannot produce numeric evidence."""

    state_dim = 2

    def lift(self, states: np.ndarray) -> np.ndarray:
        del states
        return np.full((1, 2), "not-numeric")


class _WrongLiftShapeDictionary:
    """Observable map whose output dimension contradicts the matrices."""

    state_dim = 2

    def lift(self, states: np.ndarray) -> np.ndarray:
        return np.zeros((states.shape[0], 3))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"state_matrix": np.ones((2, 3))}, "state_matrix must be square"),
        ({"input_matrix": np.ones((3, 1))}, "input_matrix must have 2 rows"),
        ({"output_matrix": np.ones((2, 3))}, "output_matrix must have 2 columns"),
        (
            {"output_matrix": np.ones((3, 2))},
            "output_matrix must have 2 rows to match the dictionary",
        ),
        ({"state_matrix": np.full((2, 2), "1.0")}, "only real numbers"),
        ({"input_matrix": np.ones((2, 1), dtype=bool)}, "must not contain boolean"),
        ({"output_matrix": np.ones((2, 2), dtype=complex)}, "must be real-valued"),
        ({"dictionary": object()}, "dictionary must implement KoopmanObservables"),
        ({"fit_residual": True}, "fit_residual must be a real number"),
        ({"fit_residual": "0.0"}, "fit_residual must be a real number"),
        ({"fit_residual": -0.1}, "fit_residual must be finite and non-negative"),
        ({"fit_residual": np.inf}, "fit_residual must be finite and non-negative"),
    ],
)
def test_predictor_constructor_rejects_inconsistent_evidence(
    overrides: dict[str, object], match: str
) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        _direct_predictor(**overrides)


def test_predictor_requires_a_callable_dictionary_lift() -> None:
    with pytest.raises(TypeError, match="dictionary must implement KoopmanObservables"):
        _direct_predictor(dictionary=_InvalidDictionary())


def test_predictor_rejects_a_non_numeric_dictionary_probe() -> None:
    with pytest.raises(ValueError, match="finite real lift matrix"):
        _direct_predictor(dictionary=_BrokenLiftDictionary())


def test_predictor_rejects_a_wrong_dictionary_lift_shape() -> None:
    with pytest.raises(ValueError, match="lift output must have shape"):
        _direct_predictor(dictionary=_WrongLiftShapeDictionary())


def test_predictor_owns_immutable_matrix_evidence() -> None:
    state_matrix = np.eye(2)
    input_matrix = np.zeros((2, 1))
    output_matrix = np.eye(2)
    predictor = _direct_predictor(
        state_matrix=state_matrix,
        input_matrix=input_matrix,
        output_matrix=output_matrix,
    )

    state_matrix[0, 0] = 99.0
    input_matrix[0, 0] = 99.0
    output_matrix[0, 0] = 99.0

    assert predictor.state_matrix[0, 0] == 1.0
    assert predictor.input_matrix[0, 0] == 0.0
    assert predictor.output_matrix[0, 0] == 1.0
    assert predictor.state_matrix.flags.writeable is False
    assert predictor.input_matrix.flags.writeable is False
    assert predictor.output_matrix.flags.writeable is False
