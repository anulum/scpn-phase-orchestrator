# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — phase-reduction evaluator tests

"""Tests for the dependency-light phase-reduction evaluator (no JAX).

The evaluator is exercised with hand-built identity weights, for which the
asymptotic phase and the phase-sensitivity function have closed forms, and with
random weights against a finite-difference gradient of the phase, plus the full
input-validation surface.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.oscillators.phase_reduction import (
    PhaseReducer,
    PhaseReductionWeights,
)


def _identity_reducer() -> PhaseReducer:
    # Single-layer encoder x -> (x1, x2, 0) and decoder (Y1, Y2, Y3) -> (Y1, Y2).
    weights = PhaseReductionWeights(
        encoder_weights=(np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]),),
        encoder_biases=(np.zeros(3),),
        decoder_weights=(np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),),
        decoder_biases=(np.zeros(2),),
        omega=1.0,
        decay=-0.5,
        state_dim=2,
    )
    return PhaseReducer(weights)


def _random_reducer(seed: int) -> PhaseReducer:
    rng = np.random.default_rng(seed)
    weights = PhaseReductionWeights(
        encoder_weights=(
            rng.standard_normal((8, 2)),
            rng.standard_normal((3, 8)),
        ),
        encoder_biases=(rng.standard_normal(8), rng.standard_normal(3)),
        decoder_weights=(
            rng.standard_normal((8, 3)),
            rng.standard_normal((2, 8)),
        ),
        decoder_biases=(rng.standard_normal(8), rng.standard_normal(2)),
        omega=1.0,
        decay=-0.4,
        state_dim=2,
    )
    return PhaseReducer(weights)


# --------------------------------------------------------------------------- #
# Closed-form identity case                                                   #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("angle", [0.0, 0.7, 1.9, -2.4, np.pi / 2])
def test_identity_asymptotic_phase_is_atan2(angle: float) -> None:
    reducer = _identity_reducer()
    state = np.array([np.cos(angle), np.sin(angle)])
    assert reducer.asymptotic_phase(state) == pytest.approx(angle, abs=1.0e-12)


@pytest.mark.parametrize("angle", [0.3, 1.2, 2.5, -1.0])
def test_identity_phase_sensitivity_is_the_unit_tangent(angle: float) -> None:
    # For the identity encoder on the unit cycle, Z(θ) = (−sin θ, cos θ).
    reducer = _identity_reducer()
    sensitivity = reducer.phase_sensitivity(angle)
    np.testing.assert_allclose(
        sensitivity, [-np.sin(angle), np.cos(angle)], atol=1.0e-9
    )


def test_identity_reconstruct_returns_the_cycle_point() -> None:
    reducer = _identity_reducer()
    np.testing.assert_allclose(reducer.reconstruct(0.8), [np.cos(0.8), np.sin(0.8)])


# --------------------------------------------------------------------------- #
# Phase sensitivity matches a finite-difference gradient                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("angle", [0.5, 2.0])
def test_phase_sensitivity_matches_finite_difference(seed: int, angle: float) -> None:
    reducer = _random_reducer(seed)
    base = reducer.reconstruct(angle)
    analytic = reducer.phase_sensitivity(angle)
    epsilon = 1.0e-6
    finite = np.zeros(2)
    for axis in range(2):
        forward = base.copy()
        forward[axis] += epsilon
        backward = base.copy()
        backward[axis] -= epsilon
        delta = reducer.asymptotic_phase(forward) - reducer.asymptotic_phase(backward)
        finite[axis] = np.angle(np.exp(1j * delta)) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, finite, atol=1.0e-5)


# --------------------------------------------------------------------------- #
# Properties + validation                                                     #
# --------------------------------------------------------------------------- #
def test_omega_and_decay_properties() -> None:
    reducer = _identity_reducer()
    assert reducer.omega == 1.0
    assert reducer.decay == -0.5


def test_asymptotic_phase_rejects_a_wrong_length_state() -> None:
    reducer = _identity_reducer()
    with pytest.raises(ValueError, match="must have 2 entries"):
        reducer.asymptotic_phase(np.zeros(3))


def test_asymptotic_phase_rejects_non_finite_state() -> None:
    reducer = _identity_reducer()
    with pytest.raises(ValueError, match="only finite values"):
        reducer.asymptotic_phase(np.array([np.inf, 0.0]))


def test_encode_observables_batches_the_encoder() -> None:
    reducer = _random_reducer(3)
    states = np.array([[0.4, -0.2], [1.1, 0.7], [-0.6, 0.3]])
    batch = reducer.encode_observables(states)
    assert batch.shape == (3, 3)
    # The batch lift matches the per-state raw encoding row by row.
    for row, state in zip(batch, states, strict=True):
        np.testing.assert_allclose(row, reducer._encode_raw(state), atol=1.0e-12)


def test_encode_observables_rejects_a_wrong_shape() -> None:
    reducer = _identity_reducer()
    with pytest.raises(ValueError, match=r"\(K, 2\) array"):
        reducer.encode_observables(np.zeros((4, 3)))


def test_encode_observables_rejects_non_finite_states() -> None:
    reducer = _identity_reducer()
    with pytest.raises(ValueError, match="only finite values"):
        reducer.encode_observables(np.array([[np.inf, 0.0]]))
