# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Finite coupling projection contracts

"""Exercise finite projection, ownership and recovery through public coupling APIs."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

MAX = float(np.finfo(np.float64).max)
TINY = float(np.nextafter(0.0, 1.0))
PAIRS = [
    (MAX, MAX),
    (-MAX, -MAX),
    (MAX, MAX / 2.0),
    (-MAX, -MAX / 2.0),
    (MAX, -MAX),
    (MAX, -float(np.nextafter(MAX, 0.0))),
    (TINY, TINY),
    (-TINY, -TINY),
    (TINY, 0.0),
    (TINY, 2.0 * TINY),
    (0.3, 0.7),
]


@pytest.mark.parametrize(("left", "right"), PAIRS)
def test_symmetry_preserves_representable_means(left: float, right: float) -> None:
    """Finite extreme and subnormal pairs retain their correctly rounded mean."""
    raw = np.array([[left, left], [right, right]])
    original = raw.copy()
    expected = float((Fraction.from_float(left) + Fraction.from_float(right)) / 2)
    with np.errstate(over="raise", invalid="raise"):
        result = SymmetryConstraint().project(raw)
    np.testing.assert_array_equal(result, [[left, expected], [expected, right]])
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(SymmetryConstraint().project(result), result)
    assert not np.shares_memory(result, raw)


@pytest.mark.parametrize(("left", "right"), PAIRS)
def test_constraint_chain_preserves_finite_coupling(left: float, right: float) -> None:
    """Symmetry followed by clipping preserves finite means and zero self-coupling."""
    raw = np.array([[MAX, left], [right, -MAX]])
    raw.setflags(write=False)
    original = raw.copy()
    mean = float((Fraction.from_float(left) + Fraction.from_float(right)) / 2)
    expected = np.array([[0.0, max(mean, 0.0)], [max(mean, 0.0), 0.0]])
    with np.errstate(over="raise", invalid="raise"):
        actual = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
        validate_knm(actual)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(raw, original)
    assert not np.shares_memory(actual, raw)
    actual[0, 1] = 0.25
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize("n", [0, 1, 8])
def test_project_validate_roundtrip(n: int) -> None:
    """Empty, singleton and ordinary projected systems share the validation contract."""
    rng = np.random.default_rng(24)
    raw = rng.uniform(-0.5, 0.5, (n, n))
    original = raw.copy()
    result = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
    validate_knm(result)
    np.testing.assert_array_equal(np.diag(result), np.zeros(n))
    np.testing.assert_array_equal(result, result.T)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize(
    "raw",
    [
        np.array([[0.0, np.inf], [1.0, 0.0]]),
        np.array([[0.0, np.nan], [1.0, 0.0]]),
        np.ones((2, 3)),
        np.array([[False, True], [True, False]]),
        np.array([[0.0, 1.0j], [1.0j, 0.0]]),
        np.array([[0.0, "1.0"], ["1.0", 0.0]], dtype=object),
    ],
)
def test_public_projection_refuses_and_recovers(raw: NDArray[np.generic]) -> None:
    """Bad ingress preserves storage and permits a subsequent projection."""
    original = raw.copy()
    constraints = [SymmetryConstraint(), NonNegativeConstraint()]
    with pytest.raises(ValueError):
        project_knm(cast(NDArray[np.float64], raw), constraints)
    np.testing.assert_array_equal(raw, original)
    recovered = project_knm(np.array([[0.0, 0.6], [0.2, 0.0]]), constraints)
    np.testing.assert_allclose(recovered, [[0.0, 0.4], [0.4, 0.0]], rtol=0, atol=1e-16)
    validate_knm(recovered)


def test_projection_order_drives_real_engine() -> None:
    """Projection order changes real Kuramoto evolution with finite coherence."""
    raw = np.array([[2.0, 0.8, 0.2], [-0.8, 3.0, 0.4], [0.2, 0.4, 4.0]])
    projected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
    reversed_order = project_knm(raw, [NonNegativeConstraint(), SymmetryConstraint()])
    np.testing.assert_array_equal(projected, [[0, 0, 0.2], [0, 0, 0.4], [0.2, 0.4, 0]])
    assert reversed_order[0, 1] == pytest.approx(0.4)
    engine = UPDEEngine(3, dt=0.01, method="rk4")
    initial = np.array([0.2, 1.4, 4.0])
    omegas = np.array([0.8, 1.0, 1.2])
    alpha = np.zeros((3, 3))
    evolved = engine.run(initial, omegas, projected, 0.0, 0.0, alpha, n_steps=40)
    alternative = engine.run(
        initial, omegas, reversed_order, 0.0, 0.0, alpha, n_steps=40
    )
    assert not np.allclose(evolved, alternative)
    r, psi = compute_order_parameter(evolved)
    assert 0.0 <= r <= 1.0
    assert 0.0 <= psi < 2.0 * np.pi
    np.testing.assert_array_equal(initial, [0.2, 1.4, 4.0])
