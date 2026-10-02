# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native finite coupling projection

"""Verify the installed projection and its real Python consumer without substitutes."""

from __future__ import annotations

from fractions import Fraction
from typing import cast

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)
from scpn_phase_orchestrator.upde.engine import UPDEEngine

MAX = float(np.finfo(np.float64).max)
TINY = float(np.nextafter(0.0, 1.0))


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (MAX, MAX),
        (-MAX, -MAX),
        (MAX, MAX / 2),
        (MAX, -MAX),
        (MAX, -float(np.nextafter(MAX, 0))),
        (TINY, TINY),
        (TINY, 2 * TINY),
        (-TINY, -TINY),
        (0.3, 0.7),
    ],
)
def test_installed_projection_matches_exact_mean(left: float, right: float) -> None:
    """Actual Rust agrees with exact means and the public Python projection."""
    raw = np.array([[MAX, left], [right, -MAX]])
    original = raw.copy()
    mean = float((Fraction.from_float(left) + Fraction.from_float(right)) / 2)
    expected = np.array([[0.0, max(mean, 0.0)], [max(mean, 0.0), 0.0]])
    native = np.array(spo_kernel.PyCouplingBuilder.project(raw.ravel(), 2)).reshape(
        2, 2
    )
    public = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
    np.testing.assert_array_equal(native, expected)
    np.testing.assert_array_equal(native, public)
    np.testing.assert_array_equal(raw, original)
    validate_knm(native)


@pytest.mark.parametrize("n", [0, 1, 16])
def test_native_projection_roundtrip(n: int) -> None:
    """Native results retain cardinality, zero diagonal and public validation."""
    raw = np.random.default_rng(7).uniform(-0.4, 0.4, (n, n))
    original = raw.copy()
    native = np.array(spo_kernel.PyCouplingBuilder.project(raw.ravel(), n)).reshape(
        n, n
    )
    np.testing.assert_array_equal(
        native, project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
    )
    validate_knm(native)
    np.testing.assert_array_equal(raw, original)


@pytest.mark.parametrize(
    "raw",
    [
        np.array([0.0, np.inf, 1.0, 0.0]),
        np.array([0.0, np.nan, 1.0, 0.0]),
        np.array([False, True, True, False]),
        np.array([0.0, 1.0j, 1.0j, 0.0]),
        np.array([0, "1.0", "1.0", 0], dtype=object),
        np.ones(3),
    ],
)
def test_native_refusal_preserves_source_and_recovery(raw: NDArray[np.generic]) -> None:
    """Native source-type, finiteness and cardinality refusals retain caller storage."""
    original = raw.copy()
    with pytest.raises(ValueError):
        spo_kernel.PyCouplingBuilder.project(raw, 2)
    np.testing.assert_array_equal(raw, original)
    np.testing.assert_array_equal(
        spo_kernel.PyCouplingBuilder.project([0, 0.6, 0.2, 0], 2), [0, 0.4, 0.4, 0]
    )


def test_native_projection_drives_public_engine() -> None:
    """A genuine projected native matrix drives the public RK4 engine unchanged."""
    raw = np.array([[2.0, -0.2, 0.8], [0.4, 3.0, 0.2], [0.6, 0.4, 4.0]])
    native = np.array(spo_kernel.PyCouplingBuilder.project(raw.ravel(), 3)).reshape(
        3, 3
    )
    expected = project_knm(raw, [SymmetryConstraint(), NonNegativeConstraint()])
    engine = UPDEEngine(3, dt=0.01, method="rk4")
    initial = np.array([0.2, 1.4, 4.0])
    frequencies = np.array([0.8, 1.0, 1.2])
    lag = np.zeros((3, 3))
    actual = engine.run(initial, frequencies, native, 0.0, 0.0, lag, n_steps=40)
    reference = engine.run(initial, frequencies, expected, 0.0, 0.0, lag, n_steps=40)
    np.testing.assert_array_equal(actual, reference)
    assert not np.allclose(actual, initial)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(initial, [0.2, 1.4, 4.0])


@pytest.mark.parametrize(
    ("n", "error"),
    [
        (True, ValueError),
        (-1, OverflowError),
        ("2", ValueError),
        (1 << (np.dtype(np.uintp).itemsize * 4), ValueError),
    ],
)
def test_native_projection_refuses_bad_dimensions(
    n: object, error: type[Exception]
) -> None:
    """Count aliases, negative dimensions and products outside usize refuse."""
    raw = np.array([0.0, 0.4, 0.4, 0.0])
    with pytest.raises(error):
        spo_kernel.PyCouplingBuilder.project(raw, n)
    np.testing.assert_array_equal(raw, [0.0, 0.4, 0.4, 0.0])
    np.testing.assert_array_equal(
        spo_kernel.PyCouplingBuilder.project(cast(NDArray[np.float64], raw), 2), raw
    )
