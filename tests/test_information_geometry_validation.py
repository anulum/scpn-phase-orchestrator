# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — information-geometry validator tests

"""Validation tests for the information-geometry input helpers.

Exercises the reachable error paths of the backend-name, simplex, gradient,
float-array, finite-real, and non-empty-string validators (dimension, emptiness,
zero mass, shape, and finiteness failures).
"""

from __future__ import annotations

import numpy as np
import pytest

import scpn_phase_orchestrator.supervisor.information_geometry as _ig
from scpn_phase_orchestrator.supervisor.information_geometry import (
    _as_finite_real,
    _as_float_array,
    _as_non_empty_str,
    _normalise_backend_name,
    _normalise_simplex,
    _validate_gradient,
)

assert _ig is not None


def test_backend_name_rejects_a_non_string() -> None:
    with pytest.raises(ValueError, match="backend must be one of: jax, numpy"):
        _normalise_backend_name(123)


def test_simplex_rejects_a_multidimensional_array() -> None:
    with pytest.raises(ValueError, match="must be a one-dimensional array"):
        _normalise_simplex(np.zeros((2, 2)), "p")


def test_simplex_rejects_an_empty_array() -> None:
    with pytest.raises(ValueError, match="must contain at least one element"):
        _normalise_simplex(np.array([]), "p")


def test_simplex_rejects_a_zero_mass_distribution() -> None:
    with pytest.raises(ValueError, match="must have positive mass"):
        _normalise_simplex(np.array([0.0, 0.0]), "p")


def test_gradient_rejects_a_multidimensional_array() -> None:
    with pytest.raises(ValueError, match="must be a one-dimensional array"):
        _validate_gradient(np.zeros((2, 2)), (2, 2), "g")


def test_float_array_rejects_non_finite_values() -> None:
    with pytest.raises(ValueError, match="must contain finite values"):
        _as_float_array(np.array([1.0, np.nan]), "x")


def test_finite_real_rejects_a_non_finite_value() -> None:
    with pytest.raises(ValueError, match="must be finite"):
        _as_finite_real(float("nan"), "x", allow_non_positive=True)


def test_non_empty_str_rejects_a_blank_value() -> None:
    with pytest.raises(ValueError, match="must be a non-empty string"):
        _as_non_empty_str("   ", "x")
