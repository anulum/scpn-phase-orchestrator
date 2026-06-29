# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lag model input-validation tests

"""Input-validation tests for the LagModel public surface.

Covers the defensive validators reached through ``estimate_from_distances``,
``estimate_lag``, and ``build_alpha_matrix``: non-numeric, non-square, and
non-finite distance matrices; non-numeric and wrong-dimensional signals; a
non-integer and a non-positive layer count; and a malformed lag-index pair.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.coupling.lags as _lags
from scpn_phase_orchestrator.coupling.lags import LagModel

assert _lags is not None


def test_distances_non_numeric_is_rejected() -> None:
    distances = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(ValueError, match="finite non-negative square matrix"):
        LagModel.estimate_from_distances(cast("np.ndarray", distances), speed=1.0)


def test_distances_non_square_is_rejected() -> None:
    with pytest.raises(ValueError, match="finite non-negative square matrix"):
        LagModel.estimate_from_distances(np.zeros((2, 3)), speed=1.0)


def test_distances_non_finite_is_rejected() -> None:
    distances = np.array([[0.0, np.nan], [np.nan, 0.0]])
    with pytest.raises(ValueError, match="distances must contain only finite values"):
        LagModel.estimate_from_distances(distances, speed=1.0)


def test_distances_boolean_array_is_rejected() -> None:
    # Exercises the constant-time dtype fast path of the boolean-alias guard.
    distances = np.array([[False, True], [True, False]])
    with pytest.raises(ValueError, match="distances must not contain boolean values"):
        LagModel.estimate_from_distances(distances, speed=1.0)


def test_distances_complex_array_is_rejected() -> None:
    # Exercises the constant-time dtype fast path of the complex-alias guard.
    distances = np.array([[0.0 + 0j, 1.0 + 0j], [1.0 + 0j, 0.0 + 0j]])
    with pytest.raises(ValueError, match="distances must contain real-valued samples"):
        LagModel.estimate_from_distances(cast("np.ndarray", distances), speed=1.0)


def test_estimate_lag_rejects_a_non_numeric_signal() -> None:
    signal_a = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="same finite one-dimensional arrays"):
        LagModel().estimate_lag(
            cast("np.ndarray", signal_a), np.ones(2), sample_rate=10.0
        )


def test_estimate_lag_rejects_a_two_dimensional_signal() -> None:
    with pytest.raises(ValueError, match="same finite one-dimensional arrays"):
        LagModel().estimate_lag(np.ones((2, 3)), np.ones(6), sample_rate=10.0)


def test_build_alpha_matrix_rejects_a_non_integer_layer_count() -> None:
    with pytest.raises(ValueError, match="n_layers must be a positive integer"):
        LagModel().build_alpha_matrix({}, n_layers=cast("int", 1.5))


def test_build_alpha_matrix_rejects_a_non_positive_layer_count() -> None:
    with pytest.raises(ValueError, match="n_layers must be a positive integer"):
        LagModel().build_alpha_matrix({}, n_layers=0)


def test_build_alpha_matrix_rejects_a_malformed_lag_index() -> None:
    lags = cast("dict[tuple[int, int], float]", {(0, 1, 2): 0.1})
    with pytest.raises(ValueError, match="lag index must be a pair of layer indices"):
        LagModel().build_alpha_matrix(lags, n_layers=3)


def test_estimate_from_distances_returns_antisymmetric_matrix() -> None:
    distances = np.array([[0.0, 2.0], [2.0, 0.0]])
    alpha = LagModel.estimate_from_distances(distances, speed=4.0)
    assert alpha.shape == (2, 2)
    np.testing.assert_allclose(alpha, -alpha.T, atol=1.0e-12)
