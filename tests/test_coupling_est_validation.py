# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling estimation input-validation tests

"""Input-validation tests for the coupling-estimation public surface.

Covers the defensive branches of ``_validate_inputs`` and ``_validate_n_harmonics``
reached through :func:`estimate_coupling` and :func:`estimate_coupling_harmonics`:
boolean-dtype, non-numeric, wrong-dimensional, and non-finite phase/frequency
arrays, plus a non-positive harmonic count. An empty boolean array is the input
that slips past the element-wise boolean-alias scan and exercises the dtype guard.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from scpn_phase_orchestrator.autotune.coupling_est import (
    estimate_coupling,
    estimate_coupling_harmonics,
)

_DT = 0.01


def _phases() -> np.ndarray:
    """Return a small valid two-oscillator phase trajectory."""
    return np.zeros((2, 8), dtype=np.float64)


def _omegas() -> np.ndarray:
    """Return a valid two-oscillator frequency vector."""
    return np.ones(2, dtype=np.float64)


def test_phases_boolean_dtype_is_rejected() -> None:
    # An empty boolean array passes the element-wise alias scan but trips the
    # dtype guard.
    with pytest.raises(ValueError, match="phases must not contain boolean values"):
        estimate_coupling(np.array([], dtype=bool), _omegas(), _DT)


def test_phases_non_numeric_is_rejected() -> None:
    phases = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(ValueError, match="phases must be a finite 2-D trajectory"):
        estimate_coupling(cast("np.ndarray", phases), _omegas(), _DT)


def test_phases_wrong_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="phases must be a finite 2-D trajectory"):
        estimate_coupling(np.array([1.0, 2.0, 3.0]), np.ones(3), _DT)


def test_phases_non_finite_is_rejected() -> None:
    phases = np.array([[1.0, np.nan], [2.0, 3.0]])
    with pytest.raises(ValueError, match="phases must contain only finite values"):
        estimate_coupling(phases, _omegas(), _DT)


def test_omegas_boolean_dtype_is_rejected() -> None:
    with pytest.raises(ValueError, match="omegas must not contain boolean values"):
        estimate_coupling(_phases(), np.array([], dtype=bool), _DT)


def test_omegas_non_numeric_is_rejected() -> None:
    omegas = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="omegas must be a finite 1-D frequency"):
        estimate_coupling(_phases(), cast("np.ndarray", omegas), _DT)


def test_omegas_wrong_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="omegas must be a finite 1-D frequency"):
        estimate_coupling(_phases(), np.array([[1.0], [2.0]]), _DT)


def test_omegas_non_finite_is_rejected() -> None:
    with pytest.raises(ValueError, match="omegas must contain only finite values"):
        estimate_coupling(_phases(), np.array([1.0, np.inf]), _DT)


def test_non_positive_harmonic_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="n_harmonics must be a positive integer"):
        estimate_coupling_harmonics(_phases(), _omegas(), _DT, n_harmonics=0)


def test_valid_inputs_return_a_square_matrix() -> None:
    rng = np.random.default_rng(7)
    phases = rng.uniform(0.0, 2.0 * np.pi, (3, 64))
    knm = estimate_coupling(phases, np.ones(3), _DT)
    assert knm.shape == (3, 3)
