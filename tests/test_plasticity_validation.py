# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — three-factor plasticity validation tests

"""Validation tests for the three-factor plasticity public surface.

Exercises the phase-vector and square-matrix guards (boolean and non-numeric
arrays) and the eligibility/coupling shape-mismatch check, all reached through
``compute_eligibility`` and ``three_factor_update``.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.coupling.plasticity as _plasticity
from scpn_phase_orchestrator.coupling.plasticity import (
    compute_eligibility,
    three_factor_update,
)

assert _plasticity is not None


def test_phase_vector_rejects_an_empty_boolean_array() -> None:
    # An empty boolean array slips past the element-wise alias scan and reaches
    # the dtype guard.
    with pytest.raises(ValueError, match="phases must not contain boolean values"):
        compute_eligibility(cast("np.ndarray", np.array([], dtype=bool)))


def test_phase_vector_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="phases must be a finite 1-D phase vector"):
        compute_eligibility(cast("np.ndarray", np.array(["a", "b"], dtype=object)))


def test_coupling_matrix_rejects_an_empty_boolean_array() -> None:
    with pytest.raises(ValueError, match="knm must not contain boolean values"):
        three_factor_update(
            cast("np.ndarray", np.empty((0, 0), dtype=bool)),
            np.zeros((2, 2)),
            0.5,
            True,
        )


def test_coupling_matrix_rejects_non_numeric_values() -> None:
    with pytest.raises(ValueError, match="knm must be a finite square matrix"):
        three_factor_update(
            cast("np.ndarray", np.array([["a", "b"], ["c", "d"]], dtype=object)),
            np.zeros((2, 2)),
            0.5,
            True,
        )


def test_three_factor_update_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match knm shape"):
        three_factor_update(np.zeros((2, 2)), np.zeros((3, 3)), 0.5, True)
