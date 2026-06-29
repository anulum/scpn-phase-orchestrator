# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SSGF cost numeric-coercion guard tests

"""Numeric-coercion guard tests for the SSGF cost public surface.

Covers the ``astype(np.float64)`` failure branches of the phase-vector and
weight-matrix validators, reached when an object array cannot be coerced to a
real numeric array.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_phase_orchestrator.ssgf.costs as _costs
from scpn_phase_orchestrator.ssgf.costs import compute_ssgf_costs

assert _costs is not None


def test_non_numeric_phases_are_rejected() -> None:
    phases = np.array(["a", "b"], dtype=object)
    with pytest.raises(ValueError, match="phases must be numeric"):
        compute_ssgf_costs(np.eye(2), cast("np.ndarray", phases))


def test_non_numeric_weight_matrix_is_rejected() -> None:
    weight_matrix = np.array([["a", "b"], ["c", "d"]], dtype=object)
    with pytest.raises(ValueError, match="W must be numeric"):
        compute_ssgf_costs(cast("np.ndarray", weight_matrix), np.array([0.1, 0.2]))
