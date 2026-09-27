# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry constraints measurement ingress

"""Verify original measurement types through public coupling operations."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.geometry_constraints import (
    NonNegativeConstraint,
    SymmetryConstraint,
    project_knm,
    validate_knm,
)

FloatArray: TypeAlias = NDArray[np.float64]


@pytest.mark.parametrize(
    "dtype", ["U8", "timedelta64[ms]", "datetime64[ms]", "object-time"]
)
@pytest.mark.parametrize("project", [False, True])
def test_geometry_measurements_reject_coercion_aliases(
    dtype: str, project: bool
) -> None:
    """Projection and validation reject text and temporal source matrices."""
    matrix = cast(
        FloatArray,
        np.full((2, 2), np.timedelta64(1, "ms"), dtype=object)
        if dtype == "object-time"
        else np.array([[0.0, 1.0], [1.0, 0.0]]).astype(dtype),
    )
    with pytest.raises(ValueError):
        if project:
            project_knm(matrix, [SymmetryConstraint(), NonNegativeConstraint()])
        else:
            validate_knm(matrix)


def test_geometry_numeric_objects_preserve_projection() -> None:
    """Preserve real object input semantics through an actual constraint chain."""
    matrix = np.array([[0.0, 2.0], [1.0, 0.0]])
    objects = cast(FloatArray, matrix.astype(object))
    constraints = [SymmetryConstraint(), NonNegativeConstraint()]
    actual = project_knm(objects, constraints)
    np.testing.assert_array_equal(actual, project_knm(matrix, constraints))
    validate_knm(actual)
