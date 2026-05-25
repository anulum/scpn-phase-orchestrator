# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry constraints and Knm validation

"""Projection and validation helpers for coupling-matrix geometry.

Geometry constraints project candidate `K_nm` matrices onto simple feasible
sets such as symmetry and non-negativity. `validate_knm` enforces the runtime
matrix contract used by domainpacks and UPDE handoff: square, symmetric,
non-negative, and zero diagonal within tolerance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GeometryConstraint",
    "SymmetryConstraint",
    "NonNegativeConstraint",
    "project_knm",
    "validate_knm",
]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, bool) for item in raw.ravel())


class GeometryConstraint(ABC):
    """Base class for K_nm matrix geometry constraints."""

    @abstractmethod
    def project(self, knm: FloatArray) -> FloatArray:
        """Project *knm* onto the feasible set defined by this constraint."""
        ...


class SymmetryConstraint(GeometryConstraint):
    """Enforce K_nm symmetry: K -> (K + K^T) / 2."""

    def project(self, knm: FloatArray) -> FloatArray:
        """Return the symmetric part of *knm*."""
        result: FloatArray = 0.5 * (knm + knm.T)
        return result


class NonNegativeConstraint(GeometryConstraint):
    """Clamp negative entries to zero."""

    def project(self, knm: FloatArray) -> FloatArray:
        """Return *knm* with all negative entries replaced by 0."""
        result: FloatArray = np.maximum(knm, 0.0)
        return result


def validate_knm(knm: FloatArray, *, atol: float = 1e-12) -> None:
    """Check that a coupling matrix is square, symmetric, non-negative, zero-diagonal.

    Raises ValueError with a specific message on the first violation found.
    """
    if _contains_boolean_alias(knm):
        raise ValueError("Knm must not contain boolean values")
    knm = np.asarray(knm, dtype=np.float64)
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError(f"Knm must be square, got shape {knm.shape}")
    if not np.all(np.isfinite(knm)):
        raise ValueError("Knm must contain only finite values")
    if not np.allclose(knm, knm.T, atol=atol):
        raise ValueError("Knm is not symmetric")
    if np.any(knm < -atol):
        raise ValueError("Knm contains negative entries")
    diag_max = float(np.max(np.abs(np.diag(knm))))
    if diag_max > atol:
        raise ValueError(f"Knm diagonal is non-zero (max |diag| = {diag_max:.2e})")


def project_knm(knm: FloatArray, constraints: list[GeometryConstraint]) -> FloatArray:
    """Apply all geometry constraints sequentially, then zero the diagonal."""
    if _contains_boolean_alias(knm):
        raise ValueError("Knm must not contain boolean values")
    result = np.asarray(knm, dtype=np.float64).copy()
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"Knm must be square, got shape {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError("Knm must contain only finite values")
    for c in constraints:
        result = c.project(result)
        if not np.all(np.isfinite(result)):
            raise ValueError("geometry constraint returned non-finite Knm values")
    np.fill_diagonal(result, 0.0)
    return result
