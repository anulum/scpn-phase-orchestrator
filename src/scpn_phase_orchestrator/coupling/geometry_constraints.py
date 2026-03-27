# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Geometry constraints and Knm validation

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "GeometryConstraint",
    "SymmetryConstraint",
    "NonNegativeConstraint",
    "project_knm",
    "validate_knm",
]


class GeometryConstraint(ABC):
    """Base class for K_nm matrix geometry constraints."""

    @abstractmethod
    def project(self, knm: NDArray) -> NDArray:
        """Project *knm* onto the feasible set defined by this constraint."""
        ...


class SymmetryConstraint(GeometryConstraint):
    """Enforce K_nm symmetry: K -> (K + K^T) / 2."""

    def project(self, knm: NDArray) -> NDArray:
        """Return the symmetric part of *knm*."""
        result: NDArray = 0.5 * (knm + knm.T)
        return result


class NonNegativeConstraint(GeometryConstraint):
    """Clamp negative entries to zero."""

    def project(self, knm: NDArray) -> NDArray:
        """Return *knm* with all negative entries replaced by 0."""
        result: NDArray = np.maximum(knm, 0.0)
        return result


def validate_knm(knm: NDArray, *, atol: float = 1e-12) -> None:
    """Check that a coupling matrix is square, symmetric, non-negative, zero-diagonal.

    Raises ValueError with a specific message on the first violation found.
    """
    if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
        raise ValueError(f"Knm must be square, got shape {knm.shape}")
    if not np.allclose(knm, knm.T, atol=atol):
        raise ValueError("Knm is not symmetric")
    if np.any(knm < -atol):
        raise ValueError("Knm contains negative entries")
    diag_max = float(np.max(np.abs(np.diag(knm))))
    if diag_max > atol:
        raise ValueError(f"Knm diagonal is non-zero (max |diag| = {diag_max:.2e})")


def project_knm(knm: NDArray, constraints: list[GeometryConstraint]) -> NDArray:
    """Apply all geometry constraints sequentially, then zero the diagonal."""
    result = knm.copy()
    for c in constraints:
        result = c.project(result)
    np.fill_diagonal(result, 0.0)
    return result
