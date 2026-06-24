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
    """Return whether the value contains any boolean alias."""
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.ravel())


def _validate_knm_matrix(value: object, *, name: str = "Knm") -> FloatArray:
    """Return the coupling as a validated finite square matrix, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    if _contains_complex_alias(value):
        raise ValueError(f"{name} must be real-valued")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        if name == "geometry constraint output":
            raise ValueError("geometry constraint returned non-finite Knm values")
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(matrix, dtype=np.float64)


class GeometryConstraint(ABC):
    """Base class for K_nm matrix geometry constraints."""

    @abstractmethod
    def project(self, knm: FloatArray) -> FloatArray:
        """Project *knm* onto the feasible set defined by this constraint.

        Parameters
        ----------
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.

        Returns
        -------
        FloatArray
            The projection of *knm* onto the constraint's feasible set.
        """
        ...


class SymmetryConstraint(GeometryConstraint):
    """Enforce K_nm symmetry: K -> (K + K^T) / 2."""

    def project(self, knm: FloatArray) -> FloatArray:
        """Return the symmetric part of *knm*.

        Parameters
        ----------
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.

        Returns
        -------
        FloatArray
            The symmetric part of *knm*.
        """
        knm = _validate_knm_matrix(knm)
        result: FloatArray = 0.5 * (knm + knm.T)
        return result


class NonNegativeConstraint(GeometryConstraint):
    """Clamp negative entries to zero."""

    def project(self, knm: FloatArray) -> FloatArray:
        """Return *knm* with all negative entries replaced by 0.

        Parameters
        ----------
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.

        Returns
        -------
        FloatArray
            *knm* with negative entries clipped to zero.
        """
        knm = _validate_knm_matrix(knm)
        result: FloatArray = np.maximum(knm, 0.0)
        return result


def validate_knm(knm: FloatArray, *, atol: float = 1e-12) -> None:
    """Check that a coupling matrix is square, symmetric, non-negative, zero-diagonal.

    Raises ValueError with a specific message on the first violation found.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    atol : float
        Absolute tolerance for the validity checks.

    Raises
    ------
    ValueError
        If ``knm`` is not square, symmetric, non-negative, and zero-diagonal.
    """
    knm = _validate_knm_matrix(knm)
    if not np.allclose(knm, knm.T, atol=atol):
        raise ValueError("Knm is not symmetric")
    if np.any(knm < -atol):
        raise ValueError("Knm contains negative entries")
    diag_max = float(np.max(np.abs(np.diag(knm))))
    if diag_max > atol:
        raise ValueError(f"Knm diagonal is non-zero (max |diag| = {diag_max:.2e})")


def project_knm(knm: FloatArray, constraints: list[GeometryConstraint]) -> FloatArray:
    """Apply all geometry constraints sequentially, then zero the diagonal.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    constraints : list[GeometryConstraint]
        Geometry constraints applied in sequence.

    Returns
    -------
    FloatArray
        The coupling matrix after applying every constraint and zeroing the diagonal.

    Raises
    ------
    ValueError
        If a constraint produces an invalid coupling matrix.
    """
    result = _validate_knm_matrix(knm).copy()
    for c in constraints:
        if not isinstance(c, GeometryConstraint):
            raise ValueError("geometry constraint must be a GeometryConstraint")
        projected = _validate_knm_matrix(
            c.project(result), name="geometry constraint output"
        )
        if projected.shape != result.shape:
            raise ValueError("geometry constraint output shape must match Knm shape")
        result = projected
    np.fill_diagonal(result, 0.0)
    return result
