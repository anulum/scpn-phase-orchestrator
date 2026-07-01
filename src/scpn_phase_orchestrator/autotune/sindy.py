# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-SINDy Symbolic Discovery

"""Sparse symbolic discovery of phase-dynamics equations from trajectories.

``PhaseSINDy`` builds per-node trigonometric libraries, fits sparse regression
coefficients, and formats discovered equations after a successful fit. Threshold
and iteration counts are validated at construction, and the optional Rust path
is remapped into the same Python coefficient layout. The class mutates only its
own coefficients and feature-name history; it does not update live coupling
state.
"""

from __future__ import annotations

from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import lstsq

try:
    from spo_kernel import (
        sindy_fit_rust as _rust_sindy_fit,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["PhaseSINDy"]

FloatArray: TypeAlias = NDArray[np.float64]


def _is_boolean_alias(value: object) -> bool:
    """Return whether the value is a boolean alias."""
    return isinstance(value, (bool, np.bool_))


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    raw = np.asarray(value, dtype=object)
    return any(_is_boolean_alias(item) for item in raw.ravel())


def _coerce_lstsq_coefficients(values: object, expected_size: int) -> FloatArray:
    """Return the least-squares coefficients as a finite array."""
    try:
        coefficients = np.asarray(values, dtype=np.float64).ravel()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "PhaseSINDy least-squares returned non-numeric coefficients"
        ) from exc
    if coefficients.size != expected_size:
        raise ValueError(
            "PhaseSINDy least-squares returned wrong coefficient count: "
            f"{coefficients.size} != {expected_size}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError("PhaseSINDy least-squares returned non-finite coefficients")
    return coefficients


class PhaseSINDy:
    """Symbolic Discovery of Phase Dynamics using SINDy.

    Discovers the governing equations of a coupled oscillator network
    by performing sparse regression on a library of trigonometric
    interaction terms.
    """

    def __init__(self, threshold: float = 0.05, max_iter: int = 10):
        """Create a SINDy estimator with validated sparsity controls."""
        if _is_boolean_alias(threshold) or not isinstance(threshold, Real):
            raise ValueError("threshold must be finite and non-negative")
        parsed_threshold = float(threshold)
        if not isfinite(parsed_threshold) or parsed_threshold < 0.0:
            raise ValueError("threshold must be non-negative and finite")
        if _is_boolean_alias(max_iter) or not isinstance(max_iter, Integral):
            raise ValueError("max_iter must be an integer >= 1")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        parsed_max_iter = int(max_iter)
        self.threshold: float = parsed_threshold
        self.max_iter: int = parsed_max_iter
        self.coefficients: list[FloatArray] = []
        self.feature_names: list[list[str]] = []

    def fit(self, phases: FloatArray, dt: float) -> list[FloatArray]:
        """Discover equations node-by-node to handle independent coupling.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        dt : float
            Integration step size.

        Returns
        -------
        list[FloatArray]
            Equations node-by-node to handle independent coupling.

        Raises
        ------
        ValueError
            If the inputs are invalid or inconsistent.
        """
        if _is_boolean_alias(dt) or not isinstance(dt, Real):
            raise ValueError("dt must be a finite and positive scalar")
        parsed_dt = float(dt)
        if not isfinite(parsed_dt) or parsed_dt <= 0.0:
            raise ValueError("dt must be a finite and positive scalar")

        if _contains_boolean_alias(phases):
            raise ValueError("phases must not contain boolean values")
        raw_phases = np.asarray(phases)
        if raw_phases.dtype == np.bool_:
            raise ValueError("phases must not contain boolean values")
        if np.iscomplexobj(raw_phases):
            raise ValueError("phases must be a finite 2D numeric array")

        try:
            phases_array = np.asarray(raw_phases, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("phases must be a finite 2D numeric array") from exc

        if phases_array.ndim != 2:
            raise ValueError("phases must be a 2D array [T, N]")

        if not np.isfinite(phases_array).all():
            raise ValueError("phases must be finite and numeric")

        T, N = phases_array.shape

        if T < 2 or N < 1:
            self.coefficients = []
            self.feature_names = []
            raise ValueError(
                "phases must contain at least two time samples and one oscillator"
            )
        if T - 1 < N:
            self.coefficients = []
            self.feature_names = []
            raise ValueError(
                "phases must provide at least one derivative sample per feature"
            )

        if _HAS_RUST:
            p_flat = np.ascontiguousarray(phases_array, dtype=np.float64).ravel()
            result_flat = _rust_sindy_fit(
                p_flat,
                N,
                T,
                parsed_dt,
                self.threshold,
                self.max_iter,
            )
            try:
                result_flat = np.asarray(result_flat, dtype=np.float64).ravel()
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Rust SINDy returned non-numeric coefficients"
                ) from exc
            expected = N * N
            if result_flat.size != expected:
                raise ValueError(
                    "Rust SINDy returned wrong number of coefficients: "
                    f"{result_flat.size} != {expected}"
                )
            if not np.all(np.isfinite(result_flat)):
                raise ValueError("Rust SINDy returned non-finite coefficients")
            result = result_flat.reshape(N, N)
            # Remap: Rust stores [ω at diagonal, K_ij off-diagonal]
            # Python expects [ω, K_j1, K_j2, ...] (constant first, then j≠i)
            self.coefficients = []
            self.feature_names = []
            for i in range(N):
                xi = [result[i, i]]  # constant (ω)
                names = ["1"]
                for j in range(N):
                    if j != i:
                        xi.append(result[i, j])
                        names.append(f"sin(theta_{j} - theta_{i})")
                self.coefficients.append(np.array(xi, dtype=np.float64))
                self.feature_names.append(names)
            return self.coefficients

        unwrapped = np.unwrap(phases_array, axis=0)
        theta_dot = np.diff(unwrapped, axis=0) / parsed_dt
        X = phases_array[:-1, :]

        self.coefficients = []
        self.feature_names = []

        for i in range(N):
            # 1. Build library for node i: [1, sin(theta_j - theta_i) for all j != i]
            library = [np.ones((T - 1, 1))]
            f_names = ["1"]

            for j in range(N):
                if i == j:
                    continue
                diff = X[:, j] - X[:, i]
                library.append(np.sin(diff)[:, np.newaxis])
                f_names.append(f"sin(theta_{j} - theta_{i})")

            Theta = np.hstack(library)

            # 2. STLSQ for this node
            xi = _coerce_lstsq_coefficients(
                lstsq(Theta, theta_dot[:, i])[0],
                Theta.shape[1],
            )

            for _ in range(self.max_iter):
                small_indices = np.abs(xi) < self.threshold
                xi[small_indices] = 0
                big_indices = ~small_indices
                if np.any(big_indices):
                    xi[big_indices] = _coerce_lstsq_coefficients(
                        lstsq(Theta[:, big_indices], theta_dot[:, i])[0],
                        int(np.count_nonzero(big_indices)),
                    )

            self.coefficients.append(xi)
            self.feature_names.append(f_names)

        return self.coefficients

    def get_equations(self) -> list[str]:
        """Format fitted sparse coefficients as per-node phase equations.

        Returns
        -------
        list[str]
            Format fitted sparse coefficients as per-node phase equations.

        Raises
        ------
        RuntimeError
            If the operation fails.
        """
        if not self.coefficients:
            raise RuntimeError("PhaseSINDy.get_equations() called before fit()")
        equations = []
        for i, xi in enumerate(self.coefficients):
            terms = []
            for j, val in enumerate(xi):
                if abs(val) > 1e-6:
                    terms.append(f"{val:.4f} * {self.feature_names[i][j]}")
            equations.append(
                f"d(theta_{i})/dt = " + (" + ".join(terms) if terms else "0")
            )
        return equations
