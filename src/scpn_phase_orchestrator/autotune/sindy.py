# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Phase-SINDy Symbolic Discovery

from __future__ import annotations

import numpy as np
from scipy.linalg import lstsq

try:
    from spo_kernel import (
        sindy_fit_rust as _rust_sindy_fit,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["PhaseSINDy"]


class PhaseSINDy:
    """Symbolic Discovery of Phase Dynamics using SINDy.

    Discovers the governing equations of a coupled oscillator network
    by performing sparse regression on a library of trigonometric
    interaction terms.
    """

    def __init__(self, threshold: float = 0.05, max_iter: int = 10):
        if threshold < 0.0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.threshold = threshold
        self.max_iter = max_iter
        self.coefficients: list[np.ndarray] = []
        self.feature_names: list[list[str]] = []

    def fit(self, phases: np.ndarray, dt: float) -> list[np.ndarray]:
        """Discover equations node-by-node to handle independent coupling."""
        T, N = phases.shape

        if _HAS_RUST:
            p_flat = np.ascontiguousarray(phases, dtype=np.float64).ravel()
            result_flat = _rust_sindy_fit(
                p_flat,
                N,
                T,
                dt,
                self.threshold,
                self.max_iter,
            )
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
                self.coefficients.append(np.array(xi))
                self.feature_names.append(names)
            return self.coefficients

        unwrapped = np.unwrap(phases, axis=0)
        theta_dot = np.diff(unwrapped, axis=0) / dt
        X = phases[:-1, :]

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
            xi = lstsq(Theta, theta_dot[:, i])[0]

            for _ in range(self.max_iter):
                small_indices = np.abs(xi) < self.threshold
                xi[small_indices] = 0
                big_indices = ~small_indices
                if np.any(big_indices):
                    xi[big_indices] = lstsq(Theta[:, big_indices], theta_dot[:, i])[0]

            self.coefficients.append(xi)
            self.feature_names.append(f_names)

        return self.coefficients

    def get_equations(self) -> list[str]:
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
