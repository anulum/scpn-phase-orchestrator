# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — operator-splitting (ADMM) convex QP solver

"""A deterministic operator-splitting solver for convex quadratic programs.

Solves the standard form used by the Koopman-MPC controller

    minimise   ½ xᵀ P x + qᵀ x
    subject to l ≤ A x ≤ u

with the ADMM iteration of Stellato et al. (2020, the OSQP algorithm). The
constant KKT matrix

    ⎡ P + σI    Aᵀ   ⎤
    ⎣ A       −ρ⁻¹I ⎦

is factorised once; each iteration solves it, relaxes the primal iterate, clips
the auxiliary variable onto ``[l, u]`` and updates the dual. The solver is pure
NumPy/SciPy — the always-present floor of the QP layer — and is validated
against the optional ``osqp`` C solver by the parity gate.

References
----------
* Stellato, Banjac, Goulart, Bemporad & Boyd 2020, *Math. Program. Comput.* 12,
  637-672 (arXiv:1711.08013) — OSQP: an operator splitting solver for quadratic
  programs.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
import scipy.linalg
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["QPSolution", "solve_qp_admm"]

_DEFAULT_SIGMA = 1.0e-6
_DEFAULT_RHO = 0.1
_DEFAULT_ALPHA = 1.6
_DEFAULT_MAX_ITER = 6000
_DEFAULT_EPS = 1.0e-8
_EQUALITY_RHO_SCALE = 1.0e3
_RHO_MIN = 1.0e-6
_RHO_MAX = 1.0e6
_RHO_ADAPT_INTERVAL = 25
_RHO_ADAPT_TRIGGER = 5.0


@dataclass(frozen=True)
class QPSolution:
    """The result of an ADMM convex-QP solve.

    Parameters
    ----------
    x : numpy.ndarray
        The primal solution of shape ``(n,)``.
    objective : float
        The objective value ``½ xᵀ P x + qᵀ x`` at the solution.
    iterations : int
        The number of ADMM iterations performed.
    primal_residual : float
        The infinity-norm primal residual ``‖A x − z‖∞`` at termination.
    dual_residual : float
        The infinity-norm dual residual ``‖P x + q + Aᵀ y‖∞`` at termination.
    converged : bool
        Whether both residuals fell below the tolerance before ``max_iter``.
    """

    x: FloatArray
    objective: float
    iterations: int
    primal_residual: float
    dual_residual: float
    converged: bool


def _validate_matrix(value: object, *, name: str) -> FloatArray:
    """Return the value as a validated finite matrix, else raise."""
    raw = np.asarray(value, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.all(np.isfinite(raw)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(raw, dtype=np.float64)


def _validate_vector(value: object, *, name: str) -> FloatArray:
    """Return the value as a validated finite vector, else raise."""
    raw = np.asarray(value, dtype=np.float64).ravel()
    if not np.all(np.isfinite(raw) | np.isinf(raw)):
        raise ValueError(f"{name} must not contain NaN values")
    return np.ascontiguousarray(raw, dtype=np.float64)


def _positive_real(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    parsed = float(value)
    if not np.isfinite(parsed) or parsed <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return parsed


def solve_qp_admm(
    objective_matrix: FloatArray,
    objective_vector: FloatArray,
    constraint_matrix: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    sigma: float = _DEFAULT_SIGMA,
    rho: float = _DEFAULT_RHO,
    alpha: float = _DEFAULT_ALPHA,
    max_iter: int = _DEFAULT_MAX_ITER,
    eps_abs: float = _DEFAULT_EPS,
    eps_rel: float = _DEFAULT_EPS,
) -> QPSolution:
    """Solve ``min ½xᵀPx + qᵀx`` s.t. ``l ≤ Ax ≤ u`` by ADMM (OSQP algorithm).

    Parameters
    ----------
    objective_matrix : numpy.ndarray
        The symmetric positive-semidefinite Hessian ``P`` of shape ``(n, n)``.
    objective_vector : numpy.ndarray
        The linear term ``q`` of shape ``(n,)``.
    constraint_matrix : numpy.ndarray
        The constraint matrix ``A`` of shape ``(m, n)``.
    lower, upper : numpy.ndarray
        The bounds ``l`` and ``u`` of shape ``(m,)``; ``±inf`` encodes a
        one-sided or absent bound and ``l == u`` encodes an equality.
    sigma : float
        The primal regularisation ``σ > 0``.
    rho : float
        The base step size ``ρ > 0``; equality rows are scaled up internally.
    alpha : float
        The over-relaxation parameter ``α ∈ (0, 2)``.
    max_iter : int
        The maximum number of ADMM iterations.
    eps_abs, eps_rel : float
        The absolute and relative residual tolerances.

    Returns
    -------
    QPSolution
        The primal solution and convergence diagnostics.

    Raises
    ------
    ValueError
        If the matrix and vector shapes are inconsistent or contain NaNs.
    TypeError
        If a scalar parameter is not a real number.
    """
    hessian = _validate_matrix(objective_matrix, name="objective_matrix")
    linear = _validate_vector(objective_vector, name="objective_vector")
    constraint = _validate_matrix(constraint_matrix, name="constraint_matrix")
    lower_bound = _validate_vector(lower, name="lower")
    upper_bound = _validate_vector(upper, name="upper")
    sigma = _positive_real(sigma, name="sigma")
    rho_base = _positive_real(rho, name="rho")
    alpha = _positive_real(alpha, name="alpha")
    if not 0.0 < alpha < 2.0:
        raise ValueError("alpha must lie in (0, 2)")
    if isinstance(max_iter, bool) or not isinstance(max_iter, Integral):
        raise TypeError("max_iter must be an integer")
    max_iter = int(max_iter)
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")

    n_vars = linear.size
    n_cons = lower_bound.size
    if hessian.shape != (n_vars, n_vars):
        raise ValueError("objective_matrix must be (n, n) matching objective_vector")
    if constraint.shape != (n_cons, n_vars):
        raise ValueError("constraint_matrix must be (m, n) matching the bounds")
    if upper_bound.size != n_cons:
        raise ValueError("lower and upper must share the constraint count")
    if np.any(upper_bound < lower_bound):
        raise ValueError("upper bound must not be below the lower bound")

    # Per-constraint step: boost equalities so they are enforced tightly.
    equality = np.isclose(lower_bound, upper_bound)
    identity_n = sigma * np.eye(n_vars)

    def _build_rho(base: float) -> FloatArray:
        """Return the ADMM penalty parameter (rho) for the problem."""
        vector = np.full(n_cons, base, dtype=np.float64)
        vector[equality] *= _EQUALITY_RHO_SCALE
        return vector

    def _factor(rho_vector: FloatArray) -> tuple[FloatArray, NDArray[np.intp]]:
        """Return the cached KKT-matrix factorisation for the solver."""
        kkt = np.zeros((n_vars + n_cons, n_vars + n_cons), dtype=np.float64)
        kkt[:n_vars, :n_vars] = hessian + identity_n
        kkt[:n_vars, n_vars:] = constraint.T
        kkt[n_vars:, :n_vars] = constraint
        # A validated constraint always has at least one row (n_cons ≥ 1).
        kkt[n_vars:, n_vars:] = -np.diag(1.0 / rho_vector)
        return scipy.linalg.lu_factor(kkt)

    rho_vector = _build_rho(rho_base)
    lu_piv = _factor(rho_vector)

    x = np.zeros(n_vars, dtype=np.float64)
    z = np.zeros(n_cons, dtype=np.float64)
    y = np.zeros(n_cons, dtype=np.float64)

    primal_residual = np.inf
    dual_residual = np.inf
    converged = False
    iteration = 0
    while iteration < max_iter:
        iteration += 1
        rhs = np.concatenate((sigma * x - linear, z - y / rho_vector))
        solution = scipy.linalg.lu_solve(lu_piv, rhs)
        x_tilde = solution[:n_vars]
        nu = solution[n_vars:]
        z_tilde = z + (nu - y) / rho_vector

        x = alpha * x_tilde + (1.0 - alpha) * x
        z_relaxed = alpha * z_tilde + (1.0 - alpha) * z
        z_new = np.clip(z_relaxed + y / rho_vector, lower_bound, upper_bound)
        y = y + rho_vector * (z_relaxed - z_new)
        z = z_new

        constraint_x = constraint @ x
        primal_residual = float(np.max(np.abs(constraint_x - z))) if n_cons else 0.0
        dual_residual = float(np.max(np.abs(hessian @ x + linear + constraint.T @ y)))
        primal_scale = max(
            float(np.max(np.abs(constraint_x))) if n_cons else 0.0,
            float(np.max(np.abs(z))) if n_cons else 0.0,
        )
        dual_scale = max(
            float(np.max(np.abs(hessian @ x))),
            float(np.max(np.abs(linear))),
            float(np.max(np.abs(constraint.T @ y))) if n_cons else 0.0,
        )
        if primal_residual <= eps_abs + eps_rel * primal_scale and (
            dual_residual <= eps_abs + eps_rel * dual_scale
        ):
            converged = True
            break

        # Adaptive ρ (OSQP heuristic): rescale to balance the residuals and
        # re-factorise only when the step moves by more than the trigger ratio.
        if n_cons and iteration % _RHO_ADAPT_INTERVAL == 0:
            numerator = primal_residual / max(primal_scale, _DEFAULT_EPS)
            denominator = dual_residual / max(dual_scale, _DEFAULT_EPS)
            ratio = numerator / max(denominator, _DEFAULT_EPS)
            rho_new = float(np.clip(rho_base * np.sqrt(ratio), _RHO_MIN, _RHO_MAX))
            if (
                rho_new > _RHO_ADAPT_TRIGGER * rho_base
                or rho_new < rho_base / _RHO_ADAPT_TRIGGER
            ):
                rho_base = rho_new
                rho_vector = _build_rho(rho_base)
                lu_piv = _factor(rho_vector)

    objective = float(0.5 * x @ hessian @ x + linear @ x)
    return QPSolution(
        x=np.ascontiguousarray(x, dtype=np.float64),
        objective=objective,
        iterations=iteration,
        primal_residual=primal_residual,
        dual_residual=dual_residual,
        converged=converged,
    )
