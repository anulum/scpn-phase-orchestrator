# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — convex QP layer (deterministic ADMM + optional OSQP)

"""The convex-QP layer for the Koopman-MPC controller.

The deterministic ADMM solver (:mod:`._qp_admm`) is the always-present floor and
the **canonical** path: a review-only controller must produce a reproducible,
content-hashable decision regardless of which solver libraries happen to be
installed. The optional ``osqp`` C solver is offered as a faster alternative and
is held to the ADMM result by the parity gate; it is never the silent default.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from ._qp_admm import QPSolution, solve_qp_admm

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = ["QPSolution", "available_qp_backends", "solve_qp"]

_QP_BACKENDS = ("admm", "osqp")


def _osqp_available() -> bool:
    """Return whether the OSQP solver is importable."""
    try:
        import osqp  # noqa: F401
    except ImportError:
        return False
    return True


def available_qp_backends() -> tuple[str, ...]:
    """Return the QP backends present in this environment, canonical-first."""
    if _osqp_available():
        return _QP_BACKENDS
    return ("admm",)


def _solve_qp_osqp(
    objective_matrix: FloatArray,
    objective_vector: FloatArray,
    constraint_matrix: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    eps_abs: float,
    eps_rel: float,
    max_iter: int,
) -> QPSolution:
    """Solve the convex QP via OSQP, else raise."""
    import osqp
    import scipy.sparse as sparse

    problem = osqp.OSQP()
    problem.setup(
        P=sparse.csc_matrix(objective_matrix),
        q=np.ascontiguousarray(objective_vector, dtype=np.float64),
        A=sparse.csc_matrix(constraint_matrix),
        l=np.ascontiguousarray(lower, dtype=np.float64),
        u=np.ascontiguousarray(upper, dtype=np.float64),
        eps_abs=eps_abs,
        eps_rel=eps_rel,
        max_iter=max_iter,
        verbose=False,
    )
    result = problem.solve()
    x = np.ascontiguousarray(np.asarray(result.x, dtype=np.float64))
    hessian = np.asarray(objective_matrix, dtype=np.float64)
    linear = np.asarray(objective_vector, dtype=np.float64)
    objective = float(0.5 * x @ hessian @ x + linear @ x)
    return QPSolution(
        x=x,
        objective=objective,
        iterations=int(result.info.iter),
        primal_residual=float(result.info.prim_res),
        dual_residual=float(result.info.dual_res),
        converged=str(result.info.status) == "solved",
    )


def solve_qp(
    objective_matrix: FloatArray,
    objective_vector: FloatArray,
    constraint_matrix: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    *,
    backend: str = "admm",
    eps_abs: float = 1.0e-8,
    eps_rel: float = 1.0e-8,
    max_iter: int = 6000,
) -> QPSolution:
    """Solve a convex QP with the chosen backend.

    Parameters
    ----------
    objective_matrix : numpy.ndarray
        The Hessian ``P`` of shape ``(n, n)``.
    objective_vector : numpy.ndarray
        The linear term ``q`` of shape ``(n,)``.
    constraint_matrix : numpy.ndarray
        The constraint matrix ``A`` of shape ``(m, n)``.
    lower, upper : numpy.ndarray
        The bounds ``l`` and ``u`` of shape ``(m,)``.
    backend : str
        ``"admm"`` for the deterministic floor (default, used by the audited
        controller) or ``"osqp"`` for the optional C solver.
    eps_abs, eps_rel : float
        Residual tolerances.
    max_iter : int
        The iteration cap.

    Returns
    -------
    QPSolution
        The primal solution and convergence diagnostics.

    Raises
    ------
    ValueError
        If ``backend`` is unknown or the requested backend is unavailable.
    """
    if backend == "admm":
        return solve_qp_admm(
            objective_matrix,
            objective_vector,
            constraint_matrix,
            lower,
            upper,
            max_iter=max_iter,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
        )
    if backend == "osqp":
        if not _osqp_available():
            raise ValueError("the osqp backend is not installed")
        return _solve_qp_osqp(
            objective_matrix,
            objective_vector,
            constraint_matrix,
            lower,
            upper,
            eps_abs=eps_abs,
            eps_rel=eps_rel,
            max_iter=max_iter,
        )
    raise ValueError("backend must be one of: " + ", ".join(_QP_BACKENDS))
