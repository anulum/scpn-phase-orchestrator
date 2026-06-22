# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — convex QP layer tests

"""Tests for the ADMM convex-QP floor and the QP backend layer.

The deterministic ADMM solver is checked against closed-form optima
(unconstrained and equality-constrained), against the optional ``osqp`` C solver
on random convex programmes, and across its full input-validation surface; the
backend layer is checked for canonical-first dispatch and graceful behaviour
when ``osqp`` is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation import _qp
from scpn_phase_orchestrator.actuation._qp import (
    available_qp_backends,
    solve_qp,
)
from scpn_phase_orchestrator.actuation._qp_admm import QPSolution, solve_qp_admm


def _random_convex_qp(
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_vars = int(rng.integers(3, 8))
    n_cons = int(rng.integers(4, 10))
    root = rng.standard_normal((n_vars, n_vars))
    hessian = root @ root.T + 0.5 * np.eye(n_vars)
    linear = rng.standard_normal(n_vars)
    constraint = rng.standard_normal((n_cons, n_vars))
    lower = -rng.uniform(0.5, 2.0, n_cons)
    upper = rng.uniform(0.5, 2.0, n_cons)
    return hessian, linear, constraint, lower, upper


def _osqp_installed() -> bool:
    try:
        import osqp  # noqa: F401
    except ImportError:
        return False
    return True


# --------------------------------------------------------------------------- #
# Closed-form optima                                                          #
# --------------------------------------------------------------------------- #
def test_unconstrained_qp_matches_the_normal_equation() -> None:
    hessian = np.array([[3.0, 0.5], [0.5, 2.0]])
    linear = np.array([1.0, -2.0])
    constraint = np.zeros((1, 2))
    wide = np.array([1.0e9])
    solution = solve_qp_admm(hessian, linear, constraint, -wide, wide)
    expected = np.linalg.solve(hessian, -linear)
    np.testing.assert_allclose(solution.x, expected, atol=1.0e-6)
    assert solution.converged


def test_equality_constraint_is_enforced() -> None:
    hessian = np.eye(2)
    linear = np.zeros(2)
    constraint = np.array([[1.0, 1.0]])
    target = np.array([2.0])
    solution = solve_qp_admm(hessian, linear, constraint, target, target)
    assert solution.x.sum() == pytest.approx(2.0, abs=1.0e-5)
    np.testing.assert_allclose(solution.x, [1.0, 1.0], atol=1.0e-5)


def test_active_box_constraint_clips_the_solution() -> None:
    # Unconstrained optimum is x = 5, but the upper bound caps it at 1.
    hessian = np.array([[2.0]])
    linear = np.array([-10.0])
    constraint = np.array([[1.0]])
    solution = solve_qp_admm(
        hessian, linear, constraint, np.array([-1.0]), np.array([1.0])
    )
    assert solution.x[0] == pytest.approx(1.0, abs=1.0e-6)


# --------------------------------------------------------------------------- #
# Parity against osqp                                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _osqp_installed(), reason="osqp is not installed")
@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_admm_matches_osqp(seed: int) -> None:
    problem = _random_convex_qp(seed)
    admm = solve_qp(*problem, backend="admm")
    osqp_solution = solve_qp(*problem, backend="osqp")
    np.testing.assert_allclose(admm.x, osqp_solution.x, atol=1.0e-5, rtol=1.0e-5)
    assert admm.objective == pytest.approx(osqp_solution.objective, abs=1.0e-5)


# --------------------------------------------------------------------------- #
# Backend layer                                                               #
# --------------------------------------------------------------------------- #
def test_available_backends_are_canonical_first() -> None:
    backends = available_qp_backends()
    assert backends[0] == "admm"
    assert "osqp" in backends or backends == ("admm",)


def test_solve_qp_rejects_an_unknown_backend() -> None:
    problem = _random_convex_qp(0)
    with pytest.raises(ValueError, match="backend must be one of"):
        solve_qp(*problem, backend="gurobi")


def test_solve_qp_osqp_errors_when_absent(monkeypatch) -> None:
    monkeypatch.setattr(_qp, "_osqp_available", lambda: False)
    problem = _random_convex_qp(1)
    with pytest.raises(ValueError, match="osqp backend is not installed"):
        solve_qp(*problem, backend="osqp")


def test_solve_qp_admm_returns_qp_solution() -> None:
    problem = _random_convex_qp(2)
    result = solve_qp(*problem, backend="admm")
    assert isinstance(result, QPSolution)
    assert result.converged


def test_available_backends_are_admm_only_without_osqp(monkeypatch) -> None:
    monkeypatch.setattr(_qp, "_osqp_available", lambda: False)
    assert available_qp_backends() == ("admm",)


def test_osqp_availability_is_false_when_the_import_fails(monkeypatch) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "osqp", None)
    assert _qp._osqp_available() is False


def test_admm_rejects_nan_bounds() -> None:
    with pytest.raises(ValueError, match="must not contain NaN values"):
        solve_qp_admm(
            np.eye(1), np.zeros(1), np.eye(1), np.array([np.nan]), np.array([1.0])
        )


# --------------------------------------------------------------------------- #
# Convergence diagnostics                                                     #
# --------------------------------------------------------------------------- #
def test_iteration_cap_reports_non_convergence() -> None:
    problem = _random_convex_qp(3)
    solution = solve_qp_admm(*problem, max_iter=1)
    assert solution.iterations == 1
    assert solution.converged is False


# --------------------------------------------------------------------------- #
# Validation                                                                  #
# --------------------------------------------------------------------------- #
def test_admm_rejects_a_non_square_hessian() -> None:
    with pytest.raises(ValueError, match="must be \\(n, n\\)"):
        solve_qp_admm(
            np.zeros((2, 3)), np.zeros(3), np.zeros((1, 3)), np.zeros(1), np.zeros(1)
        )


def test_admm_rejects_a_constraint_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="must be \\(m, n\\)"):
        solve_qp_admm(
            np.eye(2), np.zeros(2), np.zeros((1, 3)), np.zeros(1), np.zeros(1)
        )


def test_admm_rejects_mismatched_bounds() -> None:
    with pytest.raises(ValueError, match="share the constraint count"):
        solve_qp_admm(
            np.eye(2), np.zeros(2), np.zeros((2, 2)), np.zeros(2), np.zeros(1)
        )


def test_admm_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="must not be below the lower bound"):
        solve_qp_admm(
            np.eye(1), np.zeros(1), np.eye(1), np.array([1.0]), np.array([-1.0])
        )


def test_admm_rejects_a_non_two_dimensional_hessian() -> None:
    with pytest.raises(ValueError, match="must be a 2-D array"):
        solve_qp_admm(np.zeros(2), np.zeros(2), np.eye(2), np.zeros(2), np.zeros(2))


def test_admm_rejects_non_finite_objective_matrix() -> None:
    bad = np.array([[np.inf, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="only finite values"):
        solve_qp_admm(bad, np.zeros(2), np.eye(2), np.zeros(2), np.ones(2))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma": 0.0}, "sigma must be finite and positive"),
        ({"rho": -1.0}, "rho must be finite and positive"),
        ({"alpha": 2.5}, "alpha must lie in"),
        ({"alpha": "x"}, "alpha must be a real number"),
        ({"max_iter": 0}, "max_iter must be at least 1"),
        ({"max_iter": 2.5}, "max_iter must be an integer"),
    ],
)
def test_admm_rejects_invalid_parameters(kwargs, match) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        solve_qp_admm(
            np.eye(1),
            np.zeros(1),
            np.eye(1),
            np.array([-1.0]),
            np.array([1.0]),
            **kwargs,
        )


def test_qp_parity_gate_benchmark_passes() -> None:
    from benchmarks.koopman_mpc_benchmark import (
        benchmark_koopman_mpc_qp_parity_gate,
    )

    result = benchmark_koopman_mpc_qp_parity_gate(calls=1)
    assert result["parity_ok"] is True
