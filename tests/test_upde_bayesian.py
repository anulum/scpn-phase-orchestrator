# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bayesian UPDE uncertainty tests

from __future__ import annotations

import json

import numpy as np
import pytest

from scpn_phase_orchestrator.upde import (
    BayesianUPDEConfig,
    GaussianArrayDistribution,
    audit_bayesian_backend_status,
    bayesian_upde_run,
    fit_gaussian_upde_posterior,
)
from scpn_phase_orchestrator.upde.engine import upde_run


def _base_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phases = np.array([0.0, 0.4, 1.1, 1.9], dtype=np.float64)
    omegas = np.array([0.9, 1.0, 1.08, 1.16], dtype=np.float64)
    knm = np.full((4, 4), 0.18, dtype=np.float64)
    np.fill_diagonal(knm, 0.0)
    alpha = np.zeros((4, 4), dtype=np.float64)
    return phases, omegas, knm, alpha


def test_bayesian_upde_reports_reproducible_r_uncertainty() -> None:
    phases, omegas, knm, alpha = _base_problem()

    config = BayesianUPDEConfig(
        n_samples=96,
        seed=1234,
        dt=0.01,
        n_steps=12,
        method="rk4",
        credible_interval=0.90,
    )
    omega_dist = GaussianArrayDistribution(
        mean=omegas,
        std=np.full_like(omegas, 0.015),
    )
    knm_dist = GaussianArrayDistribution(
        mean=knm,
        std=np.full_like(knm, 0.01),
        non_negative=True,
        zero_diagonal=True,
    )

    result = bayesian_upde_run(
        phases,
        omega=omega_dist,
        knm=knm_dist,
        alpha=alpha,
        zeta=0.02,
        psi=0.1,
        config=config,
    )
    repeated = bayesian_upde_run(
        phases,
        omega=omega_dist,
        knm=knm_dist,
        alpha=alpha,
        zeta=0.02,
        psi=0.1,
        config=config,
    )

    assert result.backend == "numpy"
    assert result.r_samples.shape == (96,)
    assert result.final_phase_samples.shape == (96, 4)
    assert 0.0 <= result.r_mean <= 1.0
    assert result.r_sigma > 0.0
    assert result.r_lower <= result.r_mean <= result.r_upper
    np.testing.assert_allclose(result.r_samples, repeated.r_samples)
    np.testing.assert_allclose(result.final_phase_samples, repeated.final_phase_samples)


def test_bayesian_upde_collapses_to_zero_sigma_for_deterministic_inputs() -> None:
    phases, omegas, knm, alpha = _base_problem()

    result = bayesian_upde_run(
        phases,
        omega=omegas,
        knm=knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=8, seed=1, dt=0.01, n_steps=4),
    )

    assert result.r_sigma == pytest.approx(0.0, abs=1e-15)
    assert result.r_lower == pytest.approx(result.r_mean)
    assert result.r_upper == pytest.approx(result.r_mean)
    assert result.omega_mean.shape == (4,)
    assert result.knm_mean.shape == (4, 4)


def test_bayesian_upde_audit_record_is_json_safe() -> None:
    phases, omegas, knm, alpha = _base_problem()

    result = bayesian_upde_run(
        phases,
        omega=GaussianArrayDistribution(omegas, np.full_like(omegas, 0.01)),
        knm=knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=16, seed=22, dt=0.02, n_steps=5),
    )
    record = result.to_audit_record()
    encoded = json.dumps(record, sort_keys=True)

    assert "bayesian_upde" in encoded
    assert record["sample_count"] == 16
    assert record["r_summary"]["sigma"] == result.r_sigma
    assert record["diagnostics"]["finite_samples"] is True


def test_bayesian_upde_rejects_invalid_distributions_and_backends() -> None:
    phases, omegas, knm, alpha = _base_problem()

    with pytest.raises(ValueError, match="std must have shape"):
        GaussianArrayDistribution(omegas, np.ones((2, 2)))
    with pytest.raises(ValueError, match="n_samples"):
        BayesianUPDEConfig(n_samples=1)
    with pytest.raises(NotImplementedError, match="numpyro"):
        bayesian_upde_run(
            phases,
            omega=omegas,
            knm=knm,
            alpha=alpha,
            zeta=0.0,
            psi=0.0,
            config=BayesianUPDEConfig(backend="numpyro"),
        )


def test_bayesian_backend_status_audits_reserved_fail_closed_names() -> None:
    phases, omegas, knm, alpha = _base_problem()

    statuses = audit_bayesian_backend_status(
        phases,
        omega=omegas,
        knm=knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=8, seed=19, n_steps=2),
    )
    records = {status.backend: status for status in statuses}

    assert set(records) == {"numpy", "numpyro", "blackjax"}
    assert records["numpy"].available is True
    assert records["numpy"].fail_closed is False
    assert records["numpy"].sample_count == 8
    for backend in ("numpyro", "blackjax"):
        assert records[backend].available is False
        assert records[backend].fail_closed is True
        assert records[backend].sample_count == 0
        assert backend in records[backend].reason
        json.dumps(records[backend].to_audit_record(), allow_nan=False)


def _posterior_fit_trajectory(
    *,
    n_steps: int = 72,
    dt: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    phases = np.array([0.0, 0.4, 1.1], dtype=np.float64)
    omegas = np.array([0.92, 1.03, 1.11], dtype=np.float64)
    knm = np.array(
        [
            [0.0, 0.11, 0.04],
            [0.18, 0.0, 0.07],
            [0.09, 0.14, 0.0],
        ],
        dtype=np.float64,
    )
    alpha = np.zeros((3, 3), dtype=np.float64)
    rows = [phases.copy()]
    current = phases
    for _ in range(n_steps):
        current = upde_run(
            current,
            omegas,
            knm,
            alpha,
            0.0,
            0.0,
            dt,
            1,
            method="rk4",
        )
        rows.append(current.copy())
    return np.asarray(rows, dtype=np.float64), omegas, knm, alpha


def test_fit_gaussian_upde_posterior_recovers_deterministic_trajectory() -> None:
    trajectory, omegas, knm, alpha = _posterior_fit_trajectory()

    fit = fit_gaussian_upde_posterior(trajectory, dt=0.02, alpha=alpha)

    assert fit.backend == "numpy_lstsq"
    assert fit.sample_count == trajectory.shape[0]
    assert fit.residual_rmse < 1e-3
    np.testing.assert_allclose(fit.omega.mean, omegas, atol=3e-2)
    np.testing.assert_allclose(fit.knm.mean, knm, atol=6e-2)
    assert np.all(np.asarray(fit.omega.std) > 0.0)
    assert np.allclose(np.diag(np.asarray(fit.knm.mean)), 0.0)
    assert np.allclose(np.diag(np.asarray(fit.knm.std)), 0.0)


def test_fit_gaussian_upde_posterior_feeds_bayesian_rollout() -> None:
    trajectory, _, _, alpha = _posterior_fit_trajectory()
    fit = fit_gaussian_upde_posterior(trajectory, dt=0.02, alpha=alpha)

    result = bayesian_upde_run(
        trajectory[-1],
        omega=fit.omega,
        knm=fit.knm,
        alpha=alpha,
        zeta=0.0,
        psi=0.0,
        config=BayesianUPDEConfig(n_samples=32, seed=5, dt=0.02, n_steps=4),
    )

    assert result.sample_count == 32
    assert result.r_sigma >= 0.0
    assert result.r_lower <= result.r_mean <= result.r_upper
    assert result.to_audit_record()["diagnostics"]["finite_samples"] is True


def test_fit_gaussian_upde_posterior_audit_record_is_json_safe() -> None:
    trajectory, _, _, alpha = _posterior_fit_trajectory()

    fit = fit_gaussian_upde_posterior(trajectory, dt=0.02, alpha=alpha)
    record = fit.to_audit_record()

    assert record["kind"] == "gaussian_upde_posterior_fit"
    assert record["diagnostics"]["finite"] is True
    assert record["diagnostics"]["zero_diagonal"] is True
    json.dumps(record, allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("trajectory", "match"),
    [
        (np.array([0.0, 1.0]), "2-D"),
        (np.array([[0.0, 0.1], [0.2, 0.3]]), "at least three"),
        (np.array([[0.0, np.nan], [0.2, 0.3], [0.4, 0.5]]), "finite"),
    ],
)
def test_fit_gaussian_upde_posterior_rejects_invalid_trajectories(
    trajectory: np.ndarray,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        fit_gaussian_upde_posterior(trajectory, dt=0.02)


def test_fit_gaussian_upde_posterior_rejects_invalid_config() -> None:
    trajectory, _, _, alpha = _posterior_fit_trajectory()

    with pytest.raises(ValueError, match="dt"):
        fit_gaussian_upde_posterior(trajectory, dt=0.0, alpha=alpha)
    with pytest.raises(ValueError, match="ridge"):
        fit_gaussian_upde_posterior(trajectory, dt=0.02, alpha=alpha, ridge=-1.0)
    with pytest.raises(ValueError, match="alpha"):
        fit_gaussian_upde_posterior(trajectory, dt=0.02, alpha=np.zeros((2, 2)))
