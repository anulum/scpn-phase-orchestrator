# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Bayesian UPDE uncertainty propagation

"""Uncertainty propagation for Kuramoto UPDE rollouts.

The shipped backend is deterministic NumPy Monte Carlo over explicit
distributions for ``omega`` and ``K_nm``. Probabilistic-programming backends
are reserved as fail-closed names until their samplers are implemented and
benchmarked against this reproducible baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Literal, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde.engine import upde_run
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

FloatArray: TypeAlias = NDArray[np.float64]
BackendName: TypeAlias = Literal["numpy", "numpyro", "blackjax"]
MethodName: TypeAlias = Literal["euler", "rk4", "rk45"]

__all__ = [
    "BackendName",
    "BayesianBackendStatus",
    "BayesianUPDEConfig",
    "BayesianUPDEResult",
    "FloatArray",
    "GaussianUPDEPosteriorFit",
    "GaussianArrayDistribution",
    "MethodName",
    "audit_bayesian_backend_status",
    "bayesian_upde_run",
    "fit_gaussian_upde_posterior",
]


@runtime_checkable
class _ArrayDistribution(Protocol):
    """Protocol for sampleable array-valued input distributions."""

    shape: tuple[int, ...]

    def sample(self, rng: np.random.Generator, n_samples: int) -> FloatArray:
        """Draw ``n_samples`` arrays with leading sample dimension."""


@dataclass(frozen=True, slots=True)
class GaussianArrayDistribution:
    """Independent Gaussian array distribution with optional matrix guards."""

    mean: object
    std: object
    non_negative: bool = False
    zero_diagonal: bool = False

    def __post_init__(self) -> None:
        mean = _as_finite_array(self.mean, name="mean")
        std = _as_finite_array(self.std, name="std")
        if std.shape != mean.shape:
            raise ValueError(f"std must have shape {mean.shape}, got {std.shape}")
        if np.any(std < 0.0):
            raise ValueError("std must be non-negative")
        if self.zero_diagonal:
            _validate_square(mean, name="mean")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the event shape sampled by this Gaussian distribution.

        Returns
        -------
        tuple[int, ...]
            Return the event shape sampled by this Gaussian distribution.
        """
        return tuple(np.asarray(self.mean).shape)

    def sample(self, rng: np.random.Generator, n_samples: int) -> FloatArray:
        """Draw finite Gaussian samples with optional support guards applied.

        Parameters
        ----------
        rng : np.random.Generator
            NumPy random generator used for sampling.
        n_samples : int
            Number of samples to draw.

        Returns
        -------
        FloatArray
            Finite Gaussian samples, shape ``(n_samples, *event_shape)``.

        Raises
        ------
        TypeError
            If ``rng`` is not a NumPy random generator.
        """
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        sample_count = _validate_positive_integer(
            n_samples,
            name="n_samples",
            minimum=1,
        )
        draws = rng.normal(
            loc=np.asarray(self.mean, dtype=np.float64),
            scale=np.asarray(self.std, dtype=np.float64),
            size=(sample_count, *self.shape),
        )
        samples = np.asarray(draws, dtype=np.float64)
        if self.non_negative:
            samples = np.maximum(samples, 0.0)
        if self.zero_diagonal:
            diag = np.arange(self.shape[0])
            samples[:, diag, diag] = 0.0
        return np.ascontiguousarray(samples, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class BayesianUPDEConfig:
    """Configuration for Bayesian UPDE uncertainty propagation."""

    n_samples: int = 128
    seed: int | None = None
    dt: float = 0.01
    n_steps: int = 1
    method: MethodName = "rk4"
    credible_interval: float = 0.95
    backend: BackendName = "numpy"
    n_substeps: int = 1
    atol: float = 1e-6
    rtol: float = 1e-3

    def __post_init__(self) -> None:
        _validate_positive_integer(self.n_samples, name="n_samples", minimum=2)
        _validate_positive_integer(self.n_steps, name="n_steps", minimum=1)
        _validate_positive_integer(self.n_substeps, name="n_substeps", minimum=1)
        if self.seed is not None:
            _validate_positive_integer(self.seed, name="seed", minimum=0)
        _validate_positive_finite(self.dt, name="dt")
        _validate_positive_finite(self.atol, name="atol")
        _validate_positive_finite(self.rtol, name="rtol")
        if self.method not in {"euler", "rk4", "rk45"}:
            raise ValueError("method must be one of: euler, rk4, rk45")
        if self.backend not in {"numpy", "numpyro", "blackjax"}:
            raise ValueError("backend must be one of: numpy, numpyro, blackjax")
        _validate_open_unit_interval(
            self.credible_interval,
            name="credible_interval",
        )


@dataclass(frozen=True, slots=True)
class BayesianUPDEResult:
    """Posterior predictive order-parameter summary."""

    r_samples: FloatArray
    final_phase_samples: FloatArray
    omega_mean: FloatArray
    knm_mean: FloatArray
    r_mean: float
    r_sigma: float
    r_lower: float
    r_upper: float
    psi_mean: float
    sample_count: int
    credible_interval: float
    backend: str
    method: str

    @property
    def r_plus_minus(self) -> tuple[float, float]:
        """Return the compact ``R ± sigma`` pair.

        Returns
        -------
        tuple[float, float]
            Return the compact ``R ± sigma`` pair.
        """
        return self.r_mean, self.r_sigma

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe uncertainty diagnostics.

        Returns
        -------
        dict[str, object]
            Return JSON-safe uncertainty diagnostics.
        """
        return {
            "kind": "bayesian_upde",
            "backend": self.backend,
            "method": self.method,
            "sample_count": self.sample_count,
            "credible_interval": self.credible_interval,
            "r_summary": {
                "mean": self.r_mean,
                "sigma": self.r_sigma,
                "lower": self.r_lower,
                "upper": self.r_upper,
                "plus_minus": [self.r_mean, self.r_sigma],
            },
            "psi_mean": self.psi_mean,
            "omega_mean": self.omega_mean.tolist(),
            "knm_mean": self.knm_mean.tolist(),
            "final_phase_mean": np.mean(self.final_phase_samples, axis=0).tolist(),
            "diagnostics": {
                "finite_samples": bool(
                    np.all(np.isfinite(self.r_samples))
                    and np.all(np.isfinite(self.final_phase_samples))
                ),
                "r_min": float(np.min(self.r_samples)),
                "r_max": float(np.max(self.r_samples)),
            },
        }


@dataclass(frozen=True, slots=True)
class BayesianBackendStatus:
    """Execution status for one Bayesian UPDE backend name."""

    backend: str
    available: bool
    fail_closed: bool
    reason: str
    sample_count: int

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe backend availability diagnostics.

        Returns
        -------
        dict[str, object]
            Return JSON-safe backend availability diagnostics.
        """
        return {
            "kind": "bayesian_backend_status",
            "backend": self.backend,
            "available": self.available,
            "fail_closed": self.fail_closed,
            "reason": self.reason,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class GaussianUPDEPosteriorFit:
    """Gaussian posterior approximation fitted from observed phase trajectories."""

    omega: GaussianArrayDistribution
    knm: GaussianArrayDistribution
    residual_rmse: float
    sample_count: int
    dt: float
    ridge: float
    backend: str = "numpy_lstsq"

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe posterior-fit diagnostics.

        Returns
        -------
        dict[str, object]
            Return JSON-safe posterior-fit diagnostics.
        """
        return {
            "kind": "gaussian_upde_posterior_fit",
            "backend": self.backend,
            "sample_count": self.sample_count,
            "dt": self.dt,
            "ridge": self.ridge,
            "residual_rmse": self.residual_rmse,
            "omega": {
                "mean": np.asarray(self.omega.mean, dtype=np.float64).tolist(),
                "std": np.asarray(self.omega.std, dtype=np.float64).tolist(),
            },
            "knm": {
                "mean": np.asarray(self.knm.mean, dtype=np.float64).tolist(),
                "std": np.asarray(self.knm.std, dtype=np.float64).tolist(),
            },
            "diagnostics": {
                "finite": bool(
                    np.all(np.isfinite(np.asarray(self.omega.mean)))
                    and np.all(np.isfinite(np.asarray(self.omega.std)))
                    and np.all(np.isfinite(np.asarray(self.knm.mean)))
                    and np.all(np.isfinite(np.asarray(self.knm.std)))
                    and np.isfinite(self.residual_rmse)
                ),
                "zero_diagonal": bool(
                    np.allclose(np.diag(np.asarray(self.knm.mean)), 0.0)
                    and np.allclose(np.diag(np.asarray(self.knm.std)), 0.0)
                ),
            },
        }


def _as_finite_array(value: object, *, name: str) -> FloatArray:
    """Return the value as a validated finite array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, not a scalar")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        array = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in array.flat)


def _validate_square(array: FloatArray, *, name: str) -> None:
    """Return the value as a validated finite square matrix, else raise."""
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")


def _validate_positive_finite(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real")
    return coerced


def _validate_positive_integer(
    value: object,
    *,
    name: str,
    minimum: int,
) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    coerced = int(value)
    if coerced < minimum:
        raise ValueError(f"{name} must be a non-boolean integer >= {minimum}")
    return coerced


def _validate_open_unit_interval(value: object, *, name: str) -> float:
    """Return ``value`` as a float in (0, 1), else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must lie in (0, 1)")
    coerced = float(value)
    if not np.isfinite(coerced) or not (0.0 < coerced < 1.0):
        raise ValueError(f"{name} must lie in (0, 1)")
    return coerced


def fit_gaussian_upde_posterior(
    phase_trajectory: object,
    *,
    dt: float,
    alpha: object | None = None,
    ridge: float = 1e-6,
    coupling_std_floor: float = 1e-6,
    omega_std_floor: float = 1e-6,
) -> GaussianUPDEPosteriorFit:
    """Fit Gaussian ``omega`` and ``K_nm`` priors from observed phases.

    The estimator is a deterministic NumPy ridge least-squares baseline. It
    fits the Kuramoto right-hand side independently per target oscillator:

    ``d theta_i / dt = omega_i + sum_j K_ij sin(theta_j - theta_i - alpha_ij)``.

    The result is intentionally review-only: it produces distributions that can
    feed :func:`bayesian_upde_run`, but it does not apply control actions.

    Parameters
    ----------
    phase_trajectory : object
        Observed phase trajectory, shape ``(T, N)``.
    dt : float
        Integration step size.
    alpha : object | None
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    ridge : float
        Ridge-regularisation strength for the prior fit.
    coupling_std_floor : float
        Lower bound on the fitted coupling standard deviation.
    omega_std_floor : float
        Lower bound on the fitted natural-frequency standard deviation.

    Returns
    -------
    GaussianUPDEPosteriorFit
        The fitted Gaussian ``omega`` and ``K_nm`` prior.

    Raises
    ------
    ValueError
        If the phase trajectory is empty or non-finite.
    """
    trajectory = _as_finite_array(phase_trajectory, name="phase_trajectory")
    if trajectory.ndim != 2:
        raise ValueError(
            f"phase_trajectory must be a 2-D array, got shape {trajectory.shape}"
        )
    if trajectory.shape[0] < 3:
        raise ValueError("phase_trajectory must contain at least three samples")
    dt_value = _validate_positive_finite(dt, name="dt")
    ridge_value = _validate_non_negative_finite(ridge, name="ridge")
    coupling_floor = _validate_non_negative_finite(
        coupling_std_floor,
        name="coupling_std_floor",
    )
    omega_floor = _validate_non_negative_finite(
        omega_std_floor,
        name="omega_std_floor",
    )
    n_samples, n_oscillators = trajectory.shape
    alpha_array = (
        np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
        if alpha is None
        else _as_finite_array(alpha, name="alpha")
    )
    if alpha_array.shape != (n_oscillators, n_oscillators):
        raise ValueError(
            f"alpha must have shape {(n_oscillators, n_oscillators)}, "
            f"got {alpha_array.shape}"
        )

    unwrapped = np.unwrap(trajectory, axis=0)
    derivatives = np.diff(unwrapped, axis=0) / dt_value
    theta = trajectory[:-1]
    omega_mean = np.empty(n_oscillators, dtype=np.float64)
    omega_std = np.empty(n_oscillators, dtype=np.float64)
    knm_mean = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    knm_std = np.zeros((n_oscillators, n_oscillators), dtype=np.float64)
    residuals: list[FloatArray] = []

    for target in range(n_oscillators):
        source_indices = [source for source in range(n_oscillators) if source != target]
        features = np.column_stack(
            [
                np.ones(theta.shape[0], dtype=np.float64),
                *[
                    np.sin(
                        theta[:, source]
                        - theta[:, target]
                        - alpha_array[target, source]
                    )
                    for source in source_indices
                ],
            ]
        )
        target_derivative = derivatives[:, target]
        gram = features.T @ features
        penalty = ridge_value * np.eye(gram.shape[0], dtype=np.float64)
        penalty[0, 0] = 0.0
        coeffs = np.linalg.solve(gram + penalty, features.T @ target_derivative)
        predicted = features @ coeffs
        residual = target_derivative - predicted
        residuals.append(residual)
        dof = max(1, features.shape[0] - features.shape[1])
        residual_sigma = float(np.sqrt(float(residual @ residual) / dof))
        covariance = residual_sigma**2 * np.linalg.pinv(gram + penalty)
        coefficient_std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        omega_mean[target] = coeffs[0]
        omega_std[target] = max(float(coefficient_std[0]), omega_floor)
        for offset, source in enumerate(source_indices, start=1):
            value = max(float(coeffs[offset]), 0.0)
            knm_mean[target, source] = value
            knm_std[target, source] = (
                max(float(coefficient_std[offset]), coupling_floor)
                if value > 0.0
                else 0.0
            )

    residual_vector = np.concatenate(residuals)
    residual_rmse = float(np.sqrt(float(np.mean(residual_vector**2))))
    return GaussianUPDEPosteriorFit(
        omega=GaussianArrayDistribution(omega_mean, omega_std),
        knm=GaussianArrayDistribution(
            knm_mean,
            knm_std,
            non_negative=True,
            zero_diagonal=True,
        ),
        residual_rmse=residual_rmse,
        sample_count=n_samples,
        dt=dt_value,
        ridge=ridge_value,
    )


def audit_bayesian_backend_status(
    phases: object,
    *,
    omega: object,
    knm: object,
    alpha: object,
    zeta: float,
    psi: float,
    config: BayesianUPDEConfig | None = None,
    backends: tuple[BackendName, ...] = ("numpy", "numpyro", "blackjax"),
) -> tuple[BayesianBackendStatus, ...]:
    """Probe Bayesian backend names without silently accepting unsupported ones.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    omega : object
        Natural-frequency distribution or array.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    config : BayesianUPDEConfig | None
        Optional configuration object, or ``None`` for defaults.
    backends : tuple[BackendName, ...]
        Backend names to probe, in priority order.

    Returns
    -------
    tuple[BayesianBackendStatus, ...]
        Per-backend availability diagnostics.
    """
    base_config = config or BayesianUPDEConfig(n_samples=8, seed=0, n_steps=1)
    statuses: list[BayesianBackendStatus] = []
    for backend in backends:
        backend_config = replace(base_config, backend=backend)
        try:
            result = bayesian_upde_run(
                phases,
                omega=omega,
                knm=knm,
                alpha=alpha,
                zeta=zeta,
                psi=psi,
                config=backend_config,
            )
        except NotImplementedError as exc:
            statuses.append(
                BayesianBackendStatus(
                    backend=backend,
                    available=False,
                    fail_closed=True,
                    reason=str(exc),
                    sample_count=0,
                )
            )
        else:
            statuses.append(
                BayesianBackendStatus(
                    backend=backend,
                    available=True,
                    fail_closed=False,
                    reason="executed",
                    sample_count=result.sample_count,
                )
            )
    return tuple(statuses)


def _validate_non_negative_finite(value: object, *, name: str) -> float:
    """Return ``value`` as a non-negative finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be non-negative finite real")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced < 0.0:
        raise ValueError(f"{name} must be non-negative finite real")
    return coerced


def _sample_array(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    n_samples: int,
    rng: np.random.Generator,
    name: str,
) -> FloatArray:
    """Return a sampled input array from the distribution."""
    if isinstance(value, _ArrayDistribution):
        if value.shape != expected_shape:
            raise ValueError(
                f"{name} distribution must have shape {expected_shape}, "
                f"got {value.shape}"
            )
        samples = value.sample(rng, n_samples)
    else:
        array = _as_finite_array(value, name=name)
        if array.shape != expected_shape:
            raise ValueError(
                f"{name} must have shape {expected_shape}, got {array.shape}"
            )
        samples = np.broadcast_to(array, (n_samples, *expected_shape)).copy()
    if samples.shape != (n_samples, *expected_shape):
        raise ValueError(
            f"{name} samples must have shape {(n_samples, *expected_shape)}, "
            f"got {samples.shape}"
        )
    if not np.all(np.isfinite(samples)):
        raise ValueError(f"{name} samples must be finite")
    return np.ascontiguousarray(samples, dtype=np.float64)


def bayesian_upde_run(
    phases: object,
    *,
    omega: object,
    knm: object,
    alpha: object,
    zeta: float,
    psi: float,
    config: BayesianUPDEConfig | None = None,
) -> BayesianUPDEResult:
    """Run UPDE over sampled ``omega`` and ``K_nm`` distributions.

    Parameters
    ----------
    phases : object
        Oscillator phases in radians, shape ``(N,)``.
    omega : object
        Natural-frequency distribution or array.
    knm : object
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : object
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    zeta : float
        External drive strength ``ζ``.
    psi : float
        External drive reference phase ``Ψ`` in radians.
    config : BayesianUPDEConfig | None
        Optional configuration object, or ``None`` for defaults.

    Returns
    -------
    BayesianUPDEResult
        The Bayesian UPDE result with uncertainty diagnostics.

    Raises
    ------
    NotImplementedError
        If the requested backend is not implemented.
    ValueError
        If the sampled inputs are invalid.
    """
    resolved = config or BayesianUPDEConfig()
    if resolved.backend != "numpy":
        raise NotImplementedError(
            f"{resolved.backend} Bayesian UPDE backend is not implemented; "
            "use backend='numpy' for reproducible Monte Carlo propagation"
        )

    phase_array = _as_finite_array(phases, name="phases")
    if phase_array.ndim != 1:
        raise ValueError(f"phases must be 1-D, got shape {phase_array.shape}")
    n_oscillators = int(phase_array.shape[0])
    alpha_array = _as_finite_array(alpha, name="alpha")
    if alpha_array.shape != (n_oscillators, n_oscillators):
        raise ValueError(
            f"alpha must have shape {(n_oscillators, n_oscillators)}, "
            f"got {alpha_array.shape}"
        )
    zeta_value = float(zeta)
    psi_value = float(psi)
    if not np.isfinite(zeta_value) or not np.isfinite(psi_value):
        raise ValueError("zeta and psi must be finite real values")

    rng = np.random.default_rng(resolved.seed)
    omega_samples = _sample_array(
        omega,
        expected_shape=(n_oscillators,),
        n_samples=resolved.n_samples,
        rng=rng,
        name="omega",
    )
    knm_samples = _sample_array(
        knm,
        expected_shape=(n_oscillators, n_oscillators),
        n_samples=resolved.n_samples,
        rng=rng,
        name="knm",
    )

    final_phase_samples = np.empty(
        (resolved.n_samples, n_oscillators), dtype=np.float64
    )
    r_samples = np.empty(resolved.n_samples, dtype=np.float64)
    psi_samples = np.empty(resolved.n_samples, dtype=np.float64)
    for idx in range(resolved.n_samples):
        final_phases = upde_run(
            phase_array,
            omega_samples[idx],
            knm_samples[idx],
            alpha_array,
            zeta_value,
            psi_value,
            resolved.dt,
            resolved.n_steps,
            method=resolved.method,
            n_substeps=resolved.n_substeps,
            atol=resolved.atol,
            rtol=resolved.rtol,
        )
        final_phase_samples[idx] = final_phases
        order_r, order_psi = compute_order_parameter(final_phases)
        r_samples[idx] = order_r
        psi_samples[idx] = order_psi

    tail = (1.0 - resolved.credible_interval) / 2.0
    lower, upper = np.quantile(r_samples, [tail, 1.0 - tail])
    return BayesianUPDEResult(
        r_samples=np.ascontiguousarray(r_samples, dtype=np.float64),
        final_phase_samples=np.ascontiguousarray(
            final_phase_samples,
            dtype=np.float64,
        ),
        omega_mean=np.mean(omega_samples, axis=0),
        knm_mean=np.mean(knm_samples, axis=0),
        r_mean=float(np.mean(r_samples)),
        r_sigma=float(np.std(r_samples, ddof=1)),
        r_lower=float(lower),
        r_upper=float(upper),
        psi_mean=float(np.mean(psi_samples)),
        sample_count=resolved.n_samples,
        credible_interval=resolved.credible_interval,
        backend=resolved.backend,
        method=resolved.method,
    )
