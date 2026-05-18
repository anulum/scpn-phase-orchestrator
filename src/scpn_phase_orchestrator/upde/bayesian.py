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

from dataclasses import dataclass
from numbers import Real
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
    "BayesianUPDEConfig",
    "BayesianUPDEResult",
    "FloatArray",
    "GaussianArrayDistribution",
    "MethodName",
    "bayesian_upde_run",
]


@runtime_checkable
class _ArrayDistribution(Protocol):
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
        """Return the event shape sampled by this Gaussian distribution."""

        return tuple(np.asarray(self.mean).shape)

    def sample(self, rng: np.random.Generator, n_samples: int) -> FloatArray:
        """Draw finite Gaussian samples with optional support guards applied."""

        draws = rng.normal(
            loc=np.asarray(self.mean, dtype=np.float64),
            scale=np.asarray(self.std, dtype=np.float64),
            size=(n_samples, *self.shape),
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
        if isinstance(self.n_samples, bool) or self.n_samples < 2:
            raise ValueError("n_samples must be a non-boolean integer >= 2")
        if isinstance(self.n_steps, bool) or self.n_steps < 1:
            raise ValueError("n_steps must be a non-boolean integer >= 1")
        if isinstance(self.n_substeps, bool) or self.n_substeps < 1:
            raise ValueError("n_substeps must be a non-boolean integer >= 1")
        _validate_positive_finite(self.dt, name="dt")
        _validate_positive_finite(self.atol, name="atol")
        _validate_positive_finite(self.rtol, name="rtol")
        if self.method not in {"euler", "rk4", "rk45"}:
            raise ValueError("method must be one of: euler, rk4, rk45")
        if self.backend not in {"numpy", "numpyro", "blackjax"}:
            raise ValueError("backend must be one of: numpy, numpyro, blackjax")
        if not (0.0 < self.credible_interval < 1.0):
            raise ValueError("credible_interval must lie in (0, 1)")


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
        """Return the compact ``R ± sigma`` pair."""

        return self.r_mean, self.r_sigma

    def to_audit_record(self) -> dict[str, object]:
        """Return JSON-safe uncertainty diagnostics."""

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


def _as_finite_array(value: object, *, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if array.ndim == 0:
        raise ValueError(f"{name} must be an array, not a scalar")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_square(array: FloatArray, *, name: str) -> None:
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {array.shape}")


def _validate_positive_finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real")
    return coerced


def _sample_array(
    value: object,
    *,
    expected_shape: tuple[int, ...],
    n_samples: int,
    rng: np.random.Generator,
    name: str,
) -> FloatArray:
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
    """Run UPDE over sampled ``omega`` and ``K_nm`` distributions."""

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
