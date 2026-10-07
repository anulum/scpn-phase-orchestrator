# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Basin stability analysis

"""Finite-horizon Kuramoto threshold classification with five numerical owners.

The trial kernel uses full-snapshot explicit Euler and averages R after each
measurement step. Monte Carlo initial phases are drawn by NumPy in Python, so
all owners receive the same samples. Cross-language equality is numerical within
floating-point tolerance, rather than bit-exact. The direct Rust LCG sampler is
a separate legacy API and is never a public Monte Carlo shortcut.

A finite-window threshold fraction does not certify convergence, attraction-basin
volume, or linear stability. Automatic preference is Rust/Mojo/Julia/Go/Python;
it is not a measured speed ranking. A named owner is required to run or raise.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from numbers import Complex, Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import (
    _basin_stability_validation,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "BasinStabilityResult",
    "basin_stability",
    "multi_basin_stability",
    "steady_state_r",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")

FloatArray: TypeAlias = NDArray[np.float64]
TrialKernel: TypeAlias = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, int, float, float, int, int],
    float,
]


NativeTrialKernel: TypeAlias = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, int, float, float, int, int],
    object,
]


def _load_rust_fn() -> TrialKernel:
    """Load the Rust basin-stability backend callable."""
    kernel = importlib.import_module("spo_kernel")
    native = getattr(kernel, "steady_state_r_rust", None)
    if not callable(native):
        raise ImportError("Rust kernel does not export steady_state_r_rust")
    steady_state_r_rust = cast("NativeTrialKernel", native)

    def _rust(
        phases_init: FloatArray,
        omegas: FloatArray,
        knm_flat: FloatArray,
        alpha_flat: FloatArray,
        n: int,
        k_scale: float,
        dt: float,
        n_transient: int,
        n_measure: int,
    ) -> float:
        """Call the original Rust deterministic trial kernel."""
        return _basin_stability_validation.validate_basin_stability_output(
            steady_state_r_rust(
                np.ascontiguousarray(phases_init, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                int(n),
                float(k_scale),
                float(dt),
                int(n_transient),
                int(n_measure),
            )
        )

    return _rust


def _load_mojo_fn() -> TrialKernel:
    """Load the Mojo basin-stability backend callable."""
    from ..experimental.accelerators.upde._basin_stability_mojo import (
        _ensure_exe,
        steady_state_r_mojo,
    )

    _ensure_exe()
    return steady_state_r_mojo


def _load_julia_fn() -> TrialKernel:
    """Load the Julia basin-stability backend callable."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._basin_stability_julia import (
        _ensure,
        steady_state_r_julia,
    )

    _ensure()
    return steady_state_r_julia


def _load_go_fn() -> TrialKernel:
    """Load the Go basin-stability backend callable."""
    from ..experimental.accelerators.upde._basin_stability_go import (
        _load_lib,
        steady_state_r_go,
    )

    _load_lib()
    return steady_state_r_go


_LOADERS: dict[str, Callable[[], TrialKernel]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, TrialKernel] = {}


def _load_backend(name: str) -> TrialKernel:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve installed owners in the declared preference order."""
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(backend: str | None = None) -> TrialKernel | None:
    """Resolve a named owner strictly, or use the automatic preference chain."""
    if backend is not None:
        if backend not in _BACKEND_NAMES:
            raise ValueError(f"unknown basin backend: {backend!r}")
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError) as exc:
            raise ImportError(
                f"requested basin backend {backend!r} is unavailable"
            ) from exc
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    deduped: list[str] = []
    for backend in ordered_backends:
        if backend in deduped:
            continue
        deduped.append(backend)
    for backend in deduped:
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in arr.flat)


def _is_string_like(value: object) -> bool:
    """Return whether ``value`` is a Python or NumPy string scalar."""
    return isinstance(value, (str, np.str_))


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string scalar accepted by ``float``."""
    if not _is_string_like(value):
        return False
    try:
        float(str(value))
    except ValueError:
        return False
    return True


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` contains a stringified numeric scalar."""
    if _is_numeric_string_alias(value):
        return True
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(_is_numeric_string_alias(item) for item in arr.flat)


def _validate_integral(value: object, *, name: str, minimum: int) -> int:
    """Return ``value`` as a validated integer, else raise ``ValueError``."""
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not be a numeric-string alias")
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not be a numeric-string alias")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    coerced = _validate_finite_float(value, name=name)
    if coerced <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return coerced


def _validate_unit_interval(value: object, *, name: str) -> float:
    """Return ``value`` as a float in [0, 1], else raise ``ValueError``."""
    coerced = _validate_finite_float(value, name=name)
    if coerced < 0.0 or coerced > 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}")
    return coerced


def _validate_vector(value: object, *, name: str, shape: tuple[int, ...]) -> FloatArray:
    """Return a finite array with the requested shape, else raise."""
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        raw = np.asarray(value, dtype=object)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if any(
        isinstance(item, Complex) and not isinstance(item, Real) for item in raw.flat
    ):
        raise ValueError(f"{name} must be real-valued, not complex")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_omegas(value: object) -> FloatArray:
    """Return the natural frequencies as a validated finite array, else raise."""
    return _validate_nonempty_vector(value, name="omegas")


def _validate_nonempty_vector(value: object, *, name: str) -> FloatArray:
    """Return the value as a validated non-empty finite vector, else raise."""
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        raw = np.asarray(value, dtype=object)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if any(
        isinstance(item, Complex) and not isinstance(item, Real) for item in raw.flat
    ):
        raise ValueError(f"{name} must be real-valued, not complex")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} shape {arr.shape} must be one-dimensional")
    if arr.size < 1:
        raise ValueError(f"{name} must contain at least one oscillator")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_thresholds(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return the validated synchronisation thresholds, else raise."""
    if len(values) == 0:
        raise ValueError("R_thresholds must contain at least one threshold")
    thresholds = tuple(
        _validate_unit_interval(value, name=f"R_thresholds[{idx}]")
        for idx, value in enumerate(values)
    )
    labelled: dict[str, float] = {}
    for threshold in thresholds:
        label = f"R>={threshold:.2f}"
        if label in labelled and labelled[label] != threshold:
            raise ValueError(
                "R_thresholds must have distinct two-decimal result labels"
            )
        labelled[label] = threshold
    return thresholds


def _python_steady_state_r(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    k_scale: float,
    dt: float,
    n_transient: int,
    n_measure: int,
) -> float:
    """Execute the finite-horizon Euler law with exact-zero edge masking."""
    phases = phases_init.copy()
    alpha = alpha_flat.reshape(n, n)
    with np.errstate(over="ignore", invalid="ignore"):
        weights = knm_flat.reshape(n, n) * k_scale
        if not np.all(np.isfinite(weights)):
            raise ValueError("scaled coupling overflow")
        active = weights != 0.0
        r_sum = 0.0
        for step in range(n_transient + n_measure):
            angles = np.zeros((n, n), dtype=np.float64)
            np.subtract(
                phases[np.newaxis, :], phases[:, np.newaxis], out=angles, where=active
            )
            np.subtract(angles, alpha, out=angles, where=active)
            if not np.all(np.isfinite(angles)):
                raise ValueError("phase difference overflow")
            coupling = np.sum(weights * np.sin(angles), axis=1)
            velocity = omegas + coupling
            phases = phases + dt * velocity
            if not np.all(np.isfinite(velocity)) or not np.all(np.isfinite(phases)):
                raise ValueError("Euler step overflow")
            if step >= n_transient:
                r_sum += float(
                    np.hypot(np.mean(np.cos(phases)), np.mean(np.sin(phases)))
                )
    return _basin_stability_validation.validate_basin_stability_output(
        r_sum / n_measure
    )


def steady_state_r(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray | None = None,
    k_scale: float = 1.0,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    *,
    backend: str | None = None,
) -> float:
    """Measure one finite-window mean Kuramoto R through the selected owner.

    Integrates the Kuramoto ODE for ``n_transient + n_measure`` steps
    and returns the time-averaged order parameter over the latter
    window. Automatic selection uses the declared owner preference.

    Parameters
    ----------
    phases_init : FloatArray
        Initial oscillator phases in radians, shape ``(N,)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray | None
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    k_scale : float
        Multiplicative scale applied to the coupling matrix.
    dt : float
        Integration step size.
    n_transient : int
        Number of transient steps discarded before measurement.
    n_measure : int
        Number of post-step measurements; zero returns zero without integration.
    backend : str | None
        Named Rust/Mojo/Julia/Go/Python owner, or automatic preference. A named
        unavailable owner raises ImportError; computation errors propagate.

    Returns
    -------
    float
        Mean post-step Kuramoto ``R`` over the finite measurement window.
    """
    phases_init = _validate_nonempty_vector(phases_init, name="phases_init")
    N = int(phases_init.shape[0])
    omegas = _validate_vector(omegas, name="omegas", shape=(N,))
    knm = _validate_vector(knm, name="knm", shape=(N, N))
    if alpha is None:
        alpha_flat = np.zeros(N * N, dtype=np.float64)
    else:
        alpha_flat = _validate_vector(alpha, name="alpha", shape=(N, N)).ravel()
    k_scale = _validate_finite_float(k_scale, name="k_scale")
    dt = _validate_positive_float(dt, name="dt")
    n_transient = _validate_integral(n_transient, name="n_transient", minimum=0)
    n_measure = _validate_integral(n_measure, name="n_measure", minimum=0)
    backend_fn = _dispatch(backend)
    if n_measure == 0:
        return 0.0
    knm_flat = knm.ravel()
    if backend_fn is not None:
        return _basin_stability_validation.validate_basin_stability_output(
            backend_fn(
                phases_init,
                omegas,
                knm_flat,
                alpha_flat,
                N,
                k_scale,
                dt,
                n_transient,
                n_measure,
            )
        )
    return _python_steady_state_r(
        phases_init,
        omegas,
        knm_flat,
        alpha_flat,
        N,
        k_scale,
        dt,
        n_transient,
        n_measure,
    )


@dataclass
class BasinStabilityResult:
    """Consistent finite-window threshold-classification result.

    Attributes
    ----------
    S_B : float
        Classified sample fraction, canonicalized to n_converged/n_samples.
        Finite float32 rounding in a supplied fraction is accepted within
        relative tolerance 1e-7; the empty-sample convention is exactly zero.
    n_samples : int
        Total number of sampled initial conditions.
    n_converged : int
        Historical name for count(R_final >= R_threshold), without a
        convergence certificate.
    R_final : numpy.ndarray
        Per-trial mean post-step R over the finite measurement window.
    R_threshold : float
        Inclusive classification threshold in [0,1].
    """

    S_B: float
    n_samples: int
    n_converged: int
    R_final: FloatArray
    R_threshold: float

    def __post_init__(self) -> None:
        """Validate and canonicalise basin-stability result fields."""
        s_b = _validate_unit_interval(self.S_B, name="S_B")
        n_samples = _validate_integral(self.n_samples, name="n_samples", minimum=0)
        n_converged = _validate_integral(
            self.n_converged, name="n_converged", minimum=0
        )
        if n_converged > n_samples:
            raise ValueError("n_converged must be <= n_samples")

        r_final = _validate_vector(self.R_final, name="R_final", shape=(n_samples,))
        if np.any((r_final < 0.0) | (r_final > 1.0 + 1e-12)):
            raise ValueError("R_final values must lie in [0, 1]")
        r_threshold = _validate_unit_interval(self.R_threshold, name="R_threshold")
        actual_count = int(np.count_nonzero(r_final >= r_threshold))
        if n_converged != actual_count:
            raise ValueError("n_converged must match R_final >= R_threshold")
        fraction = n_converged / n_samples if n_samples else 0.0
        if not np.isclose(s_b, fraction, rtol=1e-7, atol=0.0):
            raise ValueError("S_B must match the classified sample fraction")

        self.S_B = fraction
        self.n_samples = n_samples
        self.n_converged = n_converged
        self.R_final = r_final
        self.R_threshold = r_threshold


def _monte_carlo_R_finals(
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    dt: float,
    n_transient: int,
    n_measure: int,
    n_samples: int,
    seed: int,
    backend: str | None,
) -> FloatArray:
    """Return paired finite-window R values with randomness owned by NumPy."""
    rng = np.random.default_rng(seed)
    R_finals = np.zeros(n_samples)
    backend_fn = _dispatch(backend)
    if n_measure == 0:
        return R_finals
    for i in range(n_samples):
        phases_init = rng.uniform(0, 2 * np.pi, n)
        if backend_fn is not None:
            R_finals[i] = _basin_stability_validation.validate_basin_stability_output(
                backend_fn(
                    phases_init,
                    np.ascontiguousarray(omegas, dtype=np.float64),
                    knm_flat,
                    alpha_flat,
                    n,
                    1.0,
                    dt,
                    n_transient,
                    n_measure,
                )
            )
        else:
            R_finals[i] = _python_steady_state_r(
                phases_init,
                omegas,
                knm_flat,
                alpha_flat,
                n,
                1.0,
                dt,
                n_transient,
                n_measure,
            )
    return R_finals


def basin_stability(
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray | None = None,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    n_samples: int = 100,
    R_threshold: float = 0.8,
    seed: int = 42,
    *,
    backend: str | None = None,
) -> BasinStabilityResult:
    """Estimate the fraction of finite-window trials meeting an R threshold.

    Draws ``n_samples`` random initial phase configurations from
    ``[0, 2π)^N``, evaluates each finite window via the selected trial
    kernel, and classifies trials by ``R_final ≥ R_threshold``.

    Parameters
    ----------
    omegas : FloatArray
        (N,) natural frequencies.
    knm : FloatArray
        (N, N) coupling matrix.
    alpha : FloatArray | None
        (N, N) phase lags (default: zeros).
    dt : float
        Integration timestep.
    n_transient : int
        Transient steps to discard.
    n_measure : int
        Steps to average R over.
    n_samples : int
        Number of random initial conditions.
    R_threshold : float
        Threshold for classifying as "synchronised".
    seed : int
        RNG seed (owned by Python).
    backend : str | None
        Named numerical owner or automatic preference, without named fallback.

    Returns
    -------
    BasinStabilityResult
        BasinStabilityResult with S_B, R_final array, and counts.
    """
    omegas = _validate_omegas(omegas)
    N = int(omegas.shape[0])
    knm = _validate_vector(knm, name="knm", shape=(N, N))
    if alpha is None:
        alpha_flat = np.zeros(N * N, dtype=np.float64)
    else:
        alpha_flat = _validate_vector(alpha, name="alpha", shape=(N, N)).ravel()
    dt = _validate_positive_float(dt, name="dt")
    n_transient = _validate_integral(n_transient, name="n_transient", minimum=0)
    n_measure = _validate_integral(n_measure, name="n_measure", minimum=0)
    n_samples = _validate_integral(n_samples, name="n_samples", minimum=0)
    R_threshold = _validate_unit_interval(R_threshold, name="R_threshold")
    seed = _validate_integral(seed, name="seed", minimum=0)

    R_finals = _monte_carlo_R_finals(
        omegas,
        knm.ravel(),
        alpha_flat,
        N,
        dt,
        n_transient,
        n_measure,
        n_samples,
        seed,
        backend,
    )
    n_converged = int(np.sum(R_finals >= R_threshold))
    return BasinStabilityResult(
        S_B=n_converged / n_samples if n_samples > 0 else 0.0,
        n_samples=n_samples,
        n_converged=n_converged,
        R_final=R_finals,
        R_threshold=R_threshold,
    )


def multi_basin_stability(
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray | None = None,
    dt: float = 0.01,
    n_transient: int = 500,
    n_measure: int = 200,
    n_samples: int = 100,
    R_thresholds: tuple[float, ...] = (0.3, 0.6, 0.8),
    seed: int = 42,
    *,
    backend: str | None = None,
) -> dict[str, BasinStabilityResult]:
    """Basin stability at multiple synchronisation thresholds.

    One Monte Carlo sweep; threshold classification repeated locally
    for each entry of ``R_thresholds``.

    Returns
    -------
        Dict mapping ``"R>={thresh:.2f}"`` to BasinStabilityResult.

    Parameters
    ----------
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    alpha : FloatArray | None
        Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
    dt : float
        Integration step size.
    n_transient : int
        Number of transient steps discarded before measurement.
    n_measure : int
        Number of post-step measurements; zero returns zero without integration.
    backend : str | None
        Named Rust/Mojo/Julia/Go/Python owner, or automatic preference. A named
        unavailable owner raises ImportError; computation errors propagate.
    n_samples : int
        Number of random initial-condition samples.
    R_thresholds : tuple[float, ...]
        Order-parameter thresholds to evaluate basin stability at.
    seed : int
        Seed for the deterministic RNG.
    """
    omegas = _validate_omegas(omegas)
    N = int(omegas.shape[0])
    knm = _validate_vector(knm, name="knm", shape=(N, N))
    if alpha is None:
        alpha_flat = np.zeros(N * N, dtype=np.float64)
    else:
        alpha_flat = _validate_vector(alpha, name="alpha", shape=(N, N)).ravel()
    dt = _validate_positive_float(dt, name="dt")
    n_transient = _validate_integral(n_transient, name="n_transient", minimum=0)
    n_measure = _validate_integral(n_measure, name="n_measure", minimum=0)
    n_samples = _validate_integral(n_samples, name="n_samples", minimum=0)
    R_thresholds = _validate_thresholds(R_thresholds)
    seed = _validate_integral(seed, name="seed", minimum=0)

    R_finals = _monte_carlo_R_finals(
        omegas,
        knm.ravel(),
        alpha_flat,
        N,
        dt,
        n_transient,
        n_measure,
        n_samples,
        seed,
        backend,
    )
    results: dict[str, BasinStabilityResult] = {}
    for thresh in R_thresholds:
        n_above = int(np.sum(R_finals >= thresh))
        results[f"R>={thresh:.2f}"] = BasinStabilityResult(
            S_B=n_above / n_samples if n_samples > 0 else 0.0,
            n_samples=n_samples,
            n_converged=n_above,
            R_final=R_finals,
            R_threshold=thresh,
        )
    return results
