# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ott-Antonsen mean-field reduction

"""Exact mean-field (Ott-Antonsen) reduction for globally-coupled Kuramoto.

Uses a Lorentzian ``g(ω)`` and a 5-backend fallback chain.

Dynamics
--------
On the Ott-Antonsen manifold the full N-oscillator Kuramoto system
reduces to a single complex-scalar ODE:

    dz/dt = −(Δ + iω₀)·z + (K/2)·(z − |z|²·z)

with ``z = R·e^{iψ}`` the mean-field order parameter, ``Δ`` the
half-width of the Lorentzian ``g(ω)``, ``ω₀`` its centre, and
``K`` the coupling strength.

Steady-state ``R_ss = √(1 − 2Δ/K)`` for ``K > K_c = 2Δ`` and
``R_ss = 0`` below. Reference: Ott & Antonsen 2008, *Chaos*
18(3):037113.

Numerics
--------
``run(z0, n_steps)`` is the compute-kernel path: a tight RK4 loop
on the real/imaginary components of ``z``. This is dispatched
across Rust / Mojo / Julia / Go / Python with bit-exact parity
(scalar ODE, no reduction identities, no global sums — the only
differences between backends are the rounding order of the
``k1..k4`` accumulation, which matches exactly).

The scalar-output helpers ``K_c``, ``steady_state_R`` and
``predict_from_oscillators`` stay native Python + optional Rust —
they are O(1) arithmetic or O(N) percentile work and do not
benefit from multi-language chains.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Complex, Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.upde import (
    _reduction_validation,
)
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main

try:
    from spo_kernel import (
        fit_lorentzian_rust as _rust_fit_lorentzian,
    )
    from spo_kernel import (
        steady_state_r_oa_rust as _rust_steady_state_r,
    )

    _HAS_RUST_SCALAR = True
except ImportError:
    _HAS_RUST_SCALAR = False

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "OAState",
    "OttAntonsenReduction",
]


if "OAState" not in globals():

    @dataclass
    class OAState:
        """Ott-Antonsen mean-field state: order parameter and critical coupling."""

        z: complex
        R: float
        psi: float
        K_c: float

        def __post_init__(self) -> None:
            z = _validate_finite_complex(self.z, name="z")
            r = _validate_unit_interval(self.R, name="R")
            psi = _validate_finite_real(self.psi, name="psi")
            k_c = _validate_finite_real(self.K_c, name="K_c")
            if k_c < 0.0:
                raise ValueError(f"K_c must be non-negative, got {k_c}")
            object.__setattr__(self, "z", z)
            object.__setattr__(self, "R", r)
            object.__setattr__(self, "psi", psi)
            object.__setattr__(self, "K_c", k_c)


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., tuple[float, float, float, float]]:
    """Load the Rust Ott-Antonsen backend callable."""
    from spo_kernel import oa_run_rust

    def _rust(
        z_re: float,
        z_im: float,
        omega_0: float,
        delta: float,
        k_coupling: float,
        dt: float,
        n_steps: int,
    ) -> tuple[float, float, float, float]:
        """Call the Rust Ott-Antonsen reduction step kernel."""
        re, im, r, psi = oa_run_rust(
            float(z_re),
            float(z_im),
            float(omega_0),
            float(delta),
            float(k_coupling),
            float(dt),
            int(n_steps),
        )
        return _reduction_validation.validate_oa_output(re, im, r, psi)

    return _rust


def _load_mojo_fn() -> Callable[..., tuple[float, float, float, float]]:
    # pragma: no cover — toolchain
    """Load the Mojo Ott-Antonsen backend callable."""
    from ..experimental.accelerators.upde._reduction_mojo import (
        _ensure_exe,
        oa_run_mojo,
    )

    _ensure_exe()
    return oa_run_mojo


def _load_julia_fn() -> Callable[..., tuple[float, float, float, float]]:
    # pragma: no cover — toolchain
    """Load the Julia Ott-Antonsen backend callable."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._reduction_julia import (
        oa_run_julia,
    )

    return oa_run_julia


def _load_go_fn() -> Callable[..., tuple[float, float, float, float]]:
    # pragma: no cover — toolchain
    """Load the Go Ott-Antonsen backend callable."""
    from ..experimental.accelerators.upde._reduction_go import (
        _load_lib,
        oa_run_go,
    )

    _load_lib()
    return oa_run_go


_LOADERS: dict[str, Callable[[], Callable[..., tuple[float, float, float, float]]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., tuple[float, float, float, float]]] = {}


def _load_backend(name: str) -> Callable[..., tuple[float, float, float, float]]:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
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


def _dispatch() -> Callable[..., tuple[float, float, float, float]] | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
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


def _validate_finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_finite_complex(value: object, *, name: str) -> complex:
    """Return ``value`` as a finite complex number, else raise."""
    if isinstance(value, bool) or not isinstance(value, Complex):
        raise ValueError(f"{name} must be a finite complex scalar, got {value!r}")
    coerced = complex(value)
    if not np.isfinite(coerced.real) or not np.isfinite(coerced.imag):
        raise ValueError(f"{name} must be a finite complex scalar, got {value!r}")
    if abs(coerced) > 1.0 + 1e-12:
        raise ValueError(f"{name} must lie on the OA unit disk, got {value!r}")
    return coerced


def _validate_frequency_sample(value: object, *, name: str) -> FloatArray:
    """Return the validated frequency-distribution sample, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    if arr.ndim != 1:
        raise ValueError(f"{name} shape {arr.shape} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one frequency")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in arr.flat)


def _validate_unit_interval(value: object, *, name: str) -> float:
    """Return ``value`` as a float in [0, 1], else raise ``ValueError``."""
    coerced = _validate_finite_real(value, name=name)
    if coerced < 0.0 or coerced > 1.0 + 1e-12:
        raise ValueError(f"{name} must lie in [0, 1], got {value!r}")
    return min(1.0, coerced)


def _oa_deriv(
    re: float,
    im: float,
    omega_0: float,
    delta: float,
    half_k: float,
) -> tuple[float, float]:
    """Return the Ott-Antonsen order-parameter derivative."""
    abs_sq = re * re + im * im
    lin_re = -delta * re + omega_0 * im
    lin_im = -delta * im - omega_0 * re
    cubic_factor = half_k * (1.0 - abs_sq)
    return (lin_re + cubic_factor * re, lin_im + cubic_factor * im)


def _python_oa_run(
    z_re: float,
    z_im: float,
    omega_0: float,
    delta: float,
    k_coupling: float,
    dt: float,
    n_steps: int,
) -> tuple[float, float, float, float]:
    """Python reference matching the Rust kernel exactly.

    Uses scalar-by-scalar RK4 on (re, im) with the same operation
    order and ``half_k = K/2`` factoring as
    ``spo-engine/src/reduction.rs``.
    """
    re, im = z_re, z_im
    half_k = k_coupling / 2.0
    for _ in range(n_steps):
        k1r, k1i = _oa_deriv(re, im, omega_0, delta, half_k)
        k2r, k2i = _oa_deriv(
            re + 0.5 * dt * k1r,
            im + 0.5 * dt * k1i,
            omega_0,
            delta,
            half_k,
        )
        k3r, k3i = _oa_deriv(
            re + 0.5 * dt * k2r,
            im + 0.5 * dt * k2i,
            omega_0,
            delta,
            half_k,
        )
        k4r, k4i = _oa_deriv(
            re + dt * k3r,
            im + dt * k3i,
            omega_0,
            delta,
            half_k,
        )
        re += (dt / 6.0) * (k1r + 2.0 * k2r + 2.0 * k3r + k4r)
        im += (dt / 6.0) * (k1i + 2.0 * k2i + 2.0 * k3i + k4i)
    r = (re * re + im * im) ** 0.5
    import math

    psi = math.atan2(im, re)
    return re, im, r, psi


class OttAntonsenReduction:
    """Ott-Antonsen mean-field reduction for globally-coupled Kuramoto.

    The class stores ``(ω₀, Δ, K, dt)`` and exposes ``K_c``,
    ``steady_state_R()``, ``step(z)``, ``run(z0, n_steps)`` and
    ``predict_from_oscillators(omegas, K)``. ``run`` is dispatched
    across the 5-backend chain; the scalar helpers stay Python +
    optional Rust.
    """

    def __init__(
        self,
        omega_0: float,
        delta: float,
        K: float,
        dt: float = 0.01,
    ):
        omega_0 = _validate_finite_real(omega_0, name="omega_0")
        delta = _validate_finite_real(delta, name="delta")
        K = _validate_finite_real(K, name="K")
        dt = _validate_finite_real(dt, name="dt")
        if delta < 0:
            raise ValueError(f"delta (half-width) must be non-negative, got {delta}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._omega_0 = omega_0
        self._delta = delta
        self._K = K
        self._dt = dt

    @property
    def K_c(self) -> float:
        """Critical coupling ``K_c = 2Δ``.

        Returns
        -------
        float
            Critical coupling ``K_c = 2Δ``.
        """
        return 2.0 * self._delta

    def steady_state_R(self) -> float:
        """Return the analytical steady-state ``R_ss = √(1 − 2Δ/K)`` for ``K > K_c``.

        Returns
        -------
        float
            Return the analytical steady-state ``R_ss = √(1 − 2Δ/K)`` for ``K > K_c``.
        """
        if _HAS_RUST_SCALAR:
            return float(_rust_steady_state_r(self._delta, self._K))
        if self.K_c >= self._K:
            return 0.0
        return float((1.0 - 2.0 * self._delta / self._K) ** 0.5)

    def step(self, z: complex) -> complex:
        """Single RK4 step on the OA ODE.

        Parameters
        ----------
        z : complex
            Complex Ott-Antonsen order parameter.

        Returns
        -------
        complex
            The complex order parameter after one RK4 step.
        """
        z = _validate_finite_complex(z, name="z")
        re, im, _, _ = self._run_scalar(z.real, z.imag, n_steps=1)
        return complex(re, im)

    def run(self, z0: complex, n_steps: int) -> OAState:
        """Integrate ``n_steps`` RK4 steps; return the final ``OAState``.

        Parameters
        ----------
        z0 : complex
            Initial complex Ott-Antonsen order parameter.
        n_steps : int
            Number of integration steps to run.

        Returns
        -------
        OAState
            The final ``OAState`` after ``n_steps`` RK4 steps.
        """
        z0 = _validate_finite_complex(z0, name="z0")
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        re, im, r, psi = self._run_scalar(z0.real, z0.imag, n_steps)
        return OAState(z=complex(re, im), R=r, psi=psi, K_c=self.K_c)

    def _run_scalar(
        self,
        z_re: float,
        z_im: float,
        n_steps: int,
    ) -> tuple[float, float, float, float]:
        """Run the scalar Ott-Antonsen reduction and return the trajectory."""
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        backend_fn = _dispatch()
        if backend_fn is not None:
            return _reduction_validation.validate_oa_output(
                *backend_fn(
                    z_re,
                    z_im,
                    self._omega_0,
                    self._delta,
                    self._K,
                    self._dt,
                    int(n_steps),
                )
            )
        return _python_oa_run(
            z_re,
            z_im,
            self._omega_0,
            self._delta,
            self._K,
            self._dt,
            int(n_steps),
        )

    def predict_from_oscillators(
        self,
        omegas: FloatArray,
        K: float,
    ) -> OAState:
        """Fit a Lorentzian to ``omegas`` and return the relaxed ``OAState``.

        Parameters
        ----------
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        K : float
            Global coupling strength.

        Returns
        -------
        OAState
            The relaxed ``OAState`` for the fitted Lorentzian.
        """
        omegas64 = _validate_frequency_sample(omegas, name="omegas")
        K = _validate_finite_real(K, name="K")
        if _HAS_RUST_SCALAR:
            omega_0, delta = _rust_fit_lorentzian(omegas64)
        else:
            omega_0 = float(np.median(omegas64))
            q75, q25 = np.percentile(omegas64, [75, 25])
            delta = (q75 - q25) / 2.0 if q75 > q25 else 0.01
        reducer = OttAntonsenReduction(omega_0, delta, K, dt=self._dt)
        return reducer.run(complex(0.01, 0.0), n_steps=int(10.0 / self._dt))
