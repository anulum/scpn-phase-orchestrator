# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Torus-preserving geometric integrator

"""Torus-preserving symplectic Euler integrator on ``T^N = (S¹)^N``.

Exposes a 5-backend fallback chain.

Scheme
------
Each phase is lifted to the unit circle ``z_i = exp(iθ_i)``; the
Kuramoto derivative ``ω_eff_i`` is computed in the tangent space,
and ``z_i`` is advanced by the exponential map

    z_i(t + dt) = z_i(t) · exp(i · ω_eff_i · dt)

followed by renormalisation to the unit circle. This avoids the
mod-``2π`` discontinuity that introduces subtle truncation errors
in standard integrators when trajectories cross ``θ = 0``.

Across the five backends the ``(z_re, z_im)`` state is carried in
between steps (no atan2 round-trip per step), matching the Rust
kernel ``spo-engine/src/geometric.rs`` bit-for-bit. The pairwise
derivative uses the sincos expansion on the ``alpha == 0`` branch
and the direct ``atan2 + sin(diff)`` form otherwise.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral, Real
from typing import cast

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "TorusEngine",
]

TWO_PI = 2.0 * np.pi


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., FloatArray]:
    from spo_kernel import torus_run_rust

    def _rust(
        phases: FloatArray,
        omegas: FloatArray,
        knm_flat: FloatArray,
        alpha_flat: FloatArray,
        n: int,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
    ) -> FloatArray:
        return np.asarray(
            torus_run_rust(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                int(n),
                float(zeta),
                float(psi),
                float(dt),
                int(n_steps),
            ),
            dtype=np.float64,
        )

    return _rust


def _load_mojo_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.upde._geometric_mojo import (
        _ensure_exe,
        torus_run_mojo,
    )

    _ensure_exe()
    return torus_run_mojo


def _load_julia_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401

    from ..experimental.accelerators.upde._geometric_julia import (
        torus_run_julia,
    )

    return torus_run_julia


def _load_go_fn() -> Callable[..., FloatArray]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.upde._geometric_go import (
        _load_lib,
        torus_run_go,
    )

    _load_lib()
    return torus_run_go


_LOADERS: dict[str, Callable[[], Callable[..., FloatArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, Callable[..., FloatArray]] = {}


def _load_backend(name: str) -> Callable[..., FloatArray]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
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


def _dispatch() -> Callable[..., FloatArray] | None:
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


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be >= 0 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    return coerced


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return coerced


def _validate_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _python_torus_run(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Python reference matching the Rust kernel exactly.

    Carries ``(z_re, z_im)`` state between steps (no atan2
    round-trip per step) and uses the sincos expansion
    ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` on the alpha-zero
    branch.
    """
    om = np.asarray(omegas, dtype=np.float64)
    knm = knm_flat.reshape(n, n)
    alpha = alpha_flat.reshape(n, n)
    alpha_zero = bool(np.all(alpha == 0.0))
    zs_psi = zeta * np.sin(psi) if zeta != 0.0 else 0.0
    zc_psi = zeta * np.cos(psi) if zeta != 0.0 else 0.0

    z_re = np.cos(phases).copy()
    z_im = np.sin(phases).copy()

    for _ in range(n_steps):
        if alpha_zero:
            # sin(θ_j − θ_i) = z_im[j]·z_re[i] − z_re[j]·z_im[i]
            sin_diff = (
                z_im[np.newaxis, :] * z_re[:, np.newaxis]
                - z_re[np.newaxis, :] * z_im[:, np.newaxis]
            )
            coupling = np.sum(knm * sin_diff, axis=1)
        else:
            theta = np.arctan2(z_im, z_re)
            diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
            coupling = np.sum(knm * np.sin(diff), axis=1)
        omega_eff = om + coupling
        if zeta != 0.0:
            omega_eff = omega_eff + zs_psi * z_re - zc_psi * z_im
        angle = omega_eff * dt
        sin_a = np.sin(angle)
        cos_a = np.cos(angle)
        nr = z_re * cos_a - z_im * sin_a
        ni = z_re * sin_a + z_im * cos_a
        norm = np.sqrt(nr * nr + ni * ni)
        nonzero = norm > 0.0
        # Explicit np.where to keep the zero-norm fallback
        # semantics from the Rust kernel.
        z_re = np.where(nonzero, nr / np.where(nonzero, norm, 1.0), nr)
        z_im = np.where(nonzero, ni / np.where(nonzero, norm, 1.0), ni)

    result: FloatArray = np.asarray(np.arctan2(z_im, z_re) % TWO_PI, dtype=np.float64)
    return result


class TorusEngine:
    """Symplectic Euler on ``T^N`` with 5-backend dispatch.

    Store ``(n, dt)``; step / run are stateless in ``(θ, ω, K, α,
    ζ, ψ)``.
    """

    def __init__(self, n_oscillators: int, dt: float):
        self._n = _validate_positive_int(n_oscillators, name="n_oscillators")
        self._dt = _validate_positive_float(dt, name="dt")

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        """One torus step; returns phases in ``[0, 2π)``."""
        return self.run(phases, omegas, knm, zeta, psi, alpha, n_steps=1)

    def run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
        n_steps: int,
    ) -> FloatArray:
        """Integrate torus phase dynamics for the requested number of steps."""
        n_steps = _validate_nonnegative_int(n_steps, name="n_steps")
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        alpha64 = _validate_state_array(
            alpha,
            name="alpha",
            shape=(self._n, self._n),
        )
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")
        knm_flat = knm64.ravel()
        alpha_flat = alpha64.ravel()
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                phases64,
                omegas64,
                knm_flat,
                alpha_flat,
                self._n,
                float(zeta),
                float(psi),
                float(self._dt),
                n_steps,
            )
        return _python_torus_run(
            phases64,
            omegas64,
            knm_flat,
            alpha_flat,
            self._n,
            float(zeta),
            float(psi),
            float(self._dt),
            n_steps,
        )

    def order_parameter(self, phases: FloatArray) -> float:
        """Compute the standard Kuramoto R = |<exp(iθ)>|."""
        phases64 = _validate_state_array(
            phases,
            name="phases",
            shape=(self._n,),
        )
        return float(np.abs(np.mean(np.exp(1j * phases64))))

    def _derivative(
        self,
        theta: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        """Tangent-space Kuramoto derivative ``ω_eff``.

        Kept as a private helper for external inspection tests
        (``test_torus_engine_deep``) that probe the RHS without
        lifting to ``(z_re, z_im)``. The dispatched ``run`` /
        ``step`` paths do not route through this method.
        """
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        result = omegas + coupling
        if zeta != 0.0:
            result = result + zeta * np.sin(psi - theta)
        return cast("FloatArray", result)
