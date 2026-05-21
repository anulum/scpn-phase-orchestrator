# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Lyapunov stability monitor

"""Lyapunov stability monitor with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Two public surfaces:

* :class:`LyapunovGuard` — stateful observer that tracks the Lyapunov
  function ``V(θ) = -(K/2N) Σ_ij A_ij cos(θ_i − θ_j)``, its numerical
  time derivative, and basin-of-attraction membership (van Hemmen &
  Wreszinski 1993). Single-backend NumPy; inexpensive per call.
* :func:`lyapunov_spectrum` — full Lyapunov spectrum via periodic QR
  reorthogonalisation (Benettin 1980 / Shimada-Nagashima 1979). Multi-
  backend; the heavy kernel is dispatched to Rust → Mojo → Julia → Go
  → Python in order of availability.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "LyapunovGuard",
    "LyapunovState",
    "lyapunov_spectrum",
]

FloatArray: TypeAlias = NDArray[np.float64]
LyapunovBackendFn: TypeAlias = Callable[
    [FloatArray, FloatArray, FloatArray, FloatArray, float, int, int, float, float],
    FloatArray,
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> LyapunovBackendFn:
    from spo_kernel import lyapunov_spectrum_rust

    return cast("LyapunovBackendFn", lyapunov_spectrum_rust)


def _load_mojo_fn() -> LyapunovBackendFn:
    from ..experimental.accelerators.monitor._lyapunov_mojo import (
        _ensure_exe,
        lyapunov_spectrum_mojo,
    )

    _ensure_exe()
    return lyapunov_spectrum_mojo


def _load_julia_fn() -> LyapunovBackendFn:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._lyapunov_julia import (
        lyapunov_spectrum_julia,
    )

    return lyapunov_spectrum_julia


def _load_go_fn() -> LyapunovBackendFn:
    from ..experimental.accelerators.monitor._lyapunov_go import (
        _load_lib,
        lyapunov_spectrum_go,
    )

    _load_lib()
    return lyapunov_spectrum_go


_LOADERS: dict[str, Callable[[], LyapunovBackendFn]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, LyapunovBackendFn] = {}


def _load_backend(name: str) -> LyapunovBackendFn:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> LyapunovBackendFn | None:
    if ACTIVE_BACKEND == "python":
        return None
    try:
        return _load_backend(ACTIVE_BACKEND)
    except (ImportError, RuntimeError, OSError, KeyError):
        return None


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _validate_positive_real(value: object, *, name: str) -> float:
    result = _validate_finite_real(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive, got {result}")
    return result


def _validate_non_negative_real(value: object, *, name: str) -> float:
    result = _validate_finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be non-negative, got {result}")
    return result


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite one-dimensional array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_matrix(
    value: object,
    *,
    name: str,
    expected_shape: tuple[int, int],
) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite matrix") from exc
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


@dataclass
class LyapunovState:
    """Lyapunov function V, dV/dt, basin membership, and max phase diff."""

    V: float
    dV_dt: float
    in_basin: bool
    max_phase_diff: float


class LyapunovGuard:
    """Lyapunov stability monitor for Kuramoto networks.

    V(θ) = -(K/2N) Σ_{i,j} A_ij cos(θ_i - θ_j)

    dV/dt ≤ 0 for gradient flow (Kuramoto is gradient on V).
    Basin of attraction: max|θ_i - θ_j| < π/2 for connected pairs.

    van Hemmen & Wreszinski 1993, J. Stat. Phys. 72:145-166.
    """

    def __init__(self, basin_threshold: object = np.pi / 2):
        basin_threshold = _validate_positive_real(
            basin_threshold,
            name="basin_threshold",
        )
        self._basin_threshold = basin_threshold
        self._prev_V: float | None = None

    def evaluate(self, phases: object, knm: object) -> LyapunovState:
        """Compute Lyapunov function, its time derivative, and basin check."""
        phases = _validate_vector(phases, name="phases")
        n = len(phases)
        knm = _validate_matrix(knm, name="knm", expected_shape=(n, n))
        if n == 0:
            return LyapunovState(V=0.0, dV_dt=0.0, in_basin=True, max_phase_diff=0.0)

        diff = phases[:, np.newaxis] - phases[np.newaxis, :]
        cos_diff = np.cos(diff)

        # Lyapunov fn for Kuramoto gradient system
        # V(θ) = -(1/2N) Σ K_ij cos(θ_i - θ_j)
        # van Hemmen & Wreszinski 1993, Eq. 2.3
        V = -0.5 * float(np.sum(knm * cos_diff)) / n

        # Numerical dV/dt from consecutive calls
        dV_dt = 0.0
        if self._prev_V is not None:
            dV_dt = V - self._prev_V
        self._prev_V = V

        # Basin of attraction: all connected pairs within π/2 of each other
        # (sufficient condition for gradient convergence)
        connected = knm > 0
        if np.any(connected):
            abs_diff = np.abs(diff)
            # Geodesic distance on S¹: min(|Δ|, 2π-|Δ|)
            abs_diff = np.minimum(abs_diff, 2 * np.pi - abs_diff)
            max_diff = float(np.max(abs_diff[connected]))
        else:
            max_diff = 0.0

        in_basin = max_diff < self._basin_threshold

        return LyapunovState(
            V=V,
            dV_dt=dV_dt,
            in_basin=in_basin,
            max_phase_diff=max_diff,
        )

    def reset(self) -> None:
        """Clear cached previous V, so next evaluate() reports dV/dt = 0."""
        self._prev_V = None


def _kuramoto_rhs(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
) -> FloatArray:
    """Kuramoto RHS with Sakaguchi phase lag and optional external driver."""
    diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
    coupling = np.sum(knm * np.sin(diff), axis=1)
    driving = zeta * np.sin(psi - phases) if zeta != 0.0 else 0.0
    return np.asarray(omegas + coupling + driving, dtype=np.float64)


def _kuramoto_jacobian(
    phases: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
) -> FloatArray:
    """Jacobian of the Kuramoto RHS.

    J_ij = K_ij cos(θ_j − θ_i − α_ij)                    for i ≠ j
    J_ii = −Σ_{j≠i} K_ij cos(θ_j − θ_i − α_ij)
           − ζ cos(Ψ − θ_i)                              (driver diagonal)
    """
    diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
    J: FloatArray = knm * np.cos(diff)
    np.fill_diagonal(J, 0.0)
    diag = -J.sum(axis=1)
    if zeta != 0.0:
        diag = diag - zeta * np.cos(psi - phases)
    np.fill_diagonal(J, diag)
    return J


def _rk4_step(
    phases: FloatArray,
    Q: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    dt: float,
    zeta: float,
    psi: float,
) -> tuple[FloatArray, FloatArray]:
    """One classical RK4 step on the joint (phases, Q) state."""

    def rhs(p: FloatArray, q: FloatArray) -> tuple[FloatArray, FloatArray]:
        return (
            _kuramoto_rhs(p, omegas, knm, alpha, zeta, psi),
            _kuramoto_jacobian(p, knm, alpha, zeta, psi) @ q,
        )

    k1p, k1q = rhs(phases, Q)
    k2p, k2q = rhs(phases + 0.5 * dt * k1p, Q + 0.5 * dt * k1q)
    k3p, k3q = rhs(phases + 0.5 * dt * k2p, Q + 0.5 * dt * k2q)
    k4p, k4q = rhs(phases + dt * k3p, Q + dt * k3q)
    new_phases = (phases + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)) % (
        2.0 * np.pi
    )
    new_Q = Q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    return new_phases, new_Q


def _row_qr_log_diag(Q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Row-oriented QR — returns the reorthonormalised Q and ``log|R_ii|``
    floored at ``log(1e-300)`` to avoid ``−inf``."""
    Q_t, R = np.linalg.qr(Q.T)
    diag = np.maximum(np.abs(np.diag(R)), 1e-300)
    return Q_t.T, np.log(diag)


def _lyapunov_spectrum_python(
    phases_init: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> FloatArray:
    """Benettin 1980 Lyapunov spectrum via RK4 + row-oriented MGS.

    Matches the Rust kernel line-for-line: RK4 on the joint
    ``(phases, Q)`` state, phases wrapped to ``[0, 2π)`` after each
    step, and the driver ``−ζ cos(Ψ − θ_i)`` enters the Jacobian
    diagonal. Periodic QR (Rust convention: orthonormalise rows of Q).
    """
    n = len(phases_init)
    phases = phases_init.copy()
    Q = np.eye(n, dtype=np.float64)
    exponents = np.zeros(n, dtype=np.float64)
    n_qr = 0
    total_time = 0.0
    for step in range(n_steps):
        phases, Q = _rk4_step(phases, Q, omegas, knm, alpha, dt, zeta, psi)
        total_time += dt
        if (step + 1) % qr_interval == 0:
            Q, log_diag = _row_qr_log_diag(Q)
            exponents += log_diag
            n_qr += 1
    if n_qr > 0:
        exponents /= total_time
    return np.sort(exponents)[::-1]


def lyapunov_spectrum(
    phases_init: object,
    omegas: object,
    knm: object,
    alpha: object,
    dt: object = 0.01,
    n_steps: object = 1000,
    qr_interval: object = 10,
    zeta: object = 0.0,
    psi: object = 0.0,
) -> FloatArray:
    """Full Lyapunov spectrum (all N exponents) via QR decomposition.

    Evolves N perturbation vectors alongside the Kuramoto ODE. Every
    ``qr_interval`` steps, QR-reorthogonalises and accumulates growth
    rates from the diagonal of R.

    Benettin et al. 1980, Meccanica 15:9-20.
    Shimada & Nagashima 1979, Prog. Theor. Phys. 61:1605-1616.

    Dispatches to the first available backend per the SPO fallback
    chain (Rust → Mojo → Julia → Go → Python). All five produce the
    same exponents up to floating-point rounding; the dispatcher's
    choice only affects wall-clock cost.

    Args:
        phases_init: (N,) initial phases.
        omegas: (N,) natural frequencies.
        knm: (N, N) coupling matrix.
        alpha: (N, N) phase-lag matrix.
        dt: integration timestep.
        n_steps: total integration steps.
        qr_interval: steps between QR reorthogonalisations.
        zeta: driver strength.
        psi: target driver phase.

    Returns:
        (N,) array of Lyapunov exponents, sorted descending.
    """
    p = _validate_vector(phases_init, name="phases_init")
    n = int(p.size)
    o = _validate_vector(omegas, name="omegas")
    if o.shape != p.shape:
        raise ValueError(f"omegas shape {o.shape} does not match {p.shape}")
    k = _validate_matrix(knm, name="knm", expected_shape=(n, n))
    a = _validate_matrix(alpha, name="alpha", expected_shape=(n, n))
    dt = _validate_positive_real(dt, name="dt")
    n_steps = _validate_int_at_least(n_steps, name="n_steps", minimum=0)
    qr_interval = _validate_int_at_least(
        qr_interval,
        name="qr_interval",
        minimum=1,
    )
    zeta = _validate_non_negative_real(zeta, name="zeta")
    psi = _validate_finite_real(psi, name="psi")
    backend_fn = _dispatch()
    if backend_fn is None:
        return _lyapunov_spectrum_python(
            p,
            o,
            k,
            a,
            float(dt),
            int(n_steps),
            int(qr_interval),
            float(zeta),
            float(psi),
        )
    # Rust PyO3 binding takes flat (N*N,) row-major k/alpha; the other
    # backends accept the 2-D forms directly.
    if ACTIVE_BACKEND == "rust":
        try:
            return np.asarray(
                backend_fn(
                    p,
                    o,
                    k.ravel(),
                    a.ravel(),
                    dt,
                    n_steps,
                    qr_interval,
                    zeta,
                    psi,
                ),
                dtype=np.float64,
            )
        except Exception:
            return _lyapunov_spectrum_python(
                p,
                o,
                k,
                a,
                float(dt),
                int(n_steps),
                int(qr_interval),
                float(zeta),
                float(psi),
            )
    try:
        return np.asarray(
            backend_fn(
                p,
                o,
                k,
                a,
                float(dt),
                int(n_steps),
                int(qr_interval),
                float(zeta),
                float(psi),
            ),
            dtype=np.float64,
        )
    except Exception:
        return _lyapunov_spectrum_python(
            p,
            o,
            k,
            a,
            float(dt),
            int(n_steps),
            int(qr_interval),
            float(zeta),
            float(psi),
        )
