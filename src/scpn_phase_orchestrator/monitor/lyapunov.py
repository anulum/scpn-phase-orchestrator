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
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "LyapunovGuard",
    "LyapunovState",
    "lyapunov_spectrum",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import lyapunov_spectrum_rust

    return cast("Callable[..., NDArray]", lyapunov_spectrum_rust)


def _load_mojo_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._lyapunov_mojo import (
        _ensure_exe,
        lyapunov_spectrum_mojo,
    )

    _ensure_exe()
    return lyapunov_spectrum_mojo


def _load_julia_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._lyapunov_julia import (
        lyapunov_spectrum_julia,
    )

    return lyapunov_spectrum_julia


def _load_go_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._lyapunov_go import (
        lyapunov_spectrum_go,
    )

    return lyapunov_spectrum_go


_LOADERS: dict[str, Callable[[], Callable[..., NDArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., NDArray] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


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

    def __init__(self, basin_threshold: float = np.pi / 2):
        if basin_threshold <= 0.0:
            raise ValueError(f"basin_threshold must be positive, got {basin_threshold}")
        self._basin_threshold = basin_threshold
        self._prev_V: float | None = None

    def evaluate(self, phases: NDArray, knm: NDArray) -> LyapunovState:
        """Compute Lyapunov function, its time derivative, and basin check."""
        n = len(phases)
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
    phases: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
) -> NDArray:
    """Kuramoto RHS with Sakaguchi phase lag and optional external driver."""
    diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
    coupling = np.sum(knm * np.sin(diff), axis=1)
    driving = zeta * np.sin(psi - phases) if zeta != 0.0 else 0.0
    return np.asarray(omegas + coupling + driving, dtype=np.float64)


def _kuramoto_jacobian(
    phases: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
) -> NDArray:
    """Jacobian of the Kuramoto RHS.

    J_ij = K_ij cos(θ_j − θ_i − α_ij)                    for i ≠ j
    J_ii = −Σ_{j≠i} K_ij cos(θ_j − θ_i − α_ij)
           − ζ cos(Ψ − θ_i)                              (driver diagonal)
    """
    diff = phases[np.newaxis, :] - phases[:, np.newaxis] - alpha
    J: NDArray = knm * np.cos(diff)
    np.fill_diagonal(J, 0.0)
    diag = -J.sum(axis=1)
    if zeta != 0.0:
        diag = diag - zeta * np.cos(psi - phases)
    np.fill_diagonal(J, diag)
    return J


def _rk4_step(
    phases: NDArray,
    Q: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float,
    zeta: float,
    psi: float,
) -> tuple[NDArray, NDArray]:
    """One classical RK4 step on the joint (phases, Q) state."""

    def rhs(p: NDArray, q: NDArray) -> tuple[NDArray, NDArray]:
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


def _row_qr_log_diag(Q: NDArray) -> tuple[NDArray, NDArray]:
    """Row-oriented QR — returns the reorthonormalised Q and ``log|R_ii|``
    floored at ``log(1e-300)`` to avoid ``−inf``."""
    Q_t, R = np.linalg.qr(Q.T)
    diag = np.maximum(np.abs(np.diag(R)), 1e-300)
    return Q_t.T, np.log(diag)


def _lyapunov_spectrum_python(
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float,
    n_steps: int,
    qr_interval: int,
    zeta: float,
    psi: float,
) -> NDArray:
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
    phases_init: NDArray,
    omegas: NDArray,
    knm: NDArray,
    alpha: NDArray,
    dt: float = 0.01,
    n_steps: int = 1000,
    qr_interval: int = 10,
    zeta: float = 0.0,
    psi: float = 0.0,
) -> NDArray:
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
    p = np.ascontiguousarray(phases_init, dtype=np.float64)
    o = np.ascontiguousarray(omegas, dtype=np.float64)
    k = np.ascontiguousarray(knm, dtype=np.float64)
    a = np.ascontiguousarray(alpha, dtype=np.float64)
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
