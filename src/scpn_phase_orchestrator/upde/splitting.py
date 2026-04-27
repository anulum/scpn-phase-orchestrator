# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strang operator splitting integrator

"""Strang second-order operator splitting for the Kuramoto ODE
with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Scheme
------
Split ``dθ/dt = ω + Σ_j K_ij · sin(θ_j − θ_i − α_ij) +
ζ · sin(ψ − θ_i)`` into

    A: dθ/dt = ω              (exact rotation)
    B: dθ/dt = coupling       (RK4 on the nonlinear part)

and compose symmetrically as ``A(dt/2) → B(dt) → A(dt/2)``
(Strang scheme, second-order in dt).

Why split?
----------
The ``ω`` flow is linear, so it has no truncation error; folding
it into a monolithic RK45 burns integrator budget on a solvable
direction while damping unrelated accuracy in the nonlinear
direction. Reference: Hairer, Lubich & Wanner 2006, *Geometric
Numerical Integration* §II.5.

Numerics
--------
The B-stage RK4 uses the Rust kernel's
``sin(θ_j − θ_i) = sin(θ_j)·cos(θ_i) − cos(θ_j)·sin(θ_i)``
expansion on the alpha-zero branch so that floating-point
rounding matches Rust (``spo-engine/src/splitting.rs``)
bit-for-bit. Nonzero alpha falls back to the direct
``sin(diff)`` form in all five backends.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "SplittingEngine",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import splitting_run_rust

    def _rust(
        phases: NDArray,
        omegas: NDArray,
        knm_flat: NDArray,
        alpha_flat: NDArray,
        n: int,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
    ) -> NDArray:
        # The Rust FFI reads N from ``phases.len()`` and ignores the
        # positional ``_n`` argument (hence the leading underscore in
        # its signature). We still pass ``n`` for future-proofing.
        return np.asarray(
            splitting_run_rust(
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


def _load_mojo_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._splitting_mojo import (
        _ensure_exe,
        splitting_run_mojo,
    )

    _ensure_exe()
    return splitting_run_mojo


def _load_julia_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._splitting_julia import (
        splitting_run_julia,
    )

    return splitting_run_julia


def _load_go_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._splitting_go import (
        splitting_run_go,
    )

    return splitting_run_go


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


def _coupling_deriv(
    theta: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    alpha_zero: bool,
) -> NDArray:
    s = np.sin(theta)
    c = np.cos(theta)
    if alpha_zero:
        # sin(θ_j − θ_i) = s_j·c_i − c_j·s_i
        sin_diff = (
            s[np.newaxis, :] * c[:, np.newaxis] - c[np.newaxis, :] * s[:, np.newaxis]
        )
        out = np.sum(knm * sin_diff, axis=1)
    else:
        diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
        out = np.sum(knm * np.sin(diff), axis=1)
    if zeta != 0.0:
        # ζ · sin(ψ − θ) = ζ·sin(ψ)·cos(θ) − ζ·cos(ψ)·sin(θ)
        out = out + zeta * np.sin(psi) * c - zeta * np.cos(psi) * s
    return cast("NDArray", out)


def _rk4_coupling(
    p: NDArray,
    knm: NDArray,
    alpha: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    alpha_zero: bool,
) -> NDArray:
    k1 = _coupling_deriv(p, knm, alpha, zeta, psi, alpha_zero)
    k2 = _coupling_deriv(
        (p + 0.5 * dt * k1) % TWO_PI,
        knm,
        alpha,
        zeta,
        psi,
        alpha_zero,
    )
    k3 = _coupling_deriv(
        (p + 0.5 * dt * k2) % TWO_PI,
        knm,
        alpha,
        zeta,
        psi,
        alpha_zero,
    )
    k4 = _coupling_deriv(
        (p + dt * k3) % TWO_PI,
        knm,
        alpha,
        zeta,
        psi,
        alpha_zero,
    )
    return cast("NDArray", (p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)) % TWO_PI)


def _python_run(
    phases: NDArray,
    omegas: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    """Python reference aligned to the Rust kernel exactly.

    Uses the sincos expansion on the alpha-zero branch and the
    same ``ζ·sin(ψ−θ) = ζ·sin(ψ)·cos(θ) − ζ·cos(ψ)·sin(θ)``
    expansion as the Rust kernel, giving bit-exact parity.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    om = np.asarray(omegas, dtype=np.float64)
    knm = knm_flat.reshape(n, n)
    alpha = alpha_flat.reshape(n, n)
    alpha_zero = bool(np.all(alpha == 0.0))
    half_dt = 0.5 * dt
    for _ in range(n_steps):
        p = (p + half_dt * om) % TWO_PI
        p = _rk4_coupling(p, knm, alpha, zeta, psi, dt, alpha_zero)
        p = (p + half_dt * om) % TWO_PI
    return p


class SplittingEngine:
    """Strang-split Kuramoto stepper with 5-backend dispatch.

    The engine's geometry is ``(n, dt)``; the step is stateless.
    """

    def __init__(self, n_oscillators: int, dt: float):
        if n_oscillators < 1:
            raise ValueError(f"n_oscillators must be >= 1, got {n_oscillators}")
        if dt == 0.0:
            raise ValueError(f"dt must be non-zero, got {dt}")
        # Negative dt is intentional for symplectic-reversibility
        # checks — Strang is time-reversible, so stepping forward
        # then with dt → −dt returns to the starting state.
        self._n = n_oscillators
        self._dt = dt

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
    ) -> NDArray:
        """One Strang-split step: A(dt/2) → B(dt) → A(dt/2)."""
        return self.run(phases, omegas, knm, zeta, psi, alpha, n_steps=1)

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray,
        zeta: float,
        psi: float,
        alpha: NDArray,
        n_steps: int,
    ) -> NDArray:
        knm_flat = np.ascontiguousarray(knm, dtype=np.float64).ravel()
        alpha_flat = np.ascontiguousarray(alpha, dtype=np.float64).ravel()
        backend_fn = _dispatch() if self._dt > 0.0 else None
        # Non-Rust backends also validate dt > 0 internally, so
        # negative dt (symplectic-reversibility checks) falls back
        # to the Python reference.
        if backend_fn is not None:
            return backend_fn(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                knm_flat,
                alpha_flat,
                self._n,
                float(zeta),
                float(psi),
                float(self._dt),
                int(n_steps),
            )
        return _python_run(
            np.ascontiguousarray(phases, dtype=np.float64),
            np.ascontiguousarray(omegas, dtype=np.float64),
            knm_flat,
            alpha_flat,
            self._n,
            float(zeta),
            float(psi),
            float(self._dt),
            int(n_steps),
        )

    def order_parameter(self, phases: NDArray) -> float:
        """Standard Kuramoto R = |<exp(iθ)>|."""
        return float(np.abs(np.mean(np.exp(1j * phases))))
