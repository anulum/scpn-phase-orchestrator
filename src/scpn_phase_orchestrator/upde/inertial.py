# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Second-order inertial Kuramoto

"""Second-order (swing-equation) Kuramoto with a 5-backend fallback
chain per ``feedback_module_standard_attnres.md``.

Model
-----
Each oscillator has a phase ``θ_i`` and a "frequency-deviation"
``ω_i ≡ dθ_i/dt``. The swing equation is

    M_i · d²θ_i/dt² + D_i · dθ_i/dt = P_i + Σ_j K_ij · sin(θ_j − θ_i)

and is advanced with classical explicit RK4 on the ``(θ, ω)`` pair.
This is the power-grid form used in Filatrella-Nielsen-Mallick 2008.

Numerics
--------
The derivative uses the ``sin(θ_j − θ_i) = sin(θ_j)·cos(θ_i) −
cos(θ_j)·sin(θ_i)`` expansion so that floating-point rounding
matches the Rust kernel (``spo-engine/src/inertial.rs``) bit-for-bit.
All five backends (Rust, Mojo, Julia, Go, Python) agree within
``~1e-14`` on the canonical all-to-all test problem; the
dispatcher selects the fastest available path.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

TWO_PI = 2.0 * np.pi

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "InertialKuramotoEngine",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., tuple[NDArray, NDArray]]:
    from spo_kernel import inertial_step_rust

    def _rust(
        theta: NDArray, omega_dot: NDArray, power: NDArray,
        knm_flat: NDArray, inertia: NDArray, damping: NDArray,
        n: int, dt: float,
    ) -> tuple[NDArray, NDArray]:
        new_th, new_od = inertial_step_rust(
            np.ascontiguousarray(theta, dtype=np.float64),
            np.ascontiguousarray(omega_dot, dtype=np.float64),
            np.ascontiguousarray(power, dtype=np.float64),
            np.ascontiguousarray(knm_flat, dtype=np.float64),
            np.ascontiguousarray(inertia, dtype=np.float64),
            np.ascontiguousarray(damping, dtype=np.float64),
            int(n), float(dt),
        )
        return (
            np.asarray(new_th, dtype=np.float64),
            np.asarray(new_od, dtype=np.float64),
        )

    return _rust


def _load_mojo_fn() -> Callable[..., tuple[NDArray, NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._inertial_mojo import (
        _ensure_exe,
        inertial_step_mojo,
    )

    _ensure_exe()
    return inertial_step_mojo


def _load_julia_fn() -> Callable[..., tuple[NDArray, NDArray]]:
    # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.upde._inertial_julia import (
        inertial_step_julia,
    )

    return inertial_step_julia


def _load_go_fn() -> Callable[..., tuple[NDArray, NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._inertial_go import inertial_step_go

    return inertial_step_go


_LOADERS: dict[str, Callable[[], Callable[..., tuple[NDArray, NDArray]]]] = {
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


def _dispatch() -> Callable[..., tuple[NDArray, NDArray]] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


def _python_step(
    theta: NDArray, omega_dot: NDArray, power: NDArray,
    knm_flat: NDArray, inertia: NDArray, damping: NDArray,
    n: int, dt: float,
) -> tuple[NDArray, NDArray]:
    """Python reference using the same ``sin(θ_j − θ_i) =
    s_j·c_i − c_j·s_i`` expansion as the Rust kernel."""
    knm = np.asarray(knm_flat).reshape(n, n)

    def deriv(th: NDArray, od: NDArray) -> tuple[NDArray, NDArray]:
        s = np.sin(th)
        c = np.cos(th)
        # sin(θ_j − θ_i) = s_j · c_i − c_j · s_i  (row-i, col-j)
        sin_diff = s[np.newaxis, :] * c[:, np.newaxis] \
            - c[np.newaxis, :] * s[:, np.newaxis]
        coupling = np.sum(knm * sin_diff, axis=1)
        accel = (power + coupling - damping * od) / inertia
        return od, accel

    k1t, k1o = deriv(theta, omega_dot)
    k2t, k2o = deriv(theta + 0.5 * dt * k1t, omega_dot + 0.5 * dt * k1o)
    k3t, k3o = deriv(theta + 0.5 * dt * k2t, omega_dot + 0.5 * dt * k2o)
    k4t, k4o = deriv(theta + dt * k3t, omega_dot + dt * k3o)
    new_theta = (
        theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)
    ) % TWO_PI
    new_omega = omega_dot + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
    return new_theta, new_omega


class InertialKuramotoEngine:
    """Second-order swing-equation Kuramoto stepper with 5-backend
    dispatch.

    The engine's geometry is ``(n, dt)``; the step itself is
    stateless: ``(θ, ω, P, K, M, D) → (θ', ω')``.
    """

    def __init__(self, n: int, dt: float = 0.01) -> None:
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        self._n = n
        self._dt = dt

    def step(
        self,
        theta: NDArray,
        omega_dot: NDArray,
        power: NDArray,
        knm: NDArray,
        inertia: NDArray,
        damping: NDArray,
    ) -> tuple[NDArray, NDArray]:
        knm_flat = np.ascontiguousarray(knm, dtype=np.float64).ravel()
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                theta, omega_dot, power,
                knm_flat, inertia, damping,
                self._n, self._dt,
            )
        return _python_step(
            np.asarray(theta, dtype=np.float64),
            np.asarray(omega_dot, dtype=np.float64),
            np.asarray(power, dtype=np.float64),
            knm_flat,
            np.asarray(inertia, dtype=np.float64),
            np.asarray(damping, dtype=np.float64),
            self._n, self._dt,
        )

    def run(
        self,
        theta: NDArray,
        omega_dot: NDArray,
        power: NDArray,
        knm: NDArray,
        inertia: NDArray,
        damping: NDArray,
        n_steps: int,
    ) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        theta_traj = np.empty((n_steps, self._n))
        omega_traj = np.empty((n_steps, self._n))
        th, od = theta.copy(), omega_dot.copy()
        for i in range(n_steps):
            th, od = self.step(th, od, power, knm, inertia, damping)
            theta_traj[i] = th
            omega_traj[i] = od
        return th, od, theta_traj, omega_traj

    def frequency_deviation(self, omega_dot: NDArray) -> float:
        return float(np.max(np.abs(omega_dot)) / TWO_PI)

    def coherence(self, theta: NDArray) -> float:
        return float(np.abs(np.mean(np.exp(1j * theta))))
