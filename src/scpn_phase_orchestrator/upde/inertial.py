# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Second-order inertial Kuramoto

"""Second-order (swing-equation) Kuramoto with a 5-backend fallback chain.

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
from math import isfinite
from numbers import Integral, Real

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

TWO_PI = 2.0 * np.pi

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "InertialKuramotoEngine",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


_Loader = Callable[..., tuple[FloatArray, FloatArray]]


def _load_rust_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    """Load the Rust second-order Kuramoto backend callable."""
    from spo_kernel import inertial_step_rust

    def _rust(
        theta: FloatArray,
        omega_dot: FloatArray,
        power: FloatArray,
        knm_flat: FloatArray,
        inertia: FloatArray,
        damping: FloatArray,
        n: int,
        dt: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Call the Rust second-order Kuramoto step kernel."""
        new_th, new_od = inertial_step_rust(
            np.ascontiguousarray(theta, dtype=np.float64),
            np.ascontiguousarray(omega_dot, dtype=np.float64),
            np.ascontiguousarray(power, dtype=np.float64),
            np.ascontiguousarray(knm_flat, dtype=np.float64),
            np.ascontiguousarray(inertia, dtype=np.float64),
            np.ascontiguousarray(damping, dtype=np.float64),
            int(n),
            float(dt),
        )
        return (
            np.asarray(new_th, dtype=np.float64),
            np.asarray(new_od, dtype=np.float64),
        )

    return _rust


def _load_mojo_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    """Load the Mojo second-order Kuramoto backend callable."""
    from ..experimental.accelerators.upde._inertial_mojo import (
        _ensure_exe,
        inertial_step_mojo,
    )

    _ensure_exe()
    return inertial_step_mojo


def _load_julia_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    """Load the Julia second-order Kuramoto backend callable."""
    import juliacall  # noqa: F401

    if not hasattr(juliacall, "Main"):
        raise ImportError("juliacall imported but juliacall.Main is unavailable")

    from ..experimental.accelerators.upde._inertial_julia import (
        inertial_step_julia,
    )

    return inertial_step_julia


def _load_go_fn() -> Callable[..., tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    """Load the Go second-order Kuramoto backend callable."""
    from ..experimental.accelerators.upde._inertial_go import (
        _load_lib,
        inertial_step_go,
    )

    _load_lib()
    return inertial_step_go


_LOADERS: dict[str, Callable[[], _Loader]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, _Loader] = {}


def _load_backend(name: str) -> _Loader:
    """Load and cache the named backend callable."""
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    """Resolve the active and available backends, fastest-first."""
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


def _dispatch() -> _Loader | None:
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


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    coerced = float(value)
    if not isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    return coerced


def _validate_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the state as a validated finite array, else raise."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_positive_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the state as a validated strictly positive array, else raise."""
    arr = _validate_state_array(value, name=name, shape=shape)
    if not np.all(arr > 0.0):
        raise ValueError(f"{name} must contain only positive finite values")
    return arr


def _validate_nonnegative_state_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the state as a validated non-negative array, else raise."""
    arr = _validate_state_array(value, name=name, shape=shape)
    if not np.all(arr >= 0.0):
        raise ValueError(f"{name} must contain only non-negative finite values")
    return arr


def _python_step(
    theta: FloatArray,
    omega_dot: FloatArray,
    power: FloatArray,
    knm_flat: FloatArray,
    inertia: FloatArray,
    damping: FloatArray,
    n: int,
    dt: float,
) -> tuple[FloatArray, FloatArray]:
    """Python reference using the ``sin(θ_j − θ_i)`` expansion of the Rust kernel."""
    knm = np.asarray(knm_flat).reshape(n, n)

    def deriv(th: FloatArray, od: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Return the second-order (swing-equation) Kuramoto derivative."""
        s = np.sin(th)
        c = np.cos(th)
        # sin(θ_j − θ_i) = s_j · c_i − c_j · s_i  (row-i, col-j)
        sin_diff = (
            s[np.newaxis, :] * c[:, np.newaxis] - c[np.newaxis, :] * s[:, np.newaxis]
        )
        coupling = np.sum(knm * sin_diff, axis=1)
        accel = (power + coupling - damping * od) / inertia
        return od, accel

    k1t, k1o = deriv(theta, omega_dot)
    k2t, k2o = deriv(theta + 0.5 * dt * k1t, omega_dot + 0.5 * dt * k1o)
    k3t, k3o = deriv(theta + 0.5 * dt * k2t, omega_dot + 0.5 * dt * k2o)
    k4t, k4o = deriv(theta + dt * k3t, omega_dot + dt * k3o)
    new_theta = (theta + (dt / 6.0) * (k1t + 2 * k2t + 2 * k3t + k4t)) % TWO_PI
    new_omega = omega_dot + (dt / 6.0) * (k1o + 2 * k2o + 2 * k3o + k4o)
    return new_theta, new_omega


class InertialKuramotoEngine:
    """Second-order swing-equation Kuramoto stepper with 5-backend dispatch.

    The engine's geometry is ``(n, dt)``; the step itself is
    stateless: ``(θ, ω, P, K, M, D) → (θ', ω')``.
    """

    def __init__(self, n: int, dt: float = 0.01) -> None:
        self._n = _validate_positive_int(n, name="n")
        self._dt = _validate_positive_float(dt, name="dt")

    def step(
        self,
        theta: FloatArray,
        omega_dot: FloatArray,
        power: FloatArray,
        knm: FloatArray,
        inertia: FloatArray,
        damping: FloatArray,
    ) -> tuple[FloatArray, FloatArray]:
        """Advance one second-order inertial Kuramoto timestep.

        Parameters
        ----------
        theta : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omega_dot : FloatArray
            Instantaneous frequency deviations in rad/s, shape ``(N,)``.
        power : FloatArray
            Per-oscillator power injection in the swing equation, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        inertia : FloatArray
            Per-oscillator inertia coefficients, shape ``(N,)``.
        damping : FloatArray
            Per-oscillator damping coefficients, shape ``(N,)``.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            The ``(θ, ω̇)`` state after one second-order step.
        """
        theta64 = _validate_state_array(theta, name="theta", shape=(self._n,))
        omega_dot64 = _validate_state_array(
            omega_dot,
            name="omega_dot",
            shape=(self._n,),
        )
        power64 = _validate_state_array(power, name="power", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        inertia64 = _validate_positive_state_array(
            inertia,
            name="inertia",
            shape=(self._n,),
        )
        damping64 = _validate_nonnegative_state_array(
            damping,
            name="damping",
            shape=(self._n,),
        )
        knm_flat = knm64.ravel()
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                theta64,
                omega_dot64,
                power64,
                knm_flat,
                inertia64,
                damping64,
                self._n,
                self._dt,
            )
        return _python_step(
            theta64,
            omega_dot64,
            power64,
            knm_flat,
            inertia64,
            damping64,
            self._n,
            self._dt,
        )

    def run(
        self,
        theta: FloatArray,
        omega_dot: FloatArray,
        power: FloatArray,
        knm: FloatArray,
        inertia: FloatArray,
        damping: FloatArray,
        n_steps: int,
    ) -> tuple[
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
    ]:
        """Integrate inertial Kuramoto dynamics and return final state plus traces.

        Parameters
        ----------
        theta : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omega_dot : FloatArray
            Instantaneous frequency deviations in rad/s, shape ``(N,)``.
        power : FloatArray
            Per-oscillator power injection in the swing equation, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        inertia : FloatArray
            Per-oscillator inertia coefficients, shape ``(N,)``.
        damping : FloatArray
            Per-oscillator damping coefficients, shape ``(N,)``.
        n_steps : int
            Number of integration steps to run.

        Returns
        -------
        tuple[FloatArray, FloatArray, FloatArray, FloatArray]
            The final ``(θ, ω̇)`` plus the ``θ`` and ``ω̇`` traces.
        """
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        theta_traj = np.empty((n_steps, self._n))
        omega_traj = np.empty((n_steps, self._n))
        th = _validate_state_array(theta, name="theta", shape=(self._n,)).copy()
        od = _validate_state_array(
            omega_dot,
            name="omega_dot",
            shape=(self._n,),
        ).copy()
        for i in range(n_steps):
            th, od = self.step(th, od, power, knm, inertia, damping)
            theta_traj[i] = th
            omega_traj[i] = od
        return th, od, theta_traj, omega_traj

    def frequency_deviation(self, omega_dot: FloatArray) -> float:
        """Return maximum absolute frequency deviation in cycles per unit time.

        Parameters
        ----------
        omega_dot : FloatArray
            Instantaneous frequency deviations in rad/s, shape ``(N,)``.

        Returns
        -------
        float
            The maximum absolute frequency deviation in cycles per unit time.
        """
        return float(np.max(np.abs(omega_dot)) / TWO_PI)

    def coherence(self, theta: FloatArray) -> float:
        """Return the Kuramoto order parameter for the supplied phases.

        Parameters
        ----------
        theta : FloatArray
            Oscillator phases in radians, shape ``(N,)``.

        Returns
        -------
        float
            The Kuramoto order parameter ``R``.
        """
        return float(np.abs(np.mean(np.exp(1j * theta))))
