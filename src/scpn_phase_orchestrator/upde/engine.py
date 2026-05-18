# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — UPDE integration engine (stateful class)

"""Stateful :class:`UPDEEngine`.

The batched integrator is stateless — see
:mod:`scpn_phase_orchestrator.upde._run` and the re-exported
:func:`upde_run`. This module keeps the state-heavy observer: the
class pre-allocates scratch buffers for the chosen method, holds a
reentrant lock for thread-safety, and retains ``_last_dt`` across
RK45 step calls.
"""

from __future__ import annotations

import threading
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde import _run as _run_mod

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "UPDEEngine",
    "upde_run",
]

FloatArray: TypeAlias = NDArray[np.float64]

ACTIVE_BACKEND = _run_mod.ACTIVE_BACKEND
AVAILABLE_BACKENDS = _run_mod.AVAILABLE_BACKENDS


def upde_run(
    phases: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
    method: str = "euler",
    n_substeps: int = 1,
    atol: float = 1e-6,
    rtol: float = 1e-3,
) -> FloatArray:
    """Run the stateless UPDE integrator through the engine facade.

    ``engine.ACTIVE_BACKEND`` is a public test and diagnostics control, so
    keep it synchronised with the underlying dispatcher before each call.
    """
    previous = _run_mod.ACTIVE_BACKEND
    _run_mod.ACTIVE_BACKEND = ACTIVE_BACKEND
    try:
        return _run_mod.upde_run(
            phases,
            omegas,
            knm,
            alpha,
            zeta,
            psi,
            dt,
            n_steps,
            method,
            n_substeps,
            atol,
            rtol,
        )
    finally:
        _run_mod.ACTIVE_BACKEND = previous


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


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    return coerced


def _validate_real_array(value: object, *, name: str) -> FloatArray:
    array = value if isinstance(value, np.ndarray) else np.asarray(value)
    if array.dtype == np.bool_ or not (
        np.issubdtype(array.dtype, np.integer)
        or np.issubdtype(array.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a real ndarray, got {array.dtype}")
    return array


class UPDEEngine:
    """Kuramoto UPDE integrator with pre-allocated scratch arrays.

    dtheta_i/dt = omega_i
                  + sum_j K_ij sin(theta_j - theta_i - alpha_ij)
                  + zeta sin(Psi - theta_i)
    """

    # Dormand-Prince (1980) Butcher tableau — 7-stage, FSAL property
    # Coefficients from Table 5.2 of Hairer, Norsett & Wanner, vol. I
    _DP_A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ]
    )
    # B4: 4th-order weights for error estimation (embedded pair)
    _DP_B4 = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    # B5: 5th-order weights — the accepted solution
    _DP_B5 = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0])
    # C: stage time fractions c_i, row sums of A
    _DP_C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        n_oscillators = _validate_positive_int(
            n_oscillators,
            name="n_oscillators",
        )
        dt = _validate_positive_float(dt, name="dt")
        atol = _validate_positive_float(atol, name="atol")
        rtol = _validate_positive_float(rtol, name="rtol")
        if method not in ("euler", "rk4", "rk45"):
            msg = f"Unknown method {method!r}, expected 'euler', 'rk4', or 'rk45'"
            raise ValueError(msg)
        self._n = n_oscillators
        self._dt = dt
        self._method = method
        self._atol = atol
        self._rtol = rtol
        self._last_dt = dt

        self._rust = None
        if _HAS_RUST:  # pragma: no cover
            from spo_kernel import PyUPDEStepper

            self._rust = PyUPDEStepper(n_oscillators, dt, method, atol=atol, rtol=rtol)

        self._phase_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._sin_diff = np.empty((n_oscillators, n_oscillators), dtype=np.float64)
        self._scratch_dtheta = np.empty(n_oscillators, dtype=np.float64)

        if method == "rk45":
            self._ks = [np.empty(n_oscillators, dtype=np.float64) for _ in range(7)]
            self._err_buf = np.empty(n_oscillators, dtype=np.float64)

        # Serialise concurrent step()/run() callers on this instance — the
        # pre-allocated scratch arrays above are not safe to share. Reentrant
        # so that run() → step() within the same thread does not deadlock.
        self._lock = threading.RLock()

    @property
    def last_dt(self) -> float:
        """Actual dt used on the last accepted step (relevant for rk45)."""
        return self._last_dt

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        """Advance one UPDE integration step.

        Parameters
        ----------
        phases
            Current oscillator phases, shape ``(n_oscillators,)``.
        omegas
            Natural angular frequencies in radians per second, shape
            ``(n_oscillators,)``.
        knm
            Coupling matrix ``K_nm`` with shape
            ``(n_oscillators, n_oscillators)``.
        zeta
            External drive strength.
        psi
            External drive phase target in radians.
        alpha
            Sakaguchi phase-lag matrix with the same shape as ``knm``.

        Returns
        -------
        numpy.ndarray
            New phases wrapped into ``[0, 2*pi)``.

        Raises
        ------
        ValueError
            If input shapes do not match the configured oscillator count
            or any scalar/array input is non-finite.
        """
        self._validate_inputs(phases, omegas, knm, alpha, zeta, psi)
        with self._lock:
            if self._rust is not None:  # pragma: no cover
                return self._validate_rust_output(
                    self._rust.step(
                        np.ascontiguousarray(phases.ravel()),
                        np.ascontiguousarray(omegas.ravel()),
                        np.ascontiguousarray(knm.ravel()),
                        float(zeta),
                        float(psi),
                        np.ascontiguousarray(alpha.ravel()),
                    )
                )
            if self._method == "euler":
                return self._euler_step(phases, omegas, knm, zeta, psi, alpha)
            if self._method == "rk45":
                return self._rk45_step(phases, omegas, knm, zeta, psi, alpha)
            return self._rk4_step(phases, omegas, knm, zeta, psi, alpha)

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
        """Run n_steps, return final phases. Dispatches to the fastest
        available backend via the module-level ``upde_run``."""
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        self._validate_inputs(phases, omegas, knm, alpha, zeta, psi)
        with self._lock:
            if self._rust is not None:  # pragma: no cover
                return self._validate_rust_output(
                    self._rust.run(
                        np.ascontiguousarray(phases.ravel()),
                        np.ascontiguousarray(omegas.ravel()),
                        np.ascontiguousarray(knm.ravel()),
                        float(zeta),
                        float(psi),
                        np.ascontiguousarray(alpha.ravel()),
                        int(n_steps),
                    )
                )

            return upde_run(
                phases,
                omegas,
                knm,
                alpha,
                float(zeta),
                float(psi),
                self._dt,
                n_steps,
                self._method,
                1,
                self._atol,
                self._rtol,
            )

    def compute_order_parameter(self, phases: FloatArray) -> tuple[float, float]:
        """Kuramoto order parameter: R = |<exp(i*theta)>|, psi = arg(...)."""
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        if phases.shape != (self._n,):
            raise ValueError(f"phases.shape={phases.shape}, expected {(self._n,)}")
        if not np.all(np.isfinite(phases)):
            raise ValueError("phases contains NaN/Inf")
        return compute_order_parameter(phases)

    def _validate_inputs(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        alpha: FloatArray,
        zeta: float,
        psi: float,
    ) -> None:
        """Shape and NaN/Inf guards shared by :meth:`step`."""
        zeta = _validate_finite_real(zeta, name="zeta")
        psi = _validate_finite_real(psi, name="psi")
        n = self._n
        phases = _validate_real_array(phases, name="phases")
        omegas = _validate_real_array(omegas, name="omegas")
        knm = _validate_real_array(knm, name="knm")
        alpha = _validate_real_array(alpha, name="alpha")
        checks = (
            ("phases", phases, (n,)),
            ("omegas", omegas, (n,)),
            ("knm", knm, (n, n)),
            ("alpha", alpha, (n, n)),
        )
        for name, arr, shape in checks:
            if arr.shape != shape:
                raise ValueError(f"{name}.shape={arr.shape}, expected {shape}")
            if not np.all(np.isfinite(arr)):
                raise ValueError(f"{name} contains NaN/Inf")

    def _validate_rust_output(self, result: object) -> FloatArray:
        output = np.asarray(result)
        if output.shape != (self._n,):
            raise ValueError(
                f"Rust output has malformed shape {output.shape}, expected {(self._n,)}"
            )
        if output.dtype == np.bool_ or not (
            np.issubdtype(output.dtype, np.integer)
            or np.issubdtype(output.dtype, np.floating)
        ):
            raise ValueError(
                f"Rust output must be a real numeric array, got {output.dtype}"
            )
        try:
            if not np.all(np.isfinite(output)):
                raise ValueError("Rust output contains NaN/Inf")
        except TypeError as exc:
            raise ValueError("Rust output must be finite") from exc
        return np.asarray(output, dtype=np.float64)

    def _derivative(
        self,
        theta: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        # Sakaguchi-Kuramoto coupling: K_ij sin(θ_j - θ_i - α_ij)
        # α_ij is the Sakaguchi phase-lag (Sakaguchi & Kuramoto 1986)
        np.subtract(theta[np.newaxis, :], theta[:, np.newaxis], out=self._phase_diff)
        self._phase_diff -= alpha
        np.sin(self._phase_diff, out=self._sin_diff)

        np.sum(knm * self._sin_diff, axis=1, out=self._scratch_dtheta)
        self._scratch_dtheta += omegas

        # external drive: zeta * sin(Psi - theta_i)
        if zeta != 0.0:
            self._scratch_dtheta += zeta * np.sin(psi - theta)

        return self._scratch_dtheta

    def _euler_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        dtheta = self._derivative(phases, omegas, knm, zeta, psi, alpha)
        # Mod 2π keeps phases on S¹ (circle topology)
        result: FloatArray = (phases + self._dt * dtheta) % TWO_PI
        return result

    def _rk4_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        # Classic RK4: 4th-order Runge-Kutta (1/6, 1/3, 1/3, 1/6 weights)
        dt = self._dt
        k1 = self._derivative(phases, omegas, knm, zeta, psi, alpha).copy()
        k2 = self._derivative(
            phases + 0.5 * dt * k1, omegas, knm, zeta, psi, alpha
        ).copy()
        k3 = self._derivative(
            phases + 0.5 * dt * k2, omegas, knm, zeta, psi, alpha
        ).copy()
        k4 = self._derivative(phases + dt * k3, omegas, knm, zeta, psi, alpha)
        weighted = k1 + 2.0 * k2 + 2.0 * k3 + k4
        result: FloatArray = (phases + (dt / 6.0) * weighted) % TWO_PI
        return result

    def _rk45_stage_vector(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
        dt: float,
    ) -> None:
        """Evaluate all 7 Dormand-Prince stages into ``self._ks``."""
        A = self._DP_A
        ks = self._ks
        ks[0][:] = self._derivative(phases, omegas, knm, zeta, psi, alpha)
        for i in range(1, 7):
            stage = phases + dt * np.dot(A[i, :i], np.array(ks[:i]))
            ks[i][:] = self._derivative(stage, omegas, knm, zeta, psi, alpha)

    def _rk45_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float,
        psi: float,
        alpha: FloatArray,
    ) -> FloatArray:
        """Dormand-Prince RK45 with embedded error estimation and adaptive dt."""
        dt = self._last_dt
        max_reject = 3
        ks = self._ks

        for _ in range(max_reject + 1):
            self._rk45_stage_vector(phases, omegas, knm, zeta, psi, alpha, dt)
            ks_arr = np.array(ks)
            y5 = phases + dt * np.dot(self._DP_B5, ks_arr)
            y4 = phases + dt * np.dot(self._DP_B4, ks_arr)

            # Mixed tolerance scaling (Hairer & Wanner).
            np.subtract(y5, y4, out=self._err_buf)
            np.abs(self._err_buf, out=self._err_buf)
            scale = self._atol + self._rtol * np.maximum(np.abs(phases), np.abs(y5))
            err_norm = float(np.max(self._err_buf / scale))

            if err_norm <= 1.0:
                factor = min(5.0, 0.9 * err_norm ** (-0.2)) if err_norm > 0.0 else 5.0
                self._last_dt = min(dt * factor, self._dt * 10.0)
                result: FloatArray = y5 % TWO_PI
                return result

            factor = max(0.2, 0.9 * err_norm ** (-0.25))  # pragma: no cover
            dt = dt * factor  # pragma: no cover

        self._last_dt = dt  # pragma: no cover
        result_fallback: FloatArray = y5 % TWO_PI  # pragma: no cover
        return result_fallback  # pragma: no cover
