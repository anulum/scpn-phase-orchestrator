# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Cellular Sheaf UPDE Engine

"""Cellular-sheaf UPDE integrator for multidimensional oscillator phases.

``SheafUPDEEngine`` advances ``N x D`` phase matrices using restriction-map
coupling blocks and optional Rust acceleration. It validates oscillator counts,
dimensions, timestep/tolerances, solver method, forcing scalars, phase targets,
and tensor shapes before integration. Instance-level locks protect reusable
scratch buffers so concurrent callers cannot corrupt adaptive or fixed-step
solver state.
"""

from __future__ import annotations

import threading
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["SheafUPDEEngine"]

FloatArray: TypeAlias = NDArray[np.float64]


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    return coerced


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be >= 0 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_finite_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if array.shape != shape:
        raise ValueError(f"{name}.shape={array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    return coerced


def _reshape_rust_result(
    value: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    expected_size = shape[0] * shape[1]
    if array.size != expected_size:
        raise ValueError(
            f"Rust sheaf {name} returned {array.size} values, expected {expected_size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Rust sheaf {name} returned NaN/Inf")
    return np.ascontiguousarray(array.reshape(shape), dtype=np.float64)


class SheafUPDEEngine:
    """Cellular Sheaf UPDE integrator for multi-dimensional phase vectors.

    Phase per oscillator is a vector of dimension D.
    Restriction maps (coupling blocks) B_ij are D x D matrices mapping
    the phase space of oscillator j into the space of oscillator i.

    Mathematics:
    d(theta_{i,d})/dt = omega_{i,d}
                        + sum_j sum_k B_ij^{dk} sin(theta_{j,k} - theta_{i,d})
                        + zeta * sin(Psi_d - theta_{i,d})

    This enables complex cross-frequency coupling and opinion dynamics
    over multidimensional belief spaces.
    """

    _DP_A = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        ],
        dtype=np.float64,
    )
    _DP_B4 = np.array(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40],
        dtype=np.float64,
    )
    _DP_B5 = np.array(
        [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
        dtype=np.float64,
    )

    def __init__(
        self,
        n_oscillators: int,
        d_dimensions: int,
        dt: float,
        method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        n_oscillators = _validate_positive_int(
            n_oscillators,
            name="n_oscillators",
        )
        d_dimensions = _validate_positive_int(
            d_dimensions,
            name="d_dimensions",
        )
        dt = _validate_positive_float(dt, name="dt")
        atol = _validate_positive_float(atol, name="atol")
        rtol = _validate_positive_float(rtol, name="rtol")
        self._n = n_oscillators
        self._d = d_dimensions
        self._dt = dt
        if method not in ("euler", "rk4", "rk45"):
            msg = f"Unknown method {method!r}, expected 'euler', 'rk4', or 'rk45'"
            raise ValueError(msg)
        self._method = method
        self._atol = atol
        self._rtol = rtol
        self._last_dt = dt
        self._lock = threading.RLock()

        self._rust = None
        if _HAS_RUST:
            try:
                from spo_kernel import PySheafUPDEStepper

                self._rust = PySheafUPDEStepper(
                    n_oscillators, d_dimensions, dt, method, atol=atol, rtol=rtol
                )
            except ImportError:
                pass

    @property
    def last_dt(self) -> float:
        """Return the most recent timestep used by the sheaf engine.

        Returns
        -------
        float
            Return the most recent timestep used by the sheaf engine.
        """
        return self._last_dt

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> FloatArray:
        """Advance phases by one timestep.

        Args:
            phases: Current phase matrix [theta_i,d], shape (N, D).
            omegas: Natural frequency matrix [omega_i,d], shape (N, D).
            restriction_maps: Block matrix coupling [B_ij^{dk}], shape (N, N, D, D).
            zeta: External forcing strength (global scalar).
            psi: Reference phase target vector, shape (D,).

        Returns
        -------
            New phase matrix, shape (N, D).

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        restriction_maps : FloatArray
            Sheaf restriction maps, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : FloatArray
            External drive reference phase ``Ψ`` in radians.
        """
        phases, omegas, restriction_maps, zeta, psi = self._validate_inputs(
            phases,
            omegas,
            restriction_maps,
            zeta,
            psi,
        )
        with self._lock:
            if self._rust is not None:
                res = self._rust.step(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                    np.ascontiguousarray(restriction_maps.ravel(), dtype=np.float64),
                    float(zeta),
                    np.ascontiguousarray(psi.ravel(), dtype=np.float64),
                )
                return _reshape_rust_result(
                    res,
                    name="step",
                    shape=(self._n, self._d),
                )

            if self._method == "euler":
                return self._euler_step(phases, omegas, restriction_maps, zeta, psi)
            if self._method == "rk45":
                return self._rk45_step(phases, omegas, restriction_maps, zeta, psi)
            return self._rk4_step(phases, omegas, restriction_maps, zeta, psi)

    def run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
        n_steps: int,
    ) -> FloatArray:
        """Run multiple steps in a batch, return final phases.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        restriction_maps : FloatArray
            Sheaf restriction maps, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : FloatArray
            External drive reference phase ``Ψ`` in radians.
        n_steps : int
            Number of integration steps to run.

        Returns
        -------
        FloatArray
            The final phases after ``n_steps`` sheaf steps.
        """
        n_steps = _validate_nonnegative_int(n_steps, name="n_steps")
        phases, omegas, restriction_maps, zeta, psi = self._validate_inputs(
            phases,
            omegas,
            restriction_maps,
            zeta,
            psi,
        )
        with self._lock:
            if self._rust is not None:
                res = self._rust.run(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                    np.ascontiguousarray(restriction_maps.ravel(), dtype=np.float64),
                    float(zeta),
                    np.ascontiguousarray(psi.ravel(), dtype=np.float64),
                    n_steps,
                )
                return _reshape_rust_result(
                    res,
                    name="run",
                    shape=(self._n, self._d),
                )

            p = phases.copy()
            for _ in range(n_steps):
                p = self.step(p, omegas, restriction_maps, zeta, psi)
            return p

    def _validate_inputs(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> tuple[FloatArray, FloatArray, FloatArray, float, FloatArray]:
        n, d = self._n, self._d
        zeta = _validate_finite_real(zeta, name="zeta")
        return (
            _validate_finite_matrix(phases, name="phases", shape=(n, d)),
            _validate_finite_matrix(omegas, name="omegas", shape=(n, d)),
            _validate_finite_matrix(
                restriction_maps,
                name="restriction_maps",
                shape=(n, n, d, d),
            ),
            zeta,
            _validate_finite_matrix(psi, name="psi", shape=(d,)),
        )

    def _derivative(
        self,
        theta: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> FloatArray:
        n, d = self._n, self._d
        dtheta = omegas.copy()
        for i in range(n):
            for dim in range(d):
                coupling_sum = 0.0
                for j in range(n):
                    for k in range(d):
                        b_val = restriction_maps[i, j, dim, k]
                        if b_val != 0.0:
                            coupling_sum += b_val * np.sin(theta[j, k] - theta[i, dim])
                dtheta[i, dim] += coupling_sum
                if zeta != 0.0:
                    dtheta[i, dim] += zeta * np.sin(psi[dim] - theta[i, dim])
        return dtheta

    def _rk4_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> FloatArray:
        """Single RK4 integration step (Python fallback for rk4/rk45)."""
        dt = self._dt
        args = (omegas, restriction_maps, zeta, psi)
        k1 = self._derivative(phases, *args)
        k2 = self._derivative((phases + 0.5 * dt * k1) % TWO_PI, *args)
        k3 = self._derivative((phases + 0.5 * dt * k2) % TWO_PI, *args)
        k4 = self._derivative((phases + dt * k3) % TWO_PI, *args)
        result: FloatArray = (
            phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        ) % TWO_PI
        return result

    def _rk45_stage_vector(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
        dt: float,
    ) -> list[FloatArray]:
        args = (omegas, restriction_maps, zeta, psi)
        stages = [self._derivative(phases, *args)]
        for i in range(1, 7):
            increment = sum(self._DP_A[i, j] * stages[j] for j in range(i))
            stages.append(self._derivative(phases + dt * increment, *args))
        return stages

    def _rk45_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> FloatArray:
        dt = self._last_dt
        max_reject = 3
        for _ in range(max_reject + 1):
            stages = self._rk45_stage_vector(
                phases,
                omegas,
                restriction_maps,
                zeta,
                psi,
                dt,
            )
            y5 = phases + dt * sum(self._DP_B5[i] * stages[i] for i in range(7))
            y4 = phases + dt * sum(self._DP_B4[i] * stages[i] for i in range(7))
            scale = self._atol + self._rtol * np.maximum(np.abs(phases), np.abs(y5))
            err_norm = float(np.max(np.abs(y5 - y4) / scale))
            if err_norm <= 1.0:
                factor = min(5.0, 0.9 * err_norm ** (-0.2)) if err_norm > 0.0 else 5.0
                self._last_dt = min(dt * factor, self._dt * 10.0)
                result: FloatArray = y5 % TWO_PI
                return result
            dt *= max(0.2, 0.9 * err_norm ** (-0.25))
        self._last_dt = dt
        result_fallback: FloatArray = y5 % TWO_PI
        return result_fallback

    def _euler_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> FloatArray:
        dtheta = self._derivative(phases, omegas, restriction_maps, zeta, psi)
        result: FloatArray = (phases + self._dt * dtheta) % TWO_PI
        return result
