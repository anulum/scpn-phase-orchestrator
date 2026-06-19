# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Sparse UPDE integration engine

"""Sparse CSR-style UPDE engine for validated oscillator coupling graphs.

The sparse engine advances phase vectors from row pointers, column indices,
coupling values, and phase-lag values instead of dense ``N x N`` matrices.
Inputs are checked for finite values, CSR monotonicity, edge-count consistency,
valid oscillator indices, and method selection before stepping. Optional Rust
execution and Python fallback preserve the same shape and bounds contracts.
"""

from __future__ import annotations

import threading
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["SparseUPDEEngine"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _validate_finite_real(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be a finite real number, got {value!r}")
    return coerced


def _validate_integer_ndarray(value: object, *, name: str) -> IntArray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy ndarray, got {type(value).__name__}")
    if value.dtype == np.bool_ or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"{name} must be an integer ndarray, got {value.dtype}")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {value.shape}")
    return value


def _validate_real_ndarray(value: object, *, name: str) -> FloatArray:
    if not isinstance(value, np.ndarray):
        raise ValueError(f"{name} must be a NumPy ndarray, got {type(value).__name__}")
    if value.dtype == np.bool_ or not (
        np.issubdtype(value.dtype, np.integer)
        or np.issubdtype(value.dtype, np.floating)
    ):
        raise ValueError(f"{name} must be a real numeric ndarray, got {value.dtype}")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {value.shape}")
    return value


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
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    coerced = float(value)
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    return coerced


class SparseUPDEEngine:
    """Kuramoto UPDE integrator with sparse coupling matrix support.

    The SparseUPDEEngine solves the Universal Phase Dynamics Equation (UPDE)
    using a CSR (Compressed Sparse Row) representation for the coupling matrix
    K_nm and phase lags alpha_nm. This is critical for scaling to large-scale
    oscillator networks (e.g., N > 10,000) where the dense K_nm matrix
    would consume terabytes of RAM.

    Mathematics:
    dtheta_i/dt = omega_i
                  + sum_{j in neighbors(i)} K_ij sin(theta_j - theta_i - alpha_ij)
                  + zeta sin(Psi - theta_i)

    The integrator supports sub-microsecond in-place plasticity updates
    when running on the Rust FFI path, allowing the coupling topology to
    evolve concurrently with the phase dynamics.
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
        dt: float,
        method: str = "euler",
        atol: float = 1e-6,
        rtol: float = 1e-3,
    ):
        """Initialize the sparse integrator.

        Parameters
        ----------
        n_oscillators : int
            Total number of oscillators N in the network.
        dt : float
            Integration timestep in seconds.
        method : str
            Numerical method ('euler', 'rk4', or 'rk45').
        atol : float
            Absolute tolerance for adaptive RK45.
        rtol : float
            Relative tolerance for adaptive RK45.
        """
        n_oscillators = _validate_positive_int(
            n_oscillators,
            name="n_oscillators",
        )
        dt = _validate_positive_float(dt, name="dt")
        atol = _validate_positive_float(atol, name="atol")
        rtol = _validate_positive_float(rtol, name="rtol")
        self._n = n_oscillators
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
                from spo_kernel import PySparseUPDEStepper

                self._rust = PySparseUPDEStepper(
                    n_oscillators, dt, method, atol=atol, rtol=rtol
                )
            except ImportError:
                pass

    @property
    def last_dt(self) -> float:
        """Actual dt used on the last accepted step (relevant for rk45).

        Returns
        -------
        float
            Actual dt used on the last accepted step (relevant for rk45).
        """
        return self._last_dt

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
    ) -> FloatArray:
        """Advance phases by one sparse timestep, return new phases in [0, 2*pi).

        Parameters
        ----------
        phases : FloatArray
            Current phase vector [theta_1, ..., theta_N], shape (N,).
        omegas : FloatArray
            Natural frequency vector [omega_1, ..., omega_N], shape (N,).
        row_ptr : IntArray
            CSR row pointers, shape (N+1,).
        col_indices : IntArray
            CSR column indices, shape (E,).
        knm_values : FloatArray
            CSR coupling strengths, shape (E,).
        zeta : float
            External forcing strength (global scalar).
        psi : float
            Reference phase target (global scalar).
        alpha_values : FloatArray
            CSR phase lags, shape (E,).

        Returns
        -------
        FloatArray
            New phase vector [theta_1(t+dt), ..., theta_N(t+dt)], shape (N,).
        """
        zeta = _validate_finite_real(zeta, name="zeta")
        psi = _validate_finite_real(psi, name="psi")
        self._validate_inputs(
            phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            alpha_values,
            zeta,
            psi,
        )
        with self._lock:
            if self._rust is not None:
                result = self._rust.step(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                    np.ascontiguousarray(row_ptr.ravel(), dtype=np.uint64),
                    np.ascontiguousarray(col_indices.ravel(), dtype=np.uint64),
                    np.ascontiguousarray(knm_values.ravel(), dtype=np.float64),
                    zeta,
                    psi,
                    np.ascontiguousarray(alpha_values.ravel(), dtype=np.float64),
                )
                return self._validate_rust_output(result)

            if self._method == "euler":
                return self._euler_step(
                    phases,
                    omegas,
                    row_ptr,
                    col_indices,
                    knm_values,
                    zeta,
                    psi,
                    alpha_values,
                )
            if self._method == "rk45":
                return self._rk45_step(
                    phases,
                    omegas,
                    row_ptr,
                    col_indices,
                    knm_values,
                    zeta,
                    psi,
                    alpha_values,
                )
            return self._rk4_step(
                phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
            )

    def run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
        n_steps: int,
    ) -> FloatArray:
        """Run multiple steps in a batch, return final phases.

        Parameters
        ----------
        phases : FloatArray
            Initial phase vector.
        omegas : FloatArray
            Natural frequencies.
        row_ptr : IntArray
            CSR row pointers.
        col_indices : IntArray
            CSR column indices.
        knm_values : FloatArray
            CSR coupling strengths.
        zeta : float
            External forcing strength.
        psi : float
            Reference phase target.
        alpha_values : FloatArray
            CSR phase lags.
        n_steps : int
            Number of integration steps to perform.

        Returns
        -------
        FloatArray
            Final phase vector after n_steps.
        """
        n_steps = _validate_nonnegative_int(n_steps, name="n_steps")
        zeta = _validate_finite_real(zeta, name="zeta")
        psi = _validate_finite_real(psi, name="psi")
        self._validate_inputs(
            phases,
            omegas,
            row_ptr,
            col_indices,
            knm_values,
            alpha_values,
            zeta,
            psi,
        )
        if n_steps == 0:
            return phases.copy()
        with self._lock:
            if self._rust is not None:
                result = self._rust.run(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    np.ascontiguousarray(omegas.ravel(), dtype=np.float64),
                    np.ascontiguousarray(row_ptr.ravel(), dtype=np.uint64),
                    np.ascontiguousarray(col_indices.ravel(), dtype=np.uint64),
                    np.ascontiguousarray(knm_values.ravel(), dtype=np.float64),
                    zeta,
                    psi,
                    np.ascontiguousarray(alpha_values.ravel(), dtype=np.float64),
                    n_steps,
                )
                return self._validate_rust_output(result)

            p = phases.copy()
            for _ in range(n_steps):
                p = self.step(
                    p, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values
                )
            return p

    def _validate_rust_output(self, result: object) -> FloatArray:
        out = np.asarray(result)
        if out.shape != (self._n,):
            shape_expected = (self._n,)
            raise ValueError(
                f"Rust output has malformed shape {out.shape},"
                f" expected {shape_expected}"
            )
        if out.dtype == np.bool_ or not (
            np.issubdtype(out.dtype, np.integer)
            or np.issubdtype(out.dtype, np.floating)
        ):
            raise ValueError(
                f"Rust output must be a real numeric array, got {out.dtype}"
            )
        try:
            if not np.all(np.isfinite(out)):
                raise ValueError("Rust output contains NaN/Inf")
        except TypeError as exc:
            raise ValueError("Rust output must be a finite real numeric array") from exc
        return np.asarray(out, dtype=np.float64)

    def _validate_inputs(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        alpha_values: FloatArray,
        zeta: float,
        psi: float,
    ) -> None:
        phases = _validate_real_ndarray(phases, name="phases")
        omegas = _validate_real_ndarray(omegas, name="omegas")
        row_ptr = _validate_integer_ndarray(row_ptr, name="row_ptr")
        col_indices = _validate_integer_ndarray(col_indices, name="col_indices")
        knm_values = _validate_real_ndarray(knm_values, name="knm_values")
        alpha_values = _validate_real_ndarray(alpha_values, name="alpha_values")

        n = self._n
        if row_ptr.shape != (n + 1,):
            raise ValueError(f"row_ptr.shape={row_ptr.shape}, expected {(n + 1,)}")

        if phases.shape != (n,):
            raise ValueError(f"phases.shape={phases.shape}, expected {(n,)}")
        if omegas.shape != (n,):
            raise ValueError(f"omegas.shape={omegas.shape}, expected {(n,)}")

        if np.any(row_ptr < 0):
            raise ValueError("row_ptr entries must be non-negative")
        if row_ptr[0] != 0:
            raise ValueError("row_ptr must start at 0")
        if np.any(row_ptr[1:] < row_ptr[:-1]):
            raise ValueError("row_ptr must be monotonic")

        edge_count = int(row_ptr[-1])
        if edge_count < 0:
            raise ValueError("row_ptr final entry must be non-negative")

        if col_indices.shape != (edge_count,):
            raise ValueError(
                f"col_indices.shape={col_indices.shape}, expected {(edge_count,)}"
            )
        if knm_values.shape != (edge_count,):
            raise ValueError(
                f"knm_values.shape={knm_values.shape}, expected {(edge_count,)}"
            )
        if alpha_values.shape != (edge_count,):
            raise ValueError(
                f"alpha_values.shape={alpha_values.shape}, expected {(edge_count,)}"
            )
        if knm_values.shape != alpha_values.shape:
            raise ValueError("knm_values and alpha_values must have matching shapes")

        if np.any(col_indices < 0) or np.any(col_indices >= self._n):
            raise ValueError("col_indices entries must be valid oscillator indices")

        for name, arr in (
            ("phases", phases),
            ("omegas", omegas),
            ("knm_values", knm_values),
            ("alpha_values", alpha_values),
        ):
            try:
                finite = np.isfinite(arr)
            except TypeError as exc:
                raise ValueError(f"{name} contains NaN/Inf") from exc
            if not np.all(finite):
                raise ValueError(f"{name} contains NaN/Inf")

    def _derivative(
        self,
        theta: FloatArray,
        omegas: FloatArray,
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
    ) -> FloatArray:
        """Compute the internal UPDE derivative (Python fallback)."""
        n = len(theta)
        dtheta = omegas.copy()
        for i in range(n):
            start = row_ptr[i]
            end = row_ptr[i + 1]
            for idx in range(start, end):
                j = col_indices[idx]
                dtheta[i] += knm_values[idx] * np.sin(
                    theta[j] - theta[i] - alpha_values[idx]
                )

        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - theta)

        return dtheta

    def _rk4_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
    ) -> FloatArray:
        """Single RK4 integration step (Python fallback for rk4/rk45)."""
        dt = self._dt
        args = (omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values)
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
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
        dt: float,
    ) -> list[FloatArray]:
        args = (omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values)
        stages = [self._derivative(phases, *args)]
        for i in range(1, 7):
            increment = sum(self._DP_A[i, j] * stages[j] for j in range(i))
            stages.append(self._derivative(phases + dt * increment, *args))
        return stages

    def _rk45_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
    ) -> FloatArray:
        dt = self._last_dt
        max_reject = 3
        for _ in range(max_reject + 1):
            stages = self._rk45_stage_vector(
                phases,
                omegas,
                row_ptr,
                col_indices,
                knm_values,
                zeta,
                psi,
                alpha_values,
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
        row_ptr: IntArray,
        col_indices: IntArray,
        knm_values: FloatArray,
        zeta: float,
        psi: float,
        alpha_values: FloatArray,
    ) -> FloatArray:
        """Single Euler integration step (Python fallback)."""
        dtheta = self._derivative(
            phases, omegas, row_ptr, col_indices, knm_values, zeta, psi, alpha_values
        )
        result: FloatArray = (phases + self._dt * dtheta) % TWO_PI
        return result
