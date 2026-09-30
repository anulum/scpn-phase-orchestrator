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
from numbers import Complex, Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST
from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["SheafUPDEEngine"]

FloatArray: TypeAlias = NDArray[np.float64]


def _wrap_phases(phases: FloatArray) -> FloatArray:
    """Return torus phases, mapping a rounded upper endpoint to equivalent zero."""
    wrapped: FloatArray = phases % TWO_PI
    wrapped[wrapped >= TWO_PI] = 0.0
    return wrapped


def _validate_positive_int(value: object, *, name: str) -> int:
    """Return ``value`` as a positive integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be >= 1 as a non-boolean integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    try:
        coerced = float(value)
    except OverflowError:
        raise ValueError(
            f"{name} must be positive finite real, got {value!r}"
        ) from None
    if not np.isfinite(coerced) or coerced <= 0.0:
        raise ValueError(f"{name} must be positive finite real, got {value!r}")
    return coerced


def _validate_nonnegative_int(value: object, *, name: str) -> int:
    """Return ``value`` as a non-negative integer, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be >= 0 as a non-boolean integer, got {value!r}")
    count = int(value)
    if count > np.iinfo(np.uint64).max:
        raise ValueError(f"{name} exceeds the native u64 count range")
    return count


def _as_real_numeric_array(value: object, *, name: str) -> FloatArray:
    """Return a real numeric array without coercing string or complex aliases."""
    try:
        raw = np.asarray(value)
        if isinstance(value, (list, tuple)):
            raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a numeric array") from None
    object_values = raw.dtype == np.object_
    if raw.dtype == np.bool_ or (
        object_values and any(isinstance(item, (bool, np.bool_)) for item in raw.flat)
    ):
        raise ValueError(f"{name} must be real-valued, not boolean")
    if np.iscomplexobj(raw) or (
        object_values
        and any(
            isinstance(item, Complex) and not isinstance(item, Real)
            for item in raw.flat
        )
    ):
        raise ValueError(f"{name} must be real-valued, not complex")
    numeric_object = object_values and all(isinstance(item, Real) for item in raw.flat)
    if raw.dtype.kind in ("m", "M") or (
        not np.issubdtype(raw.dtype, np.number) and not numeric_object
    ):
        raise ValueError(f"{name} must be numeric")
    try:
        return np.ascontiguousarray(raw, dtype=np.float64)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc


def _validate_finite_matrix(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> FloatArray:
    """Return the value as a validated finite matrix, else raise."""
    array = _as_real_numeric_array(value, name=name)
    if array.shape != shape:
        raise ValueError(f"{name}.shape={array.shape}, expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def _validate_finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    try:
        coerced = float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite real, got {value!r}") from None
    if not np.isfinite(coerced):
        raise ValueError(f"{name} must be finite real, got {value!r}")
    return coerced


def _reshape_rust_result(
    value: object,
    *,
    name: str,
    shape: tuple[int, int],
) -> FloatArray:
    """Validate and reshape a flat Rust result into the sheaf state shape."""
    array = _as_real_numeric_array(value, name=f"Rust sheaf {name} output")
    if array.ndim != 1:
        raise ValueError(f"Rust sheaf {name} output must be one-dimensional")
    expected_size = shape[0] * shape[1]
    if array.size != expected_size:
        raise ValueError(
            f"Rust sheaf {name} returned {array.size} values, expected {expected_size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"Rust sheaf {name} returned NaN/Inf")
    if np.any((array < 0.0) | (array >= TWO_PI)):
        raise ValueError(f"Rust sheaf {name} returned phases outside [0, 2*pi)")
    return array.reshape(shape)


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
    ) -> None:
        """Configure a fixed outer timestep and optional adaptive integration.

        Parameters
        ----------
        n_oscillators, d_dimensions : int
            Positive oscillator count and phase-vector dimension.
        dt : float
            Positive duration advanced by each successful step.
        method : str
            Euler, RK4 or adaptive Dormand-Prince RK45 integration.
        atol, rtol : float
            Positive finite absolute and relative RK45 tolerances, with
            ``rtol >= atol`` when using RK45.

        Raises
        ------
        ValueError
            Geometry overflows native cardinality, a count or numerical control
            is invalid, or the method is unsupported.
        """
        n_oscillators = _validate_positive_int(
            n_oscillators,
            name="n_oscillators",
        )
        d_dimensions = _validate_positive_int(
            d_dimensions,
            name="d_dimensions",
        )
        size = n_oscillators * d_dimensions
        if size * size > np.iinfo(np.uintp).max:
            raise ValueError("sheaf geometry overflows usize")
        dt = _validate_positive_float(dt, name="dt")
        atol = _validate_positive_float(atol, name="atol")
        rtol = _validate_positive_float(rtol, name="rtol")
        self._n = n_oscillators
        self._d = d_dimensions
        self._dt = dt
        if method not in ("euler", "rk4", "rk45"):
            msg = f"Unknown method {method!r}, expected 'euler', 'rk4', or 'rk45'"
            raise ValueError(msg)
        if method == "rk45" and rtol < atol:
            raise ValueError("for RK45, rtol must be >= atol")
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
        """Return the next adaptive substep proposal or configured fixed timestep.

        Returns
        -------
        float
            Positive finite proposal bounded by the configured outer timestep.
        """
        return self._last_dt

    def step(
        self,
        phases: ArrayLike,
        omegas: ArrayLike,
        restriction_maps: ArrayLike,
        zeta: float,
        psi: ArrayLike,
    ) -> FloatArray:
        """Advance phases through one complete configured outer interval.

        Parameters
        ----------
        phases : array_like
            Current phase matrix [theta_i,d], shape (N, D).
        omegas : array_like
            Natural frequency matrix [omega_i,d], shape (N, D).
        restriction_maps : array_like
            Block matrix coupling [B_ij^{dk}], shape (N, N, D, D).
        zeta : float
            External forcing strength (global scalar).
        psi : array_like
            Reference phase target vector, shape (D,).

        Returns
        -------
        FloatArray
            Independent float64 phase matrix, shape (N, D), in ``[0, 2*pi)``.

        Raises
        ------
        ValueError
            Inputs have invalid source types, shapes or non-finite values, or
            integration cannot produce a finite torus state. Refusal preserves
            input storage and the previously published ``last_dt`` proposal.
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
                output = _reshape_rust_result(
                    res,
                    name="step",
                    shape=(self._n, self._d),
                )
                self._last_dt = _validate_positive_float(
                    self._rust.last_dt,
                    name="Rust last_dt",
                )
                return output

            previous_dt = self._last_dt
            try:
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    if self._method == "euler":
                        output = self._euler_step(
                            phases, omegas, restriction_maps, zeta, psi
                        )
                    elif self._method == "rk45":
                        output = self._rk45_step(
                            phases, omegas, restriction_maps, zeta, psi
                        )
                    else:
                        output = self._rk4_step(
                            phases, omegas, restriction_maps, zeta, psi
                        )
                if not np.all(np.isfinite(output)) or np.any(
                    (output < 0.0) | (output >= TWO_PI)
                ):
                    raise ValueError("Sheaf step returned invalid torus phases")
                return output
            except ValueError:
                self._last_dt = previous_dt
                raise

    def run(
        self,
        phases: ArrayLike,
        omegas: ArrayLike,
        restriction_maps: ArrayLike,
        zeta: float,
        psi: ArrayLike,
        n_steps: int,
    ) -> FloatArray:
        """Advance a batch of complete outer intervals and return final phases.

        Parameters
        ----------
        phases : array_like
            Oscillator phases in radians, shape ``(N, D)``.
        omegas : array_like
            Natural frequencies in rad/s, shape ``(N, D)``.
        restriction_maps : array_like
            Sheaf restriction maps, shape ``(N, N, D, D)``.
        zeta : float
            External drive strength ``ζ``.
        psi : array_like
            External drive reference phase ``Ψ`` in radians, shape ``(D,)``.
        n_steps : int
            Non-boolean count in the native u64 range. Zero returns an
            independent, validated copy without invoking the optional backend.

        Returns
        -------
        FloatArray
            Independent float64 phases after ``n_steps * dt`` elapsed time,
            in ``[0, 2*pi)``. Zero steps preserve valid unwrapped phase values.

        Raises
        ------
        ValueError
            Count, inputs or numerical integration are invalid. A refusal at
            any interval preserves input storage and the pre-batch proposal.
        """
        n_steps = _validate_nonnegative_int(n_steps, name="n_steps")
        phases, omegas, restriction_maps, zeta, psi = self._validate_inputs(
            phases,
            omegas,
            restriction_maps,
            zeta,
            psi,
        )
        if n_steps == 0:
            return phases.copy()
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
                output = _reshape_rust_result(
                    res,
                    name="run",
                    shape=(self._n, self._d),
                )
                self._last_dt = _validate_positive_float(
                    self._rust.last_dt,
                    name="Rust last_dt",
                )
                return output

            previous_dt = self._last_dt
            try:
                p = phases.copy()
                for _ in range(n_steps):
                    p = self.step(p, omegas, restriction_maps, zeta, psi)
                return p
            except ValueError:
                self._last_dt = previous_dt
                raise

    def _validate_inputs(
        self,
        phases: ArrayLike,
        omegas: ArrayLike,
        restriction_maps: ArrayLike,
        zeta: float,
        psi: ArrayLike,
    ) -> tuple[FloatArray, FloatArray, FloatArray, float, FloatArray]:
        """Validate and normalise the sheaf-engine integration inputs."""
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
        """Return the cellular-sheaf phase derivative for the state."""
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
        """Advance a complete configured timestep by classical RK4."""
        dt = self._dt
        args = (omegas, restriction_maps, zeta, psi)
        k1 = self._derivative(phases, *args)
        k2 = self._derivative(_wrap_phases(phases + 0.5 * dt * k1), *args)
        k3 = self._derivative(_wrap_phases(phases + 0.5 * dt * k2), *args)
        k4 = self._derivative(_wrap_phases(phases + dt * k3), *args)
        return _wrap_phases(phases + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4))

    def _rk45_stage_vector(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
        dt: float,
    ) -> list[FloatArray]:
        """Return one RK45 stage derivative for the sheaf state."""
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
        """Advance the complete outer interval with bounded RK45 substeps."""
        dt = self._last_dt
        remaining = self._dt
        current = phases.copy()
        rejects = 0
        for _ in range(100_000):
            dt = min(dt, remaining)
            stages = self._rk45_stage_vector(
                current,
                omegas,
                restriction_maps,
                zeta,
                psi,
                dt,
            )
            y5 = current + dt * sum(self._DP_B5[i] * stages[i] for i in range(7))
            error = dt * sum(
                (self._DP_B4[i] - self._DP_B5[i]) * stages[i] for i in range(7)
            )
            scale = self._atol + self._rtol * np.maximum(np.abs(current), np.abs(y5))
            if (
                not all(np.all(np.isfinite(stage)) for stage in stages)
                or not np.all(np.isfinite(y5))
                or not np.all(np.isfinite(scale))
            ):
                raise ValueError("Sheaf RK45 produced non-finite arithmetic")
            err_norm = float(np.max(np.abs(error) / scale))
            if not np.isfinite(err_norm):
                raise ValueError("Sheaf RK45 produced non-finite error estimate")
            factor = (
                min(5.0, max(0.2, 0.9 * err_norm ** (-0.2))) if err_norm > 0.0 else 5.0
            )
            next_dt = min(dt * factor, self._dt)
            if not np.isfinite(next_dt) or next_dt <= 0.0:
                raise ValueError("Sheaf RK45 timestep cannot advance")
            if err_norm <= 1.0:
                next_remaining = remaining - dt
                if next_remaining == remaining:
                    raise ValueError("Sheaf RK45 timestep cannot advance")
                current = y5
                remaining = next_remaining
                self._last_dt = next_dt
                rejects = 0
                if remaining <= 0.0:
                    return _wrap_phases(current)
            else:
                rejects += 1
                if rejects >= 64:
                    raise ValueError("Sheaf RK45 rejection limit exceeded")
            dt = next_dt
        raise ValueError("Sheaf RK45 substep limit exceeded")

    def _euler_step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        restriction_maps: FloatArray,
        zeta: float,
        psi: FloatArray,
    ) -> FloatArray:
        """Advance the sheaf state one explicit-Euler step."""
        dtheta = self._derivative(phases, omegas, restriction_maps, zeta, psi)
        return _wrap_phases(phases + self._dt * dtheta)
