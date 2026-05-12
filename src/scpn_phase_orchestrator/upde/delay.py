# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Time-delayed Kuramoto coupling

from __future__ import annotations

from collections import deque
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

try:
    from spo_kernel import (
        delayed_kuramoto_run_rust as _rust_delayed_run,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["DelayBuffer", "DelayedEngine"]
FloatArray: TypeAlias = NDArray[np.float64]


def _validate_positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    return value


def _validate_finite_float(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    value = float(value)
    if not isfinite(value):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    return value


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


class DelayBuffer:
    """Circular buffer storing phase history for delayed coupling.

    Stores last `max_delay_steps` snapshots. Retrieves phases from
    `delay_steps` steps ago.
    """

    def __init__(self, n_oscillators: int, max_delay_steps: int):
        self._n = _validate_positive_int(n_oscillators, name="n_oscillators")
        self._max = _validate_positive_int(max_delay_steps, name="max_delay_steps")
        self._buffer: deque[FloatArray] = deque(maxlen=self._max)

    def push(self, phases: FloatArray) -> None:
        """Append a phase snapshot to the buffer."""
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        self._buffer.append(phases64.copy())

    def get_delayed(self, delay_steps: int) -> FloatArray | None:
        """Return phases from `delay_steps` ago, or None if not enough history."""
        if delay_steps < 1 or delay_steps > len(self._buffer):
            return None
        return self._buffer[-delay_steps]

    @property
    def length(self) -> int:
        """Number of snapshots currently stored."""
        return len(self._buffer)

    def clear(self) -> None:
        """Discard all stored phase snapshots."""
        self._buffer.clear()


class DelayedEngine:
    """Kuramoto with time-delayed coupling.

    dθ_i/dt = ω_i + Σ_j K_ij sin(θ_j(t-τ) - θ_i(t) - α_ij)
    """

    def __init__(self, n_oscillators: int, dt: float, delay_steps: int = 1):
        self._n = _validate_positive_int(n_oscillators, name="n_oscillators")
        self._dt = _validate_positive_float(dt, name="dt")
        self._delay_steps = _validate_positive_int(delay_steps, name="delay_steps")
        self._buffer: deque[FloatArray] = deque(maxlen=self._delay_steps + 1)

    @property
    def delay_steps(self) -> int:
        return self._delay_steps

    def step(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: FloatArray | None = None,
        step_idx: int = 0,
    ) -> FloatArray:
        del step_idx
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        if alpha is None:
            alpha64 = np.zeros((self._n, self._n), dtype=np.float64)
        else:
            alpha64 = _validate_state_array(
                alpha,
                name="alpha",
                shape=(self._n, self._n),
            )
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")
        self._buffer.append(phases64.copy())
        delayed = self._buffer[0] if len(self._buffer) > self._delay_steps else phases64
        diff = delayed[np.newaxis, :] - phases64[:, np.newaxis] - alpha64
        coupling = np.sum(knm64 * np.sin(diff), axis=1)
        dtheta = omegas64 + coupling
        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - phases64)
        step_out: FloatArray = (phases64 + self._dt * dtheta) % TWO_PI
        return step_out

    def run(
        self,
        phases: FloatArray,
        omegas: FloatArray,
        knm: FloatArray,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: FloatArray | None = None,
        n_steps: int = 100,
    ) -> FloatArray:
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        if alpha is None:
            alpha64 = np.zeros((self._n, self._n), dtype=np.float64)
        else:
            alpha64 = _validate_state_array(
                alpha,
                name="alpha",
                shape=(self._n, self._n),
            )
        zeta = _validate_finite_float(zeta, name="zeta")
        psi = _validate_finite_float(psi, name="psi")
        if _HAS_RUST:
            result: FloatArray = np.asarray(
                _rust_delayed_run(
                    phases64,
                    omegas64,
                    knm64.ravel(),
                    alpha64.ravel(),
                    self._n,
                    zeta,
                    psi,
                    self._dt,
                    self._delay_steps,
                    n_steps,
                )
            )
            return np.asarray(result, dtype=np.float64)
        p = phases64.copy()
        for i in range(n_steps):
            p = self.step(p, omegas64, knm64, zeta, psi, alpha64, step_idx=i)
        return p
