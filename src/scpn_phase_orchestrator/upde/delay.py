# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Time-delayed Kuramoto coupling

"""Time-delayed Kuramoto buffer and engine with validated phase history.

``DelayBuffer`` stores copied finite phase snapshots in a bounded deque, and
``DelayedEngine`` advances phases with delayed coupling, optional external
forcing, and Rust acceleration when available. Constructors and step inputs
reject non-positive dimensions, non-finite scalars, and shape-mismatched arrays
before integration so delayed history never aliases invalid caller state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "DelayBuffer",
    "DelayedEngine",
]
FloatArray: TypeAlias = NDArray[np.float64]

_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")
_DelayBackend: TypeAlias = Callable[..., FloatArray]


def _load_rust_fn() -> _DelayBackend:
    """Load the Rust delayed-Kuramoto backend callable."""
    from spo_kernel import delayed_kuramoto_run_rust

    def _rust(
        phases: FloatArray,
        omegas: FloatArray,
        knm_flat: FloatArray,
        alpha_flat: FloatArray,
        n: int,
        zeta: float,
        psi: float,
        dt: float,
        delay_steps: int,
        n_steps: int,
    ) -> FloatArray:
        """Call the Rust time-delayed Kuramoto step kernel."""
        return np.asarray(
            delayed_kuramoto_run_rust(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                int(n),
                float(zeta),
                float(psi),
                float(dt),
                int(delay_steps),
                int(n_steps),
            ),
            dtype=np.float64,
        )

    return cast("_DelayBackend", _rust)


def _load_mojo_fn() -> _DelayBackend:
    # pragma: no cover — toolchain
    """Load the Mojo delayed-Kuramoto backend callable."""
    from ..experimental.accelerators.upde._delay_mojo import (
        _ensure_exe,
        delayed_kuramoto_run_mojo,
    )

    _ensure_exe()
    return delayed_kuramoto_run_mojo


def _load_julia_fn() -> _DelayBackend:
    # pragma: no cover — toolchain
    """Load the Julia delayed-Kuramoto backend callable."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._delay_julia import (
        delayed_kuramoto_run_julia,
    )

    return delayed_kuramoto_run_julia


def _load_go_fn() -> _DelayBackend:
    # pragma: no cover — toolchain
    """Load the Go delayed-Kuramoto backend callable."""
    from ..experimental.accelerators.upde._delay_go import (
        _load_lib,
        delayed_kuramoto_run_go,
    )

    _load_lib()
    return delayed_kuramoto_run_go


_LOADERS: dict[str, Callable[[], _DelayBackend]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, _DelayBackend] = {}


def _load_backend(name: str) -> _DelayBackend:
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


def _dispatch() -> _DelayBackend | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND, *AVAILABLE_BACKENDS]
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
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
        raise ValueError(f"{name} must be a positive integer, got {value!r}")
    return int(value)


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    value = float(value)
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    return value


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
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
    """Return the state as a validated finite array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_phase_output(value: object, *, n_oscillators: int) -> FloatArray:
    """Return the backend phase output matching the reference, else raise."""
    try:
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("delayed engine output must be a finite phase vector") from exc
    if arr.shape != (n_oscillators,):
        raise ValueError(
            f"delayed engine output shape {arr.shape} does not match ({n_oscillators},)"
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError("delayed engine output must contain only finite values")
    if np.any(arr < 0.0) or np.any(arr >= TWO_PI):
        raise ValueError("delayed engine output phases must be in [0, 2*pi)")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in arr.flat)


def _python_run(
    phases: FloatArray,
    omegas: FloatArray,
    knm_flat: FloatArray,
    alpha_flat: FloatArray,
    n: int,
    zeta: float,
    psi: float,
    dt: float,
    delay_steps: int,
    n_steps: int,
) -> FloatArray:
    """NumPy reference for the delayed Kuramoto integration.

    A ring buffer of ``delay_steps + 1`` snapshots supplies the delayed source
    phase ``θ_j(t − delay_steps·dt)``; the first ``delay_steps`` steps use the
    current snapshot (zero-delay warmup).
    """
    knm = knm_flat.reshape(n, n)
    alpha = alpha_flat.reshape(n, n)
    max_buf = delay_steps + 1
    history = np.zeros((max_buf, n), dtype=np.float64)
    p = phases.copy()
    for i in range(n_steps):
        ring = i % max_buf
        history[ring] = p
        if delay_steps > 0 and i >= delay_steps:
            delayed = history[(i - delay_steps) % max_buf]
        else:
            delayed = p
        diff = delayed[np.newaxis, :] - p[:, np.newaxis] - alpha
        coupling = np.sum(knm * np.sin(diff), axis=1)
        dtheta = omegas + coupling
        if zeta != 0.0:
            dtheta = dtheta + zeta * np.sin(psi - p)
        p = (p + dt * dtheta) % TWO_PI
    return p


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
        """Append a phase snapshot to the buffer.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        """
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        self._buffer.append(phases64.copy())

    def get_delayed(self, delay_steps: int) -> FloatArray | None:
        """Return phases from `delay_steps` ago, or None if not enough history.

        Parameters
        ----------
        delay_steps : int
            Number of steps in the past to retrieve from the delay buffer.

        Returns
        -------
        FloatArray | None
            The phase snapshot from ``delay_steps`` ago, or ``None`` if history is
            short.
        """
        delay = _validate_positive_int(delay_steps, name="delay_steps")
        if delay > len(self._buffer):
            return None
        return self._buffer[-delay]

    @property
    def length(self) -> int:
        """Number of snapshots currently stored.

        Returns
        -------
        int
            Number of snapshots currently stored.
        """
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
        """Return the configured discrete coupling delay.

        Returns
        -------
        int
            Return the configured discrete coupling delay.
        """
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
        """Advance one delayed Kuramoto timestep from validated state arrays.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        alpha : FloatArray | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        step_idx : int
            Zero-based index of the current step, used to address delayed coupling
            history.

        Returns
        -------
        FloatArray
            The phases after one delayed Kuramoto step, in ``[0, 2π)``.
        """
        del step_idx
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        alpha64: FloatArray
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
        """Run delayed Kuramoto integration for ``n_steps`` validated steps.

        Parameters
        ----------
        phases : FloatArray
            Oscillator phases in radians, shape ``(N,)``.
        omegas : FloatArray
            Natural frequencies in rad/s, shape ``(N,)``.
        knm : FloatArray
            Coupling matrix ``K_nm``, shape ``(N, N)``.
        zeta : float
            External drive strength ``ζ``.
        psi : float
            External drive reference phase ``Ψ`` in radians.
        alpha : FloatArray | None
            Phase-lag matrix in radians, shape ``(N, N)``, or ``None`` for no lag.
        n_steps : int
            Number of integration steps to run.

        Returns
        -------
        FloatArray
            The final phases after ``n_steps`` delayed Kuramoto steps.
        """
        n_steps = _validate_positive_int(n_steps, name="n_steps")
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        omegas64 = _validate_state_array(omegas, name="omegas", shape=(self._n,))
        knm64 = _validate_state_array(knm, name="knm", shape=(self._n, self._n))
        alpha64: FloatArray
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
        knm_flat = np.ascontiguousarray(knm64.ravel(), dtype=np.float64)
        alpha_flat = np.ascontiguousarray(alpha64.ravel(), dtype=np.float64)
        backend_fn = _dispatch()
        if backend_fn is not None:
            try:
                return _validate_phase_output(
                    backend_fn(
                        phases64,
                        omegas64,
                        knm_flat,
                        alpha_flat,
                        self._n,
                        zeta,
                        psi,
                        self._dt,
                        self._delay_steps,
                        n_steps,
                    ),
                    n_oscillators=self._n,
                )
            except (ImportError, RuntimeError, OSError, KeyError, ValueError):
                pass
        return _python_run(
            phases64,
            omegas64,
            knm_flat,
            alpha_flat,
            self._n,
            zeta,
            psi,
            self._dt,
            self._delay_steps,
            n_steps,
        )
