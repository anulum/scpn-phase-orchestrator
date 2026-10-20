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
reject non-positive dimensions, non-finite scalars, shape-mismatched arrays,
and boolean or numeric-string aliases before integration so delayed history
never aliases invalid caller state.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from importlib import import_module
from math import isfinite
from numbers import Integral, Real
from sys import maxsize
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator._compat import TWO_PI
from scpn_phase_orchestrator.upde._julia_runtime import require_juliacall_main
from scpn_phase_orchestrator.upde._phase_wrap import wrap_phases

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "DelayBuffer",
    "DelayedEngine",
]
FloatArray: TypeAlias = NDArray[np.float64]

_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")
_DelayBackend: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        float,
        float,
        float,
        int,
        int,
    ],
    FloatArray,
]


def _load_rust_fn() -> _DelayBackend:
    """Load the Rust delayed-Kuramoto backend callable."""
    native = getattr(import_module("spo_kernel"), "delayed_kuramoto_run_rust", None)
    if not callable(native):
        raise ImportError("spo_kernel has no callable delayed_kuramoto_run_rust")
    delayed_kuramoto_run_rust = cast("_DelayBackend", native)

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

    return _rust


def _load_mojo_fn() -> _DelayBackend:
    """Load the Mojo delayed-Kuramoto backend callable."""
    from ..experimental.accelerators.upde._delay_mojo import (
        _ensure_exe,
        delayed_kuramoto_run_mojo,
    )

    _ensure_exe()
    return delayed_kuramoto_run_mojo


def _load_julia_fn() -> _DelayBackend:
    """Load the Julia delayed-Kuramoto backend callable."""
    require_juliacall_main()

    from ..experimental.accelerators.upde._delay_julia import (
        _ensure,
        delayed_kuramoto_run_julia,
    )

    _ensure()
    return delayed_kuramoto_run_julia


def _load_go_fn() -> _DelayBackend:
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
    message = f"{name} must be a positive integer fitting a platform index"
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(message)
    result = int(value)
    if not 1 <= result <= maxsize:
        raise ValueError(message)
    return result


def _validate_positive_float(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite float, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite positive real") from exc
    if not isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a finite positive real, got {value!r}")
    return value


def _validate_finite_float(value: object, *, name: str) -> float:
    """Return ``value`` as a finite float, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    try:
        value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite real") from exc
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
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    try:
        require_real_values(value, name=name, allow_object=True)
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be a finite float array") from exc
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _validate_phase_output(value: object, *, n_oscillators: int) -> FloatArray:
    """Return the backend phase output matching the reference, else raise."""
    if _contains_numeric_string_alias(value):
        raise ValueError(
            "delayed engine output must not contain numeric-string aliases"
        )
    try:
        require_real_values(value, name="delayed engine output", allow_object=True)
        arr = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
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


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string accepted by ``float``."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        float(value)
    except ValueError:
        return False
    return True


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` contains a numeric-string alias."""
    try:
        arr = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(_is_numeric_string_alias(item) for item in arr.flat)


def _euler_step(
    phases: FloatArray,
    delayed: FloatArray,
    omegas: FloatArray,
    knm: FloatArray,
    alpha: FloatArray,
    zeta: float,
    psi: float,
    dt: float,
) -> FloatArray:
    """Advance the simultaneous delayed recurrence, refusing nonfinite arithmetic."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            diff = delayed[np.newaxis, :] - phases[:, np.newaxis] - alpha
            derivative = omegas + np.sum(knm * np.sin(diff), axis=1)
            if zeta != 0.0:
                derivative = derivative + zeta * np.sin(psi - phases)
            result = wrap_phases(phases + dt * derivative)
    except FloatingPointError as exc:
        raise ValueError("delayed integration produced nonfinite phases") from exc
    return result


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
    max_buf = min(delay_steps, n_steps) + 1
    history = np.zeros((max_buf, n), dtype=np.float64)
    p = phases.copy()
    for i in range(n_steps):
        ring = i % max_buf
        history[ring] = p
        if delay_steps > 0 and i >= delay_steps:
            delayed = history[(i - delay_steps) % max_buf]
        else:
            delayed = p
        p = _euler_step(p, delayed, omegas, knm, alpha, zeta, psi, dt)
    return p


class DelayBuffer:
    """Circular buffer storing phase history for delayed coupling.

    Parameters
    ----------
    n_oscillators : int
        Positive number of phases per snapshot.
    max_delay_steps : int
        Positive maximum number of retained snapshots. Lookup index one denotes
        the latest snapshot; pushing and retrieving both preserve independent
        storage.
    """

    def __init__(self, n_oscillators: int, max_delay_steps: int) -> None:
        """Create a bounded history for positive oscillator and snapshot counts."""
        self._n = _validate_positive_int(n_oscillators, name="n_oscillators")
        self._max = _validate_positive_int(max_delay_steps, name="max_delay_steps")
        self._buffer: deque[FloatArray] = deque(maxlen=self._max)

    def push(self, phases: FloatArray) -> None:
        """Append a phase snapshot to the buffer.

        Parameters
        ----------
        phases : FloatArray
            Plain real oscillator phases in radians, shape ``(N,)``.
            Text, boolean, complex and temporal measurement aliases are rejected.
        """
        phases64 = _validate_state_array(phases, name="phases", shape=(self._n,))
        self._buffer.append(phases64.copy())

    def get_delayed(self, delay_steps: int) -> FloatArray | None:
        """Return an indexed snapshot copy, or None if history is too short.

        Parameters
        ----------
        delay_steps : int
            Positive one-based snapshot index; one retrieves the latest entry.

        Returns
        -------
        FloatArray | None
            An independent copy of the indexed snapshot, or ``None`` if history
            is short. Index one denotes the latest stored snapshot.
        """
        delay = _validate_positive_int(delay_steps, name="delay_steps")
        if delay > len(self._buffer):
            return None
        return self._buffer[-delay].copy()

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

    Parameters
    ----------
    n_oscillators : int
        Positive oscillator count.
    dt : float
        Positive finite Euler timestep, in seconds when frequencies are rad/s.
    delay_steps : int
        Positive discrete source delay. The first ``delay_steps`` accepted
        steps use their current source phases while history accumulates.

    Notes
    -----
    Coupling uses ``sum_j K_ij*sin(theta_j(t-tau)-theta_i(t)-alpha_ij)``
    with natural frequency and optional ``zeta*sin(psi-theta_i)`` drive.
    Stateful ``step`` calls preserve history; every ``run`` starts a new batch.
    """

    def __init__(self, n_oscillators: int, dt: float, delay_steps: int = 1) -> None:
        """Configure a positive timestep and discrete source-phase delay."""
        self._n = _validate_positive_int(n_oscillators, name="n_oscillators")
        self._dt = _validate_positive_float(dt, name="dt")
        self._delay_steps = _validate_positive_int(delay_steps, name="delay_steps")
        self._buffer: deque[FloatArray] = deque(maxlen=self._delay_steps)

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
            Plain real oscillator phases in radians, shape ``(N,)``.
            Text, boolean, complex and temporal measurement aliases are rejected.
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
            Legacy argument retained for call compatibility. History advances
            with accepted calls, independently of this value.

        Returns
        -------
        FloatArray
            The phases after one delayed Kuramoto step, in ``[0, 2π)``.

        Raises
        ------
        ValueError
            If inputs or computed phases are invalid. A refused step does not
            append history; the next accepted step uses the prior history.
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
        delayed = (
            self._buffer[-self._delay_steps]
            if len(self._buffer) >= self._delay_steps
            else phases64
        )
        step_out = _euler_step(
            phases64, delayed, omegas64, knm64, alpha64, zeta, psi, self._dt
        )
        self._buffer.append(phases64.copy())
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
        *,
        backend: str = "auto",
    ) -> FloatArray:
        """Run delayed Kuramoto integration for ``n_steps`` validated steps.

        Parameters
        ----------
        phases : FloatArray
            Plain real oscillator phases in radians, shape ``(N,)``.
            Text, boolean, complex and temporal measurement aliases are rejected.
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
        backend : str
            ``auto`` preserves fastest-first selection and Python fallback.
            An explicit ``python``, ``rust``, ``go``, ``julia`` or ``mojo``
            requires that owner and propagates its refusal without fallback.

        Returns
        -------
        FloatArray
            Independent final phases from a fresh batch history. This does not
            consume or extend the history accumulated by ``step``.

        Raises
        ------
        ValueError
            If controls, arrays, computed phases or the backend name are invalid.
        ImportError
            If an explicitly selected native owner cannot be loaded.
        """
        if not isinstance(backend, str) or backend not in ("auto", *_BACKEND_NAMES):
            raise ValueError(f"unknown delay backend: {backend!r}")
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
        if backend == "auto":
            backend_fn = _dispatch()
        elif backend == "python":
            backend_fn = None
        else:
            try:
                backend_fn = _load_backend(backend)
            except (ImportError, RuntimeError, OSError, KeyError) as exc:
                raise ImportError(f"delay backend {backend!r} is unavailable") from exc
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
                        min(self._delay_steps, n_steps),
                        n_steps,
                    ),
                    n_oscillators=self._n,
                )
            except (ImportError, RuntimeError, OSError, KeyError, ValueError):
                if backend != "auto":
                    raise
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
