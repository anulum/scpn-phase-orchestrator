# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Partial Information Decomposition for phase groups

r"""Partial information decomposition (PID) about global synchronisation.

Decomposes two oscillator groups with a 5-backend fallback chain.

Model
-----
Williams & Beer 2010 (*Nonnegative Decomposition of Multivariate Information*,
arXiv:1004.2515) decompose the information two sources carry about a target into
redundant, unique, and synergistic parts. Estimating it needs a *distribution*,
so the input is a phase **history** ``(T, N)`` (``T`` timesteps, ``N``
oscillators). Each timestep is reduced to three circular observables:

* target ``Y_t`` — the global order-parameter phase ``∠⟨e^{iθ}⟩`` over all
  oscillators,
* source ``A_t`` — the group-A order-parameter phase,
* source ``B_t`` — the group-B order-parameter phase.

The three series are binned into ``n_bins`` equal-width phase bins and the joint
distribution is estimated over the ``T`` samples.

Decomposition
-------------
With the specific information ``I_spec(Y=y; S) = Σ_s p(s|y)·log[p(y|s)/p(y)]``:

    redundancy  I_red = Σ_y p(y)·min( I_spec(Y=y; A), I_spec(Y=y; B) )
    synergy     I_syn = MI(A,B; Y) − MI(A; Y) − MI(B; Y) + I_red

``I_red`` is the Williams & Beer ``I_min`` redundancy; the unique information of
each source is ``MI(S; Y) − I_red`` and ``MI(A; Y) = I_red + U_A`` holds by
construction. All terms are non-negative.

A single snapshot (``T = 1``) carries no distributional information, so every
component is ``0``; meaningful decomposition needs ``T ≥ 2``.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Integral
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor._julia_runtime import require_juliacall_main

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "redundancy",
    "synergy",
]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

_DEFAULT_BINS = 32
TAU = 2.0 * np.pi

_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")
# (redundancy, synergy) from a flattened (T, N) phase history.
PidTuple: TypeAlias = tuple[float, float]
PidBackend: TypeAlias = Callable[..., PidTuple]


def _load_rust_fn() -> PidBackend:
    """Load the Rust PID backend callable."""
    from spo_kernel import pid_decomposition_rust

    def _rust(
        phase_history_flat: FloatArray,
        t: int,
        n: int,
        group_a: IntArray,
        group_b: IntArray,
        n_bins: int,
    ) -> PidTuple:
        """Call the Rust PID kernel with contiguous float arrays."""
        red, syn = pid_decomposition_rust(
            np.ascontiguousarray(phase_history_flat.ravel(), dtype=np.float64),
            int(t),
            int(n),
            np.ascontiguousarray(group_a, dtype=np.int64),
            np.ascontiguousarray(group_b, dtype=np.int64),
            int(n_bins),
        )
        return float(red), float(syn)

    return cast("PidBackend", _rust)


def _load_mojo_fn() -> PidBackend:
    # pragma: no cover — toolchain
    """Load the Mojo PID backend callable."""
    from ..experimental.accelerators.monitor._pid_mojo import (
        _ensure_exe,
        pid_decomposition_mojo,
    )

    _ensure_exe()
    return pid_decomposition_mojo


def _load_julia_fn() -> PidBackend:
    # pragma: no cover — toolchain
    """Load the Julia PID backend callable."""
    require_juliacall_main()
    from ..experimental.accelerators.monitor._pid_julia import (
        pid_decomposition_julia,
    )

    return pid_decomposition_julia


def _load_go_fn() -> PidBackend:
    # pragma: no cover — toolchain
    """Load the Go PID backend callable."""
    from ..experimental.accelerators.monitor._pid_go import (
        _load_lib,
        pid_decomposition_go,
    )

    _load_lib()
    return pid_decomposition_go


_LOADERS: dict[str, Callable[[], PidBackend]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, PidBackend] = {}


def _load_backend(name: str) -> PidBackend:
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


def _dispatch() -> PidBackend | None:
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


def _validate_n_bins(value: object) -> int:
    """Return ``n_bins`` as an integer at least 2, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError("n_bins must be an integer greater than or equal to 2")
    if value < 2:
        raise ValueError("n_bins must be greater than or equal to 2")
    return int(value)


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)


def _contains_complex_alias(value: object) -> bool:
    """Return whether the value contains any complex-number alias."""
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)


def _is_string_like(value: object) -> bool:
    """Return whether ``value`` is a Python or NumPy string scalar."""
    return isinstance(value, (str, bytes, np.str_, np.bytes_))


def _is_numeric_string_alias(value: object) -> bool:
    """Return whether ``value`` is a string scalar parsable as a float."""
    if not _is_string_like(value):
        return False
    try:
        float(cast("str | bytes", value))
    except (TypeError, ValueError):
        return False
    return True


def _contains_numeric_string_alias(value: object) -> bool:
    """Return whether the value contains numeric-string aliases."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return False
    if raw.dtype.kind not in {"O", "S", "U"}:
        return False
    saw_string = False
    for item in raw.astype(object, copy=False).flat:
        if not _is_string_like(item):
            continue
        saw_string = True
        if not _is_numeric_string_alias(item):
            return False
    return saw_string


def _has_complex_payload(value: object) -> bool:
    """Return whether the value carries a complex-number payload."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return _contains_complex_alias(value)
    return bool(np.iscomplexobj(raw) or _contains_complex_alias(value))


def _validate_phase_history(value: object) -> FloatArray:
    """Return the phase history as a validated 2-D finite array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError("phases must not contain boolean values")
    raw = np.asarray(value)
    if _has_complex_payload(value):
        raise ValueError("phases must contain real-valued samples")
    if _contains_numeric_string_alias(raw):
        raise ValueError("phases must not contain numeric-string aliases")
    try:
        history = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite (T, N) phase history") from exc
    if history.ndim == 1:
        history = history.reshape(1, -1)
    if history.ndim != 2:
        raise ValueError("phases must be a finite (T, N) phase history")
    if not np.all(np.isfinite(history)):
        raise ValueError("phases must contain only finite values")
    return np.ascontiguousarray(history, dtype=np.float64)


def _validate_group_indices(value: object, *, name: str, n_osc: int) -> IntArray:
    """Return the validated source-group indices, else raise ``ValueError``."""
    raw = np.asarray(value)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1-D integer index array")
    if _contains_boolean_alias(value):
        raise TypeError(f"{name} must contain integer indices, not booleans")
    if _has_complex_payload(value):
        raise TypeError(f"{name} must contain real integer indices")
    if _contains_numeric_string_alias(raw):
        raise TypeError(f"{name} must not contain numeric-string aliases")
    try:
        numeric = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain integer indices") from exc
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} must contain finite integer indices")
    if not np.all(numeric == np.floor(numeric)):
        raise TypeError(f"{name} must contain integer indices")
    indices = numeric.astype(np.int64)
    if indices.size > 0 and (np.any(indices < 0) or np.any(indices >= n_osc)):
        raise IndexError(f"{name} indices must be within [0, {n_osc})")
    return np.ascontiguousarray(indices, dtype=np.int64)


def _validate_pid_scalar(value: object, *, name: str) -> float:
    """Return a backend PID component as a validated non-negative float, else raise."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must not be a boolean value")
    if _has_complex_payload(value):
        raise ValueError(f"{name} must be a real scalar")
    if _contains_numeric_string_alias(value):
        raise ValueError(f"{name} must not contain numeric-string aliases")
    result = float(cast("float", value))
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _bin_series(series: FloatArray, n_bins: int) -> IntArray:
    """Bin circular angles into ``[0, n_bins)`` equal-width phase bins."""
    wrapped = np.mod(series, TAU)
    idx = np.floor(wrapped / (TAU / n_bins)).astype(np.int64)
    return np.clip(idx, 0, n_bins - 1)


def _group_phase_series(history: FloatArray, group: IntArray) -> FloatArray:
    """Order-parameter phase of a group at each timestep."""
    z = np.exp(1j * history[:, group]).mean(axis=1)
    return cast("FloatArray", np.angle(z))


def _mutual_information(
    joint: FloatArray, marg_x: FloatArray, marg_y: FloatArray
) -> float:
    """``MI(X; Y)`` from a joint count matrix and the two marginal counts."""
    total = float(marg_y.sum())
    if total <= 0.0:
        return 0.0
    mi = 0.0
    rows, cols = joint.shape
    for x in range(rows):
        if marg_x[x] <= 0.0:
            continue
        for y in range(cols):
            cxy = joint[x, y]
            if cxy <= 0.0 or marg_y[y] <= 0.0:
                continue
            p_xy = cxy / total
            mi += p_xy * np.log(p_xy / ((marg_x[x] / total) * (marg_y[y] / total)))
    return max(0.0, mi)


def _i_min_redundancy(
    cay: FloatArray,
    cby: FloatArray,
    ca: FloatArray,
    cb: FloatArray,
    cy: FloatArray,
) -> float:
    """Williams & Beer ``I_min`` redundancy from joint/marginal counts."""
    total = float(cy.sum())
    if total <= 0.0:
        return 0.0
    n_bins = cy.shape[0]
    i_red = 0.0
    for y in range(n_bins):
        if cy[y] <= 0.0:
            continue
        p_y = cy[y] / total
        ispec_a = 0.0
        for x in range(n_bins):
            if cay[x, y] <= 0.0 or ca[x] <= 0.0:
                continue
            p_a_given_y = cay[x, y] / cy[y]
            p_y_given_a = cay[x, y] / ca[x]
            ispec_a += p_a_given_y * np.log(p_y_given_a / p_y)
        ispec_b = 0.0
        for x in range(n_bins):
            if cby[x, y] <= 0.0 or cb[x] <= 0.0:
                continue
            p_b_given_y = cby[x, y] / cy[y]
            p_y_given_b = cby[x, y] / cb[x]
            ispec_b += p_b_given_y * np.log(p_y_given_b / p_y)
        i_red += p_y * min(ispec_a, ispec_b)
    return max(0.0, i_red)


def _pid_python(
    phase_history_flat: FloatArray,
    t: int,
    n: int,
    group_a: IntArray,
    group_b: IntArray,
    n_bins: int,
) -> PidTuple:
    """NumPy reference for the time-series Williams & Beer PID."""
    if t == 0 or group_a.size == 0 or group_b.size == 0:
        return 0.0, 0.0
    history = phase_history_flat.reshape(t, n)
    y = _bin_series(np.angle(np.exp(1j * history).mean(axis=1)), n_bins)
    a = _bin_series(_group_phase_series(history, group_a), n_bins)
    b = _bin_series(_group_phase_series(history, group_b), n_bins)

    cy = np.bincount(y, minlength=n_bins).astype(np.float64)
    ca = np.bincount(a, minlength=n_bins).astype(np.float64)
    cb = np.bincount(b, minlength=n_bins).astype(np.float64)
    cay = np.zeros((n_bins, n_bins), dtype=np.float64)
    cby = np.zeros((n_bins, n_bins), dtype=np.float64)
    np.add.at(cay, (a, y), 1.0)
    np.add.at(cby, (b, y), 1.0)
    ab = a * n_bins + b
    cab = np.bincount(ab, minlength=n_bins * n_bins).astype(np.float64)
    caby = np.zeros((n_bins * n_bins, n_bins), dtype=np.float64)
    np.add.at(caby, (ab, y), 1.0)

    mi_a = _mutual_information(cay, ca, cy)
    mi_b = _mutual_information(cby, cb, cy)
    mi_ab = _mutual_information(caby, cab, cy)
    i_red = _i_min_redundancy(cay, cby, ca, cb, cy)
    synergy_value = max(0.0, mi_ab - mi_a - mi_b + i_red)
    return i_red, synergy_value


def _decompose(
    phases: object,
    group_a: object,
    group_b: object,
    n_bins: object,
) -> PidTuple:
    """Validate inputs and dispatch the PID primitive to the fastest backend."""
    bin_count = _validate_n_bins(n_bins)
    history = _validate_phase_history(phases)
    t, n = int(history.shape[0]), int(history.shape[1])
    if n == 0:
        return 0.0, 0.0
    group_a_idx = _validate_group_indices(group_a, name="group_a", n_osc=n)
    group_b_idx = _validate_group_indices(group_b, name="group_b", n_osc=n)
    if group_a_idx.size == 0 or group_b_idx.size == 0:
        return 0.0, 0.0

    flat = np.ascontiguousarray(history.ravel(), dtype=np.float64)
    backend_fn = _dispatch()
    if backend_fn is not None:
        try:
            red, syn = backend_fn(flat, t, n, group_a_idx, group_b_idx, bin_count)
            return (
                _validate_pid_scalar(red, name="redundancy"),
                _validate_pid_scalar(syn, name="synergy"),
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            pass
    return _pid_python(flat, t, n, group_a_idx, group_b_idx, bin_count)


def redundancy(
    phases: FloatArray,
    group_a: list[int] | IntArray,
    group_b: list[int] | IntArray,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """Redundant information both groups share about the global phase.

    ``I_red = Σ_y p(y)·min(I_spec(Y=y; A), I_spec(Y=y; B))`` (Williams & Beer
    2010 ``I_min``). ``phases`` is a ``(T, N)`` phase history.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    group_a : list[int] | IntArray
        Indices of the first oscillator group.
    group_b : list[int] | IntArray
        Indices of the second oscillator group.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        The redundant information the groups share about the global phase.
    """
    red, _ = _decompose(phases, group_a, group_b, n_bins)
    return _validate_pid_scalar(red, name="redundancy")


def synergy(
    phases: FloatArray,
    group_a: list[int] | IntArray,
    group_b: list[int] | IntArray,
    n_bins: int = _DEFAULT_BINS,
) -> float:
    """Synergistic information present only in the joint ``(A, B)``.

    ``I_syn = MI(A,B; Y) − MI(A; Y) − MI(B; Y) + I_red``. Positive synergy means
    the combined group carries information about the global state that neither
    subgroup carries alone. ``phases`` is a ``(T, N)`` phase history.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    group_a : list[int] | IntArray
        Indices of the first oscillator group.
    group_b : list[int] | IntArray
        Indices of the second oscillator group.
    n_bins : int
        Number of histogram bins.

    Returns
    -------
    float
        The synergistic information present only in the joint ``(A, B)``.
    """
    _, syn = _decompose(phases, group_a, group_b, n_bins)
    return _validate_pid_scalar(syn, name="synergy")
