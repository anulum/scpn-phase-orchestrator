# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection

"""Chimera state detection with a 5-backend fallback chain.

Kuramoto & Battogtokh 2002, Nonlinear Phenomena in Complex Systems
5:380–385. An oscillator ``i`` is coherent when its local order
parameter ``R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|`` exceeds the
coherence threshold, incoherent when it falls below the incoherence
threshold. The chimera index is the fraction of oscillators that sit
in the boundary band in between.

Compute surface:

* :func:`local_order_parameter` — ``(N,)`` per-oscillator ``R_i`` vector;
  the coupling diagonal must be zero so self-coupling is never counted as a
  neighbour.
* :func:`detect_chimera` — classification wrapper returning
  :class:`ChimeraState`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import import_module
from numbers import Integral, Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
ChimeraBackendFn: TypeAlias = Callable[[FloatArray, FloatArray, int], FloatArray]
RustChimeraFn: TypeAlias = Callable[
    [FloatArray, FloatArray, int],
    tuple[object, object, object, object],
]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "ChimeraState",
    "detect_chimera",
    "local_order_parameter",
]


# Kuramoto & Battogtokh 2002, Nonlinear Phenom. Complex Syst. 5:380-385
_COHERENT_THRESHOLD = 0.7
_INCOHERENT_THRESHOLD = 0.3


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> ChimeraBackendFn:
    """Load the Rust chimera-detection backend callable."""
    kernel = import_module("spo_kernel")
    detect_chimera_rust = cast("RustChimeraFn", vars(kernel)["detect_chimera_rust"])

    def _rust(phases: FloatArray, knm_flat: FloatArray, n: int) -> FloatArray:
        """Call the Rust chimera-detection kernel with contiguous float arrays."""
        _coh, _incoh, _ci, local = detect_chimera_rust(
            np.ascontiguousarray(phases, dtype=np.float64),
            np.ascontiguousarray(knm_flat, dtype=np.float64),
            int(n),
        )
        return cast("FloatArray", np.asarray(local))

    return _rust


def _load_mojo_fn() -> ChimeraBackendFn:
    """Load the Mojo chimera-detection backend callable."""
    from ..experimental.accelerators.monitor._chimera_mojo import (
        _ensure_exe,
        local_order_parameter_mojo,
    )

    _ensure_exe()
    return cast("ChimeraBackendFn", local_order_parameter_mojo)


def _load_julia_fn() -> ChimeraBackendFn:
    """Load the Julia chimera-detection backend callable."""
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._chimera_julia import (
        local_order_parameter_julia,
    )

    return cast("ChimeraBackendFn", local_order_parameter_julia)


def _load_go_fn() -> ChimeraBackendFn:
    """Load the Go chimera-detection backend callable."""
    from ..experimental.accelerators.monitor._chimera_go import (
        _load_lib,
        local_order_parameter_go,
    )

    _load_lib()
    return cast("ChimeraBackendFn", local_order_parameter_go)


_LOADERS: dict[str, Callable[[], ChimeraBackendFn]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, ChimeraBackendFn] = {}


def _load_backend(name: str) -> ChimeraBackendFn:
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


def _dispatch() -> ChimeraBackendFn | None:
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


@dataclass(frozen=True)
class ChimeraState:
    """Chimera detection result: coherent/incoherent oscillator partitions and index."""

    coherent_indices: list[int] = field(default_factory=list)
    incoherent_indices: list[int] = field(default_factory=list)
    chimera_index: float = 0.0

    def __post_init__(self) -> None:
        coherent = _validate_index_list(self.coherent_indices, name="coherent_indices")
        incoherent = _validate_index_list(
            self.incoherent_indices,
            name="incoherent_indices",
        )
        overlap = set(coherent).intersection(incoherent)
        if overlap:
            raise ValueError("coherent_indices and incoherent_indices must be disjoint")
        if isinstance(self.chimera_index, (bool, np.bool_)) or not isinstance(
            self.chimera_index,
            Real,
        ):
            raise ValueError("chimera_index must be a finite real scalar in [0, 1]")
        chimera_index = float(self.chimera_index)
        if not np.isfinite(chimera_index) or not 0.0 <= chimera_index <= 1.0:
            raise ValueError("chimera_index must be finite and lie in [0, 1]")
        object.__setattr__(self, "coherent_indices", coherent)
        object.__setattr__(self, "incoherent_indices", incoherent)
        object.__setattr__(self, "chimera_index", chimera_index)


def _validate_index_list(indices: object, *, name: str) -> list[int]:
    """Return a validated list of in-range integer indices, else raise."""
    if isinstance(indices, (str, bytes)) or not isinstance(indices, Iterable):
        raise ValueError(f"{name} must be a sequence of non-negative integer indices")
    values = list(indices)
    normalised: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(f"{name} must contain only integer indices")
        index = int(value)
        if index < 0:
            raise ValueError(f"{name} must contain only non-negative indices")
        normalised.append(index)
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} must not contain duplicate indices")
    return normalised


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


def _has_complex_payload(value: object) -> bool:
    """Return whether the value carries a complex-number payload."""
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError):
        return _contains_complex_alias(value)
    return bool(np.iscomplexobj(raw) or _contains_complex_alias(value))


def _validate_chimera_inputs(
    phases: object,
    knm: object,
) -> tuple[FloatArray, FloatArray]:
    """Return the validated phase array and group indices for detection."""
    raw_phases = np.asarray(phases)
    if _contains_boolean_alias(raw_phases):
        raise ValueError("phases must not contain boolean values")
    if _has_complex_payload(phases):
        raise ValueError("phases must contain real-valued phase samples")
    try:
        phases_array = raw_phases.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a finite one-dimensional array") from exc
    if phases_array.ndim != 1:
        raise ValueError(f"phases shape {phases_array.shape} must be one-dimensional")
    if not np.all(np.isfinite(phases_array)):
        raise ValueError("phases must contain only finite values")

    n = int(phases_array.size)
    raw_knm = np.asarray(knm)
    if _contains_boolean_alias(raw_knm):
        raise ValueError("knm must not contain boolean values")
    if _has_complex_payload(knm):
        raise ValueError("knm must contain real-valued couplings")
    try:
        knm_array = raw_knm.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("knm must be a finite square coupling matrix") from exc
    if knm_array.shape != (n, n):
        raise ValueError(f"knm shape {knm_array.shape} does not match {(n, n)}")
    if not np.all(np.isfinite(knm_array)):
        raise ValueError("knm must contain only finite values")
    if not np.allclose(np.diag(knm_array), 0.0, rtol=0.0, atol=1e-15):
        raise ValueError("knm self-coupling diagonal must be zero")
    return (
        np.ascontiguousarray(phases_array, dtype=np.float64),
        np.ascontiguousarray(knm_array, dtype=np.float64),
    )


def _validate_local_order(value: object, *, n_oscillators: int) -> FloatArray:
    """Return backend local order parameters matching the reference, else raise."""
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError("local order parameter output must not contain boolean values")
    if _has_complex_payload(value):
        raise ValueError("local order parameter output must contain real values")
    try:
        local = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("local order parameter output must be numeric") from exc
    if local.shape != (n_oscillators,):
        raise ValueError(
            f"local order parameter shape {local.shape} does not match "
            f"({n_oscillators},)"
        )
    if not np.all(np.isfinite(local)):
        raise ValueError("local order parameter must contain only finite values")
    tolerance = 1e-12
    if np.any(local < -tolerance) or np.any(local > 1.0 + tolerance):
        raise ValueError("local order parameter must lie in [0, 1]")
    return np.ascontiguousarray(np.clip(local, 0.0, 1.0), dtype=np.float64)


def local_order_parameter(phases: FloatArray, knm: FloatArray) -> FloatArray:
    """Per-oscillator local order parameter.

    ``R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|`` with ``N(i) =
    {j : K_ij > 0}`` and a required zero self-coupling diagonal. Zero
    when oscillator ``i`` has no neighbours.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    FloatArray
        The per-oscillator local order parameter, shape ``(N,)``.
    """
    phases, knm = _validate_chimera_inputs(phases, knm)
    n = int(phases.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    knm_flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)

    backend_fn = _dispatch()
    if backend_fn is not None:
        try:
            return _validate_local_order(
                backend_fn(phases, knm_flat, n), n_oscillators=n
            )
        except (ImportError, RuntimeError, OSError, KeyError):
            backend_fn = None

    r_local: FloatArray = np.zeros(n, dtype=np.float64)
    knm_2d = knm_flat.reshape(n, n)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    unit = np.exp(1j * diffs)
    for i in range(n):
        mask = knm_2d[i] > 0
        if not np.any(mask):
            r_local[i] = 0.0
            continue
        r_local[i] = float(np.abs(np.mean(unit[i, mask])))
    return _validate_local_order(r_local, n_oscillators=n)


def detect_chimera(phases: FloatArray, knm: FloatArray) -> ChimeraState:
    """Detect chimera states in a Kuramoto network.

    Parameters
    ----------
    phases : FloatArray
        ``(N,)`` oscillator phases.
    knm : FloatArray
        ``(N, N)`` coupling matrix. ``K_ij > 0`` defines neighbours; diagonal
        self-coupling must be zero.

    Returns
    -------
    ChimeraState
        :class:`ChimeraState` with coherent / incoherent index lists and the
        boundary-fraction chimera index.
    """
    phases, knm = _validate_chimera_inputs(phases, knm)
    n = int(phases.size)
    if n == 0:
        return ChimeraState()

    r_local = local_order_parameter(phases, knm)
    coherent = [int(i) for i in range(n) if r_local[i] > _COHERENT_THRESHOLD]
    incoherent = [int(i) for i in range(n) if r_local[i] < _INCOHERENT_THRESHOLD]
    boundary = n - len(coherent) - len(incoherent)
    chimera_index = boundary / n if n > 0 else 0.0
    return ChimeraState(
        coherent_indices=coherent,
        incoherent_indices=incoherent,
        chimera_index=chimera_index,
    )
