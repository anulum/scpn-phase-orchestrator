# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection

"""Chimera state detection with a 5-backend fallback chain.

An oscillator ``i`` is coherent when its local order
parameter ``R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|`` exceeds the
coherence threshold, incoherent when it falls below the incoherence
threshold. The chimera index is the fraction of oscillators that sit
in the boundary band in between. These implementation thresholds are an
instantaneous adjacency diagnostic, not a dynamical chimera certificate or
the weighted nonlocal field of Kuramoto & Battogtokh (2002).

Measurements must be plain real numbers before conversion. Boolean, text,
complex and temporal values are refused; real numeric object arrays remain
supported.

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

from ._chimera_validation import _measurement_array

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


# Implementation thresholds for the instantaneous boundary-fraction diagnostic.
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
    from ..experimental.accelerators.monitor._chimera_julia import (
        _ensure,
        local_order_parameter_julia,
    )

    _ensure()
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
    """Resolve active and available backends in the declared preference order."""
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


def _dispatch(backend: str | None = None) -> ChimeraBackendFn | None:
    """Resolve an explicit owner strictly, or use the automatic fallback chain."""
    if backend is not None:
        if not isinstance(backend, str) or backend not in _BACKEND_NAMES:
            raise ValueError(f"backend must be one of {_BACKEND_NAMES}, or None")
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError) as exc:
            raise ImportError(f"chimera backend {backend!r} is unavailable") from exc
    # Import-time resolution has already cached each admitted implementation.
    # Use its selected default; computation faults remain visible to callers.
    if ACTIVE_BACKEND == "python":
        return None
    return _load_backend(ACTIVE_BACKEND)


@dataclass(frozen=True)
class ChimeraState:
    """Chimera detection result: coherent/incoherent oscillator partitions and index."""

    coherent_indices: list[int] = field(default_factory=list)
    incoherent_indices: list[int] = field(default_factory=list)
    chimera_index: float = 0.0

    def __post_init__(self) -> None:
        """Validate and copy result fields at construction."""
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
    """Return copied distinct nonnegative integer indices, else raise."""
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


def _validate_chimera_inputs(
    phases: object,
    knm: object,
) -> tuple[FloatArray, FloatArray]:
    """Return validated phase and coupling arrays for detection."""
    phases_array = _measurement_array(
        phases,
        name="phases",
        conversion_error="phases must be a finite one-dimensional array",
    )
    if phases_array.ndim != 1:
        raise ValueError(f"phases shape {phases_array.shape} must be one-dimensional")
    if not np.all(np.isfinite(phases_array)):
        raise ValueError("phases must contain only finite values")

    n = int(phases_array.size)
    knm_array = _measurement_array(
        knm,
        name="knm",
        conversion_error="knm must be a finite square coupling matrix",
    )
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
    local = _measurement_array(
        value,
        name="local order parameter output",
        conversion_error="local order parameter output must be numeric",
    )
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


def local_order_parameter(
    phases: FloatArray, knm: FloatArray, *, backend: str | None = None
) -> FloatArray:
    """Per-oscillator local order parameter.

    ``R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|`` with ``N(i) =
    {j : j != i and K_ij > 0}`` and a required zero self-coupling diagonal
    (absolute tolerance ``1e-15``). Admitted diagonal residue is ignored. Zero
    when oscillator ``i`` has no neighbours.

    Parameters
    ----------
    phases : FloatArray
        Oscillator phases in radians, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    backend : str or None, optional
        Explicit ``python``, ``rust``, ``go``, ``julia`` or ``mojo`` owner.
        An unavailable explicit owner raises ImportError. None selects the
        automatic chain. Computation and output-validation failures propagate.

    Returns
    -------
    FloatArray
        The per-oscillator local order parameter, shape ``(N,)``.
    """
    phases, knm = _validate_chimera_inputs(phases, knm)
    n = int(phases.size)
    backend_fn = _dispatch(backend)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    knm_flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)

    if backend_fn is not None:
        return _validate_local_order(backend_fn(phases, knm_flat, n), n_oscillators=n)

    r_local: FloatArray = np.zeros(n, dtype=np.float64)
    knm_2d = knm_flat.reshape(n, n)
    # Factoring out exp(-i*theta_i) preserves the magnitude and avoids
    # subtracting finite unwrapped phases whose difference could overflow.
    unit = np.cos(phases) + 1j * np.sin(phases)
    for i in range(n):
        mask = knm_2d[i] > 0
        mask[i] = False
        if not np.any(mask):
            r_local[i] = 0.0
            continue
        r_local[i] = float(np.abs(np.mean(unit[mask])))
    return _validate_local_order(r_local, n_oscillators=n)


def detect_chimera(
    phases: FloatArray, knm: FloatArray, *, backend: str | None = None
) -> ChimeraState:
    """Classify instantaneous local phase coherence in a Kuramoto network.

    Parameters
    ----------
    phases : FloatArray
        ``(N,)`` oscillator phases.
    knm : FloatArray
        ``(N, N)`` coupling matrix. ``K_ij > 0`` defines neighbours; diagonal
        self-coupling must be zero.
    backend : str or None, optional
        Strict named owner or automatic selection, as in local_order_parameter.

    Returns
    -------
    ChimeraState
        :class:`ChimeraState` with coherent / incoherent index lists and the
        boundary-fraction chimera index.
    """
    r_local = local_order_parameter(phases, knm, backend=backend)
    n = int(r_local.size)
    if n == 0:
        return ChimeraState()

    coherent = [int(i) for i in range(n) if r_local[i] > _COHERENT_THRESHOLD]
    incoherent = [int(i) for i in range(n) if r_local[i] < _INCOHERENT_THRESHOLD]
    boundary = n - len(coherent) - len(incoherent)
    chimera_index = boundary / n
    return ChimeraState(
        coherent_indices=coherent,
        incoherent_indices=incoherent,
        chimera_index=chimera_index,
    )
