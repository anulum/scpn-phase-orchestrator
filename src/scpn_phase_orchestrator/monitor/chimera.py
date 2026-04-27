# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection

"""Chimera state detection with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Kuramoto & Battogtokh 2002, Nonlinear Phenomena in Complex Systems
5:380–385. An oscillator ``i`` is coherent when its local order
parameter ``R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|`` exceeds the
coherence threshold, incoherent when it falls below the incoherence
threshold. The chimera index is the fraction of oscillators that sit
in the boundary band in between.

Compute surface:

* :func:`local_order_parameter` — ``(N,)`` per-oscillator ``R_i`` vector.
* :func:`detect_chimera` — classification wrapper returning
  :class:`ChimeraState`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

import numpy as np
from numpy.typing import NDArray

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


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import detect_chimera_rust

    def _rust(phases: NDArray, knm_flat: NDArray, n: int) -> NDArray:
        _coh, _incoh, _ci, local = detect_chimera_rust(
            np.ascontiguousarray(phases, dtype=np.float64),
            np.ascontiguousarray(knm_flat, dtype=np.float64),
            int(n),
        )
        return np.asarray(local, dtype=np.float64)

    return cast("Callable[..., NDArray]", _rust)


def _load_mojo_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._chimera_mojo import (
        _ensure_exe,
        local_order_parameter_mojo,
    )

    _ensure_exe()
    return local_order_parameter_mojo


def _load_julia_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._chimera_julia import (
        local_order_parameter_julia,
    )

    return local_order_parameter_julia


def _load_go_fn() -> Callable[..., NDArray]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._chimera_go import (
        local_order_parameter_go,
    )

    return local_order_parameter_go


_LOADERS: dict[str, Callable[[], Callable[..., NDArray]]] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}


def _resolve_backends() -> tuple[str, list[str]]:
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _LOADERS[name]()
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch() -> Callable[..., NDArray] | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()


@dataclass(frozen=True)
class ChimeraState:
    """Chimera detection result: coherent/incoherent oscillator partitions and index."""

    coherent_indices: list[int] = field(default_factory=list)
    incoherent_indices: list[int] = field(default_factory=list)
    chimera_index: float = 0.0


def local_order_parameter(phases: NDArray, knm: NDArray) -> NDArray:
    """Per-oscillator local order parameter.

    ``R_i = |⟨exp(i(θ_j − θ_i))⟩_{j ∈ N(i)}|`` with ``N(i) =
    {j : K_ij > 0}``. Zero when oscillator ``i`` has no neighbours.
    """
    phases = np.asarray(phases, dtype=np.float64)
    knm = np.asarray(knm, dtype=np.float64)
    n = int(phases.size)
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    knm_flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)

    backend_fn = _dispatch()
    if backend_fn is not None:
        return np.asarray(backend_fn(phases, knm_flat, n), dtype=np.float64)

    r_local = np.zeros(n, dtype=np.float64)
    knm_2d = knm_flat.reshape(n, n)
    diffs = phases[np.newaxis, :] - phases[:, np.newaxis]
    unit = np.exp(1j * diffs)
    for i in range(n):
        mask = knm_2d[i] > 0
        if not np.any(mask):
            r_local[i] = 0.0
            continue
        r_local[i] = float(np.abs(np.mean(unit[i, mask])))
    return r_local


def detect_chimera(phases: NDArray, knm: NDArray) -> ChimeraState:
    """Detect chimera states in a Kuramoto network.

    Args:
        phases: ``(N,)`` oscillator phases.
        knm: ``(N, N)`` coupling matrix. ``K_ij > 0`` defines neighbours.

    Returns:
        :class:`ChimeraState` with coherent / incoherent index lists and
        the boundary-fraction chimera index.
    """
    phases = np.asarray(phases, dtype=np.float64)
    knm = np.asarray(knm, dtype=np.float64)
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
