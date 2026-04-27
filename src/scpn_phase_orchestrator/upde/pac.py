# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-amplitude coupling

"""Tort 2010 phase-amplitude coupling with 5-backend fallback chain.

Follows ``feedback_module_standard_attnres.md``:

* ``modulation_index`` — scalar Tort 2010 MI on a single
  (θ_low, a_high) pair of time series.
* ``pac_matrix`` — ``(N, N)`` pairwise MI matrix over ``N``
  oscillator phase / amplitude channels.
* ``pac_gate`` — pure-Python boolean gate on an MI value (no
  backend dispatch needed — trivial comparison).

All compute kernels available in Rust, Mojo, Julia, Go, Python — the
dispatcher resolves fastest-first at import time via
``ACTIVE_BACKEND`` / ``AVAILABLE_BACKENDS``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "modulation_index",
    "pac_matrix",
    "pac_gate",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import pac_matrix_compute, pac_modulation_index

    return {
        "modulation_index": pac_modulation_index,
        "pac_matrix": pac_matrix_compute,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.upde._pac_mojo import (
        _ensure_exe,
        modulation_index_mojo,
        pac_matrix_mojo,
    )

    _ensure_exe()
    return {
        "modulation_index": modulation_index_mojo,
        "pac_matrix": pac_matrix_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.upde._pac_julia import (
        modulation_index_julia,
        pac_matrix_julia,
    )

    return {
        "modulation_index": modulation_index_julia,
        "pac_matrix": pac_matrix_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.upde._pac_go import (
        modulation_index_go,
        pac_matrix_go,
    )

    return {
        "modulation_index": modulation_index_go,
        "pac_matrix": pac_matrix_go,
    }


_LOADERS: dict[str, Callable[[], dict[str, object]]] = {
    "rust": _load_rust_fns,
    "mojo": _load_mojo_fns,
    "julia": _load_julia_fns,
    "go": _load_go_fns,
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


def _dispatch(fn_name: str) -> object:
    if ACTIVE_BACKEND == "python":
        return None
    try:
        return _LOADERS[ACTIVE_BACKEND]()[fn_name]
    except (ImportError, RuntimeError, OSError):
        return None


def modulation_index(theta_low: NDArray, amp_high: NDArray, n_bins: int = 18) -> float:
    """Phase-amplitude coupling via Tort et al. 2010, J. Neurophysiol.

    Bins amplitude by phase, computes KL divergence from uniform,
    returns the modulation index normalised to ``[0, 1]`` by
    ``log(n_bins)``.
    """
    if n_bins < 2:
        return 0.0
    if theta_low.size == 0 or amp_high.size == 0:
        return 0.0

    backend_fn = _dispatch("modulation_index")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray, NDArray, int], float]", backend_fn)
        return float(
            fn(
                np.ascontiguousarray(theta_low, dtype=np.float64),
                np.ascontiguousarray(amp_high, dtype=np.float64),
                n_bins,
            )
        )

    n = min(theta_low.size, amp_high.size)
    theta = theta_low[:n] % (2.0 * np.pi)
    amp = amp_high[:n]
    bin_edges = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    bin_indices = np.digitize(theta, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    mean_amp = np.zeros(n_bins, dtype=np.float64)
    for k in range(n_bins):
        mask = bin_indices == k
        if np.any(mask):
            mean_amp[k] = np.mean(amp[mask])
    total = mean_amp.sum()
    if total <= 0.0:
        return 0.0
    p = mean_amp / total
    log_n = np.log(n_bins)
    if log_n < 1e-15:
        return 0.0
    kl = 0.0
    for pk in p:
        if pk > 0.0:
            kl += pk * np.log(pk * n_bins)
    mi = kl / log_n
    return float(np.clip(mi, 0.0, 1.0))


def pac_matrix(
    phases_history: NDArray,
    amplitudes_history: NDArray,
    n_bins: int = 18,
) -> NDArray:
    """``(N, N)`` PAC matrix. Entry ``[i, j]`` is
    ``MI(phase_i, amplitude_j)`` over the ``T`` timesteps.

    Args:
        phases_history: ``(T, N)`` phase time series.
        amplitudes_history: ``(T, N)`` amplitude time series.
        n_bins: number of phase bins.
    """
    if phases_history.ndim != 2 or amplitudes_history.ndim != 2:
        raise ValueError("phases_history and amplitudes_history must be 2-D")
    t, n = phases_history.shape
    if amplitudes_history.shape != (t, n):
        raise ValueError("phases and amplitudes must have the same shape")

    backend_fn = _dispatch("pac_matrix")
    if backend_fn is not None:
        fn = cast(
            "Callable[[NDArray, NDArray, int, int, int], NDArray]",
            backend_fn,
        )
        flat = fn(
            np.ascontiguousarray(phases_history.ravel(), dtype=np.float64),
            np.ascontiguousarray(amplitudes_history.ravel(), dtype=np.float64),
            t,
            n,
            n_bins,
        )
        return np.asarray(flat, dtype=np.float64).reshape(n, n)

    result = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            result[i, j] = modulation_index(
                phases_history[:, i], amplitudes_history[:, j], n_bins
            )
    return result


def pac_gate(pac_value: float, threshold: float = 0.3) -> bool:
    """Binary gate: ``True`` when PAC exceeds ``threshold``.

    Pure-Python helper; no dispatcher — the comparison is trivial.
    """
    return pac_value >= threshold
