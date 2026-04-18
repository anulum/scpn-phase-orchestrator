# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Inter-Trial Phase Coherence

"""Lachaux 1999 inter-trial phase coherence with a 5-backend fallback
chain per ``feedback_module_standard_attnres.md``.

Two kernels:

* :func:`compute_itpc` — ITPC across trials at each time point.
* :func:`itpc_persistence` — mean ITPC at stimulus-pause indices.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "compute_itpc",
    "itpc_persistence",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        compute_itpc_rust,
        itpc_persistence_rust,
    )

    return {
        "itpc": compute_itpc_rust,
        "persistence": itpc_persistence_rust,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._itpc_mojo import (
        _ensure_exe,
        compute_itpc_mojo,
        itpc_persistence_mojo,
    )

    _ensure_exe()
    return {"itpc": compute_itpc_mojo, "persistence": itpc_persistence_mojo}


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    import juliacall  # type: ignore[import-untyped]  # noqa: F401

    from scpn_phase_orchestrator.monitor._itpc_julia import (
        compute_itpc_julia,
        itpc_persistence_julia,
    )

    return {"itpc": compute_itpc_julia, "persistence": itpc_persistence_julia}


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain
    from scpn_phase_orchestrator.monitor._itpc_go import (
        compute_itpc_go,
        itpc_persistence_go,
    )

    return {"itpc": compute_itpc_go, "persistence": itpc_persistence_go}


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


def _dispatch(fn_name: str) -> object | None:
    if ACTIVE_BACKEND == "python":
        return None
    return _LOADERS[ACTIVE_BACKEND]()[fn_name]


def compute_itpc(phases_trials: NDArray) -> NDArray:
    """Inter-Trial Phase Coherence at each time point.

    ``ITPC = |mean(exp(i·θ))|`` across trials (Lachaux et al. 1999).

    Args:
        phases_trials: shape ``(n_trials, n_timepoints)`` — phases in
            radians. A 1-D input is treated as a single trial.

    Returns:
        ``(n_timepoints,)`` array of ITPC values in ``[0, 1]``.
    """
    phases = np.asarray(phases_trials, dtype=np.float64)
    if phases.ndim == 1:
        return np.array([1.0])
    if phases.shape[0] == 0:
        return np.array([])
    n_trials, n_tp = phases.shape

    backend_fn = _dispatch("itpc")
    if backend_fn is not None:
        if ACTIVE_BACKEND == "rust":
            fn_rust = cast(
                "Callable[[NDArray, int, int], NDArray]", backend_fn
            )
            flat = np.ascontiguousarray(phases.ravel(), dtype=np.float64)
            return np.asarray(fn_rust(flat, n_trials, n_tp), dtype=np.float64)
        fn = cast("Callable[[NDArray, int, int], NDArray]", backend_fn)
        return np.asarray(
            fn(phases.ravel(), int(n_trials), int(n_tp)),
            dtype=np.float64,
        )

    result: NDArray = np.abs(np.mean(np.exp(1j * phases), axis=0))
    return result


def itpc_persistence(
    phases_trials: NDArray,
    pause_indices: list[int] | NDArray,
) -> float:
    """Mean ITPC at stimulus-pause indices.

    Distinguishes true neural entrainment from evoked response: if ITPC
    remains high after the driving stimulus stops, oscillators have
    genuinely phase-locked. If it drops immediately, the response was
    merely evoked.

    Args:
        phases_trials: ``(n_trials, n_timepoints)`` phases in radians.
        pause_indices: time-point indices falling within / after a pause.

    Returns:
        Mean ITPC across ``pause_indices``. ``0.0`` if empty.
    """
    pause_idx = np.asarray(pause_indices, dtype=int)
    if pause_idx.size == 0:
        return 0.0

    phases = np.asarray(phases_trials, dtype=np.float64)
    if phases.ndim == 1:
        phases = phases.reshape(1, -1)
    n_trials, n_tp = phases.shape

    backend_fn = _dispatch("persistence")
    if backend_fn is not None:
        if ACTIVE_BACKEND == "rust":
            fn_rust = cast(
                "Callable[[NDArray, int, int, NDArray], float]", backend_fn
            )
            return float(
                fn_rust(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    n_trials,
                    n_tp,
                    np.ascontiguousarray(pause_idx, dtype=np.int64),
                )
            )
        fn = cast("Callable[[NDArray, int, int, NDArray], float]", backend_fn)
        return float(
            fn(phases.ravel(), int(n_trials), int(n_tp), pause_idx)
        )

    itpc_full = compute_itpc(phases)
    valid = pause_idx[(pause_idx >= 0) & (pause_idx < itpc_full.size)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(itpc_full[valid]))
