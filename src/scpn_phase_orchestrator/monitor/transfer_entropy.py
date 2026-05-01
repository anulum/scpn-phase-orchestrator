# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase Transfer Entropy

"""Phase transfer entropy via binned histograms with a 5-backend
fallback chain per ``feedback_module_standard_attnres.md``.

Two compute kernels:

* ``phase_transfer_entropy`` — scalar ``TE(X → Y)`` on a pair of
  equal-length phase series.
* ``transfer_entropy_matrix`` — ``(N, N)`` pairwise TE matrix over
  ``N`` oscillator trajectories.

Estimator: 1-step Markov-order conditional entropy difference

    TE(X → Y) = H(Y_{t+1} | Y_t) − H(Y_{t+1} | Y_t, X_t)

with phases wrapped to ``[0, 2π)`` and binned into ``n_bins``
equal-width intervals. Higher TE indicates stronger directional
coupling from source to target.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.integer[Any]]

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "phase_transfer_entropy",
    "transfer_entropy_matrix",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import (
        phase_transfer_entropy_rust,
        transfer_entropy_matrix_rust,
    )

    return {
        "phase_te": phase_transfer_entropy_rust,
        "te_matrix": transfer_entropy_matrix_rust,
    }


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.monitor._te_mojo import (
        _ensure_exe,
        phase_te_mojo,
        te_matrix_mojo,
    )

    _ensure_exe()
    return {"phase_te": phase_te_mojo, "te_matrix": te_matrix_mojo}


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._te_julia import (
        phase_te_julia,
        te_matrix_julia,
    )

    return {"phase_te": phase_te_julia, "te_matrix": te_matrix_julia}


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.monitor._te_go import (
        _load_lib,
        phase_te_go,
        te_matrix_go,
    )

    _load_lib()
    return {"phase_te": phase_te_go, "te_matrix": te_matrix_go}


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
    return _LOADERS[ACTIVE_BACKEND]()[fn_name]


def _conditional_entropy(
    target: IntArray, condition: IntArray, n_cond_bins: int
) -> float:
    """``H(target | condition)`` via histogram."""
    n = len(target)
    h = 0.0
    for c in range(n_cond_bins):
        mask = condition == c
        count = int(np.sum(mask))
        if count < 2:
            continue
        vals = target[mask]
        _, counts = np.unique(vals, return_counts=True)
        probs = counts / count
        h -= (count / n) * float(np.sum(probs * np.log(probs + 1e-30)))
    return h


def phase_transfer_entropy(
    source: FloatArray, target: FloatArray, n_bins: int = 16
) -> float:
    """Transfer entropy ``TE(X → Y)`` on binned phase series."""
    source_values: FloatArray = np.asarray(source, dtype=np.float64)
    target_values: FloatArray = np.asarray(target, dtype=np.float64)
    backend_fn = _dispatch("phase_te")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, FloatArray, int], float]", backend_fn)
        return float(
            fn(
                np.ascontiguousarray(source_values, dtype=np.float64),
                np.ascontiguousarray(target_values, dtype=np.float64),
                n_bins,
            )
        )

    if len(source_values) < 3 or len(target_values) < 3:
        return 0.0
    n = min(len(source_values), len(target_values)) - 1
    bins = np.linspace(0, 2 * np.pi, n_bins + 1)
    src_binned: IntArray = np.clip(
        np.digitize(source_values[:n] % (2 * np.pi), bins) - 1,
        0,
        n_bins - 1,
    )
    tgt_binned: IntArray = np.clip(
        np.digitize(target_values[:n] % (2 * np.pi), bins) - 1,
        0,
        n_bins - 1,
    )
    tgt_next: IntArray = np.clip(
        np.digitize(target_values[1 : n + 1] % (2 * np.pi), bins) - 1,
        0,
        n_bins - 1,
    )
    h_y_yt = _conditional_entropy(tgt_next, tgt_binned, n_bins)
    joint_cond: IntArray = tgt_binned * n_bins + src_binned
    h_y_yt_x = _conditional_entropy(tgt_next, joint_cond, n_bins * n_bins)
    return max(0.0, h_y_yt - h_y_yt_x)


def transfer_entropy_matrix(phase_series: FloatArray, n_bins: int = 16) -> FloatArray:
    """Pairwise TE matrix; entry ``[i, j] = TE(i → j)`` for all
    oscillator pairs with zero diagonal."""
    series: FloatArray = np.asarray(phase_series, dtype=np.float64)
    if series.ndim != 2:
        raise ValueError(
            f"phase_series must be 2-D (oscillators, timesteps), "
            f"got shape {series.shape}"
        )
    n_osc, n_time = series.shape
    backend_fn = _dispatch("te_matrix")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, int, int, int], FloatArray]", backend_fn)
        flat = fn(
            np.ascontiguousarray(series.ravel(), dtype=np.float64),
            n_osc,
            n_time,
            n_bins,
        )
        return np.asarray(flat, dtype=np.float64).reshape(n_osc, n_osc)

    te: FloatArray = np.zeros((n_osc, n_osc), dtype=np.float64)
    for i in range(n_osc):
        for j in range(n_osc):
            if i != j:
                te[i, j] = phase_transfer_entropy(series[i], series[j], n_bins)
    return te
