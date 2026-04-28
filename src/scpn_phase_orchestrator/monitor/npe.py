# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Normalised Persistent Entropy (NPE)

"""Normalised Persistent Entropy from H₀ persistence diagram of the
circular phase-distance matrix. 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

* ``phase_distance_matrix`` — pairwise circular distance in ``[0, π]``.
* ``compute_npe`` — normalised H₀ persistent entropy via single-
  linkage clustering on the distance matrix.

Reference: *Scientific Reports* 2025 — NPE outperforms Kuramoto R
for synchronisation detection. The single-linkage shortcut replaces
a full Vietoris–Rips persistence (ripser) with O(N²) union-find.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "compute_npe",
    "phase_distance_matrix",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fns() -> dict[str, object]:
    from spo_kernel import compute_npe as rust_npe
    from spo_kernel import phase_distance_matrix as rust_pdm

    return {"phase_distance_matrix": rust_pdm, "compute_npe": rust_npe}


def _load_mojo_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.monitor._npe_mojo import (
        _ensure_exe,
        compute_npe_mojo,
        phase_distance_matrix_mojo,
    )

    _ensure_exe()
    return {
        "phase_distance_matrix": phase_distance_matrix_mojo,
        "compute_npe": compute_npe_mojo,
    }


def _load_julia_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.monitor._npe_julia import (
        compute_npe_julia,
        phase_distance_matrix_julia,
    )

    return {
        "phase_distance_matrix": phase_distance_matrix_julia,
        "compute_npe": compute_npe_julia,
    }


def _load_go_fns() -> dict[str, object]:  # pragma: no cover — toolchain-gated
    from scpn_phase_orchestrator.monitor._npe_go import (
        _load_lib,
        compute_npe_go,
        phase_distance_matrix_go,
    )

    _load_lib()
    return {
        "phase_distance_matrix": phase_distance_matrix_go,
        "compute_npe": compute_npe_go,
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
    return _LOADERS[ACTIVE_BACKEND]()[fn_name]


def phase_distance_matrix(phases: NDArray) -> NDArray:
    """Pairwise circular distance matrix ``d[i, j] ∈ [0, π]``."""
    n = phases.size
    backend_fn = _dispatch("phase_distance_matrix")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray], NDArray]", backend_fn)
        flat = fn(np.ascontiguousarray(phases.ravel(), dtype=np.float64))
        return np.asarray(flat, dtype=np.float64).reshape(n, n)

    diff = phases[:, np.newaxis] - phases[np.newaxis, :]
    return np.asarray(np.abs(np.arctan2(np.sin(diff), np.cos(diff))), dtype=np.float64)


def compute_npe(phases: NDArray, max_radius: float | None = None) -> float:
    """Normalised Persistent Entropy from H₀ persistence diagram.

    ``NPE = H(p) / log(|p|)`` where ``p_i = lifetime_i / Σ lifetimes``
    are the normalised birth-to-death lifetimes of the ``N − 1``
    components in the H₀ barcode. Returns values in ``[0, 1]``:
    ``~0`` means one dominant cluster (synchronised); ``~1`` means
    uniform lifetime distribution (incoherent).
    """
    n = phases.size
    if n < 2:
        return 0.0
    radius = np.pi if max_radius is None else float(max_radius)

    backend_fn = _dispatch("compute_npe")
    if backend_fn is not None:
        fn = cast("Callable[[NDArray, float], float]", backend_fn)
        return float(
            fn(
                np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                radius,
            )
        )

    dist = phase_distance_matrix(phases)
    triu_idx = np.triu_indices(n, k=1)
    edges = dist[triu_idx]
    sorted_idx = np.argsort(edges)

    parent = list(range(n))
    rank = [0] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    lifetimes: list[float] = []
    for idx in sorted_idx:
        i, j = int(triu_idx[0][idx]), int(triu_idx[1][idx])
        d = float(edges[idx])
        if d > radius:
            break
        ri, rj = find(i), find(j)
        if ri != rj:
            lifetimes.append(d)
            if rank[ri] < rank[rj]:
                parent[ri] = rj
            elif rank[ri] > rank[rj]:
                parent[rj] = ri
            else:
                parent[rj] = ri
                rank[ri] += 1

    if not lifetimes:
        return 0.0
    total = sum(lifetimes)
    if total < 1e-15:
        return 0.0
    probs = np.array(lifetimes) / total
    probs = probs[probs > 0]
    entropy = -float(np.sum(probs * np.log(probs)))
    max_entropy = np.log(len(probs)) if len(probs) > 1 else 1.0
    if max_entropy < 1e-15:
        return 0.0
    return entropy / max_entropy
