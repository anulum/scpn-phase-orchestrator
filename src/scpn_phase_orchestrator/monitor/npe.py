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
from numbers import Real
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

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


def _load_mojo_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._npe_mojo import (
        _ensure_exe,
        compute_npe_mojo,
        phase_distance_matrix_mojo,
    )

    _ensure_exe()
    return {
        "phase_distance_matrix": phase_distance_matrix_mojo,
        "compute_npe": compute_npe_mojo,
    }


def _load_julia_fns() -> dict[str, object]:
    import juliacall  # noqa: F401

    from ..experimental.accelerators.monitor._npe_julia import (
        compute_npe_julia,
        phase_distance_matrix_julia,
    )

    return {
        "phase_distance_matrix": phase_distance_matrix_julia,
        "compute_npe": compute_npe_julia,
    }


def _load_go_fns() -> dict[str, object]:
    from ..experimental.accelerators.monitor._npe_go import (
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
_BACKEND_CACHE: dict[str, dict[str, object]] = {}


def _load_backend(name: str) -> dict[str, object]:
    cached = _BACKEND_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = _LOADERS[name]()
    _BACKEND_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    _BACKEND_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            _load_backend(name)
        except (ImportError, RuntimeError, OSError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()


def _dispatch(fn_name: str) -> object:
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
            fn = _load_backend(backend).get(fn_name)
        except (ImportError, RuntimeError, OSError):
            continue
        if fn is None:
            continue
        return fn
    return None


def _contains_boolean_alias(raw: np.ndarray) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def _validate_phases(phases: object) -> FloatArray:
    raw = np.asarray(phases)
    if _contains_boolean_alias(raw):
        raise ValueError("phases must not contain boolean values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("phases must be a one-dimensional float array") from exc
    if array.ndim != 1:
        raise ValueError(f"phases shape {array.shape} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("phases must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _validate_distance_matrix(value: object, *, n_phases: int) -> FloatArray:
    if _contains_boolean_alias(np.asarray(value)):
        raise ValueError("phase distance matrix must not contain boolean values")
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("phase distance matrix must be numeric") from exc
    if matrix.shape != (n_phases, n_phases):
        raise ValueError(
            f"phase distance matrix shape {matrix.shape} does not match "
            f"({n_phases}, {n_phases})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("phase distance matrix must contain only finite values")
    tolerance = 1e-12
    if np.any(matrix < -tolerance) or np.any(matrix > np.pi + tolerance):
        raise ValueError("phase distance matrix entries must lie in [0, pi]")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1e-10):
        raise ValueError("phase distance matrix must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("phase distance matrix diagonal must be zero")
    matrix = np.clip(matrix, 0.0, np.pi)
    np.fill_diagonal(matrix, 0.0)
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _validate_max_radius(max_radius: float | None) -> float:
    if max_radius is None:
        return float(np.pi)
    if isinstance(max_radius, bool) or not isinstance(max_radius, Real):
        raise ValueError(
            f"max_radius must be a finite non-negative real, got {max_radius!r}"
        )
    radius = float(max_radius)
    if not np.isfinite(radius) or radius < 0.0:
        raise ValueError(
            f"max_radius must be finite and non-negative, got {max_radius!r}"
        )
    if radius > np.pi + 1e-12:
        raise ValueError(f"max_radius must not exceed pi, got {max_radius!r}")
    return radius


def _validate_npe_value(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"NPE must be a finite real scalar in [0, 1], got {value!r}")
    score = float(value)
    if not np.isfinite(score):
        raise ValueError(f"NPE must be finite, got {value!r}")
    tolerance = 1e-12
    if score < -tolerance or score > 1.0 + tolerance:
        raise ValueError(f"NPE must lie in [0, 1], got {value!r}")
    return min(1.0, max(0.0, score))


def phase_distance_matrix(phases: FloatArray) -> FloatArray:
    """Pairwise circular distance matrix ``d[i, j] ∈ [0, π]``."""
    phases = _validate_phases(phases)
    n = phases.size
    backend_fn = _dispatch("phase_distance_matrix")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray], FloatArray]", backend_fn)
        try:
            flat = fn(np.ascontiguousarray(phases.ravel(), dtype=np.float64))
            return _validate_distance_matrix(flat, n_phases=n)
        except Exception:
            backend_fn = None

    diff = phases[:, np.newaxis] - phases[np.newaxis, :]
    return _validate_distance_matrix(
        np.abs(np.arctan2(np.sin(diff), np.cos(diff))),
        n_phases=n,
    )


def compute_npe(phases: FloatArray, max_radius: float | None = None) -> float:
    """Normalised Persistent Entropy from H₀ persistence diagram.

    ``NPE = H(p) / log(|p|)`` where ``p_i = lifetime_i / Σ lifetimes``
    are the normalised birth-to-death lifetimes of the ``N − 1``
    components in the H₀ barcode. Returns values in ``[0, 1]``:
    ``~0`` means one dominant cluster (synchronised); ``~1`` means
    uniform lifetime distribution (incoherent).
    """
    phases = _validate_phases(phases)
    n = phases.size
    radius = _validate_max_radius(max_radius)
    if n < 2:
        return 0.0

    backend_fn = _dispatch("compute_npe")
    if backend_fn is not None:
        fn = cast("Callable[[FloatArray, float], float]", backend_fn)
        try:
            return _validate_npe_value(
                fn(
                    np.ascontiguousarray(phases.ravel(), dtype=np.float64),
                    radius,
                ),
            )
        except Exception:
            backend_fn = None

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
    return _validate_npe_value(entropy / max_entropy)
