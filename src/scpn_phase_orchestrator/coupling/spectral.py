# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral graph analysis for coupling networks

"""Symmetric eigendecomposition of the combinatorial graph
Laplacian ``L = D − |W|`` exposed through a 5-backend fallback
chain per ``feedback_module_standard_attnres.md``.

Primitive
---------
``spectral_eig(W_flat, n) → (eigvals, fiedler)`` — eigenvalues
ascending + Fiedler eigenvector (column 1 of the sorted
decomposition).

Backend chain
-------------
* **Rust**: pre-existing ``fiedler_value_rust``,
  ``fiedler_vector_rust``, ``spectral_gap_rust``,
  ``critical_coupling_rust``, ``sync_convergence_rate_rust``
  FFI fast paths are wired individually (each exposes a direct
  entry, no round-trip through the primitive).
* **Julia**: ``LinearAlgebra.eigen(Symmetric(L))`` — LAPACK
  ``dsyev`` underneath, same numerics as NumPy.
* **Go**: ``gonum.org/v1/gonum/mat :: EigenSym`` — pure-Go
  symmetric solver, sub-``1e-12`` drift vs LAPACK on
  well-conditioned Laplacians.
* **Mojo**: LAPACK ``dsyev_`` via the ``std.ffi.OwnedDLHandle``
  pattern (same as ``_lapack_test.mojo``).
* **Python**: ``np.linalg.eigh`` — LAPACK-backed reference.

Derived functions (``fiedler_value``, ``fiedler_vector``,
``spectral_gap``) route through the primitive on non-Rust
backends. ``critical_coupling`` and ``sync_convergence_rate``
are composites that reuse ``fiedler_value``.

References: Dörfler & Bullo 2014, *Automatica* 50(6):1539-1564;
Dörfler & Bullo 2013, *IEEE Proc.* 102(10):1539-1564.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "critical_coupling",
    "fiedler_partition",
    "fiedler_value",
    "fiedler_vector",
    "graph_laplacian",
    "spectral_eig",
    "spectral_gap",
    "sync_convergence_rate",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def graph_laplacian(knm: NDArray) -> NDArray:
    """Combinatorial graph Laplacian ``L = D − |W|`` with zero
    diagonal on ``W``."""
    w = np.abs(knm)
    np.fill_diagonal(w, 0.0)
    degrees = w.sum(axis=1)
    return cast("NDArray", np.diag(degrees) - w)


def _python_spectral_eig(
    knm_flat: NDArray,
    n: int,
) -> tuple[NDArray, NDArray]:
    W = knm_flat.reshape(n, n)
    L = graph_laplacian(W)
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1] if n > 1 else np.zeros(n)
    return eigvals, fiedler


def _load_rust_bundle() -> dict[str, Any]:
    from spo_kernel import (
        critical_coupling_rust,
        fiedler_value_rust,
        fiedler_vector_rust,
        spectral_gap_rust,
        sync_convergence_rate_rust,
    )

    return {
        "fv": fiedler_value_rust,
        "fvec": fiedler_vector_rust,
        "sg": spectral_gap_rust,
        "kc": critical_coupling_rust,
        "scr": sync_convergence_rate_rust,
    }


def _load_mojo_primitive() -> Callable[[NDArray, int], tuple[NDArray, NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.coupling._spectral_mojo import (
        _ensure_exe,
        spectral_eig_mojo,
    )

    _ensure_exe()
    return spectral_eig_mojo


def _load_julia_primitive() -> Callable[[NDArray, int], tuple[NDArray, NDArray]]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.coupling._spectral_julia import (
        spectral_eig_julia,
    )

    return spectral_eig_julia


def _load_go_primitive() -> Callable[[NDArray, int], tuple[NDArray, NDArray]]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.coupling._spectral_go import (
        _load_lib,
        spectral_eig_go,
    )

    _load_lib()
    return spectral_eig_go


_LOADERS: dict[str, Callable[[], Any]] = {
    "rust": _load_rust_bundle,
    "mojo": _load_mojo_primitive,
    "julia": _load_julia_primitive,
    "go": _load_go_primitive,
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

_RUST_CACHE: dict[str, Any] | None = None
_PRIM_CACHE: Callable[[NDArray, int], tuple[NDArray, NDArray]] | None = None


def _rust_bundle() -> dict[str, Any]:
    global _RUST_CACHE
    if _RUST_CACHE is None:
        _RUST_CACHE = _load_rust_bundle()
    return _RUST_CACHE


def _primitive() -> Callable[[NDArray, int], tuple[NDArray, NDArray]]:
    global _PRIM_CACHE
    if ACTIVE_BACKEND == "python":
        return _python_spectral_eig
    if ACTIVE_BACKEND == "rust":
        # Rust has direct-per-function fast paths; the primitive
        # falls back to Python so derived users never accidentally
        # double-compute.
        return _python_spectral_eig
    if _PRIM_CACHE is None:
        _PRIM_CACHE = _LOADERS[ACTIVE_BACKEND]()
    return _PRIM_CACHE


def spectral_eig(knm: NDArray) -> tuple[NDArray, NDArray]:
    """Symmetric eigendecomposition of ``L = D − |W|``.

    Returns ``(eigvals ascending, fiedler vector)``. Thin wrapper
    over the dispatched backend primitive; ``python`` reference
    is a direct ``np.linalg.eigh``.
    """
    knm = np.asarray(knm, dtype=np.float64)
    n = knm.shape[0]
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    return _primitive()(flat, n)


def fiedler_value(knm: NDArray) -> float:
    """Algebraic connectivity ``λ₂(L)`` — second smallest
    eigenvalue (Dörfler-Bullo 2014)."""
    knm = np.asarray(knm, dtype=np.float64)
    n = knm.shape[0]
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    if ACTIVE_BACKEND == "rust":
        return float(_rust_bundle()["fv"](flat, n))
    eigvals, _ = _primitive()(flat, n)
    return float(eigvals[1]) if n > 1 else 0.0


def fiedler_vector(knm: NDArray) -> NDArray:
    """Eigenvector for ``λ₂`` — partitions the graph into
    synchronisation clusters."""
    knm = np.asarray(knm, dtype=np.float64)
    n = knm.shape[0]
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    if ACTIVE_BACKEND == "rust":
        return np.asarray(_rust_bundle()["fvec"](flat, n))
    _, fiedler = _primitive()(flat, n)
    return fiedler


def critical_coupling(omegas: NDArray, knm: NDArray) -> float:
    """Dörfler-Bullo critical coupling ``K_c = Δω / λ₂``.

    Returns ``+inf`` if the graph is disconnected
    (``λ₂ ≈ 0``)."""
    knm = np.asarray(knm, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    n = knm.shape[0]
    if ACTIVE_BACKEND == "rust":
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        return float(_rust_bundle()["kc"](o, flat, n))
    lambda2 = fiedler_value(knm)
    if lambda2 < 1e-12:
        return float("inf")
    omega_spread = float(np.max(omegas) - np.min(omegas))
    return omega_spread / lambda2


def fiedler_partition(knm: NDArray) -> tuple[list[int], list[int]]:
    """Bisect the network using ``sign(v₂)``.

    Returns ``(group_positive, group_negative)`` — indices
    of oscillators in each partition.
    """
    v2 = fiedler_vector(knm)
    pos = [i for i, val in enumerate(v2) if val >= 0]
    neg = [i for i, val in enumerate(v2) if val < 0]
    return pos, neg


def spectral_gap(knm: NDArray) -> float:
    """Gap between ``λ₂`` and ``λ₃`` — larger gap means cleaner
    two-cluster structure."""
    knm = np.asarray(knm, dtype=np.float64)
    n = knm.shape[0]
    if n < 3:
        return 0.0
    off_diag = np.abs(knm[~np.eye(n, dtype=bool)])
    if off_diag.size and np.allclose(off_diag, off_diag[0], rtol=1e-12, atol=1e-12):
        return 0.0
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    if ACTIVE_BACKEND == "rust":
        return float(_rust_bundle()["sg"](flat, n))
    eigvals, _ = _primitive()(flat, n)
    return float(eigvals[2] - eigvals[1])


def sync_convergence_rate(
    knm: NDArray,
    omegas: NDArray,
    gamma_max: float = 0.0,
) -> float:
    """Estimated convergence rate
    ``μ = K_eff · λ₂ · cos(γ_max) / N``
    (Dörfler-Bullo 2014 §III.B)."""
    knm = np.asarray(knm, dtype=np.float64)
    omegas = np.asarray(omegas, dtype=np.float64)
    n = len(omegas)
    if n == 0:
        return 0.0
    if ACTIVE_BACKEND == "rust":
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        return float(_rust_bundle()["scr"](flat, o, n, gamma_max))
    lambda2 = fiedler_value(knm)
    pos_vals = knm[knm > 0]
    k_eff = float(np.mean(pos_vals)) if pos_vals.size > 0 else 0.0
    return float(k_eff * lambda2 * np.cos(gamma_max) / n)
