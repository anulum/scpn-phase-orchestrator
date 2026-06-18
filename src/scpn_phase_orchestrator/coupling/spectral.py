# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Spectral graph analysis for coupling networks

"""Symmetric eigendecomposition of the combinatorial graph Laplacian ``L = D − A``.

Exposes a 5-backend fallback chain.

For asymmetric measured coupling, the undirected adjacency is the
reciprocal magnitude average ``A = (|W| + |Wᵀ|) / 2`` with zeroed
diagonal. Degrees are then computed from ``A``. This preserves the
combinatorial Laplacian contract: symmetric positive-semidefinite
``L``, zero row sums, and ``1 ∈ ker L``.

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
from numbers import Real
from typing import Any, TypeAlias, cast

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
FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    if isinstance(value, np.ndarray):
        if value.dtype == np.bool_:
            return True
        if value.dtype != object:
            return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _contains_complex_alias(value: object) -> bool:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        return True
    if isinstance(value, np.ndarray) and raw.dtype != object:
        return False
    try:
        raw = np.asarray(value, dtype=object)
    except (TypeError, ValueError):
        return False
    return any(isinstance(item, (complex, np.complexfloating)) for item in raw.ravel())


def _validate_coupling_matrix(knm: object) -> FloatArray:
    if _contains_boolean_alias(knm):
        raise ValueError("knm must not contain boolean values")
    raw = np.asarray(knm)
    if raw.dtype == np.bool_:
        raise ValueError("knm must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_complex_alias(knm):
        raise ValueError("knm must be a finite square matrix of real-valued weights")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("knm must be a finite square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("knm must be a finite square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("knm must contain only finite values")
    return matrix


def _validate_omegas(omegas: object, *, expected_n: int) -> FloatArray:
    values = _validate_omega_vector(omegas)
    if values.shape != (expected_n,):
        raise ValueError(f"omegas shape {values.shape} does not match ({expected_n},)")
    return values


def _validate_omega_vector(omegas: object) -> FloatArray:
    if _contains_boolean_alias(omegas):
        raise ValueError("omegas must not contain boolean values")
    raw = np.asarray(omegas)
    if raw.dtype == np.bool_:
        raise ValueError("omegas must not contain boolean values")
    if np.iscomplexobj(raw) or _contains_complex_alias(omegas):
        raise ValueError("omegas must be a finite real-valued frequency vector")
    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("omegas must be a finite 1-D frequency vector") from exc
    if values.ndim != 1:
        raise ValueError("omegas must be a finite 1-D frequency vector")
    if values.size == 0:
        raise ValueError("omegas must contain at least one frequency")
    if not np.all(np.isfinite(values)):
        raise ValueError("omegas must contain only finite values")
    return values


def _validate_gamma_max(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError("gamma_max must be a finite real value")
    gamma = float(value)
    if not np.isfinite(gamma):
        raise ValueError("gamma_max must be finite")
    if gamma < 0.0:
        raise ValueError("gamma_max must be non-negative")
    return gamma


def _validate_non_negative_scalar(
    value: object, *, name: str, allow_infinite: bool = False
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a non-negative scalar")
    resolved = float(value)
    if allow_infinite and np.isposinf(resolved):
        return resolved
    if not np.isfinite(resolved) or resolved < -1e-10:
        raise ValueError(f"{name} must be a finite non-negative scalar")
    return max(resolved, 0.0)


def _validate_rust_fiedler_vector(value: object, *, n: int) -> FloatArray:
    if _contains_boolean_alias(value) or _contains_complex_alias(value):
        raise ValueError("Fiedler vector must be real-valued and non-boolean")
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Fiedler vector must be numeric") from exc
    if vector.shape != (n,):
        raise ValueError(f"Fiedler vector shape {vector.shape} must be ({n},)")
    if not np.all(np.isfinite(vector)):
        raise ValueError("Fiedler vector must contain only finite values")
    if n > 1 and np.linalg.norm(vector) <= 1e-10:
        raise ValueError("Fiedler vector must be non-zero")
    return np.ascontiguousarray(vector, dtype=np.float64)


def graph_laplacian(knm: FloatArray) -> FloatArray:
    """Combinatorial graph Laplacian ``L = D − A``.

    ``A`` is the reciprocal undirected magnitude adjacency
    ``(|W| + |Wᵀ|) / 2`` with zero diagonal, so asymmetric measured
    couplings produce one symmetric edge weight before node degrees
    are computed.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    FloatArray
        The combinatorial graph Laplacian ``L = D − A``.
    """
    knm = _validate_coupling_matrix(knm)
    w = np.abs(knm)
    np.fill_diagonal(w, 0.0)
    adjacency = 0.5 * (w + w.T)
    np.fill_diagonal(adjacency, 0.0)
    degrees = adjacency.sum(axis=1)
    return cast("FloatArray", np.diag(degrees) - adjacency)


def _python_spectral_eig(
    knm_flat: FloatArray,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    W = knm_flat.reshape(n, n)
    L = graph_laplacian(W)
    eigvals, eigvecs = np.linalg.eigh(L)
    fiedler = eigvecs[:, 1] if n > 1 else np.zeros(n)
    return eigvals, fiedler


def _validate_spectral_output(
    value: object,
    *,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("spectral primitive output must be (eigvals, fiedler)")
    eigvals_raw = value[0]
    fiedler_raw = value[1]
    if (
        _contains_boolean_alias(eigvals_raw)
        or _contains_boolean_alias(fiedler_raw)
        or _contains_complex_alias(eigvals_raw)
        or _contains_complex_alias(fiedler_raw)
    ):
        raise ValueError("spectral primitive output must be real-valued numeric arrays")
    try:
        eigvals = np.asarray(eigvals_raw, dtype=np.float64)
        fiedler = np.asarray(fiedler_raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("spectral primitive output must be numeric") from exc
    if eigvals.shape != (n,):
        raise ValueError(f"spectral eigenvalue shape {eigvals.shape} must be ({n},)")
    if fiedler.shape != (n,):
        raise ValueError(f"spectral fiedler shape {fiedler.shape} must be ({n},)")
    if not np.all(np.isfinite(eigvals)) or not np.all(np.isfinite(fiedler)):
        raise ValueError("spectral primitive output must contain only finite values")
    tolerance = 1e-10
    if np.any(eigvals < -tolerance):
        raise ValueError("spectral eigenvalues must be non-negative")
    if np.any(np.diff(eigvals) < -tolerance):
        raise ValueError("spectral eigenvalues must be sorted ascending")
    if n > 1 and np.linalg.norm(fiedler) <= tolerance:
        raise ValueError("spectral fiedler vector must be non-zero")
    return (
        np.ascontiguousarray(np.maximum(eigvals, 0.0), dtype=np.float64),
        np.ascontiguousarray(fiedler, dtype=np.float64),
    )


def _spectral_eig_checked(
    knm_flat: FloatArray, n: int
) -> tuple[FloatArray, FloatArray]:
    primitive = _primitive()
    try:
        return _validate_spectral_output(primitive(knm_flat, n), n=n)
    except (ImportError, RuntimeError, OSError, KeyError):
        if primitive is _python_spectral_eig:
            raise
    return _validate_spectral_output(_python_spectral_eig(knm_flat, n), n=n)


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


def _load_mojo_primitive() -> Callable[
    [FloatArray, int], tuple[FloatArray, FloatArray]
]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.coupling._spectral_mojo import (
        _ensure_exe,
        spectral_eig_mojo,
    )

    _ensure_exe()
    return spectral_eig_mojo


def _load_julia_primitive() -> Callable[
    [FloatArray, int], tuple[FloatArray, FloatArray]
]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401

    from ..experimental.accelerators.coupling._spectral_julia import (
        spectral_eig_julia,
    )

    return spectral_eig_julia


def _load_go_primitive() -> Callable[[FloatArray, int], tuple[FloatArray, FloatArray]]:
    # pragma: no cover — toolchain
    from ..experimental.accelerators.coupling._spectral_go import (
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

_PRIMITIVE_CACHE: dict[
    str, Callable[[FloatArray, int], tuple[FloatArray, FloatArray]]
] = {}
_PRIM_CACHE: (
    dict[str, Callable[[FloatArray, int], tuple[FloatArray, FloatArray]]] | None
) = _PRIMITIVE_CACHE


def _load_primitive_backend(
    name: str,
) -> Callable[[FloatArray, int], tuple[FloatArray, FloatArray]]:
    global _PRIM_CACHE
    if _PRIM_CACHE is None:
        _PRIM_CACHE = {}
    cached = _PRIM_CACHE.get(name)
    if cached is not None:
        return cached
    loaded = cast(
        Callable[[FloatArray, int], tuple[FloatArray, FloatArray]], _LOADERS[name]()
    )
    _PRIM_CACHE[name] = loaded
    return loaded


def _resolve_backends() -> tuple[str, list[str]]:
    global _RUST_CACHE, _PRIM_CACHE
    _RUST_CACHE = None
    if _PRIM_CACHE is not None:
        _PRIM_CACHE.clear()
    available: list[str] = []
    for name in _BACKEND_NAMES[:-1]:
        try:
            if name == "rust":
                _LOADERS[name]()
            else:
                _load_primitive_backend(name)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        available.append(name)
    available.append("python")
    return available[0], available


ACTIVE_BACKEND, AVAILABLE_BACKENDS = _resolve_backends()

_RUST_CACHE: dict[str, Any] | None = None


def _rust_bundle() -> dict[str, Any]:
    global _RUST_CACHE
    if _RUST_CACHE is None:
        _RUST_CACHE = _load_rust_bundle()
    return _RUST_CACHE


def _primitive() -> Callable[[FloatArray, int], tuple[FloatArray, FloatArray]]:
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend in {"python", "rust"}:
            if backend == "python":
                return _python_spectral_eig
            continue
        try:
            loaded = _load_primitive_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
        return loaded
    return _python_spectral_eig


def spectral_eig(knm: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Symmetric eigendecomposition of ``L = D − |W|``.

    Returns ``(eigvals ascending, fiedler vector)``. Thin wrapper
    over the dispatched backend primitive; ``python`` reference
    is a direct ``np.linalg.eigh``.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        The eigenvalues and eigenvectors of the symmetric Laplacian.
    """
    knm = _validate_coupling_matrix(knm)
    n = knm.shape[0]
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    return _spectral_eig_checked(flat, n)


def fiedler_value(knm: FloatArray) -> float:
    """Return the algebraic connectivity ``λ₂(L)`` (Dörfler-Bullo 2014).

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    float
        The algebraic connectivity ``λ₂(L)``.
    """
    knm = _validate_coupling_matrix(knm)
    n = knm.shape[0]
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    if ACTIVE_BACKEND == "rust":
        return _validate_non_negative_scalar(
            _rust_bundle()["fv"](flat, n), name="Fiedler value"
        )
    eigvals, _ = _spectral_eig_checked(flat, n)
    return float(eigvals[1]) if n > 1 else 0.0


def fiedler_vector(knm: FloatArray) -> FloatArray:
    """Return the ``λ₂`` eigenvector partitioning the graph into clusters.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    FloatArray
        The ``λ₂`` eigenvector partitioning the graph.
    """
    knm = _validate_coupling_matrix(knm)
    n = knm.shape[0]
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    if ACTIVE_BACKEND == "rust":
        return _validate_rust_fiedler_vector(_rust_bundle()["fvec"](flat, n), n=n)
    _, fiedler = _spectral_eig_checked(flat, n)
    return fiedler


def critical_coupling(omegas: FloatArray, knm: FloatArray) -> float:
    """Dörfler-Bullo critical coupling ``K_c = Δω / λ₂``.

    Returns ``+inf`` if the graph is disconnected
    (``λ₂ ≈ 0``).

    Parameters
    ----------
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    float
        The Dörfler-Bullo critical coupling ``K_c``.
    """
    knm = _validate_coupling_matrix(knm)
    n = knm.shape[0]
    omegas = _validate_omegas(omegas, expected_n=n)
    if ACTIVE_BACKEND == "rust":
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        return _validate_non_negative_scalar(
            _rust_bundle()["kc"](o, flat, n),
            name="critical coupling",
            allow_infinite=True,
        )
    lambda2 = fiedler_value(knm)
    if lambda2 < 1e-12:
        return float("inf")
    omega_spread = float(np.max(omegas) - np.min(omegas))
    return omega_spread / lambda2


def fiedler_partition(knm: FloatArray) -> tuple[list[int], list[int]]:
    """Bisect the network using ``sign(v₂)``.

    Returns ``(group_positive, group_negative)`` — indices
    of oscillators in each partition.

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    tuple[list[int], list[int]]
        The two index lists of the ``sign(v₂)`` bisection.
    """
    v2 = fiedler_vector(knm)
    pos = [i for i, val in enumerate(v2) if val >= 0]
    neg = [i for i, val in enumerate(v2) if val < 0]
    return pos, neg


def spectral_gap(knm: FloatArray) -> float:
    """Return the gap between ``λ₂`` and ``λ₃`` (two-cluster cleanliness).

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.

    Returns
    -------
    float
        The gap between ``λ₂`` and ``λ₃``.
    """
    knm = _validate_coupling_matrix(knm)
    n = knm.shape[0]
    if n < 3:
        return 0.0
    off_diag = np.abs(knm[~np.eye(n, dtype=bool)])
    if off_diag.size and np.allclose(off_diag, off_diag[0], rtol=1e-12, atol=1e-12):
        return 0.0
    flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
    if ACTIVE_BACKEND == "rust":
        return _validate_non_negative_scalar(
            _rust_bundle()["sg"](flat, n), name="spectral gap"
        )
    eigvals, _ = _spectral_eig_checked(flat, n)
    return float(eigvals[2] - eigvals[1])


def sync_convergence_rate(
    knm: FloatArray,
    omegas: FloatArray,
    gamma_max: float = 0.0,
) -> float:
    """Estimate the convergence rate from ``λ₂`` (Dörfler-Bullo 2014 §III.B).

    Parameters
    ----------
    knm : FloatArray
        Coupling matrix ``K_nm``, shape ``(N, N)``.
    omegas : FloatArray
        Natural frequencies in rad/s, shape ``(N,)``.
    gamma_max : float
        Maximum phase-lag ``γ`` across edges.

    Returns
    -------
    float
        The estimated synchronisation convergence rate.
    """
    knm = _validate_coupling_matrix(knm)
    n = knm.shape[0]
    if n == 0:
        return 0.0
    omegas = _validate_omegas(omegas, expected_n=n)
    gamma_max = _validate_gamma_max(gamma_max)
    if ACTIVE_BACKEND == "rust":
        flat = np.ascontiguousarray(knm.ravel(), dtype=np.float64)
        o = np.ascontiguousarray(omegas, dtype=np.float64)
        return _validate_non_negative_scalar(
            _rust_bundle()["scr"](flat, o, n, gamma_max),
            name="sync convergence rate",
        )
    lambda2 = fiedler_value(knm)
    pos_vals = knm[knm > 0]
    k_eff = float(np.mean(pos_vals)) if pos_vals.size > 0 else 0.0
    return float(k_eff * lambda2 * np.cos(gamma_max) / n)
