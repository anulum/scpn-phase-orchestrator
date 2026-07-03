# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Combinatorial Hodge decomposition of coupling flow

r"""Combinatorial Hodge (Helmholtz–Hodge) decomposition of the Kuramoto current.

Exposes a 5-backend fallback chain.

Model
-----
The oscillator network is treated as a simplicial complex
``(V, E, T)``: vertices ``V`` are the oscillators, edges ``E`` are the
unordered pairs ``{i, j}`` (``i < j``) carrying non-zero symmetric
coupling, and triangles ``T`` are the 2-simplices (3-cliques of the
coupling graph, or an explicit user-supplied set).

The decomposed object is the **alternating edge flow** — the Kuramoto
coupling current on the reference orientation ``i → j`` (``i < j``):

    f_{ij} = K^{sym}_{ij} · sin(θ_j − θ_i),   K^{sym} = ½(K + Kᵀ)

which satisfies ``f_{ji} = −f_{ij}`` exactly, the defining property of a
1-cochain. Using the symmetric part of ``K`` keeps the current
alternating even when ``K`` encodes directed coupling; for the standard
symmetric Kuramoto model ``K^{sym} = K``.

Boundary operators
------------------
* ``B1`` (``|V| × |E|``) is the node–edge incidence ``∂₁``: for edge
  ``e = (i, j)`` with ``i < j``, ``B1[i, e] = −1`` and ``B1[j, e] = +1``.
  The discrete gradient is ``grad(s) = B1ᵀ s`` with
  ``(B1ᵀ s)_{ij} = s_j − s_i``; the divergence is its adjoint ``B1``.
* ``B2`` (``|E| × |T|``) is the edge–triangle incidence ``∂₂``: for
  triangle ``t = {i, j, k}`` with ``i < j < k`` the simplicial boundary
  ``∂[i, j, k] = [j, k] − [i, k] + [i, j]`` gives
  ``B2[(i,j), t] = +1``, ``B2[(j,k), t] = +1``, ``B2[(i,k), t] = −1``.
  The discrete curl is ``curl(f) = B2ᵀ f``.

Because ``∂₁ ∂₂ = B1 B2 = 0`` (boundary of a boundary is empty), the
gradient image ``im(B1ᵀ)`` and the curl image ``im(B2)`` are
L²-orthogonal.

Decomposition
-------------
With graph Laplacian ``L0 = B1 B1ᵀ`` and triangle Laplacian
``L2 = B2ᵀ B2``:

    f_grad = B1ᵀ · L0⁺ · (B1 f)     (curl-free, conservative)
    f_curl = B2  · L2⁺ · (B2ᵀ f)    (divergence-free, rotational)
    f_harm = f − f_grad − f_curl    (harmonic: ker of the Hodge
                                     1-Laplacian L1 = B1ᵀB1 + B2 B2ᵀ)

The three components are mutually L²-orthogonal (Jiang, Lim, Yao & Ye
2011, Theorem 2.4). The harmonic part is both divergence-free and
curl-free; its dimension equals the first Betti number

    β₁ = |E| − rank(B1) − rank(B2)

i.e. the number of independent cycles not bounded by triangles. On a
triangle-free graph with a cycle (e.g. a square) a circulating current
is *purely harmonic* — the topological content that a plain
symmetric/antisymmetric matrix split cannot represent.

Output
------
:class:`HodgeResult` returns the three components and the input current
as antisymmetric ``(N, N)`` flow matrices (``M[i, j]`` is the flow on
edge ``i → j``), the minimum-norm node potential ``s`` such that
``f_grad = grad(s)``, and the integer ``betti_one`` (``β₁``).

Numerics
--------
The decomposition needs two least-squares solves (``L0⁺``, ``L2⁺``), so
exact cross-language parity is not attainable; the dispatcher validates
each accelerated backend against the NumPy reference within
``rtol = 1e-10`` / ``atol = 1e-12`` (matching the spectral solver) and
falls back to NumPy on any valid numerical mismatch. Malformed backend
payloads fail closed before parity fallback.

Reference
---------
Jiang, Lim, Yao & Ye 2011, *Statistical ranking and combinatorial Hodge
theory*, Math. Program. **127** (1):203–244.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling._hodge_validation import (
    validate_hodge_backend_output,
)
from scpn_phase_orchestrator.coupling._julia_runtime import require_juliacall_main

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
# (gradient, curl, harmonic) edge-flow matrices, each (N, N) antisymmetric.
HodgeTuple: TypeAlias = tuple[FloatArray, FloatArray, FloatArray]
HodgeBackend: TypeAlias = Callable[..., HodgeTuple]

# Backend acceptance tolerance: the least-squares pseudoinverse solves
# preclude bit-exact parity across LAPACK/BLAS implementations.
_BACKEND_RTOL = 1e-10
_BACKEND_ATOL = 1e-12

# Relative singular-value cutoff for the symmetric PSD pseudoinverse,
# shared by every backend so the gradient/curl projections agree.
_PINV_RCOND = 1e-9


__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "HodgeResult",
    "hodge_decomposition",
]


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> HodgeBackend:
    """Load the Rust Hodge-decomposition backend callable."""
    from spo_kernel import hodge_decomposition_rust

    def _rust(
        knm_flat: FloatArray,
        phases: FloatArray,
        n: int,
        edges_flat: IntArray,
        n_edges: int,
        tris_flat: IntArray,
        n_tris: int,
    ) -> HodgeTuple:
        """Call the Rust Hodge-decomposition kernel."""
        g, c, h = hodge_decomposition_rust(
            np.ascontiguousarray(knm_flat.ravel(), dtype=np.float64),
            np.ascontiguousarray(phases.ravel(), dtype=np.float64),
            int(n),
            np.ascontiguousarray(edges_flat.ravel(), dtype=np.int64),
            int(n_edges),
            np.ascontiguousarray(tris_flat.ravel(), dtype=np.int64),
            int(n_tris),
        )
        return validate_hodge_backend_output((g, c, h), n=n)

    return cast("HodgeBackend", _rust)


def _load_mojo_fn() -> HodgeBackend:
    # pragma: no cover — toolchain
    """Load the Mojo Hodge-decomposition backend callable."""
    from ..experimental.accelerators.coupling._hodge_mojo import (
        _ensure_exe,
        hodge_decomposition_mojo,
    )

    _ensure_exe()
    return hodge_decomposition_mojo


def _load_julia_fn() -> HodgeBackend:
    # pragma: no cover — toolchain
    """Load the Julia Hodge-decomposition backend callable."""
    require_juliacall_main()
    from ..experimental.accelerators.coupling._hodge_julia import (
        hodge_decomposition_julia,
    )

    return hodge_decomposition_julia


def _load_go_fn() -> HodgeBackend:
    # pragma: no cover — toolchain
    """Load the Go Hodge-decomposition backend callable."""
    from ..experimental.accelerators.coupling._hodge_go import (
        _load_lib,
        hodge_decomposition_go,
    )

    _load_lib()
    return hodge_decomposition_go


_LOADERS: dict[
    str,
    Callable[[], HodgeBackend],
] = {
    "rust": _load_rust_fn,
    "mojo": _load_mojo_fn,
    "julia": _load_julia_fn,
    "go": _load_go_fn,
}
_BACKEND_CACHE: dict[str, HodgeBackend] = {}


def _load_backend(name: str) -> HodgeBackend:
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


def _dispatch() -> HodgeBackend | None:
    """Return the fastest available backend callable, or ``None`` for Python."""
    ordered_backends = [ACTIVE_BACKEND] + list(AVAILABLE_BACKENDS)
    seen: set[str] = set()
    for backend in ordered_backends:
        if backend in seen:
            continue
        seen.add(backend)
        if backend == "python":
            return None
        try:
            return _load_backend(backend)
        except (ImportError, RuntimeError, OSError, KeyError):
            continue
    return None


@dataclass
class HodgeResult:
    """Decompose the Kuramoto coupling current into three L²-orthogonal flows.

    Each flow matrix is antisymmetric: ``M[i, j]`` is the flow on the
    oriented edge ``i → j`` and ``M[j, i] = −M[i, j]``.

    Attributes
    ----------
        gradient: Conservative (curl-free) component ``grad(s)``.
        curl: Rotational (divergence-free) component bounded by triangles.
        harmonic: Topological residual in ``ker(L1)``; non-zero exactly
            when the graph carries cycles not filled by triangles.
        flow: The input alternating coupling current
            ``K^{sym}_{ij} · sin(θ_j − θ_i)``.
        potential: Minimum-norm node potential ``s`` with
            ``gradient = grad(s)``.
        betti_one: First Betti number ``β₁`` — the dimension of the
            harmonic subspace.
    """

    gradient: FloatArray
    curl: FloatArray
    harmonic: FloatArray
    flow: FloatArray
    potential: FloatArray
    betti_one: int


def _contains_boolean_alias(value: object) -> bool:
    """Return whether the value contains any boolean alias."""
    raw = np.asarray(value, dtype=object)
    return any(isinstance(item, (bool, np.bool_)) for item in raw.ravel())


def _validate_phase_vector(value: object, *, name: str) -> FloatArray:
    """Return the phases as a validated 1-D finite array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    try:
        phases = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 1-D phase vector") from exc
    if phases.ndim != 1:
        raise ValueError(f"{name} must be a finite 1-D phase vector")
    if not np.all(np.isfinite(phases)):
        raise ValueError(f"{name} must contain only finite values")
    return phases


def _validate_coupling_matrix(value: object, *, expected_n: int) -> FloatArray:
    """Return the coupling as a validated finite square matrix, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError("knm must not contain boolean values")
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError("knm must not contain boolean values")
    try:
        matrix = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("knm must be a finite square matrix") from exc
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("knm must be a finite square matrix")
    if matrix.shape != (expected_n, expected_n):
        raise ValueError(
            f"knm shape {matrix.shape} does not match ({expected_n}, {expected_n})"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("knm must contain only finite values")
    return matrix


def _build_edges(k_sym: FloatArray, n: int) -> IntArray:
    """Return the ordered ``i < j`` edge list of non-zero symmetric couplings."""
    iu, ju = np.triu_indices(n, k=1)
    if iu.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    mask = k_sym[iu, ju] != 0.0
    return np.column_stack((iu[mask], ju[mask])).astype(np.int64)


def _validate_triangles(
    triangles: Sequence[Sequence[int]],
    *,
    edge_set: frozenset[tuple[int, int]],
    n: int,
) -> IntArray:
    """Validate an explicit triangle set against the edge support."""
    rows: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for raw_tri in triangles:
        nodes = tuple(raw_tri)
        if len(nodes) != 3:
            raise ValueError("each triangle must have exactly three nodes")
        for node in nodes:
            if isinstance(node, bool) or not isinstance(node, Integral):
                raise ValueError(f"triangle node must be an integer, got {node!r}")
        i, j, k = sorted(int(node) for node in nodes)
        if i == j or j == k:
            raise ValueError("triangle nodes must be distinct")
        if i < 0 or k >= n:
            raise ValueError(f"triangle node outside [0, {n})")
        if (i, j) not in edge_set or (j, k) not in edge_set or (i, k) not in edge_set:
            raise ValueError(
                f"triangle {{{i}, {j}, {k}}} requires all three edges to exist"
            )
        key = (i, j, k)
        if key in seen:
            raise ValueError(f"duplicate triangle {{{i}, {j}, {k}}}")
        seen.add(key)
        rows.append(key)
    if not rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def _build_triangles(
    edge_set: frozenset[tuple[int, int]],
    n: int,
) -> IntArray:
    """Return all 3-cliques ``i < j < k`` of the coupling graph."""
    rows: list[tuple[int, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if (i, j) not in edge_set:
                continue
            for k in range(j + 1, n):
                if (i, k) in edge_set and (j, k) in edge_set:
                    rows.append((i, j, k))
    if not rows:
        return np.empty((0, 3), dtype=np.int64)
    return np.array(rows, dtype=np.int64)


def _incidence_operators(
    edges: IntArray,
    triangles: IntArray,
    n: int,
) -> tuple[FloatArray, FloatArray]:
    """Assemble the node–edge ``B1`` and edge–triangle ``B2`` boundary matrices."""
    n_edges = int(edges.shape[0])
    n_tris = int(triangles.shape[0])
    b1 = np.zeros((n, n_edges), dtype=np.float64)
    edge_index: dict[tuple[int, int], int] = {}
    for e_idx in range(n_edges):
        i = int(edges[e_idx, 0])
        j = int(edges[e_idx, 1])
        b1[i, e_idx] = -1.0
        b1[j, e_idx] = 1.0
        edge_index[(i, j)] = e_idx
    b2 = np.zeros((n_edges, n_tris), dtype=np.float64)
    for t_idx in range(n_tris):
        i = int(triangles[t_idx, 0])
        j = int(triangles[t_idx, 1])
        k = int(triangles[t_idx, 2])
        b2[edge_index[(i, j)], t_idx] += 1.0
        b2[edge_index[(j, k)], t_idx] += 1.0
        b2[edge_index[(i, k)], t_idx] -= 1.0
    return b1, b2


def _embed_flow(values: FloatArray, edges: IntArray, n: int) -> FloatArray:
    """Embed an edge-flow vector into an antisymmetric ``(N, N)`` matrix."""
    matrix = np.zeros((n, n), dtype=np.float64)
    for e_idx in range(int(edges.shape[0])):
        i = int(edges[e_idx, 0])
        j = int(edges[e_idx, 1])
        matrix[i, j] = values[e_idx]
        matrix[j, i] = -values[e_idx]
    return matrix


def _edge_flow(k_sym: FloatArray, phases: FloatArray, edges: IntArray) -> FloatArray:
    """Sample the alternating coupling current on the edge list."""
    n_edges = int(edges.shape[0])
    if n_edges == 0:
        return np.empty(0, dtype=np.float64)
    i = edges[:, 0]
    j = edges[:, 1]
    return k_sym[i, j] * np.sin(phases[j] - phases[i])


def _psd_pinv_apply(matrix: FloatArray, vector: FloatArray) -> FloatArray:
    """Apply the Moore–Penrose pseudoinverse of an SPD matrix to a vector.

    Eigenvalues at or below ``_PINV_RCOND · λ_max`` are treated as the
    null space. Every backend mirrors this algorithm so the projections
    agree to within the dispatcher tolerance.
    """
    if matrix.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    lambda_max = float(eigenvalues[-1]) if eigenvalues.size else 0.0
    cutoff = _PINV_RCOND * lambda_max if lambda_max > 0.0 else 0.0
    keep = eigenvalues > cutoff
    inv_eigenvalues = np.divide(
        1.0,
        eigenvalues,
        out=np.zeros_like(eigenvalues),
        where=keep,
    )
    projected = eigenvectors.T @ vector
    return cast("FloatArray", eigenvectors @ (inv_eigenvalues * projected))


def _hodge_components(
    flow: FloatArray,
    b1: FloatArray,
    b2: FloatArray,
    n: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Project an edge flow onto gradient, curl, and harmonic spaces."""
    if flow.size == 0:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty.copy(), empty.copy(), np.zeros(n, dtype=np.float64)
    potential = _psd_pinv_apply(b1 @ b1.T, b1 @ flow)
    f_grad = b1.T @ potential
    if b2.shape[1] == 0:
        f_curl = np.zeros(flow.size, dtype=np.float64)
    else:
        triangle_pot = _psd_pinv_apply(b2.T @ b2, b2.T @ flow)
        f_curl = b2 @ triangle_pot
    f_harm = flow - f_grad - f_curl
    return f_grad, f_curl, f_harm, potential


def _betti_one(b1: FloatArray, b2: FloatArray, n_edges: int) -> int:
    """First Betti number ``β₁ = |E| − rank(B1) − rank(B2)``."""
    if n_edges == 0:
        return 0
    rank_b1 = int(np.linalg.matrix_rank(b1))
    rank_b2 = 0 if b2.shape[1] == 0 else int(np.linalg.matrix_rank(b2))
    return max(0, n_edges - rank_b1 - rank_b2)


def _simplicial_complex(
    k_sym: FloatArray,
    n: int,
    triangles: Sequence[Sequence[int]] | None,
) -> tuple[IntArray, IntArray]:
    """Build the ``(edges, triangles)`` index arrays for the coupling graph."""
    edges = _build_edges(k_sym, n)
    edge_set = frozenset((int(a), int(b)) for a, b in edges)
    if triangles is None:
        tri = _build_triangles(edge_set, n)
    else:
        tri = _validate_triangles(triangles, edge_set=edge_set, n=n)
    return edges, tri


def _decompose(
    k: FloatArray,
    phases: FloatArray,
    edges: IntArray,
    triangles: IntArray,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, int]:
    """Decompose the coupling current on a fixed simplicial complex.

    Returns ``(gradient, curl, harmonic, flow, potential, betti_one)``
    with the four flows as antisymmetric ``(N, N)`` matrices.
    """
    n = int(phases.size)
    k_sym = 0.5 * (k + k.T)
    flow_vec = _edge_flow(k_sym, phases, edges)
    flow_matrix = _embed_flow(flow_vec, edges, n)
    b1, b2 = _incidence_operators(edges, triangles, n)
    f_grad, f_curl, f_harm, potential = _hodge_components(flow_vec, b1, b2, n)
    betti = _betti_one(b1, b2, int(edges.shape[0]))
    return (
        _embed_flow(f_grad, edges, n),
        _embed_flow(f_curl, edges, n),
        _embed_flow(f_harm, edges, n),
        flow_matrix,
        potential,
        betti,
    )


def _python_decomposition(
    k: FloatArray,
    phases: FloatArray,
    triangles: Sequence[Sequence[int]] | None = None,
) -> HodgeTuple:
    """Return the gradient/curl/harmonic flow matrices (NumPy reference)."""
    n = int(phases.size)
    k_sym = 0.5 * (k + k.T)
    edges, tri = _simplicial_complex(k_sym, n, triangles)
    gradient, curl, harmonic, _, _, _ = _decompose(k, phases, edges, tri)
    return gradient, curl, harmonic


def _normalise_backend_output(
    output: object,
    *,
    expected_n: int,
) -> HodgeTuple:
    """Return the normalised backend Hodge decomposition output."""
    return validate_hodge_backend_output(output, n=expected_n)


def _backend_matches_reference(
    backend_output: HodgeTuple,
    reference: HodgeTuple,
) -> bool:
    """Return whether the backend output matches the reference."""
    return all(
        np.allclose(actual, expected, rtol=_BACKEND_RTOL, atol=_BACKEND_ATOL)
        for actual, expected in zip(backend_output, reference, strict=True)
    )


def hodge_decomposition(
    knm: FloatArray,
    phases: FloatArray,
    triangles: Sequence[Sequence[int]] | None = None,
) -> HodgeResult:
    """Decompose the Kuramoto coupling current into orthogonal edge flows.

    Parameters
    ----------
    knm : FloatArray
        Square ``(N, N)`` coupling matrix; the symmetric part defines the edge support
        and the current magnitude.
    phases : FloatArray
        ``(N,)`` oscillator phases.
    triangles : Sequence[Sequence[int]] | None
        Optional explicit 2-simplices as node triples; each must reference existing
        edges. When omitted, all 3-cliques of the coupling graph are used.

    Returns
    -------
    HodgeResult
        :class:`HodgeResult` with the three flow components as antisymmetric ``(N, N)``
        matrices, the input current, the node potential, and the first Betti number.
    """
    phases = _validate_phase_vector(phases, name="phases")
    n = int(phases.size)
    if n == 0:
        empty = np.zeros((0, 0), dtype=np.float64)
        return HodgeResult(
            gradient=empty,
            curl=empty.copy(),
            harmonic=empty.copy(),
            flow=empty.copy(),
            potential=np.array([], dtype=np.float64),
            betti_one=0,
        )

    k = _validate_coupling_matrix(knm, expected_n=n)
    k_sym = 0.5 * (k + k.T)
    edges, tri = _simplicial_complex(k_sym, n, triangles)

    gradient, curl, harmonic, flow_matrix, potential, betti = _decompose(
        k, phases, edges, tri
    )
    reference: HodgeTuple = (gradient, curl, harmonic)

    n_edges = int(edges.shape[0])
    n_tris = int(tri.shape[0])
    edges_flat = np.ascontiguousarray(edges.ravel(), dtype=np.int64)
    tris_flat = np.ascontiguousarray(tri.ravel(), dtype=np.int64)
    k_flat = np.ascontiguousarray(k.ravel(), dtype=np.float64)

    backend_fn = _dispatch()
    if backend_fn is not None:
        backend_output = _normalise_backend_output(
            backend_fn(
                k_flat,
                phases,
                n,
                edges_flat,
                n_edges,
                tris_flat,
                n_tris,
            ),
            expected_n=n,
        )
        if _backend_matches_reference(backend_output, reference):
            g, c, h = backend_output
            return HodgeResult(
                gradient=g,
                curl=c,
                harmonic=h,
                flow=flow_matrix,
                potential=potential,
                betti_one=betti,
            )

    return HodgeResult(
        gradient=gradient,
        curl=curl,
        harmonic=harmonic,
        flow=flow_matrix,
        potential=potential,
        betti_one=betti,
    )
