# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generalised k-body hypergraph coupling

"""Hypergraph Kuramoto: arbitrary k-body interactions beyond
pairwise, with a 5-backend fallback chain per
``feedback_module_standard_attnres.md``.

Extends the standard Kuramoto model with k-body coupling terms for
any k ≥ 2. The standard model (k=2) and simplicial model (k=3) are
special cases.

For a k-hyperedge {i₁, …, iₖ}, the coupling on oscillator ``iₘ`` is

    σₖ · sin( Σ_{j≠m} θ_{iⱼ} − (k−1)·θ_{iₘ} )

which generalises

    k = 2:  sin(θ_j − θ_i)            — standard Kuramoto
    k = 3:  sin(θ_j + θ_k − 2·θ_i)    — simplicial / triadic

The engine also accepts a dense pairwise coupling matrix
``pairwise_knm`` (optional) and an external-drive field ``(ζ, ψ)``.

Numerics
--------
The pairwise-derivative loop uses the Rust kernel's
``sin(θ_j − θ_i) = sin(θ_j)·cos(θ_i) − cos(θ_j)·sin(θ_i)``
expansion in the ``alpha == 0`` fast path so that floating-point
rounding matches Rust (``spo-engine/src/hypergraph.rs``) bit-for-bit.
Alpha ≠ 0 falls back to the direct ``sin(θ_j − θ_i − α)`` form in
all five backends.

References
----------
Tanaka & Aoyagi 2011, Phys. Rev. Lett. 106:224101.
Skardal & Arenas 2019, Comm. Phys. 2:22.
Bick, Gross, Harrington & Schaub 2023, Nat. Rev. Physics 5:307-317.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = [
    "ACTIVE_BACKEND",
    "AVAILABLE_BACKENDS",
    "Hyperedge",
    "HypergraphEngine",
]


@dataclass
class Hyperedge:
    """A k-body interaction among oscillators.

    Attributes:
        nodes: Tuple of oscillator indices in this hyperedge.
        strength: Coupling strength σₖ for this hyperedge.
    """

    nodes: tuple[int, ...]
    strength: float = 1.0

    @property
    def order(self) -> int:
        return len(self.nodes)


_BACKEND_NAMES = ("rust", "mojo", "julia", "go", "python")


def _load_rust_fn() -> Callable[..., NDArray]:
    from spo_kernel import hypergraph_run_rust

    def _rust(
        phases: NDArray,
        omegas: NDArray,
        n: int,
        edge_nodes: NDArray,
        edge_offsets: NDArray,
        edge_strengths: NDArray,
        knm_flat: NDArray,
        alpha_flat: NDArray,
        zeta: float,
        psi: float,
        dt: float,
        n_steps: int,
    ) -> NDArray:
        return np.asarray(
            hypergraph_run_rust(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                int(n),
                np.ascontiguousarray(edge_nodes, dtype=np.int64),
                np.ascontiguousarray(edge_offsets, dtype=np.int64),
                np.ascontiguousarray(edge_strengths, dtype=np.float64),
                np.ascontiguousarray(knm_flat, dtype=np.float64),
                np.ascontiguousarray(alpha_flat, dtype=np.float64),
                float(zeta),
                float(psi),
                float(dt),
                int(n_steps),
            ),
            dtype=np.float64,
        )

    return _rust


def _load_mojo_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._hypergraph_mojo import (
        _ensure_exe,
        hypergraph_run_mojo,
    )

    _ensure_exe()
    return hypergraph_run_mojo


def _load_julia_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    import juliacall  # noqa: F401
    from scpn_phase_orchestrator.upde._hypergraph_julia import (
        hypergraph_run_julia,
    )

    return hypergraph_run_julia


def _load_go_fn() -> Callable[..., NDArray]:
    # pragma: no cover — toolchain
    from scpn_phase_orchestrator.upde._hypergraph_go import (
        hypergraph_run_go,
    )

    return hypergraph_run_go


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


def _python_run(
    phases: NDArray,
    omegas: NDArray,
    n: int,
    edge_nodes: NDArray,
    edge_offsets: NDArray,
    edge_strengths: NDArray,
    knm_flat: NDArray,
    alpha_flat: NDArray,
    zeta: float,
    psi: float,
    dt: float,
    n_steps: int,
) -> NDArray:
    """Python reference aligned to the Rust kernel.

    Uses the ``sin(θ_j − θ_i) = s_j·c_i − c_j·s_i`` expansion for
    the ``alpha == 0`` fast path (bit-identical to Rust) and the
    direct ``sin(diff)`` form for nonzero alpha.
    """
    p = np.asarray(phases, dtype=np.float64).copy()
    om = np.asarray(omegas, dtype=np.float64)
    has_pairwise = knm_flat.size == n * n
    knm = knm_flat.reshape(n, n) if has_pairwise else None
    has_alpha = alpha_flat.size == n * n
    alpha = alpha_flat.reshape(n, n) if has_alpha else None
    alpha_zero = (alpha is None) or bool(np.all(alpha == 0.0))

    n_edges = int(edge_offsets.size)
    for _ in range(n_steps):
        deriv = om.copy()
        if has_pairwise:
            if knm is None:
                raise RuntimeError("pairwise coupling matrix was not initialised")
            s = np.sin(p)
            c = np.cos(p)
            if alpha_zero:
                # sin(θ_j − θ_i) = s_j · c_i − c_j · s_i
                sin_diff = (
                    s[np.newaxis, :] * c[:, np.newaxis]
                    - c[np.newaxis, :] * s[:, np.newaxis]
                )
                deriv += np.sum(knm * sin_diff, axis=1)
            else:
                if alpha is None:
                    raise RuntimeError("phase frustration matrix was not initialised")
                diff = p[np.newaxis, :] - p[:, np.newaxis] - alpha
                deriv += np.sum(knm * np.sin(diff), axis=1)
        if zeta != 0.0:
            deriv += zeta * np.sin(psi - p)
        for e in range(n_edges):
            start = int(edge_offsets[e])
            stop = int(edge_offsets[e + 1]) if e + 1 < n_edges else int(edge_nodes.size)
            nodes = edge_nodes[start:stop]
            k = stop - start
            phase_sum = float(np.sum(p[nodes]))
            sigma = float(edge_strengths[e])
            for m_idx in nodes:
                m = int(m_idx)
                deriv[m] += sigma * np.sin(phase_sum - k * p[m])
        p = (p + dt * deriv) % TWO_PI
    return p


class HypergraphEngine:
    """Kuramoto engine with arbitrary k-body hypergraph coupling and
    a 5-backend fallback chain.

    Supports mixed-order interactions: some edges can be pairwise,
    some 3-body, some 4-body, etc. Each ``Hyperedge`` specifies
    which oscillators participate and the coupling strength.
    """

    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        hyperedges: list[Hyperedge] | None = None,
    ):
        self._n = n_oscillators
        self._dt = dt
        self._hyperedges: list[Hyperedge] = hyperedges or []

    def add_edge(self, nodes: tuple[int, ...], strength: float = 1.0) -> None:
        self._hyperedges.append(Hyperedge(nodes=nodes, strength=strength))

    def add_all_to_all(self, order: int, strength: float = 1.0) -> None:
        """Add all C(N, order) hyperedges of given order."""
        from itertools import combinations

        for combo in combinations(range(self._n), order):
            self._hyperedges.append(Hyperedge(nodes=combo, strength=strength))

    @property
    def n_edges(self) -> int:
        return len(self._hyperedges)

    def _encode_edges(self) -> tuple[NDArray, NDArray, NDArray]:
        edge_nodes_list: list[int] = []
        edge_offsets_list: list[int] = []
        edge_strengths_list: list[float] = []
        for edge in self._hyperedges:
            edge_offsets_list.append(len(edge_nodes_list))
            edge_nodes_list.extend(edge.nodes)
            edge_strengths_list.append(edge.strength)
        return (
            np.array(edge_nodes_list, dtype=np.int64),
            np.array(edge_offsets_list, dtype=np.int64),
            np.array(edge_strengths_list, dtype=np.float64),
        )

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        pairwise_knm: NDArray | None = None,
        alpha: NDArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
    ) -> NDArray:
        """One explicit-Euler step."""
        return self.run(
            phases,
            omegas,
            n_steps=1,
            pairwise_knm=pairwise_knm,
            alpha=alpha,
            zeta=zeta,
            psi=psi,
        )

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        n_steps: int,
        pairwise_knm: NDArray | None = None,
        alpha: NDArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
    ) -> NDArray:
        """Integrate ``n_steps`` Euler steps through the fastest
        available backend; return final phases."""
        en, eo, es = self._encode_edges()
        knm_flat = (
            np.ascontiguousarray(pairwise_knm, dtype=np.float64).ravel()
            if pairwise_knm is not None
            else np.empty(0, dtype=np.float64)
        )
        alpha_flat = (
            np.ascontiguousarray(alpha, dtype=np.float64).ravel()
            if alpha is not None
            else np.empty(0, dtype=np.float64)
        )
        backend_fn = _dispatch()
        if backend_fn is not None:
            return backend_fn(
                np.ascontiguousarray(phases, dtype=np.float64),
                np.ascontiguousarray(omegas, dtype=np.float64),
                self._n,
                en,
                eo,
                es,
                knm_flat,
                alpha_flat,
                float(zeta),
                float(psi),
                float(self._dt),
                int(n_steps),
            )
        return _python_run(
            np.ascontiguousarray(phases, dtype=np.float64),
            np.ascontiguousarray(omegas, dtype=np.float64),
            self._n,
            en,
            eo,
            es,
            knm_flat,
            alpha_flat,
            float(zeta),
            float(psi),
            float(self._dt),
            int(n_steps),
        )

    def order_parameter(self, phases: NDArray) -> float:
        """Standard Kuramoto R = |<exp(iθ)>|."""
        return float(np.abs(np.mean(np.exp(1j * phases))))
