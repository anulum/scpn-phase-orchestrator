# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generalized k-body hypergraph coupling

"""Hypergraph Kuramoto: arbitrary k-body interactions beyond pairwise.

Extends the standard Kuramoto model with k-body coupling terms for
any k >= 2. The standard model (k=2) and simplicial model (k=3) are
special cases.

For a k-hyperedge {i₁, ..., iₖ}, the coupling on oscillator iₘ is:
    σₖ * sin(Σ_{j≠m} θ_{iⱼ} - (k-1)*θ_{iₘ})

This generalizes:
    k=2: sin(θ_j - θ_i)         — standard Kuramoto
    k=3: sin(θ_j + θ_k - 2θ_i)  — simplicial/triadic

References:
    Tanaka & Aoyagi 2011, Phys. Rev. Lett. 106:224101.
    Skardal & Arenas 2019, Comm. Phys. 2:22.
    Bick et al. 2023, Nat. Rev. Physics 5:307-317.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

try:
    from spo_kernel import (  # type: ignore[import-untyped]
        hypergraph_run_rust as _rust_hypergraph_run,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = [
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


class HypergraphEngine:
    """Kuramoto engine with arbitrary k-body hypergraph coupling.

    Supports mixed-order interactions: some edges can be pairwise,
    some 3-body, some 4-body, etc. Each Hyperedge specifies which
    oscillators participate and the coupling strength.
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

    def _derivative(
        self,
        theta: NDArray,
        omegas: NDArray,
        pairwise_knm: NDArray | None = None,
        alpha: NDArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
    ) -> NDArray:
        n = self._n
        dtheta = omegas.copy()

        # Standard pairwise coupling (if provided)
        if pairwise_knm is not None:
            if alpha is None:
                alpha = np.zeros((n, n))
            diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
            dtheta += np.sum(pairwise_knm * np.sin(diff), axis=1)

        # Hypergraph k-body coupling
        for edge in self._hyperedges:
            k = edge.order
            nodes = edge.nodes
            sigma = edge.strength

            # For each node m in the hyperedge:
            # coupling_m = σ * sin(Σ_{j∈edge, j≠m} θ_j - (k-1)*θ_m)
            # Total phase of all nodes in this hyperedge
            phase_sum = sum(theta[j] for j in nodes)

            for m in nodes:
                # Tanaka & Aoyagi 2011, Eq. 2: argument is
                # Σ_{j≠m} θ_j - (k-1)θ_m = (Σ_all θ) - k·θ_m
                arg = phase_sum - k * theta[m]
                dtheta[m] += sigma * np.sin(arg)

        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - theta)

        return dtheta

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        pairwise_knm: NDArray | None = None,
        alpha: NDArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
    ) -> NDArray:
        """Euler step."""
        d = self._derivative(phases, omegas, pairwise_knm, alpha, zeta, psi)
        return (phases + self._dt * d) % TWO_PI

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
        """Run multiple steps, return final phases."""
        if _HAS_RUST and self._hyperedges:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            # Flat-encode hyperedges
            edge_nodes: list[int] = []
            edge_offsets: list[int] = []
            edge_strengths: list[float] = []
            for edge in self._hyperedges:
                edge_offsets.append(len(edge_nodes))
                edge_nodes.extend(edge.nodes)
                edge_strengths.append(edge.strength)
            en = np.array(edge_nodes, dtype=np.int64)
            eo = np.array(edge_offsets, dtype=np.int64)
            es = np.array(edge_strengths, dtype=np.float64)
            kn = (
                np.ascontiguousarray(pairwise_knm.ravel(), dtype=np.float64)
                if pairwise_knm is not None
                else np.empty(0, dtype=np.float64)
            )
            al = (
                np.ascontiguousarray(alpha.ravel(), dtype=np.float64)
                if alpha is not None
                else np.empty(0, dtype=np.float64)
            )
            result: NDArray = _rust_hypergraph_run(
                p,
                o,
                self._n,
                en,
                eo,
                es,
                kn,
                al,
                zeta,
                psi,
                self._dt,
                n_steps,
            )
            return result
        p = phases.copy()
        for _ in range(n_steps):
            p = self.step(p, omegas, pairwise_knm, alpha, zeta, psi)
        return p

    def order_parameter(self, phases: NDArray) -> float:
        """Standard Kuramoto R = |<exp(iθ)>|."""
        return float(np.abs(np.mean(np.exp(1j * phases))))
