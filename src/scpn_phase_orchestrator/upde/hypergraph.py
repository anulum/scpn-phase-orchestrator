# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Generalized k-body hypergraph coupling

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

try:
    from spo_kernel import (
        PyHypergraphStepper as _HypergraphStepper,
    )

    _HAS_RUST = True
except ImportError:
    _HAS_RUST = False

__all__ = ["Hyperedge", "HypergraphEngine"]


@dataclass
class Hyperedge:
    nodes: tuple[int, ...]
    strength: float = 1.0

    @property
    def order(self) -> int:
        return len(self.nodes)


class HypergraphEngine:
    def __init__(
        self,
        n_oscillators: int,
        dt: float,
        hyperedges: list[Hyperedge] | None = None,
    ):
        self._n = n_oscillators
        self._dt = dt
        self._hyperedges: list[Hyperedge] = hyperedges or []
        if _HAS_RUST:
            self._stepper = _HypergraphStepper(n_oscillators, dt)
        else:
            self._stepper = None

    def add_edge(self, nodes: tuple[int, ...], strength: float = 1.0) -> None:
        self._hyperedges.append(Hyperedge(nodes=nodes, strength=strength))

    def step(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: NDArray | None = None,
    ) -> NDArray:
        if _HAS_RUST:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            k = np.ascontiguousarray(
                knm.ravel() if knm is not None else np.array([]),
                dtype=np.float64,
            )
            a = np.ascontiguousarray(
                alpha.ravel() if alpha is not None else np.array([]),
                dtype=np.float64,
            )
            edges_raw = [(list(e.nodes), e.strength) for e in self._hyperedges]
            result: NDArray = np.asarray(
                self._stepper.step(p, o, edges_raw, k, a, zeta, psi)
            )
            return result
        dtheta = self._derivative(phases, omegas, knm, alpha, zeta, psi)
        return (phases + self._dt * dtheta) % TWO_PI

    def run(
        self,
        phases: NDArray,
        omegas: NDArray,
        knm: NDArray | None = None,
        zeta: float = 0.0,
        psi: float = 0.0,
        alpha: NDArray | None = None,
        n_steps: int = 100,
    ) -> NDArray:
        if _HAS_RUST:
            p = np.ascontiguousarray(phases, dtype=np.float64)
            o = np.ascontiguousarray(omegas, dtype=np.float64)
            k = np.ascontiguousarray(
                knm.ravel() if knm is not None else np.array([]),
                dtype=np.float64,
            )
            a = np.ascontiguousarray(
                alpha.ravel() if alpha is not None else np.array([]),
                dtype=np.float64,
            )
            edges_raw = [(list(e.nodes), e.strength) for e in self._hyperedges]
            result_run: NDArray = np.asarray(
                self._stepper.run(p, o, edges_raw, k, a, zeta, psi, n_steps)
            )
            return result_run
        p = phases.copy()
        for _ in range(n_steps):
            p = self.step(p, omegas, knm, zeta, psi, alpha)
        return p

    def _derivative(
        self,
        theta: NDArray,
        omegas: NDArray,
        knm: NDArray | None,
        alpha: NDArray | None,
        zeta: float,
        psi: float,
    ) -> NDArray:
        n = self._n
        dtheta = omegas.copy()
        if knm is not None:
            if alpha is None:
                alpha = np.zeros((n, n))
            diff = theta[np.newaxis, :] - theta[:, np.newaxis] - alpha
            dtheta += np.sum(knm * np.sin(diff), axis=1)
        for edge in self._hyperedges:
            k = edge.order
            nodes = edge.nodes
            sigma = edge.strength
            phase_sum = sum(theta[j] for j in nodes)
            for m in nodes:
                dtheta[m] += sigma * np.sin(phase_sum - k * theta[m])
        if zeta != 0.0:
            dtheta += zeta * np.sin(psi - theta)
        return dtheta

    def order_parameter(self, phases: NDArray) -> float:
        return float(np.abs(np.mean(np.exp(1j * phases))))
