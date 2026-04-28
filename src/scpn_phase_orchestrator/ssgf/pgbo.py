# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator - Phase-Geometry Bidirectional Observer

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.ssgf.costs import SSGFCosts, compute_ssgf_costs
from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

__all__ = ["PGBO", "PGBOSnapshot"]


@dataclass
class PGBOSnapshot:
    """Observation from the Phase-Geometry Bidirectional Observer at one timestep."""

    R: float
    psi: float
    costs: SSGFCosts
    phase_geometry_alignment: float
    gauge_curvature: float
    step: int


class PGBO:
    """Phase-Geometry Bidirectional Observer.

    Monitors the alignment between phase dynamics (Kuramoto state) and
    geometry (SSGF carrier W). Computes a scalar curvature proxy based
    on the rank-2 metric tensor h_{mu nu}. This represents the structural
    coupling between the phase-manifold and the underlying physical space.

    The bidirectionality:
        1. Phases -> Cost -> Gradient -> Geometry (forward control)
        2. Geometry -> Coupling -> Phases (backward influence)

    PGBO observes the emergence of curvature as synchronization patterns
    stretch or curve the geometric coupling field W.
    """

    def __init__(self, cost_weights: tuple[float, ...] = (1.0, 0.5, 0.1, 0.1)):
        if len(cost_weights) < 1:
            raise ValueError("cost_weights must contain at least one weight")
        if any(w < 0.0 for w in cost_weights):
            raise ValueError(f"cost_weights must be non-negative, got {cost_weights}")
        self._weights = cost_weights
        self._step = 0
        self._history: list[PGBOSnapshot] = []

    def observe(self, phases: NDArray, W: NDArray) -> PGBOSnapshot:
        """Compute coherence, SSGF costs, and phase-geometry alignment.

        Args:
            phases: Current phase vector theta_i.
            W: Current geometric coupling matrix W_ij.

        Returns:
            A PGBOSnapshot containing order parameter R, Psi, SSGF costs,
            and the gauge curvature proxy.
        """
        self._step += 1
        R, psi = compute_order_parameter(phases)
        costs = compute_ssgf_costs(W, phases, weights=self._weights)

        # Phase-geometry alignment: correlation between pairwise PLV
        # and coupling strength W_ij
        n = len(phases)
        if n < 2:
            alignment = 0.0
            gauge_curvature = 0.0
        else:
            diff = phases[:, np.newaxis] - phases[np.newaxis, :]
            plv_matrix = np.cos(diff)
            triu = np.triu_indices(n, k=1)
            plv_flat = plv_matrix[triu]
            w_flat = W[triu]
            if np.std(plv_flat) < 1e-12 or np.std(w_flat) < 1e-12:
                alignment = 0.0
            else:
                alignment = float(np.corrcoef(plv_flat, w_flat)[0, 1])
                if not np.isfinite(alignment):
                    alignment = 0.0

            # Compute Gauge-Theoretic Metric Tensor h_munu
            # h_ij = W_ij * cos(theta_i - theta_j)
            if np.sum(np.abs(W)) < 1e-12:
                gauge_curvature = 0.0
            else:
                h_munu = W * plv_matrix
                # Scalar curvature proxy: sum(h_ij) / sum(|W_ij|)
                gauge_curvature = float(np.sum(h_munu) / np.sum(np.abs(W)))

        snap = PGBOSnapshot(
            R=R,
            psi=psi,
            costs=costs,
            phase_geometry_alignment=alignment,
            gauge_curvature=gauge_curvature,
            step=self._step,
        )
        self._history.append(snap)
        return snap

    @property
    def history(self) -> list[PGBOSnapshot]:
        """All snapshots recorded so far."""
        return list(self._history)

    def alignment_trend(self, window: int = 10) -> float:
        """Mean alignment over last window observations."""
        if not self._history:
            return 0.0
        recent = self._history[-window:]
        return float(np.mean([s.phase_geometry_alignment for s in recent]))

    def reset(self) -> None:
        """Clear step counter and observation history."""
        self._step = 0
        self._history.clear()
