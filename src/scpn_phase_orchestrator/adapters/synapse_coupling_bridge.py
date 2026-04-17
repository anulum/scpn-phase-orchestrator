# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — sc-neurocore synapse → coupling bridge

"""Bridge sc-neurocore spike-level synapse dynamics into SPO
phase-level coupling matrix K_nm.

Maps three synapse types to SPO coupling parameters:

1. **STDP weight changes → K_nm deltas**: spike-timing dependent
   plasticity modifies pairwise coupling strengths. Potentiation
   (dW > 0) strengthens K_ij; depression (dW < 0) weakens it.

2. **Gap junction conductance → phase coupling**: electrical
   synapses provide direct bidirectional coupling. g_c maps
   linearly to K_ij (symmetric).

3. **Tripartite astrocyte Ca²⁺ → imprint modulation**: astrocyte
   oscillations modulate the imprint memory vector m_k. High Ca²⁺
   enhances imprint accumulation; low Ca²⁺ accelerates decay.

Requires: pip install sc-neurocore>=3.13.0
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SynapseCouplingBridge",
    "SynapseSnapshot",
]


@dataclass
class SynapseSnapshot:
    """Snapshot of synapse state mapped to SPO parameters."""

    knm_delta: NDArray
    gap_coupling: NDArray
    astrocyte_modulation: NDArray
    mean_weight_change: float
    mean_conductance: float
    mean_ca: float


class SynapseCouplingBridge:
    """Map sc-neurocore synapse dynamics to SPO coupling parameters.

    Usage::

        from sc_neurocore.synapses.triplet_stdp import TripletSTDP
        from sc_neurocore.synapses.gap_junction import GapJunction

        bridge = SynapseCouplingBridge(n_oscillators=8)

        # After each SNN step, feed weight changes
        bridge.update_stdp_weights(weight_matrix)
        bridge.update_gap_conductances(conductance_matrix)
        bridge.update_astrocyte_ca(ca_levels)

        # Get SPO coupling delta
        snap = bridge.snapshot()
        knm_new = knm_base + snap.knm_delta
    """

    def __init__(
        self,
        n_oscillators: int,
        stdp_scale: float = 1.0,
        gap_scale: float = 1.0,
        ca_scale: float = 1.0,
    ):
        self._n = n_oscillators
        self._stdp_scale = stdp_scale
        self._gap_scale = gap_scale
        self._ca_scale = ca_scale

        self._stdp_weights: NDArray = np.zeros((n_oscillators, n_oscillators))
        self._prev_weights: NDArray = np.zeros((n_oscillators, n_oscillators))
        self._gap_conductances: NDArray = np.zeros((n_oscillators, n_oscillators))
        self._ca_levels: NDArray = np.zeros(n_oscillators)

    def update_stdp_weights(self, weights: NDArray) -> None:
        """Feed current STDP weight matrix from sc-neurocore.

        The bridge computes dW = weights - prev_weights and maps
        to K_nm deltas.
        """
        self._prev_weights = self._stdp_weights.copy()
        self._stdp_weights = np.asarray(weights, dtype=np.float64)

    def update_gap_conductances(self, conductances: NDArray) -> None:
        """Feed gap junction conductance matrix.

        Symmetric: g_c(i,j) = g_c(j,i). Maps directly to K_ij.
        """
        g = np.asarray(conductances, dtype=np.float64)
        self._gap_conductances = 0.5 * (g + g.T)
        np.fill_diagonal(self._gap_conductances, 0.0)

    def update_astrocyte_ca(self, ca_levels: NDArray) -> None:
        """Feed astrocyte Ca²⁺ concentration per oscillator.

        High Ca²⁺ → strong imprint modulation (facilitates learning).
        """
        self._ca_levels = np.asarray(ca_levels, dtype=np.float64).ravel()

    def snapshot(self) -> SynapseSnapshot:
        """Compute SPO coupling parameters from current synapse state."""
        # STDP → K_nm delta
        dw = self._stdp_weights - self._prev_weights
        knm_delta = dw * self._stdp_scale
        np.fill_diagonal(knm_delta, 0.0)

        # Gap junction → symmetric coupling
        gap_coupling = self._gap_conductances * self._gap_scale

        # Astrocyte Ca²⁺ → imprint modulation vector
        ca_norm = self._ca_levels / max(self._ca_levels.max(), 1e-10)
        astro_mod = ca_norm * self._ca_scale

        return SynapseSnapshot(
            knm_delta=knm_delta,
            gap_coupling=gap_coupling,
            astrocyte_modulation=astro_mod,
            mean_weight_change=float(np.mean(np.abs(dw))),
            mean_conductance=float(np.mean(self._gap_conductances)),
            mean_ca=float(np.mean(self._ca_levels)),
        )

    def apply_to_knm(self, knm_base: NDArray) -> NDArray:
        """Apply all synapse-derived modifications to a base K_nm."""
        snap = self.snapshot()
        knm = knm_base + snap.knm_delta + snap.gap_coupling
        result: NDArray = np.maximum(knm, 0.0)
        np.fill_diagonal(result, 0.0)
        return result

    def apply_to_imprint(self, m_k: NDArray) -> NDArray:
        """Modulate imprint vector by astrocyte Ca²⁺ levels."""
        snap = self.snapshot()
        result: NDArray = m_k * (1.0 + snap.astrocyte_modulation)
        return result
