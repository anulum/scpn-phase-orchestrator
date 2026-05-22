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
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "SynapseCouplingBridge",
    "SynapseSnapshot",
]
FloatArray: TypeAlias = NDArray[np.float64]


def _validate_n_oscillators(n_oscillators: int) -> int:
    if (
        isinstance(n_oscillators, bool)
        or not isinstance(n_oscillators, int)
        or n_oscillators <= 0
    ):
        raise ValueError("n_oscillators must be a positive integer")
    return n_oscillators


def _validate_positive_scale(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite positive real number")
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a finite positive real number")
    return result


def _validate_nonnegative_real(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative real number")
    return result


def _finite_array(value: FloatArray, name: str) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_square_matrix(value: FloatArray, name: str, n: int) -> FloatArray:
    array = _finite_array(value, name)
    if array.shape != (n, n):
        raise ValueError(f"{name} must have shape ({n}, {n})")
    return array.copy()


def _validate_nonnegative_square_matrix(
    value: FloatArray,
    name: str,
    n: int,
) -> FloatArray:
    array = _validate_square_matrix(value, name, n)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


def _validate_vector(value: FloatArray, name: str, n: int) -> FloatArray:
    array = _finite_array(value, name)
    if array.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},)")
    return array.copy()


def _validate_nonnegative_vector(value: FloatArray, name: str, n: int) -> FloatArray:
    array = _validate_vector(value, name, n)
    if np.any(array < 0.0):
        raise ValueError(f"{name} must contain only non-negative values")
    return array


@dataclass
class SynapseSnapshot:
    """Snapshot of synapse state mapped to SPO parameters."""

    knm_delta: FloatArray
    gap_coupling: FloatArray
    astrocyte_modulation: FloatArray
    mean_weight_change: float
    mean_conductance: float
    mean_ca: float

    def __post_init__(self) -> None:
        n = _validate_n_oscillators(self.knm_delta.shape[0])
        self.knm_delta = _validate_square_matrix(self.knm_delta, "knm_delta", n)
        self.gap_coupling = _validate_nonnegative_square_matrix(
            self.gap_coupling,
            "gap_coupling",
            n,
        )
        self.astrocyte_modulation = _validate_nonnegative_vector(
            self.astrocyte_modulation,
            "astrocyte_modulation",
            n,
        )
        self.mean_weight_change = _validate_nonnegative_real(
            "mean_weight_change",
            self.mean_weight_change,
        )
        self.mean_conductance = _validate_nonnegative_real(
            "mean_conductance",
            self.mean_conductance,
        )
        self.mean_ca = _validate_nonnegative_real(
            "mean_ca",
            self.mean_ca,
        )


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
        self._n = _validate_n_oscillators(n_oscillators)
        self._stdp_scale = _validate_positive_scale("stdp_scale", stdp_scale)
        self._gap_scale = _validate_positive_scale("gap_scale", gap_scale)
        self._ca_scale = _validate_positive_scale("ca_scale", ca_scale)

        self._stdp_weights: FloatArray = np.zeros(
            (self._n, self._n),
            dtype=np.float64,
        )
        self._prev_weights: FloatArray = np.zeros(
            (self._n, self._n),
            dtype=np.float64,
        )
        self._gap_conductances: FloatArray = np.zeros(
            (self._n, self._n),
            dtype=np.float64,
        )
        self._ca_levels: FloatArray = np.zeros(self._n, dtype=np.float64)

    def update_stdp_weights(self, weights: FloatArray) -> None:
        """Feed current STDP weight matrix from sc-neurocore.

        The bridge computes dW = weights - prev_weights and maps
        to K_nm deltas.
        """
        self._prev_weights = self._stdp_weights.copy()
        self._stdp_weights = _validate_square_matrix(weights, "weights", self._n)

    def update_gap_conductances(self, conductances: FloatArray) -> None:
        """Feed gap junction conductance matrix.

        Symmetric: g_c(i,j) = g_c(j,i). Maps directly to K_ij.
        """
        g = _validate_nonnegative_square_matrix(
            conductances,
            "conductances",
            self._n,
        )
        self._gap_conductances = 0.5 * (g + g.T)
        np.fill_diagonal(self._gap_conductances, 0.0)

    def update_astrocyte_ca(self, ca_levels: FloatArray) -> None:
        """Feed astrocyte Ca²⁺ concentration per oscillator.

        High Ca²⁺ → strong imprint modulation (facilitates learning).
        """
        self._ca_levels = _validate_nonnegative_vector(
            ca_levels,
            "ca_levels",
            self._n,
        )

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

    def apply_to_knm(self, knm_base: FloatArray) -> FloatArray:
        """Apply all synapse-derived modifications to a base K_nm."""
        validated_knm_base = _validate_square_matrix(knm_base, "knm_base", self._n)
        snap = self.snapshot()
        knm = validated_knm_base + snap.knm_delta + snap.gap_coupling
        result: FloatArray = np.maximum(knm, 0.0)
        np.fill_diagonal(result, 0.0)
        return result

    def apply_to_imprint(self, m_k: FloatArray) -> FloatArray:
        """Modulate imprint vector by astrocyte Ca²⁺ levels."""
        validated_m_k = _validate_vector(m_k, "m_k", self._n)
        snap = self.snapshot()
        result: FloatArray = validated_m_k * (1.0 + snap.astrocyte_modulation)
        return result
