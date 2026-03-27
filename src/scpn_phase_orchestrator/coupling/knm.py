# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Coupling matrix builder

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import HAS_RUST as _HAS_RUST

__all__ = [
    "CouplingState",
    "CouplingBuilder",
    "SCPN_LAYER_TIMESCALES",
    "SCPN_LAYER_NAMES",
    "SCPN_CALIBRATION_ANCHORS",
]

# 16-layer SCPN hierarchy timescales (seconds).
# Source: HolonomicAtlas/src/knm_tools/knm_matrix_calculator.py,
# Paper 0 (UPDE framework), Paper 4 (synchronization).
SCPN_LAYER_TIMESCALES: dict[int, float] = {
    1: 0.1,  # Quantum Biological (~100ms)
    2: 0.004,  # Neurochemical (~25ms, 40Hz)
    3: 3600.0,  # Genomic (~1 hour)
    4: 2.0,  # Synchronization (~2s)
    5: 1.0,  # Psychoemotional (~1s)
    6: 86400.0,  # Planetary (~24h)
    7: 10.0,  # Geometrical-Symbolic (~10s)
    8: 31557600.0,  # Cosmic Phase Locking (~1 year)
    9: 3.154e9,  # Memory Imprint (~100 years)
    10: 1.0,  # Boundary Control (~1s)
    11: 86400.0,  # Noospheric (~24h)
    12: 31557600.0,  # Ecological Gaian (~1 year)
    13: 0.001,  # Source Field (~1ms)
    14: 1e-20,  # Transdimensional (~Planck)
    15: 1.0,  # Consilium Oversoul (~1s)
    16: 1e-30,  # Meta Director (atemporal, placeholder)
}

SCPN_LAYER_NAMES: dict[int, str] = {
    1: "Quantum",
    2: "Neural",
    3: "Genomic",
    4: "Tissue",
    5: "Psycho",
    6: "Planetary",
    7: "Symbolic",
    8: "Cosmic",
    9: "Memory",
    10: "Boundary",
    11: "Noospheric",
    12: "Gaian",
    13: "Source",
    14: "Transdim",
    15: "Consilium",
    16: "Meta",
}

# Calibration anchors from Paper 0 / HolonomicAtlas.
# These override the computed values for adjacent layers 1-5.
SCPN_CALIBRATION_ANCHORS: dict[tuple[int, int], float] = {
    (1, 2): 0.302,
    (2, 3): 0.201,
    (3, 4): 0.252,
    (4, 5): 0.154,
}


@dataclass(frozen=True)
class CouplingState:
    """Immutable snapshot of phase/amplitude coupling matrices and template."""

    knm: NDArray
    alpha: NDArray
    active_template: str
    knm_r: NDArray | None = None


class CouplingBuilder:
    """Builds Knm coupling matrices."""

    def build(
        self, n_layers: int, base_strength: float, decay_alpha: float
    ) -> CouplingState:
        """K_ij = base_strength * exp(-decay_alpha * |i - j|), zero diagonal."""
        if _HAS_RUST:  # pragma: no cover
            from spo_kernel import PyCouplingBuilder  # type: ignore[import-untyped]

            d = PyCouplingBuilder().build(n_layers, base_strength, decay_alpha)
            n = d["n"]
            rust_knm = np.asarray(d["knm"], dtype=np.float64).reshape(n, n)
            rust_alpha = np.asarray(d["alpha"], dtype=np.float64).reshape(n, n)
            return CouplingState(
                knm=rust_knm, alpha=rust_alpha, active_template="default"
            )
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm = base_strength * np.exp(-decay_alpha * dist)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n_layers, n_layers), dtype=np.float64)
        return CouplingState(knm=knm, alpha=alpha, active_template="default")

    def build_scpn_physics(
        self,
        k_base: float = 0.45,
        alpha_decay: float = 0.3,
    ) -> CouplingState:
        """Build 16×16 K_nm using SCPN layer physics.

        Three coupling mechanisms (Paper 0, HolonomicAtlas v2.4.0):
        - Adjacent: timescale matching with calibration anchors
        - Near-neighbor (|n-m|=2): geometric mean of intermediate path
        - Distant (|n-m|>=3): exponential decay with cross-hierarchy boosts

        Returns CouplingState with 16×16 matrix.
        """
        K = np.zeros((16, 16))

        # Pass 1: Adjacent layers. Use anchors where available.
        for n in range(1, 16):
            m = n + 1
            if (n, m) in SCPN_CALIBRATION_ANCHORS:
                val = SCPN_CALIBRATION_ANCHORS[(n, m)]
            else:
                val = self._adjacent_coupling(n, m, k_base)
            K[n - 1, m - 1] = val
            K[m - 1, n - 1] = val

        # Pass 2: Near-neighbor (|n-m|=2), geometric mean of intermediate path
        for n in range(1, 15):
            m = n + 2
            mid = (n + m) // 2
            k1 = K[n - 1, mid - 1]
            k2 = K[mid - 1, m - 1]
            val = float(np.sqrt(k1 * k2))
            # Frequency penalty for large timescale mismatch
            tau_n = SCPN_LAYER_TIMESCALES.get(n, 1.0)
            tau_m = SCPN_LAYER_TIMESCALES.get(m, 1.0)
            if n != 16 and m != 16 and tau_n > 0 and tau_m > 0:
                omega_n = 2.0 * np.pi / tau_n
                omega_m = 2.0 * np.pi / tau_m
                omega_avg = (omega_n + omega_m) / 2.0
                if omega_avg > 0:
                    penalty = 1.0 + abs(omega_n - omega_m) / omega_avg * 0.1
                    val /= penalty
            val = float(np.clip(val, 0.01, 0.4))
            K[n - 1, m - 1] = val
            K[m - 1, n - 1] = val

        # Pass 3: Distant (|n-m|>=3), exponential decay
        for n in range(1, 17):
            for m in range(n + 3, 17):
                dist = abs(n - m)
                val = k_base * np.exp(-alpha_decay * dist)
                val = float(np.clip(val, 0.001, 0.2))
                K[n - 1, m - 1] = val
                K[m - 1, n - 1] = val

        # Cross-hierarchy boosts (Paper 0, Section 5)
        _set_symmetric(K, 1, 16, max(K[0, 15], 0.05))  # Quantum-Meta
        _set_symmetric(K, 5, 7, max(K[4, 6], 0.15))  # Psycho-Symbolic

        alpha = np.zeros((16, 16), dtype=np.float64)
        return CouplingState(knm=K, alpha=alpha, active_template="scpn_physics")

    def apply_handshakes(
        self, state: CouplingState, handshakes_path: str | Path
    ) -> CouplingState:
        """Overlay documented inter-layer couplings from JSON spec.

        Reads the KNM_MATRIX_COMPLETE_SPECIFICATION.json format:
        each entry has from_layer, to_layer, coupling_strength.
        Negative values are preserved (inhibitory coupling).
        """
        path = Path(handshakes_path)
        data = json.loads(path.read_text(encoding="utf-8"))
        knm = state.knm.copy()
        for entry in data["matrix"]:
            fr = int(entry["from_layer"])
            to = int(entry["to_layer"])
            strength = float(entry["coupling_strength"])
            knm[fr - 1, to - 1] = strength
            # Symmetric unless negative (directional inhibition)
            if strength >= 0:
                knm[to - 1, fr - 1] = strength
        return CouplingState(
            knm=knm,
            alpha=state.alpha.copy(),
            active_template="scpn_handshakes",
            knm_r=state.knm_r,
        )

    def build_with_amplitude(
        self,
        n_layers: int,
        base_strength: float,
        decay_alpha: float,
        amp_strength: float,
        amp_decay: float,
    ) -> CouplingState:
        """Build phase + amplitude coupling matrices together."""
        phase = self.build(n_layers, base_strength, decay_alpha)
        idx = np.arange(n_layers)
        dist = np.abs(idx[:, np.newaxis] - idx[np.newaxis, :])
        knm_r = amp_strength * np.exp(-amp_decay * dist)
        np.fill_diagonal(knm_r, 0.0)
        return CouplingState(
            knm=phase.knm,
            alpha=phase.alpha,
            active_template=phase.active_template,
            knm_r=knm_r,
        )

    def switch_template(
        self,
        state: CouplingState,
        template_name: str,
        templates: dict[str, NDArray],
    ) -> CouplingState:
        """Replace the active K_nm with a named template matrix."""
        if template_name not in templates:
            raise KeyError(f"Template {template_name!r} not found")
        return CouplingState(
            knm=templates[template_name].copy(),
            alpha=state.alpha.copy(),
            active_template=template_name,
            knm_r=state.knm_r,
        )

    @staticmethod
    def _adjacent_coupling(n: int, m: int, k_base: float) -> float:
        """Adjacent layer coupling via timescale matching."""
        if n == 16 or m == 16:
            return 0.2
        tau_n = SCPN_LAYER_TIMESCALES[n]
        tau_m = SCPN_LAYER_TIMESCALES[m]
        mismatch = abs(np.log(tau_n / tau_m))
        # Adjusted for stiff biological hierarchies (beta=0.05)
        val = k_base / (1.0 + 0.05 * mismatch)
        return float(np.clip(val, 0.1, 0.5))


def _set_symmetric(K: NDArray, n: int, m: int, val: float) -> None:
    """Set K[n-1, m-1] and K[m-1, n-1] to val (1-indexed layer numbers)."""
    K[n - 1, m - 1] = val
    K[m - 1, n - 1] = val
