# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.base import PhaseExtractor, PhaseState

TWO_PI = 2.0 * np.pi


class SymbolicExtractor(PhaseExtractor):
    """Phase extraction from discrete symbolic state sequences.

    Maps discrete state indices to phases on the unit circle via
    theta = 2*pi*s/N (ring-phase) or via graph-walk position.
    """

    def __init__(self, n_states: int, node_id: str = "sym", mode: str = "ring"):
        """
        Args:
            n_states: total number of discrete states N
            node_id: identifier for generated PhaseState objects
            mode: "ring" for ring-phase, "graph" for graph-walk phase
        """
        if n_states < 2:
            raise ValueError(f"n_states must be >= 2, got {n_states}")
        if mode not in ("ring", "graph"):
            raise ValueError(f"mode must be 'ring' or 'graph', got {mode!r}")
        self._n_states = n_states
        self._node_id = node_id
        self._mode = mode

    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]:
        indices = np.asarray(signal, dtype=np.int64)
        if self._mode == "ring":
            thetas = TWO_PI * indices / self._n_states
        else:
            # Graph-walk: normalise sequential position to [0, 2*pi)
            positions = np.arange(len(indices), dtype=np.float64)
            thetas = TWO_PI * positions / max(len(indices), 1)

        thetas = thetas % TWO_PI
        dt = 1.0 / sample_rate if sample_rate > 0 else 1.0
        omegas = np.zeros_like(thetas)
        if len(thetas) > 1:
            dtheta = np.diff(thetas)
            # Unwrap jumps larger than pi
            dtheta = (dtheta + np.pi) % TWO_PI - np.pi
            omegas[1:] = dtheta / dt

        states = []
        for i in range(len(thetas)):
            states.append(
                PhaseState(
                    theta=float(thetas[i]),
                    omega=float(omegas[i]),
                    amplitude=1.0,
                    quality=self._transition_quality(indices, i),
                    channel="S",
                    node_id=self._node_id,
                )
            )
        return states

    def quality_score(self, phase_states: list[PhaseState]) -> float:
        if not phase_states:
            return 0.0
        return float(np.mean([ps.quality for ps in phase_states]))

    def _transition_quality(self, indices, i):
        """Quality based on transition regularity: penalise repeated or large jumps."""
        if i == 0 or len(indices) < 2:
            return 0.5
        step = abs(int(indices[i]) - int(indices[i - 1]))
        if step == 0:
            return 0.2  # stalled
        if step == 1:
            return 1.0  # ideal single-step transition
        # Penalise large jumps proportionally
        return max(0.1, 1.0 - (step - 1) / self._n_states)
