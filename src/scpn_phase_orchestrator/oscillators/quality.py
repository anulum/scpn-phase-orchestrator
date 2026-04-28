# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase quality scorer

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.base import PhaseState

__all__ = ["PhaseQualityScorer"]


class PhaseQualityScorer:
    """Aggregate quality scoring and collapse detection for phase state arrays."""

    def score(self, phase_states: list[PhaseState]) -> float:
        """Weighted average quality across all phase states."""
        if not phase_states:
            return 0.0
        qualities = np.array([ps.quality for ps in phase_states])
        amplitudes = np.array([ps.amplitude for ps in phase_states])
        weights = np.maximum(amplitudes, 1e-12)
        return float(np.average(qualities, weights=weights))

    # Thresholds: see docs/ASSUMPTIONS.md § Quality Gating
    def detect_collapse(
        self, phase_states: list[PhaseState], threshold: float = 0.1
    ) -> bool:
        """True if quality is below threshold for the majority of states."""
        if not phase_states:
            return True
        below = sum(1 for ps in phase_states if ps.quality < threshold)
        return below > len(phase_states) / 2

    def downweight_mask(
        self, phase_states: list[PhaseState], min_quality: float = 0.3
    ) -> NDArray:
        """Weight array in [0,1], zeros below min_quality."""
        if not phase_states:
            return np.array([], dtype=np.float64)
        qualities = np.array([ps.quality for ps in phase_states])
        mask = np.where(qualities >= min_quality, qualities, 0.0)
        return mask.astype(np.float64)
