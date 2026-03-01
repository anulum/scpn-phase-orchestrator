# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.base import PhaseState


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

    def detect_collapse(self, phase_states: list[PhaseState], threshold=0.1) -> bool:
        """True if quality is below threshold for the majority of states."""
        if not phase_states:
            return True
        below = sum(1 for ps in phase_states if ps.quality < threshold)
        return below > len(phase_states) / 2

    def downweight_mask(
        self, phase_states: list[PhaseState], min_quality=0.3
    ) -> NDArray:
        """Weight array in [0,1], zeros below min_quality."""
        if not phase_states:
            return np.array([], dtype=np.float64)
        qualities = np.array([ps.quality for ps in phase_states])
        mask = np.where(qualities >= min_quality, qualities, 0.0)
        return mask.astype(np.float64)
