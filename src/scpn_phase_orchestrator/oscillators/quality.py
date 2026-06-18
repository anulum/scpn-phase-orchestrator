# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase quality scorer

"""Quality aggregation and collapse detection for extracted phase states.

The scorer turns per-oscillator extraction quality into weighted aggregate
signals for runtime gating and diagnostics. Empty state sets collapse to safe
defaults, low-quality states can be masked, and amplitude weighting prevents
near-zero signals from dominating quality summaries.
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.base import PhaseState

__all__ = ["PhaseQualityScorer"]

FloatArray: TypeAlias = NDArray[np.float64]

try:
    from spo_kernel import PyPhaseQualityScorer as _RustPhaseQualityScorer
except ImportError:
    _RustPhaseQualityScorer = None


class PhaseQualityScorer:
    """Aggregate quality scoring and collapse detection for phase state arrays."""

    def __init__(self, collapse_threshold: float = 0.1, min_quality: float = 0.3):
        if not np.isfinite(collapse_threshold):
            raise ValueError("collapse_threshold must be finite")
        if not np.isfinite(min_quality):
            raise ValueError("min_quality must be finite")
        if not 0.0 <= collapse_threshold <= 1.0:
            raise ValueError("collapse_threshold must be in [0, 1]")
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        self._collapse_threshold = float(collapse_threshold)
        self._min_quality = float(min_quality)
        self._rust = (
            _RustPhaseQualityScorer(
                collapse_threshold=self._collapse_threshold,
                min_quality=self._min_quality,
            )
            if _RustPhaseQualityScorer is not None
            else None
        )

    def score(self, phase_states: list[PhaseState]) -> float:
        """Weighted average quality across all phase states."""
        if not phase_states:
            return 0.0
        qualities = np.array([ps.quality for ps in phase_states])
        amplitudes = np.array([ps.amplitude for ps in phase_states])
        if self._rust is not None:
            return float(self._rust.score(qualities.tolist(), amplitudes.tolist()))
        weights = np.maximum(amplitudes, 1e-12)
        return float(np.average(qualities, weights=weights))

    # Thresholds: see docs/ASSUMPTIONS.md § Quality Gating
    def detect_collapse(
        self, phase_states: list[PhaseState], threshold: float = 0.1
    ) -> bool:
        """Return True if quality is below threshold for the majority of states."""
        if not phase_states:
            return True
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self._rust is not None and threshold == self._collapse_threshold:
            qualities = [ps.quality for ps in phase_states]
            return bool(self._rust.is_collapsed(qualities))
        below = sum(1 for ps in phase_states if ps.quality < threshold)
        return below > len(phase_states) / 2

    def downweight_mask(
        self, phase_states: list[PhaseState], min_quality: float = 0.3
    ) -> FloatArray:
        """Weight array in [0,1], zeros below min_quality."""
        if not phase_states:
            return np.array([], dtype=np.float64)
        if not np.isfinite(min_quality):
            raise ValueError("min_quality must be finite")
        min_quality = float(min_quality)
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        qualities = np.array([ps.quality for ps in phase_states])
        if self._rust is not None and min_quality == self._min_quality:
            return np.asarray(
                self._rust.downweight_mask(qualities.tolist()),
                dtype=np.float64,
            )
        mask = np.where(qualities >= min_quality, qualities, 0.0)
        return mask.astype(np.float64)
