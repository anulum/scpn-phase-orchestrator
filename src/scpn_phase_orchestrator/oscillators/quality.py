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

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values
from scpn_phase_orchestrator.oscillators.base import PhaseState

__all__ = ["PhaseQualityScorer"]

FloatArray: TypeAlias = NDArray[np.float64]

try:
    from spo_kernel import PyPhaseQualityScorer as _RustPhaseQualityScorer
except ImportError:
    _RustPhaseQualityScorer = None


def _state_values(
    states: list[PhaseState], *, name: Literal["quality", "amplitude"]
) -> FloatArray:
    """Preserve source types until plain real state measurements are validated."""
    values = [
        state.quality if name == "quality" else state.amplitude for state in states
    ]
    require_real_values(values, name=name, allow_object=True)
    return np.asarray(values, dtype=np.float64)


class PhaseQualityScorer:
    """Aggregate quality scoring and collapse detection for phase state arrays."""

    def __init__(self, collapse_threshold: float = 0.1, min_quality: float = 0.3):
        require_real_values(collapse_threshold, name="collapse_threshold")
        require_real_values(min_quality, name="min_quality")
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
        """Weighted average quality across all phase states.

        Parameters
        ----------
        phase_states : list[PhaseState]
            Extracted per-oscillator phase states with plain real quality and
            amplitude values; text, boolean and temporal aliases are rejected.

        Returns
        -------
        float
            Weighted average quality across all phase states. States whose quality
            or amplitude is not finite are skipped, quality is clamped to
            ``[0, 1]``, and ``0.0`` is returned when no state remains; the Rust
            and Python paths give the same result.
        """
        if not phase_states:
            return 0.0
        qualities = _state_values(phase_states, name="quality")
        amplitudes = _state_values(phase_states, name="amplitude")
        if self._rust is not None:
            return float(self._rust.score(qualities.tolist(), amplitudes.tolist()))
        usable = np.isfinite(qualities) & np.isfinite(amplitudes)
        if not bool(np.any(usable)):
            return 0.0
        weights = np.maximum(amplitudes[usable], 1e-12)
        return float(np.average(np.clip(qualities[usable], 0.0, 1.0), weights=weights))

    # Thresholds: see docs/ASSUMPTIONS.md § Quality Gating
    def detect_collapse(
        self, phase_states: list[PhaseState], threshold: float = 0.1
    ) -> bool:
        """Return True if quality is below threshold for the majority of states.

        Parameters
        ----------
        phase_states : list[PhaseState]
            Extracted per-oscillator phase states with plain real quality
            values; text, boolean and temporal aliases are rejected.
        threshold : float
            Decision threshold.

        Returns
        -------
        bool
            True if quality is below threshold for the majority of states. A
            state whose quality is not finite counts as below the threshold,
            on both the Rust and the Python path.

        Raises
        ------
        ValueError
            If the inputs are invalid or inconsistent.
        """
        if not phase_states:
            return True
        require_real_values(threshold, name="threshold")
        qualities = _state_values(phase_states, name="quality")
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        threshold = float(threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self._rust is not None and threshold == self._collapse_threshold:
            return bool(self._rust.is_collapsed(qualities.tolist()))
        below = sum(
            1
            for quality in qualities
            if not np.isfinite(quality) or quality < threshold
        )
        return below > len(phase_states) / 2

    def downweight_mask(
        self, phase_states: list[PhaseState], min_quality: float = 0.3
    ) -> FloatArray:
        """Weight array in [0,1], zeros below min_quality.

        Parameters
        ----------
        phase_states : list[PhaseState]
            Extracted per-oscillator phase states with plain real quality
            values; text, boolean and temporal aliases are rejected.
        min_quality : float
            Minimum extraction quality.

        Returns
        -------
        FloatArray
            Weight array in [0,1], zeros below min_quality and for qualities that
            are not finite; passing qualities are clamped to ``[0, 1]``.

        Raises
        ------
        ValueError
            If the inputs are invalid or inconsistent.
        """
        if not phase_states:
            return np.array([], dtype=np.float64)
        require_real_values(min_quality, name="min_quality")
        if not np.isfinite(min_quality):
            raise ValueError("min_quality must be finite")
        min_quality = float(min_quality)
        if not 0.0 <= min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        qualities = _state_values(phase_states, name="quality")
        if self._rust is not None and min_quality == self._min_quality:
            return np.asarray(
                self._rust.downweight_mask(qualities.tolist()),
                dtype=np.float64,
            )
        passing = np.isfinite(qualities) & (qualities >= min_quality)
        mask = np.where(passing, np.clip(qualities, 0.0, 1.0), 0.0)
        return mask.astype(np.float64)
