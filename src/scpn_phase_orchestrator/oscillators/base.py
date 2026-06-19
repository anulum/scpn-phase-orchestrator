# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Oscillator base class

"""Shared phase-state contract and extractor interface.

`PhaseState` is the typed handoff record from channel-specific extractors into
binding, quality scoring, and UPDE initialisation. `PhaseExtractor` defines the
minimal extraction/quality interface implemented by physical waveform,
informational event, and symbolic state-sequence extractors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["PhaseState", "PhaseExtractor"]

FloatArray: TypeAlias = NDArray[np.float64]


@dataclass
class PhaseState:
    """Extracted phase, frequency, amplitude, and quality for one oscillator."""

    theta: float  # radians [0, 2*pi)
    omega: float  # instantaneous frequency, rad/s
    amplitude: float
    quality: float  # 0..1
    channel: str
    node_id: str


class PhaseExtractor(ABC):
    """Abstract base for signal-to-phase extraction algorithms."""

    @abstractmethod
    def extract(self, signal: FloatArray, sample_rate: float) -> list[PhaseState]:
        """Extract phase states from a raw signal at the given sample rate.

        Parameters
        ----------
        signal : FloatArray
            Input signal, shape ``(T,)``.
        sample_rate : float
            Sampling rate in Hz.

        Returns
        -------
        list[PhaseState]
            Phase states from a raw signal at the given sample rate.
        """
        ...

    @abstractmethod
    def quality_score(self, phase_states: list[PhaseState]) -> float:
        """Aggregate quality metric (0..1) over a set of extracted phase states.

        Parameters
        ----------
        phase_states : list[PhaseState]
            Extracted per-oscillator phase states.

        Returns
        -------
        float
            Aggregate quality metric (0..1) over a set of extracted phase states.
        """
        ...
