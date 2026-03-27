# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Oscillator base class

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray

__all__ = ["PhaseState", "PhaseExtractor"]


@dataclass
class PhaseState:
    """Extracted phase, frequency, amplitude, and quality for one oscillator."""

    theta: float  # radians [0, 2*pi)
    omega: float  # instantaneous frequency, rad/s
    amplitude: float
    quality: float  # 0..1
    channel: str  # "P", "I", or "S"
    node_id: str


class PhaseExtractor(ABC):
    """Abstract base for signal-to-phase extraction algorithms."""

    @abstractmethod
    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]:
        """Extract phase states from a raw signal at the given sample rate."""
        ...

    @abstractmethod
    def quality_score(self, phase_states: list[PhaseState]) -> float:
        """Aggregate quality metric (0..1) over a set of extracted phase states."""
        ...
