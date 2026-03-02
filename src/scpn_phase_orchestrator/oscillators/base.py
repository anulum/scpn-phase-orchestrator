# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from numpy.typing import NDArray

__all__ = ["PhaseState", "PhaseExtractor"]


@dataclass
class PhaseState:
    theta: float  # radians [0, 2*pi)
    omega: float  # instantaneous frequency, rad/s
    amplitude: float
    quality: float  # 0..1
    channel: str  # "P", "I", or "S"
    node_id: str


class PhaseExtractor(ABC):
    @abstractmethod
    def extract(self, signal: NDArray, sample_rate: float) -> list[PhaseState]: ...

    @abstractmethod
    def quality_score(self, phase_states: list[PhaseState]) -> float: ...
