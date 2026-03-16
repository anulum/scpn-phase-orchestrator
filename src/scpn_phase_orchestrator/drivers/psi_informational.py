# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational Psi driver

from __future__ import annotations

from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["InformationalDriver"]


class InformationalDriver:
    """External drive Psi_I(t) = 2*pi*cadence_hz*t (mod 2*pi)."""

    def __init__(self, cadence_hz: float):
        if cadence_hz <= 0.0:
            raise ValueError(f"cadence_hz must be positive, got {cadence_hz}")
        self._cadence_hz = cadence_hz

    def compute(self, t: float) -> float:
        return (TWO_PI * self._cadence_hz * t) % TWO_PI

    def compute_batch(self, t_array: NDArray) -> NDArray:
        return (TWO_PI * self._cadence_hz * t_array) % TWO_PI
