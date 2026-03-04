# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.imprint.state import ImprintState

__all__ = ["ImprintModel"]


class ImprintModel:
    """Exponential exposure accumulation with decay and saturation.

    m_k(t+dt) = m_k(t) * exp(-decay_rate * dt) + exposure * dt,
    clipped to [0, saturation].
    """

    def __init__(self, decay_rate: float, saturation: float):
        if decay_rate < 0.0:
            raise ValueError(f"decay_rate must be non-negative, got {decay_rate}")
        if saturation <= 0.0:
            raise ValueError(f"saturation must be positive, got {saturation}")
        self._decay_rate = decay_rate
        self._saturation = saturation

    def update(self, state: ImprintState, exposure: NDArray, dt: float) -> ImprintState:
        decayed = state.m_k * np.exp(-self._decay_rate * dt)
        m_new = np.clip(decayed + exposure * dt, 0.0, self._saturation)
        return ImprintState(
            m_k=m_new,
            last_update=state.last_update + dt,
            attribution=state.attribution.copy(),
        )

    def modulate_coupling(self, knm: NDArray, imprint: ImprintState) -> NDArray:
        """Scale Knm rows by (1 + m_k)."""
        result: NDArray = knm * (1.0 + imprint.m_k)[:, np.newaxis]
        return result

    def modulate_lag(self, alpha: NDArray, imprint: ImprintState) -> NDArray:
        """Shift phase lags by antisymmetric imprint offset."""
        offset = imprint.m_k[:, np.newaxis] - imprint.m_k[np.newaxis, :]
        result: NDArray = alpha + offset
        return result

    def modulate_mu(self, mu: NDArray, imprint: ImprintState) -> NDArray:
        """Scale bifurcation parameter: μ_k * (1 + m_k)."""
        result: NDArray = mu * (1.0 + imprint.m_k)
        return result
