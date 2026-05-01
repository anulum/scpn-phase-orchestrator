# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SCPN-control bridge adapter

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["SCPNControlBridge"]

FloatArray: TypeAlias = NDArray[np.float64]


class SCPNControlBridge:
    """Adapter between scpn-control telemetry and phase-orchestrator types."""

    def __init__(self, scpn_config: dict):
        self._config = scpn_config

    def import_knm(self, scpn_knm: FloatArray) -> CouplingState:
        """Wrap an external Knm matrix into a CouplingState."""
        knm: FloatArray = np.asarray(scpn_knm, dtype=np.float64)
        if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
            raise ValueError(f"Knm must be square, got shape {knm.shape}")
        n = knm.shape[0]
        return CouplingState(
            knm=knm,
            alpha=np.zeros((n, n), dtype=np.float64),
            active_template="scpn_import",
        )

    def import_omega(self, scpn_omega: FloatArray) -> FloatArray:
        """Validate and pass through natural frequencies."""
        omega: FloatArray = np.asarray(scpn_omega, dtype=np.float64)
        if omega.ndim != 1:
            raise ValueError(f"omega must be 1-D, got ndim={omega.ndim}")
        if np.any(omega <= 0.0):
            raise ValueError("All natural frequencies must be positive")
        return omega

    def export_state(self, upde_state: UPDEState) -> dict:
        """Convert UPDEState to scpn-control compatible telemetry dict."""
        return {
            "regime": upde_state.regime_id,
            "stability": upde_state.stability_proxy,
            "layers": [
                {
                    "R": ls.R,
                    "psi": ls.psi,
                    "locks": {
                        k: {"plv": v.plv, "lag": v.mean_lag}
                        for k, v in ls.lock_signatures.items()
                    },
                }
                for ls in upde_state.layers
            ],
            "cross_alignment": upde_state.cross_layer_alignment.tolist(),
        }
