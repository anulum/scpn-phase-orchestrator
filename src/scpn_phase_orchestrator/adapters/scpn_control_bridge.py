# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SCPN-control bridge adapter

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["SCPNControlBridge"]

FloatArray: TypeAlias = NDArray[np.float64]
JSONConfig: TypeAlias = dict[str, object]


def _validate_config_value(value: object, *, path: str) -> object:
    if value is None or isinstance(value, str | bool):
        return value
    if isinstance(value, Real):
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{path} must be finite")
        return int(value) if isinstance(value, int) else result
    if isinstance(value, list):
        return [
            _validate_config_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return _validate_scpn_config(value, path=path)
    raise ValueError(f"{path} must be JSON-compatible")


def _validate_scpn_config(config: dict, *, path: str = "scpn_config") -> JSONConfig:
    validated: JSONConfig = {}
    for key, value in config.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path} keys must be non-empty strings")
        validated[key] = _validate_config_value(value, path=f"{path}.{key}")
    return validated


class SCPNControlBridge:
    """Adapter between scpn-control telemetry and phase-orchestrator types."""

    def __init__(self, scpn_config: dict):
        if not isinstance(scpn_config, dict):
            raise ValueError("scpn_config must be a dict")
        self._config = _validate_scpn_config(scpn_config)

    def import_knm(self, scpn_knm: FloatArray) -> CouplingState:
        """Wrap an external Knm matrix into a CouplingState."""
        knm: FloatArray = np.asarray(scpn_knm, dtype=np.float64)
        if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
            raise ValueError(f"Knm must be square, got shape {knm.shape}")
        if not np.all(np.isfinite(knm)):
            raise ValueError("Knm must contain only finite values")
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
        if not np.all(np.isfinite(omega)):
            raise ValueError("omega must contain only finite values")
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
