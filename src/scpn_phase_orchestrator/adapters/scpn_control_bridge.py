# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SCPN-control bridge adapter

"""SCPN-control bridge for validated coupling, frequency, and telemetry exchange.

The bridge accepts JSON-compatible configuration, imports finite dense coupling
matrices and positive natural-frequency vectors, and exports reduced UPDE state
telemetry with layer locks and cross-alignment. It is a data-shape adapter only;
it does not invoke an external control engine or apply actions.
"""

from __future__ import annotations

from math import isfinite
from numbers import Real
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.knm import CouplingState
from scpn_phase_orchestrator.upde.metrics import UPDEState

__all__ = ["SCPNControlBridge"]

FloatArray: TypeAlias = NDArray[np.float64]
JSONConfig: TypeAlias = dict[str, object]


def _has_non_real_numeric_alias(value: object) -> bool:
    """Return whether the value contains a non-real numeric alias."""
    if isinstance(value, bool | np.bool_):
        return True
    if isinstance(value, complex | np.complexfloating):
        return True
    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"b", "c"}:
            return True
        if value.dtype.kind == "O":
            return any(_has_non_real_numeric_alias(item) for item in value.flat)
        return False
    if isinstance(value, list | tuple):
        return any(_has_non_real_numeric_alias(item) for item in value)
    return not isinstance(value, Real)


def _as_real_numeric_array(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated real numeric array, else raise."""
    if _has_non_real_numeric_alias(value):
        raise ValueError(f"{name} must be real-valued numeric data")
    return np.asarray(value, dtype=np.float64)


def _validate_config_value(value: object, *, path: str) -> object:
    """Return a validated SCPN-control config value, else raise."""
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


def _validate_scpn_config(
    config: dict[str, Any], *, path: str = "scpn_config"
) -> JSONConfig:
    """Validate and normalise the SCPN-control configuration, else raise."""
    validated: JSONConfig = {}
    for key, value in config.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{path} keys must be non-empty strings")
        validated[key] = _validate_config_value(value, path=f"{path}.{key}")
    return validated


class SCPNControlBridge:
    """Adapter between scpn-control telemetry and phase-orchestrator types."""

    def __init__(self, scpn_config: dict[str, Any]):
        if not isinstance(scpn_config, dict):
            raise ValueError("scpn_config must be a dict")
        self._config = _validate_scpn_config(scpn_config)

    def import_knm(self, scpn_knm: FloatArray) -> CouplingState:
        """Wrap an external Knm matrix into a CouplingState.

        Parameters
        ----------
        scpn_knm : FloatArray
            An external coupling matrix, shape ``(N, N)``.

        Returns
        -------
        CouplingState
            The coupling state wrapping the external matrix.

        Raises
        ------
        ValueError
            If the coupling matrix is invalid.
        """
        knm = _as_real_numeric_array(scpn_knm, name="Knm")
        if knm.ndim != 2 or knm.shape[0] != knm.shape[1]:
            raise ValueError(f"Knm must be square, got shape {knm.shape}")
        if knm.size == 0:
            raise ValueError("Knm must be non-empty")
        if not np.all(np.isfinite(knm)):
            raise ValueError("Knm must contain only finite values")
        if np.any(np.diag(knm) != 0.0):
            raise ValueError("Knm self-coupling diagonal must be zero")
        n = knm.shape[0]
        return CouplingState(
            knm=knm,
            alpha=np.zeros((n, n), dtype=np.float64),
            active_template="scpn_import",
        )

    def import_omega(self, scpn_omega: FloatArray) -> FloatArray:
        """Validate and pass through natural frequencies.

        Parameters
        ----------
        scpn_omega : FloatArray
            External natural frequencies, shape ``(N,)``.

        Returns
        -------
        FloatArray
            The validated natural frequencies.

        Raises
        ------
        ValueError
            If the natural frequencies are invalid.
        """
        omega = _as_real_numeric_array(scpn_omega, name="omega")
        if omega.ndim != 1:
            raise ValueError(f"omega must be 1-D, got ndim={omega.ndim}")
        if omega.size == 0:
            raise ValueError("omega must be non-empty")
        if not np.all(np.isfinite(omega)):
            raise ValueError("omega must contain only finite values")
        if np.any(omega <= 0.0):
            raise ValueError("All natural frequencies must be positive")
        return omega

    def export_state(self, upde_state: UPDEState) -> dict[str, Any]:
        """Convert UPDEState to scpn-control compatible telemetry dict.

        Parameters
        ----------
        upde_state : UPDEState
            The UPDE state to export.

        Returns
        -------
        dict[str, Any]
            The scpn-control-compatible telemetry dict.
        """
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
