# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Informational Psi driver

"""Informational-channel cadence reference-phase driver.

`InformationalDriver` maps a positive event cadence to wrapped phase-drive
values in `[0, 2*pi)`. Constructor and compute paths reject invalid numeric
inputs so cadence-driven forcing remains bounded and deterministic.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["InformationalDriver"]

FloatArray: TypeAlias = NDArray[np.float64]


def _contains_boolean_alias(value: object) -> bool:
    """Return whether ``value`` carries a Python or NumPy boolean alias."""
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return True
        if value.dtype == object:
            return any(_contains_boolean_alias(item) for item in value.flat)
        return False
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Iterable):
        return any(_contains_boolean_alias(item) for item in value)
    return False


def _require_finite_real(value: object, *, name: str) -> float:
    """Return ``value`` as a finite real float, else raise ``ValueError``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value}")
    return parsed


def _require_finite_real_array(value: object, *, name: str) -> FloatArray:
    """Return ``value`` as a validated finite real array, else raise."""
    if _contains_boolean_alias(value):
        raise ValueError(f"{name} must be finite")
    array = np.asarray(value)
    dtype = array.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.number)
    ):
        raise ValueError(f"{name} must be finite")
    parsed = array.astype(np.float64, copy=False)
    if not np.all(np.isfinite(parsed)):
        raise ValueError(f"{name} must be finite")
    return parsed


class InformationalDriver:
    """External drive Psi_I(t) = 2*pi*cadence_hz*t (mod 2*pi)."""

    def __init__(self, cadence_hz: float):
        """Initialise an informational cadence reference driver.

        Parameters
        ----------
        cadence_hz : float
            Positive event cadence in hertz.

        Raises
        ------
        ValueError
            If the cadence is a boolean alias, non-real, non-finite, or not
            positive.
        """
        parsed_cadence = _require_finite_real(cadence_hz, name="cadence_hz")
        if not isfinite(parsed_cadence) or parsed_cadence <= 0.0:
            raise ValueError(
                f"cadence_hz must be finite and positive, got {cadence_hz}"
            )
        self._cadence_hz = parsed_cadence

    def compute(self, t: float) -> float:
        """Return Psi_I at time *t*, wrapped to [0, 2*pi).

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Psi_I at time *t*, wrapped to [0, 2*pi).
        """
        t = _require_finite_real(t, name="t")
        return (TWO_PI * self._cadence_hz * t) % TWO_PI

    def compute_batch(self, t_array: FloatArray) -> FloatArray:
        """Vectorised Psi_I over an array of time values.

        Parameters
        ----------
        t_array : FloatArray
            Time samples, shape ``(T,)``.

        Returns
        -------
        FloatArray
            Vectorised Psi_I over an array of time values.
        """
        t_array = _require_finite_real_array(t_array, name="t_array")
        result: FloatArray = (TWO_PI * self._cadence_hz * t_array) % TWO_PI
        return result
