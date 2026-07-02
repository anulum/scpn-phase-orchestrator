# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Physical Psi driver

"""Physical-channel sinusoidal reference-phase driver.

`PhysicalDriver` generates finite sinusoidal drive values from a positive
frequency and non-negative amplitude. Scalar and vector compute paths reject
boolean, complex, non-numeric, and non-finite time inputs so external forcing
cannot inject invalid values into UPDE integration.
"""

from __future__ import annotations

from collections.abc import Iterable
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._compat import TWO_PI

__all__ = ["PhysicalDriver"]

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


class PhysicalDriver:
    """Sinusoidal external drive: Psi_P(t) = amplitude * sin(2*pi*frequency*t)."""

    def __init__(self, frequency: float, amplitude: float = 1.0):
        """Initialise a physical sinusoidal reference driver.

        Parameters
        ----------
        frequency : float
            Positive drive frequency in hertz.
        amplitude : float, default=1.0
            Non-negative peak drive amplitude.

        Raises
        ------
        ValueError
            If either parameter is a boolean alias, non-real, non-finite, or
            outside its allowed numeric range.
        """
        if isinstance(frequency, bool) or isinstance(amplitude, bool):
            raise ValueError("frequency and amplitude must be finite real values")
        parsed_frequency = _require_finite_real(frequency, name="frequency")
        parsed_amplitude = _require_finite_real(amplitude, name="amplitude")
        if parsed_frequency <= 0.0:
            raise ValueError(f"frequency must be finite and positive, got {frequency}")
        if parsed_amplitude < 0.0:
            raise ValueError(
                f"amplitude must be finite and non-negative, got {amplitude}"
            )
        self._frequency = parsed_frequency
        self._amplitude = parsed_amplitude

    def compute(self, t: float) -> float:
        """Return Psi_P at time *t*.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        float
            Psi_P at time *t*.
        """
        t = _require_finite_real(t, name="t")
        return float(self._amplitude * np.sin(TWO_PI * self._frequency * t))

    def compute_batch(self, t_array: FloatArray) -> FloatArray:
        """Vectorised Psi_P over an array of time values.

        Parameters
        ----------
        t_array : FloatArray
            Time samples, shape ``(T,)``.

        Returns
        -------
        FloatArray
            Vectorised Psi_P over an array of time values.
        """
        t_array = _require_finite_real_array(t_array, name="t_array")
        result: FloatArray = self._amplitude * np.sin(
            TWO_PI * self._frequency * t_array
        )
        return result
