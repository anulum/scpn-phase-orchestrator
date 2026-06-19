# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Symbolic Psi driver

"""Symbolic-channel cyclic sequence reference driver.

`SymbolicDriver` exposes deterministic cyclic lookup over finite real symbolic
phase values. It rejects empty, boolean-containing, multi-dimensional,
non-finite, or non-integer step inputs before returning drive values for
symbolic or semiotic simulations.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ["SymbolicDriver"]

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]


def _contains_bool(value: object) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (str, bytes)):
        return False
    if isinstance(value, Iterable):
        return any(_contains_bool(item) for item in value)
    return False


def _validate_step(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("step must be an integer")
    return int(value)


def _validate_steps(value: object) -> IntArray:
    steps = np.asarray(value)
    dtype = steps.dtype
    if (
        np.issubdtype(dtype, np.bool_)
        or np.issubdtype(dtype, np.complexfloating)
        or not np.issubdtype(dtype, np.integer)
    ):
        raise ValueError("steps must be integer")
    return steps.astype(np.int64, copy=False)


class SymbolicDriver:
    """Deterministic phase sequence driver for symbolic/semiotic channels."""

    def __init__(self, sequence: list[float]):
        if not sequence:
            raise ValueError("sequence must be non-empty")
        if _contains_bool(sequence):
            raise ValueError("sequence values must be finite real numbers")
        parsed_sequence = np.asarray(sequence, dtype=np.float64)
        if parsed_sequence.ndim != 1:
            raise ValueError("sequence must be one-dimensional")
        if not np.all(np.isfinite(parsed_sequence)):
            raise ValueError("sequence values must be finite real numbers")
        self._sequence = parsed_sequence
        self._n = len(parsed_sequence)

    def compute(self, step: int) -> float:
        """Return symbolic phase at discrete *step* (cyclic).

        Parameters
        ----------
        step : int
            Zero-based step index.

        Returns
        -------
        float
            Symbolic phase at discrete *step* (cyclic).
        """
        step = _validate_step(step)
        return float(self._sequence[step % self._n])

    def compute_batch(self, steps: IntArray) -> FloatArray:
        """Vectorised symbolic phase lookup over an array of step indices.

        Parameters
        ----------
        steps : IntArray
            Number of replay steps.

        Returns
        -------
        FloatArray
            Vectorised symbolic phase lookup over an array of step indices.
        """
        steps = _validate_steps(steps)
        result: FloatArray = self._sequence[steps % self._n]
        return result
