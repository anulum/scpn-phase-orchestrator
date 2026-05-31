# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct embedding backend validation

"""Shared typed validation for direct delay-embedding backend bridge calls."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "FloatArray",
    "validate_delay_embed_backend_inputs",
    "validate_mutual_information_backend_inputs",
    "validate_nearest_neighbor_backend_inputs",
]


def _contains_boolean_alias(raw: np.ndarray) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype != object:
        return False
    return any(isinstance(value, (bool, np.bool_)) for value in raw.flat)


def _validate_int_at_least(value: object, *, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {result}")
    return result


def _validate_float_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if _contains_boolean_alias(raw):
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real values")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a finite one-dimensional float array"
        raise ValueError(msg) from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_delay_embed_backend_inputs(
    signal: object,
    delay: object,
    dimension: object,
) -> tuple[FloatArray, int, int, int]:
    """Return validated direct-backend delay-embedding arguments."""

    s = _validate_float_vector(signal, name="signal")
    delay_int = _validate_int_at_least(delay, name="delay", minimum=1)
    dimension_int = _validate_int_at_least(dimension, name="dimension", minimum=1)
    t_effective = int(s.size) - (dimension_int - 1) * delay_int
    if t_effective <= 0:
        raise ValueError(
            "signal is too short for the requested delay embedding "
            f"(T={s.size}, delay={delay_int}, dimension={dimension_int})"
        )
    return s, delay_int, dimension_int, t_effective


def validate_mutual_information_backend_inputs(
    signal: object,
    lag: object,
    n_bins: object,
) -> tuple[FloatArray, int, int]:
    """Return validated direct-backend mutual-information arguments."""

    return (
        _validate_float_vector(signal, name="signal"),
        _validate_int_at_least(lag, name="lag", minimum=0),
        _validate_int_at_least(n_bins, name="n_bins", minimum=2),
    )


def validate_nearest_neighbor_backend_inputs(
    embedded: object,
    t: object,
    m: object,
) -> tuple[FloatArray, int, int]:
    """Return validated direct-backend nearest-neighbor arguments."""

    t_int = _validate_int_at_least(t, name="t", minimum=0)
    m_int = _validate_int_at_least(m, name="m", minimum=1)
    e = _validate_float_vector(embedded, name="embedded")
    expected = t_int * m_int
    if e.size != expected:
        raise ValueError(f"embedded length {e.size} does not match t*m = {expected}")
    return e, t_int, m_int
