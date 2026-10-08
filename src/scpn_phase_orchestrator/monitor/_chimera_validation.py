# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — chimera backend boundary validation

"""Shared validation for direct chimera accelerator calls.

Original boolean, text, complex and temporal measurement aliases are refused;
real numeric object arrays are supported and counts must be plain integers.
"""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values

FloatArray: TypeAlias = NDArray[np.float64]


def _measurement_array(
    value: object, *, name: str, conversion_error: str
) -> FloatArray:
    """Convert an original real measurement while preserving its dimensions.

    Parameters
    ----------
    value : object
        Original array or sequence, before dtype coercion.
    name : str
        Owning field name used in alias diagnostics.
    conversion_error : str
        Owning error for malformed or nonrepresentable Float64 payloads.

    Returns
    -------
    FloatArray
        An independent Float64 array retaining the original shape. The owning
        contract checks dimensions, finiteness and physical bounds afterward.

    Raises
    ------
    ValueError
        For malformed arrays, boolean/text/complex/temporal aliases or values
        that cannot be represented as Float64 measurements.
    """
    try:
        raw = np.asarray(value)
        # Primitive NumPy numeric storage cannot hide an original source alias.
        # Sequences and other dtypes retain original elements for admission.
        if isinstance(value, np.ndarray) and raw.dtype.kind in "iuf":
            objects = None
        else:
            objects = np.asarray(value, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(conversion_error) from exc
    if objects is not None:
        if any(isinstance(item, (bool, np.bool_)) for item in objects.flat):
            raise ValueError(f"{name} must not contain boolean values")
        if np.iscomplexobj(raw) or any(
            isinstance(item, (complex, np.complexfloating)) for item in objects.flat
        ):
            raise ValueError(f"{name} must be real-valued")
        strings = [item for item in objects.flat if isinstance(item, (str, bytes))]
        if strings:
            try:
                for item in strings:
                    float(item)
            except ValueError as exc:
                raise ValueError(conversion_error) from exc
            raise ValueError(f"{name} must not contain numeric-string aliases")
    try:
        require_real_values(value, name=name, allow_object=True)
        result: FloatArray = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as exc:
        raise ValueError(conversion_error) from exc
    return result


def _validate_n(value: object) -> int:
    """Return the validated oscillator count, else raise."""
    if isinstance(
        value, (bool, np.bool_, np.timedelta64, np.datetime64)
    ) or not isinstance(value, Integral):
        raise ValueError("n must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"n must be non-negative, got {result}")
    return result


def _validate_float_vector(value: object, name: str) -> FloatArray:
    """Return ``value`` as a validated finite float vector, else raise."""
    array = _measurement_array(
        value,
        name=name,
        conversion_error=f"{name} must be a finite one-dimensional float array",
    )
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def validate_chimera_backend_inputs(
    phases: object,
    knm_flat: object,
    n: object,
    *,
    maximum_n: int | None = None,
) -> tuple[FloatArray, FloatArray, int]:
    """Validate direct local-order inputs before optional runtime loading."""
    n_int = _validate_n(n)
    if maximum_n is not None and n_int > maximum_n:
        raise ValueError(f"n exceeds backend integer range {maximum_n}")
    phases_vec = _validate_float_vector(phases, "phases")
    if phases_vec.size != n_int:
        raise ValueError(f"phases length {phases_vec.size} does not match n={n_int}")
    knm_vec = _validate_float_vector(knm_flat, "knm_flat")
    expected = n_int * n_int
    if knm_vec.size != expected:
        raise ValueError(
            f"knm_flat length {knm_vec.size} does not match n*n={expected}"
        )
    if n_int:
        diagonal = np.diag(knm_vec.reshape(n_int, n_int))
        if not np.allclose(diagonal, 0.0, rtol=0.0, atol=1e-15):
            raise ValueError("knm_flat self-coupling diagonal must be zero")
    return phases_vec, knm_vec, n_int


def validate_chimera_backend_output(
    local_order: object,
    n: object,
) -> FloatArray:
    """Validate direct backend local-order output before returning it."""
    n_int = _validate_n(n)
    values = _validate_float_vector(local_order, "local_order")
    if values.size != n_int:
        raise ValueError(f"local_order length {values.size} does not match n={n_int}")
    if np.any((values < -1e-12) | (values > 1.0 + 1e-12)):
        raise ValueError("local_order values must lie in the physical interval [0, 1]")
    return np.ascontiguousarray(np.clip(values, 0.0, 1.0), dtype=np.float64)
