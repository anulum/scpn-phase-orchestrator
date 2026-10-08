# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Connectome count and structural-weight admission

"""Validate synthetic and neurolib ingress before weights enter public coupling."""

from __future__ import annotations

from numbers import Integral
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator._array_types import require_real_values

FloatArray: TypeAlias = NDArray[np.float64]

_MAX_SEED = 2**64 - 1


def _validate_n_regions(value: object, *, max_regions: int | None = None) -> int:
    """Admit integer counts before dense allocation or optional data access.

    Parameters
    ----------
    value : object
        Original scalar; boolean, text, float and temporal aliases are refused.
    max_regions : int or None
        Optional dataset-specific region limit.

    Returns
    -------
    int
        At least two regions whose dense float64 storage is addressable.

    Raises
    ------
    TypeError
        If the original scalar is not a genuine non-boolean integer.
    ValueError
        If its source type, minimum, dataset limit or byte capacity is invalid.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("n_regions must be an integer")
    require_real_values(value, name="n_regions", allow_object=True)
    n_regions = int(value)
    if n_regions < 2:
        msg = f"n_regions must be >= 2, got {n_regions}"
        raise ValueError(msg)
    if max_regions is not None and n_regions > max_regions:
        msg = f"n_regions must be <= {max_regions}, got {n_regions}"
        raise ValueError(msg)
    if n_regions * n_regions > np.iinfo(np.intp).max // np.dtype(np.float64).itemsize:
        raise ValueError("connectome matrix exceeds addressable float64 storage")
    return n_regions


def _validate_seed(value: object) -> int:
    """Admit the original non-boolean unsigned 64-bit seed.

    Parameters
    ----------
    value : object
        Original scalar seed, including NumPy integer scalars.

    Returns
    -------
    int
        Exact seed in the inclusive range zero through ``2**64 - 1``.

    Raises
    ------
    TypeError
        If the original scalar is not a genuine non-boolean integer.
    ValueError
        If it is a temporal alias or lies outside the unsigned seed range.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError("seed must be an integer in the u64 range")
    require_real_values(value, name="seed", allow_object=True)
    seed = int(value)
    if seed < 0 or seed > _MAX_SEED:
        raise ValueError("seed must be an integer in the u64 range")
    return seed


def _validate_connectome_matrix(
    value: object, *, n_regions: int, source: str
) -> FloatArray:
    """Admit structural weights with an exact zero self-coupling diagonal.

    Parameters
    ----------
    value : object
        Original producer matrix before float conversion.
    n_regions : int
        Required number of rows and columns.
    source : str
        Producer identity used in refusal messages.

    Returns
    -------
    FloatArray
        Finite non-negative symmetric C-contiguous float64 weights.

    Raises
    ------
    ValueError
        If the producer matrix violates source, shape or structural admission.
    """
    matrix = _coerce_connectome_matrix(value, n_regions=n_regions, source=source)
    if np.any(np.diag(matrix) != 0.0):
        raise ValueError(f"{source} connectome output diagonal must be zero")
    return np.ascontiguousarray(matrix, dtype=np.float64)


def _coerce_connectome_matrix(
    value: object, *, n_regions: int, source: str
) -> FloatArray:
    """Validate source types, shape and graph weights before float publication.

    Parameters
    ----------
    value : object
        Original matrix; finite real object entries remain compatible.
    n_regions : int
        Required square dimension.
    source : str
        Producer identity for refusal messages.

    Returns
    -------
    FloatArray
        Symmetric finite non-negative contiguous weights. The caller owns
        diagonal admission or canonicalisation for its ingress contract.

    Raises
    ------
    ValueError
        If source elements, conversion, shape, finiteness or graph weights fail.
    """
    raw = np.asarray(value, dtype=object)
    if any(isinstance(item, bool | np.bool_) for item in raw.ravel()):
        raise ValueError(f"{source} connectome output must not contain boolean values")
    if any(isinstance(item, complex | np.complexfloating) for item in raw.ravel()):
        raise ValueError(f"{source} connectome output must contain real-valued weights")
    for item in raw.ravel():
        if not isinstance(item, str | bytes | np.str_ | np.bytes_):
            continue
        try:
            float(item)
        except (TypeError, ValueError):
            continue
        raise ValueError(
            f"{source} connectome output must not contain numeric-string aliases"
        )
    try:
        require_real_values(
            value, name=f"{source} connectome output", allow_object=True
        )
        matrix = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{source} connectome output must be a float matrix") from exc
    if matrix.shape != (n_regions, n_regions):
        raise ValueError(
            f"{source} connectome output must have shape "
            f"({n_regions}, {n_regions}), got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{source} connectome output must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError(f"{source} connectome output must be non-negative")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0.0):
        raise ValueError(f"{source} connectome output must be symmetric")
    return np.ascontiguousarray(matrix, dtype=np.float64)
