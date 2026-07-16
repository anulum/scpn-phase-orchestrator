# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Shared assertions for public typing contract tests."""

from __future__ import annotations

from typing import Any, get_args, get_origin

import numpy as np
from numpy.typing import NDArray


def assert_precise_ndarray_hint(hint: object, *, context: str | None = None) -> None:
    """Assert that a hint contains only concrete dtype-specialized NDArrays."""

    context_prefix = f"{context}: " if context else ""

    if not _contains_precise_ndarray(hint, context_prefix=context_prefix):
        raise AssertionError(
            f"{context_prefix}expected a numpy.ndarray typing hint, got {hint!r}"
        )


def _contains_precise_ndarray(hint: object, *, context_prefix: str) -> bool:
    if isinstance(hint, (list, tuple)):
        return any(
            _contains_precise_ndarray(arg, context_prefix=context_prefix)
            for arg in hint
        )

    origin = get_origin(hint)

    # numpy >= 2.5 keeps ``NDArray[dtype]`` as its alias — ``get_origin`` is the
    # ``NDArray`` alias and the sole arg is the concrete dtype. Older numpy expands
    # it to ``ndarray[shape, dtype[dtype]]`` (``get_origin`` is ``np.ndarray``, the
    # dtype nested under ``args[1]``). Accept both so the contract test is not
    # hostage to the numpy repr of a hint.
    if origin is NDArray:
        alias_args = get_args(hint)
        if len(alias_args) != 1:
            raise AssertionError(
                f"{context_prefix}expected a single NDArray dtype arg, got {hint!r}"
            )
        _assert_concrete_dtype(alias_args[0], hint, context_prefix=context_prefix)
        return True

    if origin is not np.ndarray:
        return any(
            _contains_precise_ndarray(arg, context_prefix=context_prefix)
            for arg in get_args(hint)
        )

    args = get_args(hint)
    if len(args) < 2:
        raise AssertionError(
            f"{context_prefix}expected shape and dtype args on the hint, got {hint!r}"
        )

    dtype_hint = args[1]
    dtype_args = get_args(dtype_hint)
    if len(dtype_args) != 1:
        raise AssertionError(
            f"{context_prefix}expected a typed numpy.dtype spec, got {hint!r}"
        )

    _assert_concrete_dtype(dtype_args[0], hint, context_prefix=context_prefix)
    return True


def _assert_concrete_dtype(
    dtype_arg: object, hint: object, *, context_prefix: str
) -> None:
    """Raise unless ``dtype_arg`` is a concrete (non-``Any``) numpy dtype."""
    if dtype_arg is Any:
        raise AssertionError(f"{context_prefix}NDArray dtype must not be Any")
    try:
        np.dtype(dtype_arg)
    except TypeError as exc:
        raise AssertionError(
            f"{context_prefix}NDArray dtype must be concrete; got {dtype_arg!r}"
        ) from exc
