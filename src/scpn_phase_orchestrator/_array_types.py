# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Real array source types

"""Reject measurement aliases before callers convert arrays to floating point."""

from __future__ import annotations

from collections.abc import Sequence
from numbers import Real

import numpy as np


def require_real_values(
    value: object, *, name: str, allow_object: bool = False
) -> None:
    """Require plain real array elements without converting their source types.

    Parameters
    ----------
    value : object
        Array or sequence before conversion. Shape and finiteness are checked
        by the owning numerical contract.
    name : str
        Argument name for error messages.
    allow_object : bool
        Whether object arrays containing only real numbers are supported.

    Raises
    ------
    ValueError
        If elements are boolean, text, complex, temporal or otherwise not
        real numbers, or object storage is unsupported by this contract.
    """
    raw = np.asarray(value)
    if raw.dtype.kind == "O":
        if not allow_object or any(
            isinstance(item, (bool, np.bool_, np.datetime64, np.timedelta64))
            or not isinstance(item, Real)
            for item in raw.flat
        ):
            raise ValueError(f"{name} must contain only plain real numbers")
    elif raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain only plain real numbers")
    if isinstance(value, Sequence) and any(
        isinstance(item, (bool, np.bool_))
        for item in np.asarray(value, dtype=object).flat
    ):
        raise ValueError(f"{name} must not contain boolean values")
