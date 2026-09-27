# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Original measurement source samples

"""Construct source types whose silent float conversion changes their meaning."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray


def source_alias(kind: str, shape: tuple[int, ...]) -> NDArray[np.float64]:
    """Return deliberately mistyped measurement storage for an API refusal test.

    Parameters
    ----------
    kind : str
        Numeric text, NumPy duration storage or object duration elements.
    shape : tuple[int, ...]
        The valid numerical shape required by the production API.

    Returns
    -------
    NDArray[np.float64]
        Storage with an intentionally false static dtype annotation. Tests use
        the runtime dtype to check that callers refuse the original types.
    """
    if kind == "text":
        return cast(NDArray[np.float64], np.full(shape, "0"))
    if kind == "duration":
        return cast(NDArray[np.float64], np.zeros(shape, dtype="timedelta64[ns]"))
    if kind == "object_duration":
        return cast(
            NDArray[np.float64],
            np.array(
                [np.timedelta64(0, "ns")] * int(np.prod(shape)), dtype=object
            ).reshape(shape),
        )
    raise ValueError(f"unknown source kind {kind!r}")
