# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — E/I balance measurement ingress

"""Verify original measurement types through public coupling operations."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.ei_balance import (
    adjust_ei_ratio,
    compute_ei_balance,
)

FloatArray: TypeAlias = NDArray[np.float64]


@pytest.mark.parametrize(
    "dtype", ["U8", "timedelta64[ms]", "datetime64[ms]", "object-time"]
)
@pytest.mark.parametrize("adjust", [False, True])
def test_ei_measurements_reject_coercion_aliases(dtype: str, adjust: bool) -> None:
    """Require real measurements before E/I summary or native adjustment."""
    matrix = cast(
        FloatArray,
        np.full((2, 2), np.timedelta64(1, "ms"), dtype=object)
        if dtype == "object-time"
        else np.array([[0.0, 1.0], [1.0, 0.0]]).astype(dtype),
    )
    with pytest.raises(ValueError):
        if adjust:
            adjust_ei_ratio(matrix, [0], [1])
        else:
            compute_ei_balance(matrix, [0], [1])


@pytest.mark.parametrize(
    "value", [np.timedelta64(1, "ms"), np.datetime64("2026-01-01")]
)
def test_ei_target_rejects_temporal_scalar(value: object) -> None:
    """Temporal target ratios cannot be interpreted as dimensionless controls."""
    with pytest.raises((TypeError, ValueError)):
        adjust_ei_ratio(
            np.array([[0.0, 1.0], [1.0, 0.0]]), [0], [1], cast(float, value)
        )


def test_ei_numeric_objects_preserve_native_results() -> None:
    """Keep real object matrices identical through actual public operations."""
    matrix = np.array([[0.0, 2.0], [1.0, 0.0]])
    objects = cast(FloatArray, matrix.astype(object))
    assert compute_ei_balance(matrix, [0], [1]) == compute_ei_balance(objects, [0], [1])
    np.testing.assert_allclose(
        adjust_ei_ratio(matrix, [0], [1]), adjust_ei_ratio(objects, [0], [1])
    )
