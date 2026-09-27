# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Adaptive coupling measurement ingress

"""Validate adaptive coupling measurements through the actual public pipeline."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.te_adaptive import te_adapt_coupling

FloatArray: TypeAlias = NDArray[np.float64]


@pytest.mark.parametrize("field", ["knm", "phase_history"])
@pytest.mark.parametrize(
    "dtype",
    [
        "U8",
        "timedelta64[ms]",
        "timedelta64[ns]",
        "datetime64[ms]",
        "datetime64[ns]",
        "object-time",
    ],
)
def test_adaptive_coupling_rejects_source_aliases(field: str, dtype: str) -> None:
    """Text and unit-bearing time values cannot become coupling or phases."""
    k = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.vstack([np.linspace(0.0, 5.0, 32), np.linspace(0.2, 5.2, 32)])
    original = k if field == "knm" else h
    invalid = cast(
        FloatArray,
        np.array([np.timedelta64(1, "ns")] * original.size, dtype=object).reshape(
            original.shape
        )
        if dtype == "object-time"
        else original.astype(dtype),
    )
    with pytest.raises(ValueError):
        te_adapt_coupling(
            invalid if field == "knm" else k, invalid if field == "phase_history" else h
        )


@pytest.mark.parametrize("field", ["lr", "decay", "n_bins"])
def test_adaptive_coupling_rejects_temporal_controls(field: str) -> None:
    """Real/Integral timedelta aliases are rejected before parameter conversion."""
    k = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.vstack([np.linspace(0.0, 5.0, 32), np.linspace(0.2, 5.2, 32)])
    value = np.timedelta64(2, "ms")
    with pytest.raises(ValueError):
        te_adapt_coupling(
            k,
            h,
            lr=cast(float, value) if field == "lr" else 0.01,
            decay=cast(float, value) if field == "decay" else 0.0,
            n_bins=cast(int, value) if field == "n_bins" else 8,
        )


def test_adaptive_coupling_numeric_objects_preserve_update() -> None:
    """Real object measurements preserve the full TE estimation and update."""
    k = np.array([[0.0, 1.0], [1.0, 0.0]])
    h = np.vstack([np.linspace(0.0, 5.0, 32), np.linspace(0.2, 5.2, 32)])
    expected = te_adapt_coupling(k, h)
    actual = te_adapt_coupling(
        cast(FloatArray, k.astype(object)), cast(FloatArray, h.astype(object))
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
