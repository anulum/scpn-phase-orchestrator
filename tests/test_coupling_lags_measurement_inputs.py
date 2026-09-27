# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Phase-lag measurement ingress

"""Check measurement source types through all public lag-model operations."""

from __future__ import annotations

from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.lags import LagModel

FloatArray: TypeAlias = NDArray[np.float64]


@pytest.mark.parametrize("surface", ["distances", "signal_a", "signal_b"])
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
def test_lag_measurements_reject_aliases(surface: str, dtype: str) -> None:
    """Reject unit-bearing times and text before physical lag arithmetic."""
    source = (
        np.array([[0.0, 1.0], [1.0, 0.0]])
        if surface == "distances"
        else np.array([0.0, 1.0, 0.0, -1.0])
    )
    raw = (
        np.array([np.timedelta64(1, "ns")] * source.size, dtype=object).reshape(
            source.shape
        )
        if dtype == "object-time"
        else source.astype(dtype)
    )
    invalid = cast(FloatArray, raw)
    with pytest.raises(ValueError):
        if surface == "distances":
            LagModel.estimate_from_distances(invalid, 1.0)
        else:
            LagModel().estimate_lag(
                invalid if surface == "signal_a" else source,
                invalid if surface == "signal_b" else source,
                10.0,
            )


@pytest.mark.parametrize(
    "field",
    [
        "speed",
        "sample_rate",
        "carrier",
        "n_layers",
        "source_index",
        "target_index",
        "lag",
    ],
)
def test_lag_controls_reject_temporal_aliases(field: str) -> None:
    """Reject timedelta scalars that satisfy numbers.Real and Integral."""
    value = np.timedelta64(1, "ms")
    with pytest.raises(ValueError):
        if field == "speed":
            LagModel.estimate_from_distances(
                np.array([[0.0, 1.0], [1.0, 0.0]]), cast(float, value)
            )
        elif field == "sample_rate":
            LagModel().estimate_lag(
                np.array([0.0, 1.0, 0.0, -1.0]),
                np.array([0.0, 1.0, 0.0, -1.0]),
                cast(float, value),
            )
        else:
            pair = (
                cast(int, value) if field == "source_index" else 0,
                cast(int, value) if field == "target_index" else 1,
            )
            LagModel().build_alpha_matrix(
                {pair: cast(float, value) if field == "lag" else 0.1},
                cast(int, value) if field == "n_layers" else 2,
                cast(float, value) if field == "carrier" else 1.0,
            )


def test_lag_real_numeric_objects_preserve_results() -> None:
    """Retain equivalent real-object measurements through actual estimators."""
    distances = np.array([[0.0, 1.0], [1.0, 0.0]])
    signal = np.array([0.0, 1.0, 0.0, -1.0])
    np.testing.assert_array_equal(
        LagModel.estimate_from_distances(
            cast(FloatArray, distances.astype(object)), 2.0
        ),
        LagModel.estimate_from_distances(distances, 2.0),
    )
    assert LagModel().estimate_lag(
        cast(FloatArray, signal.astype(object)), signal, 10.0
    ) == LagModel().estimate_lag(signal, signal, 10.0)
