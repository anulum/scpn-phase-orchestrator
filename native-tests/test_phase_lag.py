# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native phase-lag source types

"""Exercise the installed phase-lag Python-to-Rust boundary."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize(
    "value",
    [True, np.bool_(True), "1", np.timedelta64(1, "ms"), np.datetime64("2026-01-01")],
)
def test_native_phase_lag_rejects_measurement_aliases(value: object) -> None:
    """Original flat distance values must survive validation before extraction."""
    with pytest.raises(ValueError):
        spo_kernel.PyLagModel.estimate([0.0, value, value, 0.0], 2, 1.0)


@pytest.mark.parametrize("field", ["n", "speed"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "1", np.timedelta64(1, "ms")])
def test_native_phase_lag_rejects_control_aliases(field: str, value: object) -> None:
    """Native counts and propagation speed reject coercible aliases."""
    with pytest.raises(ValueError):
        spo_kernel.PyLagModel.estimate(
            [0.0, 1.0, 1.0, 0.0],
            value if field == "n" else 2,
            value if field == "speed" else 1.0,
        )
    if field == "n":
        with pytest.raises(ValueError):
            spo_kernel.PyLagModel.zeros(value)


def test_native_phase_lag_numeric_objects_preserve_estimate() -> None:
    """Real object distance values remain equivalent at the installed boundary."""
    objects = np.array([0, 1.0, np.float32(1.0), 0], dtype=object)
    actual = spo_kernel.PyLagModel.estimate(objects, np.int64(2), np.float64(2.0))
    expected = spo_kernel.PyLagModel.estimate([0.0, 1.0, 1.0, 0.0], 2, 2.0)
    assert actual.n == expected.n == 2
    np.testing.assert_array_equal(actual.alpha, expected.alpha)


@pytest.mark.parametrize(
    "distances",
    [
        [0.0, 1.0, 1.0],
        [0.0, np.nan, np.nan, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 2.0, 0.0],
    ],
)
def test_native_phase_lag_preserves_physical_refusals(distances: list[float]) -> None:
    """Source validation still delegates shape and physics faults to the engine."""
    with pytest.raises(ValueError):
        spo_kernel.PyLagModel.estimate(distances, 2, 1.0)


def test_native_zero_phase_lag_preserves_empty_transport() -> None:
    """A valid zero-lag constructor retains the complete oscillator matrix."""
    model = spo_kernel.PyLagModel.zeros(np.int64(3))
    assert model.n == 3
    np.testing.assert_array_equal(
        np.asarray(model.alpha).reshape(3, 3), np.zeros((3, 3))
    )
