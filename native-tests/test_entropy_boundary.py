# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native entropy production boundary tests

"""Exercise the installed Rust dissipation ingress without substitution."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

Rate = Callable[
    [NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], object, object],
    float,
]
rate = cast(Rate, spo_kernel.entropy_production_rate)


@pytest.mark.parametrize("field", ["alpha", "dt"])
@pytest.mark.parametrize(
    "value",
    [
        True,
        np.bool_(True),
        "1",
        1 + 0j,
        np.timedelta64(1, "ms"),
        np.datetime64("2026-01-01"),
    ],
)
def test_scalar_alias_refusal(field: str, value: object) -> None:
    """Only original real controls reach the Rust core."""
    with pytest.raises((ValueError, TypeError)):
        rate(
            np.zeros(3),
            np.ones(3),
            np.zeros(9),
            value if field == "alpha" else 1.0,
            value if field == "dt" else 0.1,
        )


@pytest.mark.parametrize("field", ["phases", "omegas", "knm"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_measurement_refusal(field: str, value: float) -> None:
    """Nonfinite samples cannot produce a published dissipation rate."""
    p = np.zeros(3)
    o = np.ones(3)
    k = np.zeros(9)
    target = p if field == "phases" else o if field == "omegas" else k
    target[0] = value
    with pytest.raises(ValueError):
        rate(p, o, k, 1.0, 0.1)


@pytest.mark.parametrize("field", ["alpha", "dt"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_control_refusal(field: str, value: float) -> None:
    """Nonfinite real controls fail before dissipation calculation."""
    with pytest.raises(ValueError):
        rate(
            np.zeros(3),
            np.ones(3),
            np.zeros(9),
            value if field == "alpha" else 1.0,
            value if field == "dt" else 0.1,
        )


@pytest.mark.parametrize(
    "p_size,o_size,k_size", [(3, 2, 9), (3, 3, 8), (0, 1, 0), (0, 0, 1)]
)
def test_cardinality_refusal(p_size: int, o_size: int, k_size: int) -> None:
    """Malformed buffers are refused even for an empty phase vector."""
    with pytest.raises(ValueError):
        rate(np.zeros(p_size), np.zeros(o_size), np.zeros(k_size), 1.0, 0.1)


def test_negative_timestep_refusal() -> None:
    """Negative time intervals do not silently return zero."""
    with pytest.raises(ValueError):
        rate(np.zeros(3), np.ones(3), np.zeros(9), 1.0, -0.1)


@pytest.mark.parametrize("n,dt", [(0, 0.1), (3, 0.0)])
def test_zero_contract(n: int, dt: float) -> None:
    """Valid empty systems and zero intervals keep zero dissipation."""
    assert rate(np.zeros(n), np.ones(n), np.zeros(n * n), 1.0, dt) == 0.0


def test_output_overflow_refusal() -> None:
    """Finite input that overflows the dissipation sum fails closed."""
    with pytest.raises(ValueError):
        rate(np.zeros(1), np.array([1e308]), np.zeros(1), 1.0, 0.1)
