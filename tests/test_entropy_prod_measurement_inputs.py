# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Entropy production measurement source types

"""Exercise source measurement and control types through real entropy APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _entropy_prod_go,
    _entropy_prod_julia,
    _entropy_prod_mojo,
)
from scpn_phase_orchestrator.monitor.entropy_prod import entropy_production_rate

FloatArray = NDArray[np.float64]
Rate = Callable[[FloatArray, FloatArray, FloatArray, float, float], float]
BACKENDS: list[Rate] = [
    entropy_production_rate,
    _entropy_prod_go.entropy_production_rate_go,
    _entropy_prod_julia.entropy_production_rate_julia,
    _entropy_prod_mojo.entropy_production_rate_mojo,
]


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("field", ["phases", "omegas", "knm"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "datetime64[ms]", "U", "bool", "complex128"]
)
def test_source_refusal(backend: Rate, field: str, dtype: str) -> None:
    """Temporal, textual, boolean and complex measurements are refused."""
    p = np.arange(3, dtype=np.float64)
    o = np.ones(3)
    k = np.ones((3, 3)) - np.eye(3)
    if field == "phases":
        p = cast(FloatArray, p.astype(dtype))
    elif field == "omegas":
        o = cast(FloatArray, o.astype(dtype))
    else:
        k = cast(FloatArray, k.astype(dtype))
    with pytest.raises(ValueError):
        backend(p, o, k, 1.0, 0.1)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("field", ["alpha", "dt"])
def test_temporal_control_refusal(backend: Rate, field: str) -> None:
    """Temporal scalars fail with a contract refusal before float extraction."""
    alpha: float = 1.0
    dt: float = 0.1
    if field == "alpha":
        alpha = cast(float, np.timedelta64(1, "ms"))
    else:
        dt = cast(float, np.timedelta64(1, "ms"))
    with pytest.raises(ValueError):
        backend(np.zeros(3), np.ones(3), np.zeros((3, 3)), alpha, dt)


@pytest.mark.parametrize("backend", BACKENDS)
def test_object_numeric_parity(backend: Rate) -> None:
    """Supported numeric object storage retains real runtime parity."""
    p = np.arange(3, dtype=np.float64)
    o = np.ones(3)
    k = np.ones((3, 3)) - np.eye(3)
    expected = entropy_production_rate(p, o, k, 1.0, 0.1)
    assert backend(
        cast(FloatArray, p.astype(object)),
        cast(FloatArray, o.astype(object)),
        cast(FloatArray, k.astype(object)),
        1.0,
        0.1,
    ) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("field", ["phases", "omegas", "knm"])
def test_object_temporal_refusal(backend: Rate, field: str) -> None:
    """Object arrays retain temporal source identity for refusal."""
    p = np.zeros(3)
    o = np.ones(3)
    k = np.zeros((3, 3))
    temporal = np.array([np.timedelta64(i, "ms") for i in range(3)], dtype=object)
    if field == "phases":
        p = cast(FloatArray, temporal)
    elif field == "omegas":
        o = cast(FloatArray, temporal)
    else:
        k = cast(FloatArray, np.tile(temporal, (3, 1)))
    with pytest.raises(ValueError):
        backend(p, o, k, 1.0, 0.1)
