# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — AttnRes measurement ingress

"""Exercise original measurement types through AttnRes public and bridge APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.coupling.attention_residuals import (
    attnres_modulate,
    default_projections,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_go import (
    attnres_modulate_go,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_julia import (
    attnres_modulate_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.coupling._attnres_mojo import (
    attnres_modulate_mojo,
)

FloatArray: TypeAlias = NDArray[np.float64]
Bridge: TypeAlias = Callable[
    [
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        FloatArray,
        int,
        int,
        int,
        float,
        float,
    ],
    FloatArray,
]


def _inputs() -> list[FloatArray]:
    """Return a connected graph, phase vector and actual default projections."""
    q, k, v, o = default_projections()
    return [np.ones((3, 3)) - np.eye(3), np.array([0.0, 1.0, 2.0]), q, k, v, o]


@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]", "object-time"])
def test_public_attnres_rejects_temporal_arrays(field: int, dtype: str) -> None:
    """Reject temporal phase, coupling and projection measurements before dispatch."""
    values = _inputs()
    if dtype == "object-time":
        invalid = np.full(values[field].shape, np.timedelta64(1, "ms"), dtype=object)
    else:
        invalid = values[field].astype(dtype)
    values[field] = cast(FloatArray, invalid)
    k, theta, q, key, v, o = values
    with pytest.raises(ValueError):
        attnres_modulate(k, theta, w_q=q, w_k=key, w_v=v, w_o=o)


@pytest.mark.parametrize(
    "bridge", [attnres_modulate_go, attnres_modulate_julia, attnres_modulate_mojo]
)
@pytest.mark.parametrize("field", range(6))
@pytest.mark.parametrize("dtype", ["timedelta64[ms]", "datetime64[ms]", "object-time"])
def test_direct_attnres_rejects_temporal_arrays(
    bridge: Bridge, field: int, dtype: str
) -> None:
    """Reject original temporal values before optional runtime loading."""
    values = [a.ravel() for a in _inputs()]
    if dtype == "object-time":
        invalid = np.full(values[field].shape, np.timedelta64(1, "ms"), dtype=object)
    else:
        invalid = values[field].astype(dtype)
    values[field] = cast(FloatArray, invalid)
    k, theta, q, key, v, o = values
    with pytest.raises(ValueError):
        bridge(k, theta, q, key, v, o, 3, 4, -1, 1.0, 0.5)


@pytest.mark.parametrize(
    "field", ["n_heads", "block_size", "projection_seed", "temperature", "lambda_"]
)
def test_public_attnres_rejects_temporal_controls(field: str) -> None:
    """Reject timedelta scalars even when numbers.Real or Integral accepts them."""
    k, theta, q, key, v, o = _inputs()
    temporal = np.timedelta64(1, "ms")
    with pytest.raises(ValueError):
        attnres_modulate(
            k,
            theta,
            w_q=q,
            w_k=key,
            w_v=v,
            w_o=o,
            n_heads=cast(int, temporal) if field == "n_heads" else 4,
            block_size=cast(int, temporal) if field == "block_size" else None,
            projection_seed=cast(int, temporal) if field == "projection_seed" else 0,
            temperature=cast(float, temporal) if field == "temperature" else 1.0,
            lambda_=cast(float, temporal) if field == "lambda_" else 0.5,
        )


@pytest.mark.parametrize(
    "bridge", [attnres_modulate_go, attnres_modulate_julia, attnres_modulate_mojo]
)
@pytest.mark.parametrize(
    "field", ["n", "n_heads", "block_size", "temperature", "lambda_"]
)
def test_direct_attnres_rejects_temporal_controls(bridge: Bridge, field: str) -> None:
    """Validate scalar source types before any native backend call."""
    k, theta, q, key, v, o = [a.ravel() for a in _inputs()]
    temporal = np.timedelta64(1, "ms")
    with pytest.raises(ValueError):
        bridge(
            k,
            theta,
            q,
            key,
            v,
            o,
            cast(int, temporal) if field == "n" else 3,
            cast(int, temporal) if field == "n_heads" else 4,
            cast(int, temporal) if field == "block_size" else -1,
            cast(float, temporal) if field == "temperature" else 1.0,
            cast(float, temporal) if field == "lambda_" else 0.5,
        )


def test_public_attnres_preserves_numeric_objects() -> None:
    """Keep real numeric object inputs equivalent through the actual dispatcher."""
    values = _inputs()
    k, theta, q, key, v, o = values
    expected = attnres_modulate(k, theta, w_q=q, w_k=key, w_v=v, w_o=o)
    k, theta, q, key, v, o = [cast(FloatArray, a.astype(object)) for a in values]
    actual = attnres_modulate(k, theta, w_q=q, w_k=key, w_v=v, w_o=o)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
