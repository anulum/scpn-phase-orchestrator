# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera measurement source types

"""Exercise original measurement types through public and actual backend APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_go import (
    local_order_parameter_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_julia import (
    local_order_parameter_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._chimera_mojo import (
    local_order_parameter_mojo,
)
from scpn_phase_orchestrator.monitor import chimera

FloatArray = NDArray[np.float64]
Backend = Callable[[FloatArray, FloatArray, int], FloatArray]


@pytest.mark.parametrize("field", ["phases", "knm"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "datetime64[ms]", "U", "bool", "complex128"]
)
def test_public_source_refusal(field: str, dtype: str) -> None:
    """Both public entry points reject aliased measurement arrays."""
    p = np.arange(3, dtype=np.float64)
    k = np.ones((3, 3)) - np.eye(3)
    if field == "phases":
        p = cast(FloatArray, p.astype(dtype))
    else:
        k = cast(FloatArray, k.astype(dtype))
    for fn in (chimera.local_order_parameter, chimera.detect_chimera):
        with pytest.raises(ValueError):
            fn(p, k)


@pytest.mark.parametrize(
    "backend",
    [local_order_parameter_go, local_order_parameter_julia, local_order_parameter_mojo],
)
@pytest.mark.parametrize("field", ["phases", "knm"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "datetime64[ms]", "U", "bool", "complex128"]
)
def test_direct_source_refusal(backend: Backend, field: str, dtype: str) -> None:
    """Direct accelerator ingress refuses aliases before numerical transport."""
    p = np.arange(3, dtype=np.float64)
    k = (np.ones((3, 3)) - np.eye(3)).ravel()
    if field == "phases":
        p = cast(FloatArray, p.astype(dtype))
    else:
        k = cast(FloatArray, k.astype(dtype))
    with pytest.raises(ValueError):
        backend(p, k, 3)


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param(fn, marks=pytest.mark.native_runtime)
        for fn in (
            local_order_parameter_go,
            local_order_parameter_julia,
            local_order_parameter_mojo,
        )
    ],
)
def test_numeric_object_parity(backend: Backend) -> None:
    """Real object arrays retain numerical parity through installed accelerators."""
    p = np.arange(3, dtype=np.float64)
    k = np.ones((3, 3)) - np.eye(3)
    expected = chimera.local_order_parameter(p, k)
    np.testing.assert_allclose(
        chimera.local_order_parameter(
            cast(FloatArray, p.astype(object)), cast(FloatArray, k.astype(object))
        ),
        expected,
        atol=1e-9,
        rtol=0,
    )
    np.testing.assert_allclose(
        backend(
            cast(FloatArray, p.astype(object)),
            cast(FloatArray, k.ravel().astype(object)),
            3,
        ),
        expected,
        atol=1e-9,
        rtol=0,
    )


@pytest.mark.parametrize(
    "backend",
    [local_order_parameter_go, local_order_parameter_julia, local_order_parameter_mojo],
)
def test_temporal_count_refusal(backend: Backend) -> None:
    """Temporal integer-like metadata cannot select oscillator cardinality."""
    with pytest.raises(ValueError):
        backend(np.zeros(3), np.zeros(9), cast(int, np.timedelta64(3, "ms")))


@pytest.mark.parametrize("field", ["phases", "knm"])
def test_object_temporal_refusal(field: str) -> None:
    """Object storage cannot turn temporal measurements into numeric samples."""
    p = np.arange(3, dtype=np.float64)
    k = np.ones((3, 3)) - np.eye(3)
    if field == "phases":
        p = cast(
            FloatArray,
            np.array([np.timedelta64(i, "ms") for i in range(3)], dtype=object),
        )
    else:
        k = cast(
            FloatArray,
            np.array(
                [[np.timedelta64(int(v), "ms") for v in row] for row in k], dtype=object
            ),
        )
    with pytest.raises(ValueError):
        chimera.local_order_parameter(p, k)


def test_public_numeric_object_parity() -> None:
    """Public chimera measurements preserve real object samples without a runtime."""
    phases = np.arange(3, dtype=np.float64)
    knm = np.ones((3, 3)) - np.eye(3)
    np.testing.assert_allclose(
        chimera.local_order_parameter(
            cast(FloatArray, phases.astype(object)),
            cast(FloatArray, knm.astype(object)),
        ),
        chimera.local_order_parameter(phases, knm),
        atol=1e-9,
        rtol=0,
    )
