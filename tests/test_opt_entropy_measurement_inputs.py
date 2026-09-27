# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Ordinal entropy measurement source types

"""Test original types through public and real compiled bridge APIs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _opt_entropy_go,
    _opt_entropy_julia,
    _opt_entropy_mojo,
)
from scpn_phase_orchestrator.monitor.opt_entropy import (
    ordinal_pattern_sequence,
    transition_entropy,
)

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
CodesFn: TypeAlias = Callable[[FloatArray, int, int], IntArray]
EntropyFn: TypeAlias = Callable[[FloatArray, int, int], float]

BACKENDS: dict[str, tuple[CodesFn, EntropyFn]] = {
    "public": (ordinal_pattern_sequence, transition_entropy),
    "go": (
        _opt_entropy_go.ordinal_pattern_sequence_go,
        _opt_entropy_go.transition_entropy_go,
    ),
    "julia": (
        _opt_entropy_julia.ordinal_pattern_sequence_julia,
        _opt_entropy_julia.transition_entropy_julia,
    ),
    "mojo": (
        _opt_entropy_mojo.ordinal_pattern_sequence_mojo,
        _opt_entropy_mojo.transition_entropy_mojo,
    ),
}


@pytest.mark.parametrize("backend", list(BACKENDS))
@pytest.mark.parametrize("surface", ["codes", "entropy"])
@pytest.mark.parametrize(
    "dtype", ["timedelta64[ms]", "datetime64[ns]", "object-time", "U16", "bool"]
)
def test_ordinal_api_rejects_measurement_aliases(
    backend: str, surface: str, dtype: str
) -> None:
    """Original temporal/object aliases are refused before bridge execution."""
    source = np.arange(20, dtype=np.float64)
    value = (
        np.array([np.timedelta64(i, "ns") for i in range(20)], dtype=object)
        if dtype == "object-time"
        else source.astype(dtype)
    )
    codes, entropy = BACKENDS[backend]
    with pytest.raises(ValueError):
        if surface == "codes":
            codes(cast(FloatArray, value), 3, 1)
        else:
            entropy(cast(FloatArray, value), 3, 1)


@pytest.mark.parametrize("backend", list(BACKENDS))
@pytest.mark.parametrize("surface", ["codes", "entropy"])
@pytest.mark.parametrize("field", ["dimension", "delay"])
def test_ordinal_api_rejects_temporal_controls(
    backend: str, surface: str, field: str
) -> None:
    """Durations cannot select the ordinal embedding dimension or delay."""
    value = cast(int, np.timedelta64(3, "ns"))
    dimension, delay = (value, 1) if field == "dimension" else (3, value)
    codes, entropy = BACKENDS[backend]
    with pytest.raises(ValueError):
        if surface == "codes":
            codes(np.arange(20, dtype=np.float64), dimension, delay)
        else:
            entropy(np.arange(20, dtype=np.float64), dimension, delay)


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_ordinal_api_preserves_numeric_objects(backend: str) -> None:
    """Real object samples preserve tied ordinal codes and entropy."""
    value = np.array([0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    codes, entropy = BACKENDS[backend]
    np.testing.assert_array_equal(
        codes(value, 3, 1), codes(cast(FloatArray, value.astype(object)), 3, 1)
    )
    assert entropy(value, 3, 1) == entropy(cast(FloatArray, value.astype(object)), 3, 1)


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_ordinal_api_rejects_boolean_promoted_by_sequence(backend: str) -> None:
    """A Boolean mixed with float samples cannot disappear during promotion."""
    values = cast(FloatArray, [0.0, 2.0, True, 1.0, 3.0, 0.0])
    codes, entropy = BACKENDS[backend]
    with pytest.raises(ValueError):
        codes(values, 3, 1)
    with pytest.raises(ValueError):
        entropy(values, 3, 1)


@pytest.mark.parametrize("backend", list(BACKENDS))
def test_ordinal_api_large_delay_has_no_embedding_windows(backend: str) -> None:
    """A huge valid delay never wraps at the native transport boundary."""
    codes, entropy = BACKENDS[backend]
    series = np.arange(20, dtype=np.float64)
    delay = int(np.iinfo(np.uintp).max)
    np.testing.assert_array_equal(codes(series, 7, delay), [])
    assert entropy(series, 7, delay) == 0.0
