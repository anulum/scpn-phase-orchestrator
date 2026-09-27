# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native ordinal entropy source types

"""Exercise installed typed-array ordinal entropy and parameter boundaries."""

from __future__ import annotations

import numpy as np
import pytest
import spo_kernel


@pytest.mark.parametrize("surface", ["ordinal_pattern_sequence", "transition_entropy"])
@pytest.mark.parametrize("field", ["dimension", "delay"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "3", np.timedelta64(3, "ns")])
def test_native_ordinal_rejects_control_aliases(
    surface: str, field: str, value: object
) -> None:
    """Native parameter extraction cannot reinterpret measurement aliases."""
    controls: dict[str, object] = {"dimension": 3, "delay": 1}
    controls[field] = value
    with pytest.raises(ValueError):
        getattr(spo_kernel, surface)(np.arange(20, dtype=np.float64), **controls)


@pytest.mark.parametrize("surface", ["ordinal_pattern_sequence", "transition_entropy"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_native_ordinal_rejects_nonfinite_samples(surface: str, value: float) -> None:
    """Ordinal ranking is undefined for nonfinite measurements."""
    with pytest.raises(ValueError):
        getattr(spo_kernel, surface)(np.array([0.0, value, 1.0, 2.0]), 3, 1)


@pytest.mark.parametrize("surface", ["ordinal_pattern_sequence", "transition_entropy"])
@pytest.mark.parametrize("controls", [(1, 1), (8, 1), (3, 0)])
def test_native_ordinal_rejects_invalid_embedding_domain(
    surface: str, controls: tuple[int, int]
) -> None:
    """Dimensions and delays obey the public embedding contract."""
    with pytest.raises(ValueError):
        getattr(spo_kernel, surface)(np.arange(20, dtype=np.float64), *controls)


def test_native_ordinal_large_delay_has_no_windows() -> None:
    """A huge delay yields empty evidence without native multiplication overflow."""
    series = np.arange(20, dtype=np.float64)
    delay = int(np.iinfo(np.uintp).max)
    np.testing.assert_array_equal(
        spo_kernel.ordinal_pattern_sequence(series, 7, delay), []
    )
    assert spo_kernel.transition_entropy(series, 7, delay) == 0.0


def test_native_ordinal_defaults_and_numpy_integer_controls() -> None:
    """Default and NumPy integer metadata retain the same real ordinal result."""
    series = np.array([0.0, 2.0, 1.0, 1.0, 3.0, 0.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        spo_kernel.ordinal_pattern_sequence(series),
        spo_kernel.ordinal_pattern_sequence(series, np.int64(3), np.int64(1)),
    )
    assert spo_kernel.transition_entropy(series) == spo_kernel.transition_entropy(
        series, np.int64(3), np.int64(1)
    )


@pytest.mark.parametrize("surface", ["ordinal_pattern_sequence", "transition_entropy"])
@pytest.mark.parametrize(
    "dtype", ["bool", "U8", "complex128", "timedelta64[ns]", "object"]
)
def test_native_ordinal_retains_float64_array_abi(surface: str, dtype: str) -> None:
    """The installed native ABI refuses alternative source storage types."""
    with pytest.raises(TypeError):
        getattr(spo_kernel, surface)(
            np.arange(20, dtype=np.float64).astype(dtype), 3, 1
        )
