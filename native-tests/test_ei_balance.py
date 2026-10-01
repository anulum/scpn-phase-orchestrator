# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Installed E/I native contracts

"""Exercise the real PyO3 E/I entry points, including refusal and recovery."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

Metrics = tuple[float, float, float, bool, float, float, float, float]


@pytest.mark.parametrize("operation", ["compute", "adjust"])
@pytest.mark.parametrize(
    "fault",
    ["short", "long", "nan", "infinity", "negative-e", "negative-i", "overflow-count"],
)
def test_native_refusal_preserves_buffers_and_recovers(
    operation: str, fault: str
) -> None:
    """Refuse malformed source inputs before any native indexing or mutation."""
    matrix = np.array([0.0, 2.0, 1.0, 0.0])
    e = np.array([0], dtype=np.int64)
    i = np.array([1], dtype=np.int64)
    n = 2
    if fault == "short":
        matrix = matrix[:1]
    elif fault == "long":
        matrix = np.append(matrix, 0.0)
    elif fault == "nan":
        matrix[1] = np.nan
    elif fault == "infinity":
        matrix[1] = np.inf
    elif fault == "negative-e":
        e[0] = -1
    elif fault == "negative-i":
        i[0] = -1
    else:
        n = int(np.iinfo(np.uintp).max)
    snapshots = [v.copy() for v in (matrix, e, i)]
    with pytest.raises(ValueError):
        if operation == "compute":
            spo_kernel.compute_ei_balance_rust(matrix, n, e, i)
        else:
            spo_kernel.adjust_ei_ratio_rust(matrix, n, e, i)
    for source, snapshot in zip((matrix, e, i), snapshots, strict=True):
        np.testing.assert_array_equal(source, snapshot)
    good = np.array([0.0, 2.0, 1.0, 0.0])
    e, i = np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)
    assert spo_kernel.compute_ei_balance_rust(good, 2, e, i)[0] == 2.0
    np.testing.assert_array_equal(
        spo_kernel.adjust_ei_ratio_rust(good, 2, e, i), [0, 2, 2, 0]
    )


@pytest.mark.parametrize("operation", ["compute", "adjust"])
@pytest.mark.parametrize("buffer", ["matrix", "exc", "inh"])
def test_native_strides_refuse_without_mutation(operation: str, buffer: str) -> None:
    """Keep the direct contiguous-buffer admission explicit and recoverable."""
    matrix = np.array([0.0, 2.0, 1.0, 0.0])
    e, i = np.array([0, 0], dtype=np.int64), np.array([1, 1], dtype=np.int64)
    if buffer == "matrix":
        matrix = np.repeat(matrix, 2)[::2]
    elif buffer == "exc":
        e = np.repeat(e, 2)[::2]
    else:
        i = np.repeat(i, 2)[::2]
    before = matrix.copy()
    with pytest.raises(ValueError):
        if operation == "compute":
            spo_kernel.compute_ei_balance_rust(matrix, 2, e, i)
        else:
            spo_kernel.adjust_ei_ratio_rust(matrix, 2, e, i)
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf"), 1e-310])
def test_native_target_or_scale_refuses(target: float) -> None:
    """Reject invalid dimensionless targets and unrepresentable scale values."""
    matrix = np.array([0.0, 2.0, 1.0, 0.0])
    before = matrix.copy()
    with pytest.raises(ValueError):
        spo_kernel.adjust_ei_ratio_rust(
            matrix,
            2,
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            target,
        )
    np.testing.assert_array_equal(matrix, before)


@pytest.mark.parametrize("value", [True, np.bool_(True), "1", 1 + 0j])
def test_native_target_types_refuse(value: object) -> None:
    """Reject coercion aliases before reading the real coupling buffers."""
    matrix = np.array([0.0, 2.0, 1.0, 0.0])
    with pytest.raises((TypeError, ValueError)):
        spo_kernel.adjust_ei_ratio_rust(
            matrix,
            2,
            np.array([0], dtype=np.int64),
            np.array([1], dtype=np.int64),
            value,
        )
    np.testing.assert_array_equal(matrix, [0.0, 2.0, 1.0, 0.0])


def test_native_duplicate_and_ignored_indices_form_sets() -> None:
    """Replay analytic directed means and apply inhibitory scaling once."""
    matrix = np.array([0.0, 2.0, 4.0, 6.0, 0.0, 8.0, 10.0, 12.0, 0.0])
    e, i = np.array([0, 0, 1, 99], dtype=np.int64), np.array([2, 2, 99], dtype=np.int64)
    result = cast(Metrics, spo_kernel.compute_ei_balance_rust(matrix, 3, e, i))
    assert result[0] == pytest.approx(5 / 11)
    assert result[1:3] == pytest.approx((10 / 3, 22 / 3))
    assert result[3] is False
    assert result[4:] == pytest.approx((2.0, 6.0, 11.0, 0.0))
    adjusted = cast(
        NDArray[np.float64], spo_kernel.adjust_ei_ratio_rust(matrix, 3, e, i)
    )
    np.testing.assert_allclose(
        adjusted, [0.0, 2.0, 4.0, 6.0, 0.0, 8.0, 50 / 11, 60 / 11, 0.0]
    )
    assert not np.shares_memory(adjusted, matrix)


@pytest.mark.parametrize("value", [0.0, 1e308, -1e308])
def test_native_finite_large_mean_and_copy(value: float) -> None:
    """Avoid sum overflow and preserve independent no-op native allocations."""
    matrix = np.full(4, value)
    e, i = np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)
    result = cast(Metrics, spo_kernel.compute_ei_balance_rust(matrix, 2, e, i))
    assert result == (1.0, value, value, True, value, value, value, value)
    adjusted = cast(
        NDArray[np.float64], spo_kernel.adjust_ei_ratio_rust(matrix, 2, e, i)
    )
    np.testing.assert_array_equal(adjusted, matrix)
    assert not np.shares_memory(adjusted, matrix)


def test_native_empty_is_neutral() -> None:
    """Retain the real zero-sized matrix contract across both entry points."""
    matrix = np.empty(0)
    indices = np.array([99], dtype=np.int64)
    assert spo_kernel.compute_ei_balance_rust(matrix, 0, indices, indices) == (
        1.0,
        0.0,
        0.0,
        True,
        0.0,
        0.0,
        0.0,
        0.0,
    )
    adjusted = cast(
        NDArray[np.float64],
        spo_kernel.adjust_ei_ratio_rust(matrix, 0, indices, indices),
    )
    assert adjusted.size == 0
    assert not np.shares_memory(adjusted, matrix)


@pytest.mark.parametrize("excitation", [-2.0, 2.0])
@pytest.mark.parametrize("inhibition", [-1e308, 1e308])
def test_native_underflowing_scale_refuses_and_recovers(
    excitation: float, inhibition: float
) -> None:
    """Reject signed zero scales at the actual PyO3 entry point without mutation."""
    matrix = np.array([0.0, excitation, inhibition, 0.0])
    e, i = np.array([0, 0], dtype=np.int64), np.array([1, 1], dtype=np.int64)
    snapshots = [v.copy() for v in (matrix, e, i)]
    with pytest.raises(ValueError, match="non-zero scale"):
        spo_kernel.adjust_ei_ratio_rust(matrix, 2, e, i, 1e308)
    adjusted = cast(
        NDArray[np.float64], spo_kernel.adjust_ei_ratio_rust(matrix, 2, e, i)
    )
    assert adjusted[2] == pytest.approx(excitation)
    assert spo_kernel.compute_ei_balance_rust(adjusted, 2, e, i)[0] == pytest.approx(
        1.0
    )
    assert not np.shares_memory(adjusted, matrix)
    for source, snapshot in zip((matrix, e, i), snapshots, strict=True):
        np.testing.assert_array_equal(source, snapshot)


def test_native_representable_scale_preserves_silent_summary() -> None:
    """Exercise a valid tiny product without claiming a finite summary target."""
    matrix = np.array([0.0, 2.0, 1.0, 0.0])
    e, i = np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)
    adjusted = cast(
        NDArray[np.float64], spo_kernel.adjust_ei_ratio_rust(matrix, 2, e, i, 1e20)
    )
    assert adjusted[2] == pytest.approx(2e-20, rel=1e-15, abs=0.0)
    assert spo_kernel.compute_ei_balance_rust(adjusted, 2, e, i)[0] == float("inf")
    np.testing.assert_array_equal(matrix, [0.0, 2.0, 1.0, 0.0])
