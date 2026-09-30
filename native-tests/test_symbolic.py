# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Native symbolic extraction contracts

"""Exercise signed and unsigned symbolic arrays through the installed extension."""

from __future__ import annotations

import sys
from types import FrameType
from typing import Literal, cast

import numpy as np
import pytest
import spo_kernel
from numpy.typing import NDArray

from scpn_phase_orchestrator.oscillators.symbolic import SymbolicExtractor

IntegerArray = NDArray[np.int64] | NDArray[np.uint64]
IntegerDtype = type[np.int64] | type[np.uint64]
Operation = Literal["ring", "graph", "quality"]
OPERATIONS: tuple[Operation, ...] = ("ring", "graph", "quality")
TWO_PI = 2.0 * np.pi


def integer_labels(
    values: list[int], dtype: IntegerDtype, *, unaligned: bool = False
) -> IntegerArray:
    """Create signed or unsigned labels with an optional one-byte buffer offset.

    Parameters
    ----------
    values : list[int]
        Labels representable in the selected 64-bit integer domain.
    dtype : type[np.int64] or type[np.uint64]
        Signedness of the real NumPy observations.
    unaligned : bool
        Allocate with a one-byte offset to exercise direct buffer refusal.

    Returns
    -------
    IntegerArray
        One-dimensional observations containing the supplied labels.
    """
    result: IntegerArray
    if dtype is np.int64:
        result = (
            np.ndarray(
                (len(values),),
                dtype=np.int64,
                buffer=bytearray(8 * len(values) + 1),
                offset=1,
            )
            if unaligned
            else np.array(values, dtype=np.int64)
        )
    else:
        result = (
            np.ndarray(
                (len(values),),
                dtype=np.uint64,
                buffer=bytearray(8 * len(values) + 1),
                offset=1,
            )
            if unaligned
            else np.array(values, dtype=np.uint64)
        )
    if unaligned:
        result[:] = values
    return result


def extract_native(
    operation: Operation, signal: IntegerArray, n_states: int = 8
) -> NDArray[np.float64]:
    """Call the actual registered vector owner, retaining its typed float output.

    Parameters
    ----------
    operation : Operation
        Ring mapping, graph walk or linear transition quality.
    signal : IntegerArray
        Real observations passed unchanged to the installed extension.
    n_states : int
        Native vocabulary count; unlike the public extractor, zero is admitted.

    Returns
    -------
    NDArray[np.float64]
        The actual native phases or qualities in logical observation order.
    """
    if operation == "ring":
        result = spo_kernel.ring_phases_rust(signal, n_states)
    elif operation == "graph":
        result = spo_kernel.graph_walk_phases_rust(signal, n_states)
    else:
        result = spo_kernel.transition_qualities_rust(signal, n_states, 0.5)
    return np.asarray(result, dtype=np.float64)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "readonly"])
def test_vector_layouts_preserve_logical_labels(
    operation: Operation, dtype: IntegerDtype, layout: str
) -> None:
    """Logical label order and source bytes agree across real NumPy view layouts."""
    signal = integer_labels([0, 2, 5, 1], dtype)
    if layout == "strided":
        signal = integer_labels([0, 0, 2, 2, 5, 5, 1, 1], dtype)[::2]
    elif layout == "reversed":
        signal = signal[::-1].copy()[::-1]
    elif layout == "readonly":
        signal.setflags(write=False)
    original = signal.copy()
    actual = extract_native(operation, signal)
    expected = {
        "ring": [0.0, np.pi / 2, 5 * np.pi / 4, np.pi / 4],
        "graph": [0.0, 4 * np.pi / 9, 10 * np.pi / 9, 0.0],
        "quality": [0.5, 0.875, 0.75, 0.625],
    }[operation]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(signal, original)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_zero_stride_stalls_and_empty_arrays(
    operation: Operation, dtype: IntegerDtype
) -> None:
    """Readonly broadcast strides retain stalls; empty views fabricate no result."""
    signal: IntegerArray
    if dtype is np.int64:
        signal = np.broadcast_to(np.array([3], dtype=np.int64), (4,))
    else:
        signal = np.broadcast_to(np.array([3], dtype=np.uint64), (4,))
    actual = extract_native(operation, signal)
    expected = {
        "ring": [3 * np.pi / 4] * 4,
        "graph": [0.0] * 4,
        "quality": [0.5, 0.2, 0.2, 0.2],
    }[operation]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_array_equal(signal, [3, 3, 3, 3])
    empty = extract_native(operation, integer_labels([], dtype))
    assert empty.shape == (0,)


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_unaligned_direct_arrays_refuse_before_element_views(
    operation: Operation, dtype: IntegerDtype
) -> None:
    """Direct FFI refuses unaligned buffers and leaves their recorded labels intact."""
    signal = integer_labels([0, 1, 0], dtype, unaligned=True)
    original = signal.copy()
    with pytest.raises(ValueError, match="state_indices must be aligned"):
        extract_native(operation, signal)
    np.testing.assert_array_equal(signal, original)
    np.testing.assert_allclose(
        extract_native(operation, original),
        {
            "ring": [0.0, np.pi / 4, 0.0],
            "graph": [0.0, np.pi, 0.0],
            "quality": [0.5, 1.0, 1.0],
        }[operation],
        atol=1e-12,
    )


@pytest.mark.parametrize("operation", OPERATIONS)
@pytest.mark.parametrize(
    "signal",
    [
        np.array([0.0, 1.0]),
        np.array([True, False]),
        np.array([0, 1], dtype=np.int32),
        np.array([0, 1], dtype=np.dtype(np.int64).newbyteorder("S")),
        np.array([0, 1], dtype=np.dtype(np.uint64).newbyteorder("S")),
        np.array([[0, 1]], dtype=np.int64),
        [0, 1],
    ],
)
def test_direct_array_type_and_rank_refusal(
    operation: Operation, signal: object
) -> None:
    """Direct 64-bit one-dimensional entry points refuse other array contracts."""
    with pytest.raises(TypeError, match="one-dimensional int64 or uint64"):
        extract_native(operation, cast(IntegerArray, signal))


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_full_span_graph_totals_and_quality(dtype: IntegerDtype) -> None:
    """Both integer domains retain a midpoint after sums exceed uint64."""
    limits = np.iinfo(dtype)
    signal = integer_labels([int(limits.min), int(limits.max), int(limits.min)], dtype)
    np.testing.assert_allclose(
        extract_native("graph", signal, 4), [0.0, np.pi, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        extract_native("quality", signal, 4), [0.5, 0.1, 0.1], atol=1e-12
    )
    singleton = integer_labels([int(limits.max)], dtype)
    expected = TWO_PI * (int(limits.max) % 4) / 4
    np.testing.assert_allclose(
        extract_native("graph", singleton, 4), [expected], atol=1e-12
    )


def test_scalar_registration_and_machine_width_vocabulary() -> None:
    """All scalar exports retain their laws; the vector ring handles usize maximum."""
    maximum = int(np.iinfo(np.uintp).max)
    assert spo_kernel.ring_phase(maximum, 4) == pytest.approx(3 * np.pi / 2)
    assert spo_kernel.graph_walk_phase(5, 10) == pytest.approx(np.pi)
    assert spo_kernel.graph_walk_phase(0, 0) == 0.0
    assert spo_kernel.transition_quality(0, 4) == 0.2
    assert spo_kernel.transition_quality(1, 0) == 1.0
    assert spo_kernel.transition_quality(2, 0) == 0.1
    for dtype in (np.int64, np.uint64):
        actual = extract_native("ring", integer_labels([0, 1, 0], dtype), maximum)
        np.testing.assert_allclose(
            actual, [0.0, TWO_PI * (1 / maximum), 0.0], rtol=1e-15, atol=0.0
        )
    with pytest.raises(OverflowError):
        extract_native("graph", np.array([0, 1]), maximum + 1)


@pytest.mark.parametrize("initial_quality", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_initial_quality_refuses(initial_quality: float) -> None:
    """Nonfinite quality cannot enter native transition output."""
    with pytest.raises(ValueError, match="initial_quality must be a finite float"):
        spo_kernel.transition_qualities_rust(np.array([0, 1]), 4, initial_quality)


@pytest.mark.parametrize("mode", ["ring", "graph"])
@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_public_extraction_executes_installed_native_functions(
    mode: str, dtype: IntegerDtype
) -> None:
    """Observe actual C calls without replacing the backend or toggling availability."""
    owners = {
        "ring_phases_rust": spo_kernel.ring_phases_rust,
        "graph_walk_phases_rust": spo_kernel.graph_walk_phases_rust,
        "transition_qualities_rust": spo_kernel.transition_qualities_rust,
    }
    executed: set[str] = set()

    def observe_call(frame: FrameType, event: str, argument: object) -> None:
        """Record C-call identities belonging to the installed production extension."""
        if event == "c_call":
            for name, owner in owners.items():
                if argument is owner:
                    executed.add(name)

    previous_profile = sys.getprofile()
    signal = integer_labels([0, 2, 5, 1], dtype)
    try:
        sys.setprofile(observe_call)
        states = SymbolicExtractor(n_states=8, mode=mode).extract(signal, 8.0)
    finally:
        sys.setprofile(previous_profile)
    expected_calls = (
        {"ring_phases_rust"}
        if mode == "ring"
        else {"graph_walk_phases_rust", "transition_qualities_rust"}
    )
    assert executed == expected_calls
    expected = (
        [0.0, np.pi / 2, 5 * np.pi / 4, np.pi / 4]
        if mode == "ring"
        else [0.0, 4 * np.pi / 9, 10 * np.pi / 9, 0.0]
    )
    np.testing.assert_allclose([state.theta for state in states], expected, atol=1e-12)
