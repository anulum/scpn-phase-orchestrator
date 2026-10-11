# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Direct native Hodge input admission

"""Actual installed PyO3 Hodge cardinality, simplex and numerical contracts."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
NativeHodge = Callable[
    [object, object, object, object, object, object, object],
    tuple[FloatArray, FloatArray, FloatArray],
]
pytestmark = pytest.mark.native_runtime


def test_rust_hodge_empty_count_rejects_nonempty_coupling() -> None:
    """An actual empty native request cannot conceal a nonempty coupling array."""
    native = cast(NativeHodge, import_module("spo_kernel").hodge_decomposition_rust)
    with pytest.raises(ValueError, match="cardinality"):
        native(
            np.ones(1),
            np.empty(0),
            0,
            np.empty(0, dtype=np.int64),
            0,
            np.empty(0, dtype=np.int64),
            0,
        )


@pytest.fixture
def native_hodge() -> NativeHodge:
    """Resolve the original installed PyO3 callable without a substitute."""
    return cast(NativeHodge, import_module("spo_kernel").hodge_decomposition_rust)


@pytest.mark.parametrize(
    "position,value,error",
    [
        (1, np.ones(1), ValueError),
        (3, np.array([0, 1], dtype=np.int64), ValueError),
        (5, np.array([0, 1, 2], dtype=np.int64), ValueError),
        (2, False, ValueError),
        (4, False, ValueError),
        (6, False, ValueError),
        (2, np.bool_(False), ValueError),
        (2, "0", ValueError),
        (2, -1, OverflowError),
        (0, np.empty(0, dtype=np.bool_), TypeError),
        (1, np.empty(0, dtype=np.complex128), TypeError),
    ],
    ids=[
        "phases",
        "edges",
        "triangles",
        "boolean-n",
        "boolean-edge-count",
        "boolean-face-count",
        "numpy-boolean-n",
        "text-n",
        "negative-n",
        "boolean-coupling-dtype",
        "complex-phase-dtype",
    ],
)
def test_rust_hodge_refuses_inconsistent_empty_request(
    native_hodge: NativeHodge,
    position: int,
    value: object,
    error: type[Exception],
) -> None:
    """Empty arrays do not erase source dtype, count or cardinality contracts."""
    args: list[object] = [
        np.empty(0),
        np.empty(0),
        0,
        np.empty(0, dtype=np.int64),
        0,
        np.empty(0, dtype=np.int64),
        0,
    ]
    args[position] = value
    with pytest.raises(error):
        native_hodge(*args)


@pytest.mark.parametrize("position", [2, 4, 6], ids=["n", "edges", "triangles"])
def test_rust_hodge_refuses_overflowing_dimensions(
    native_hodge: NativeHodge, position: int
) -> None:
    """Declared dimensions are checked before multiplication or allocation."""
    args: list[object] = [
        np.empty(0),
        np.empty(0),
        0,
        np.empty(0, dtype=np.int64),
        0,
        np.empty(0, dtype=np.int64),
        0,
    ]
    args[position] = np.iinfo(np.uintp).max
    with pytest.raises(ValueError, match="overflows"):
        native_hodge(*args)


@pytest.mark.parametrize(
    "position", [0, 1, 3, 5], ids=["coupling", "phases", "edges", "triangles"]
)
def test_rust_hodge_refuses_noncontiguous_arrays(
    native_hodge: NativeHodge, position: int
) -> None:
    """Typed strided arrays are refused before they can be borrowed as flat slices."""
    args: list[object] = [
        (np.ones((3, 3)) - np.eye(3)).ravel(),
        np.array([0.0, 1.0, 2.3]),
        3,
        np.array([0, 1, 0, 2, 1, 2]),
        3,
        np.array([0, 1, 2]),
        1,
    ]
    args[position] = np.repeat(np.asarray(args[position]), 2)[::2]
    with pytest.raises(ValueError, match="contiguous"):
        native_hodge(*args)


@pytest.mark.parametrize(
    "edges,n_edges,triangles,n_triangles,match",
    [
        (np.array([0, 1, 0, 2, 1, 2]), 3, np.array([-1, 1, 2]), 1, "0 <= i"),
        (np.array([0, 1, 0, 2, 1, 2]), 3, np.array([1, 0, 2]), 1, "0 <= i"),
        (np.array([-1, 1]), 1, np.empty(0, dtype=np.int64), 0, "0 <= i"),
        (np.array([0, 3]), 1, np.empty(0, dtype=np.int64), 0, "0 <= i"),
        (np.array([1, 0]), 1, np.empty(0, dtype=np.int64), 0, "0 <= i"),
        (np.array([0, 0]), 1, np.empty(0, dtype=np.int64), 0, "0 <= i"),
        (np.array([0, 1, 0, 1]), 2, np.empty(0, dtype=np.int64), 0, "unique"),
        (np.array([0, 1, 0, 2, 1, 2]), 3, np.array([0, 2, 1]), 1, "0 <= i"),
        (np.array([0, 1, 0, 2, 1, 2]), 3, np.array([0, 1, 3]), 1, "0 <= i"),
        (np.array([0, 1, 0, 2, 1, 2]), 3, np.array([0, 1, 2, 0, 1, 2]), 2, "unique"),
        (np.array([0, 1, 1, 2]), 2, np.array([0, 1, 2]), 1, "missing edge"),
    ],
    ids=[
        "negative-face",
        "reversed-first-face-edge",
        "negative-edge",
        "outside-edge",
        "reversed-edge",
        "self-edge",
        "duplicate-edge",
        "reversed-face",
        "outside-face",
        "duplicate-face",
        "missing-face-edge",
    ],
)
def test_rust_hodge_refuses_invalid_simplices(
    native_hodge: NativeHodge,
    edges: NDArray[np.int64],
    n_edges: int,
    triangles: NDArray[np.int64],
    n_triangles: int,
    match: str,
) -> None:
    """Malformed topology is refused before the actual native projection starts."""
    with pytest.raises(ValueError, match=match):
        native_hodge(
            (np.ones((3, 3)) - np.eye(3)).ravel(),
            np.array([0.0, 1.0, 2.3]),
            3,
            edges,
            n_edges,
            triangles,
            n_triangles,
        )


@pytest.mark.parametrize("field", ["coupling", "phases"])
def test_rust_hodge_refuses_nonfinite_measurements(
    native_hodge: NativeHodge, field: str
) -> None:
    """NaN weights and infinite phases cannot reach a native solve."""
    coupling = (np.ones((3, 3)) - np.eye(3)).ravel()
    phases = np.array([0.0, 1.0, 2.3])
    if field == "coupling":
        coupling[1] = np.nan
    else:
        phases[1] = np.inf
    with pytest.raises(ValueError, match="finite"):
        native_hodge(
            coupling, phases, 3, np.array([0, 1, 0, 2, 1, 2]), 3, np.array([0, 1, 2]), 1
        )


def test_rust_hodge_refuses_nonfinite_computed_components(
    native_hodge: NativeHodge,
) -> None:
    """Finite input that overflows its current cannot publish nonfinite flows."""
    coupling = np.array([0.0, np.finfo(np.float64).max, np.finfo(np.float64).max, 0.0])
    with pytest.raises(ValueError, match="nonfinite components"):
        native_hodge(
            coupling,
            np.array([0.0, np.pi / 2.0]),
            2,
            np.array([0, 1]),
            1,
            np.empty(0, dtype=np.int64),
            0,
        )


def test_rust_hodge_empty_and_plain_numpy_counts_remain_valid(
    native_hodge: NativeHodge,
) -> None:
    """A genuine empty native request retains three typed empty flow vectors."""
    result = native_hodge(
        np.empty(0),
        np.empty(0),
        np.int64(0),
        np.empty(0, dtype=np.int64),
        np.int64(0),
        np.empty(0, dtype=np.int64),
        np.int64(0),
    )
    for component in result:
        assert component.shape == (0,)
        assert component.dtype == np.float64


def test_rust_hodge_triangle_matches_independent_circulation(
    native_hodge: NativeHodge,
) -> None:
    """A filled equilateral phase cycle is pure curl in the real PyO3 kernel."""
    coupling = (np.ones((3, 3)) - np.eye(3)).ravel()
    phases = np.arange(3, dtype=np.float64) * (2.0 * np.pi / 3.0)
    before = coupling.copy(), phases.copy()
    gradient, curl, harmonic = native_hodge(
        coupling, phases, 3, np.array([0, 1, 0, 2, 1, 2]), 3, np.array([0, 1, 2]), 1
    )
    expected = (
        np.sqrt(3.0)
        / 2.0
        * np.array([[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    )
    np.testing.assert_allclose(gradient.reshape(3, 3), 0.0, atol=1e-10)
    np.testing.assert_allclose(curl.reshape(3, 3), expected, atol=1e-10)
    np.testing.assert_allclose(harmonic.reshape(3, 3), 0.0, atol=1e-10)
    np.testing.assert_array_equal(coupling, before[0])
    np.testing.assert_array_equal(phases, before[1])
