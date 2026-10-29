# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actual dimension native ABI contracts

"""Qualify installed PyO3 and Go C boundaries with owned, bounded buffers."""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from importlib import import_module
from typing import Protocol, TypeAlias, cast

import numpy as np
import pytest
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
pytestmark = pytest.mark.native_runtime


def _load_lib() -> ctypes.CDLL:
    """Load the Go library only for Go cases, so Rust-only lanes need no package."""
    module = import_module(
        "scpn_phase_orchestrator.experimental.accelerators.monitor._dimension_go"
    )
    return cast(ctypes.CDLL, module._load_lib())


class NativeIntegral(Protocol):
    """The actual Python-callable dimension argument-binding boundary."""

    def __call__(
        self,
        trajectory: object,
        t: object,
        d: object,
        epsilons: object,
        max_pairs: object = 50000,
        seed: object = 42,
    ) -> FloatArray:
        """Count pairs or refuse an invalid raw Python request."""
        ...


class NativeSpectrum(Protocol):
    """The actual positional/keyword spectrum boundary."""

    def __call__(self, exponents: object) -> float:
        """Compute a dimension or refuse a malformed native spectrum."""
        ...


@pytest.fixture
def native_integral() -> NativeIntegral:
    """Resolve the installed extension without replacing a numerical callable."""
    return cast(NativeIntegral, import_module("spo_kernel").correlation_integral_rust)


@pytest.mark.parametrize("binding", ["positional", "keyword", "mixed"])
def test_rust_argument_binding_and_output_lifetime(
    native_integral: NativeIntegral, binding: str
) -> None:
    """Real positional/keyword binding returns independent strict-radius fractions."""
    trajectory = np.array([0.0, 1.0, 3.0])
    radii = np.array([1.0, 2.0, 4.0])
    if binding == "positional":
        actual = native_integral(trajectory, 3, 1, radii, 10, 42)
    elif binding == "keyword":
        actual = native_integral(
            trajectory=trajectory, t=3, d=1, epsilons=radii, max_pairs=10, seed=42
        )
    else:
        actual = native_integral(
            trajectory,
            np.int64(3),
            d=np.int64(1),
            epsilons=radii,
            max_pairs=np.int64(10),
        )
    np.testing.assert_allclose(actual, [0.0, 1.0 / 3.0, 1.0], rtol=0.0, atol=1e-15)
    assert actual.dtype == np.float64
    assert not np.shares_memory(actual, trajectory)
    assert not np.shares_memory(actual, radii)
    actual[:] = -1
    np.testing.assert_array_equal(trajectory, [0.0, 1.0, 3.0])
    np.testing.assert_array_equal(radii, [1.0, 2.0, 4.0])
    repeated = native_integral(trajectory, 3, 1, radii)
    del trajectory, radii
    np.testing.assert_allclose(repeated, [0.0, 1.0 / 3.0, 1.0], rtol=0.0, atol=1e-15)


@pytest.mark.parametrize("binding", ["missing", "extra", "duplicate", "unknown"])
def test_rust_refuses_malformed_argument_binding(
    native_integral: NativeIntegral, binding: str
) -> None:
    """Actual PyO3 binding rejects missing, excess and conflicting arguments."""
    args: list[object] = [np.array([0.0, 1.0]), 2, 1, np.array([2.0])]
    kwargs: dict[str, object] = {}
    if binding == "missing":
        args.pop()
    elif binding == "extra":
        args.extend([10, 42, 99])
    elif binding == "duplicate":
        kwargs["t"] = 2
    else:
        kwargs["unknown"] = 2
    with pytest.raises(TypeError):
        native_integral(*args, **kwargs)


@pytest.mark.parametrize(
    "position,value,error",
    [
        (0, np.array([0, 1], dtype=np.int64), TypeError),
        (0, np.array([0.0, 1.0], dtype=np.float32), TypeError),
        (0, np.array([0.0 + 0j, 1.0 + 0j]), TypeError),
        (0, np.array([[0.0, 1.0]]), TypeError),
        (0, np.array([0.0, 8.0, 1.0, 8.0])[::2], ValueError),
        (0, np.array([0.0, np.inf]), ValueError),
        (0, np.array([0.0, np.nan]), ValueError),
        (0, np.zeros(3), ValueError),
        (1, True, ValueError),
        (1, np.bool_(True), ValueError),
        (1, "2", ValueError),
        (1, 2.0, ValueError),
        (1, np.timedelta64(2, "s"), ValueError),
        (1, -1, OverflowError),
        (1, int(np.iinfo(np.uintp).max) + 1, OverflowError),
        (2, 0, ValueError),
        (2, False, ValueError),
        (2, int(np.iinfo(np.uintp).max), ValueError),
        (3, np.array([1], dtype=np.int64), TypeError),
        (3, np.array([0.5, 9.0, 2.0, 9.0])[::2], ValueError),
        (3, np.array([-1.0]), ValueError),
        (3, np.array([np.nan]), ValueError),
        (3, np.array([np.inf]), ValueError),
        (3, np.array([2.0, 1.0]), ValueError),
        (4, 0, ValueError),
        (4, False, ValueError),
        (4, -1, OverflowError),
        (5, True, ValueError),
        (5, -1, OverflowError),
        (5, 2**64, OverflowError),
    ],
)
def test_rust_refuses_invalid_arrays_and_metadata(
    native_integral: NativeIntegral,
    position: int,
    value: object,
    error: type[Exception],
) -> None:
    """Source dtype/count/cardinality checks precede native allocation and sampling."""
    args: list[object] = [np.array([0.0, 1.0]), 2, 1, np.array([0.5, 2.0]), 10, 42]
    args[position] = value
    with pytest.raises(error):
        native_integral(*args)


def test_rust_empty_singleton_and_maximal_metadata(
    native_integral: NativeIntegral,
) -> None:
    """Large valid native controls do not allocate the whole requested quota."""
    limit = int(np.iinfo(np.uintp).max)
    for samples in [np.empty(0), np.array([1.0])]:
        np.testing.assert_array_equal(
            native_integral(
                samples, samples.size, 1, np.array([1.0]), limit, 2**64 - 1
            ),
            [0.0],
        )
    np.testing.assert_array_equal(
        native_integral(np.array([0.0, 1.0]), 2, 1, np.array([2.0]), limit, 2**64 - 1),
        [1.0],
    )
    assert native_integral(np.array([0.0, 1.0]), 2, 1, np.empty(0)).size == 0


@pytest.mark.parametrize(
    "value,error",
    [
        (np.array([1, -2]), TypeError),
        (np.array([[1.0, -2.0]]), TypeError),
        (np.array([1.0, 9.0, -2.0, 9.0])[::2], ValueError),
        (np.array([np.nan]), ValueError),
        (np.array([np.inf]), ValueError),
    ],
)
def test_rust_kaplan_yorke_refuses_invalid_spectrum(
    value: object, error: type[Exception]
) -> None:
    """The raw spectrum boundary rejects wrong dtype, layout and nonfinite values."""
    native = cast(
        NativeSpectrum,
        import_module("spo_kernel").kaplan_yorke_dimension_rust,
    )
    with pytest.raises(error):
        native(value)


def test_rust_kaplan_yorke_input_is_preserved() -> None:
    """The actual native sort leaves caller order unchanged, including empty input."""
    native = cast(
        NativeSpectrum,
        import_module("spo_kernel").kaplan_yorke_dimension_rust,
    )
    values = np.array([-2.0, 1.0, 0.0])
    assert native(values) == 2.5
    assert native(exponents=values) == 2.5
    np.testing.assert_array_equal(values, [-2.0, 1.0, 0.0])
    assert native(np.empty(0)) == 0.0


class GoIntegral(Protocol):
    """Signed C metadata and caller-owned pointer capacities for the real Go ABI."""

    def __call__(
        self,
        trajectory: object,
        t: int,
        d: int,
        first: object,
        second: object,
        pairs: int,
        radii: object,
        count: int,
        out: object,
    ) -> int:
        """Return a status code and write output only after validating the request."""
        ...


def _double_pointer(values: FloatArray) -> object:
    """Keep NumPy-owned contiguous buffers alive while exposing their C address."""
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


@pytest.mark.parametrize(
    "case,expected",
    [
        ("valid", 0),
        ("negative-t", -1),
        ("zero-d", -1),
        ("negative-pairs", -1),
        ("negative-radii", -1),
        ("null-trajectory", -1),
        ("null-first", -1),
        ("null-second", -1),
        ("null-radii", -1),
        ("null-output", -1),
        ("nonfinite-trajectory", -2),
        ("negative-index", -3),
        ("outside-index", -3),
        ("self-pair", -3),
        ("negative-radius", -4),
        ("nonfinite-radius", -4),
        ("unsorted-radii", -4),
    ],
)
def test_go_c_abi_status_and_output_atomicity(case: str, expected: int) -> None:
    """Actual C calls use valid-capacity buffers or nulls, never invalid addresses."""
    trajectory = np.array([0.0, 1.0])
    first = np.array([0], dtype=np.int64)
    second = np.array([1], dtype=np.int64)
    radii = np.array([0.5, 2.0])
    output = np.full(2, -9.0)
    if case == "nonfinite-trajectory":
        trajectory[0] = np.nan
    if case == "negative-index":
        first[0] = -1
    if case == "outside-index":
        second[0] = 2
    if case == "self-pair":
        second[0] = 0
    if case == "negative-radius":
        radii[0] = -1
    if case == "nonfinite-radius":
        radii[0] = np.inf
    if case == "unsorted-radii":
        radii[:] = [2.0, 0.5]
    ci = cast(GoIntegral, _load_lib().CorrelationIntegral)
    actual = ci(
        None if case == "null-trajectory" else _double_pointer(trajectory),
        -1 if case == "negative-t" else 2,
        0 if case == "zero-d" else 1,
        None
        if case == "null-first"
        else first.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        None
        if case == "null-second"
        else second.ctypes.data_as(ctypes.POINTER(ctypes.c_longlong)),
        -1 if case == "negative-pairs" else 1,
        None if case == "null-radii" else _double_pointer(radii),
        -1 if case == "negative-radii" else 2,
        None if case == "null-output" else _double_pointer(output),
    )
    assert actual == expected
    np.testing.assert_array_equal(output, [0.0, 1.0] if expected == 0 else [-9.0, -9.0])


def test_go_c_abi_degenerate_null_buffers() -> None:
    """Zero-capacity arrays permit null pointers without concealing positive buffers."""
    ci = cast(GoIntegral, _load_lib().CorrelationIntegral)
    out = np.array([-9.0])
    radii = np.array([1.0])
    assert (
        ci(None, 0, 1, None, None, 0, _double_pointer(radii), 1, _double_pointer(out))
        == 0
    )
    np.testing.assert_array_equal(out, [0.0])
    assert ci(None, 0, 1, None, None, 0, None, 0, None) == 0


@pytest.mark.parametrize(
    "case,expected",
    [
        ("valid", 0),
        ("empty", 0),
        ("negative-count", -1),
        ("null-input", -1),
        ("null-output", -1),
        ("nonfinite", -2),
    ],
)
def test_go_kaplan_yorke_c_status(case: str, expected: int) -> None:
    """The actual Go scalar ABI leaves output unchanged on refused spectra."""
    function = cast(
        "Callable[[object, int, object], int]", _load_lib().KaplanYorkeDimension
    )
    values = np.array([-2.0, 1.0])
    if case == "nonfinite":
        values[0] = np.nan
    before = values.copy()
    out = ctypes.c_double(-9.0)
    count = -1 if case == "negative-count" else 0 if case == "empty" else 2
    status = function(
        None if case in {"empty", "null-input"} else _double_pointer(values),
        count,
        None if case == "null-output" else ctypes.byref(out),
    )
    assert status == expected
    expected_output = (0.0 if case == "empty" else 1.5) if status == 0 else -9.0
    assert out.value == expected_output
    np.testing.assert_array_equal(values, before)
