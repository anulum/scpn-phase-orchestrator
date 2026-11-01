# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare defensive contract tests

"""Defensive contract coverage for the monitor Poincare API."""

from __future__ import annotations

from collections.abc import Callable
from typing import NoReturn, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import poincare as poincare_module
from scpn_phase_orchestrator.monitor.poincare import (
    PoincareResult,
    poincare_section,
)
from tests._runtime_owners import NEEDS_KERNEL

FloatArray = NDArray[np.float64]
SectionBackend = Callable[
    [FloatArray, int, int, FloatArray, float, int],
    tuple[FloatArray, FloatArray, int],
]
PhaseBackend = Callable[
    [FloatArray, int, int, int, float],
    tuple[FloatArray, FloatArray, int],
]


class _ExplodingArray:
    """Array-like value whose conversion always fails."""

    def __array__(
        self,
        dtype: object | None = None,
        copy: object | None = None,
    ) -> NoReturn:
        """Raise during NumPy coercion to exercise defensive fallback paths."""
        raise ValueError("cannot materialize array")


def _base_result_payload() -> dict[str, object]:
    """Return a valid public result payload for mutation-based tests."""
    return {
        "crossings": [[0.0], [1.0]],
        "crossing_times": [0.5, 2.5],
        "return_times": [2.0],
        "mean_return_time": 2.0,
        "std_return_time": 0.0,
    }


def test_alias_helpers_fail_closed_when_array_protocol_raises() -> None:
    """Array-protocol failures must be treated as non-alias payloads."""
    value = _ExplodingArray()

    assert poincare_module._contains_boolean_alias(value) is False
    assert poincare_module._contains_complex_alias(value) is False
    assert poincare_module._has_complex_payload(value) is False
    assert poincare_module._is_numeric_string_alias(1.0) is False
    assert poincare_module._is_numeric_string_alias("not-a-number") is False
    assert poincare_module._contains_numeric_string_alias(value) is False
    assert (
        poincare_module._contains_numeric_string_alias(
            np.array([1.0, "not-a-number"], dtype=object)
        )
        is False
    )
    assert (
        poincare_module._contains_numeric_string_alias(
            np.array([1.0, "2.0"], dtype=object)
        )
        is True
    )


@NEEDS_KERNEL
def test_rust_backend_loader_wraps_and_pads_kernel_outputs() -> None:
    """The compiled Rust loader pads crossings to the trajectory size."""
    if "rust" not in poincare_module.AVAILABLE_BACKENDS:
        pytest.skip("rust unavailable")
    loaded = poincare_module._load_rust_fns()
    section_fn_raw = loaded["section"]
    assert callable(section_fn_raw)
    section_fn: SectionBackend = section_fn_raw
    phase_fn_raw = loaded["phase"]
    assert callable(phase_fn_raw)
    phase_fn: PhaseBackend = phase_fn_raw

    # One downward crossing of x = 0.25 between the two samples, at 3/8.
    traj = np.array([1.0, 2.0, -1.0, 4.0], dtype=np.float64)
    crossings, times, count = section_fn(traj, 2, 2, np.array([1.0, 0.0]), 0.25, 1)
    assert count == 1
    assert crossings.shape == (4,) and times.shape == (2,)
    np.testing.assert_allclose(crossings, [0.25, 2.75, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(times, [0.375, 0.0], atol=1e-12)

    # Oscillator 1 passes 0.5 (mod 2π) once, a third of the way through.
    phases = np.array([0.0, 6.0, 0.3, 6.0 + 0.9], dtype=np.float64)
    crossings, times, count = phase_fn(phases, 2, 2, 1, 0.5)
    alpha = (2 * np.pi - (6.0 - 0.5)) / ((2 * np.pi - 5.5) + (6.9 - 2 * np.pi - 0.5))
    assert count == 1
    assert crossings.shape == (4,) and times.shape == (2,)
    np.testing.assert_allclose(
        crossings, [0.3 * alpha, 6.0 + 0.9 * alpha, 0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(times, [alpha, 0.0], atol=1e-12)


def test_julia_backend_loader_returns_poincare_callables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Julia backend loader must expose both public dispatch callables."""
    monkeypatch.setattr(poincare_module, "require_juliacall_main", lambda: None)

    loaded = poincare_module._load_julia_fns()

    assert callable(loaded["section"])
    assert callable(loaded["phase"])


def test_dispatch_returns_none_when_backend_lacks_requested_callable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dispatch must fall back when a backend loads but omits the callable."""
    previous_backend = poincare_module.ACTIVE_BACKEND
    previous_available = list(poincare_module.AVAILABLE_BACKENDS)
    previous_loader = poincare_module._LOADERS["go"]
    poincare_module.ACTIVE_BACKEND = "go"
    poincare_module.AVAILABLE_BACKENDS = ["go"]
    poincare_module._BACKEND_CACHE.clear()

    try:
        monkeypatch.setitem(poincare_module._LOADERS, "go", lambda: {})
        assert poincare_module._dispatch("section") == ("python", None)
    finally:
        poincare_module.ACTIVE_BACKEND = previous_backend
        poincare_module.AVAILABLE_BACKENDS = previous_available
        monkeypatch.setitem(poincare_module._LOADERS, "go", previous_loader)
        poincare_module._BACKEND_CACHE.clear()


def test_public_section_rejects_three_dimensional_history() -> None:
    """The public section API must reject rank-3 trajectories."""
    with pytest.raises(ValueError, match="trajectory must be 1D or 2D"):
        poincare_section(np.zeros((2, 2, 2)), normal=[1.0, 0.0])


def test_public_section_rejects_numeric_string_backend_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public backend replay must reject numeric-string crossing payloads."""

    def _numeric_string_section(
        _traj_flat: FloatArray,
        _t: int,
        _d: int,
        _normal: FloatArray,
        _offset: float,
        _direction_id: int,
    ) -> tuple[FloatArray, FloatArray, int]:
        return (
            cast(FloatArray, np.array(["0.0", "1.0"], dtype=object)),
            cast(FloatArray, np.array(["0.5", "0.0"], dtype=object)),
            1,
        )

    monkeypatch.setattr(
        poincare_module,
        "_dispatch",
        lambda _fn_name, _backend="auto": ("go", _numeric_string_section),
    )

    with pytest.raises(ValueError, match="numeric-string"):
        poincare_section([[-1.0], [1.0]], normal=[1.0], offset=0.0)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"crossings": [["bad"], ["payload"]]}, "crossings must be"),
        (
            {"crossing_times": [True, False]},
            "crossing_times must not contain boolean values",
        ),
        ({"crossing_times": ["bad", "payload"]}, "crossing_times must be"),
        (
            {"crossing_times": [[0.5, 2.5]]},
            "crossing_times must be one-dimensional",
        ),
        ({"return_times": [True]}, "return_times must not contain boolean values"),
        ({"return_times": ["bad"]}, "return_times must be"),
        ({"return_times": [[2.0]]}, "return_times must be one-dimensional"),
        (
            {"return_times": [np.inf]},
            "return_times must contain only finite values",
        ),
    ],
)
def test_public_result_record_rejects_defensive_payload_variants(
    override: dict[str, object],
    match: str,
) -> None:
    """Public result validation must reject malformed scalar arrays."""
    payload = _base_result_payload()
    payload.update(override)
    # A widened view lets the deliberately malformed fields reach runtime validation.
    unchecked_result: Callable[..., PoincareResult] = PoincareResult

    with pytest.raises(ValueError, match=match):
        unchecked_result(**payload)


def test_result_assembly_rejects_nonpositive_dimension() -> None:
    """Backend result assembly must reject impossible crossing dimensions."""
    with pytest.raises(ValueError, match="dim must be positive"):
        poincare_module._assemble_result(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0,
            0,
        )
