# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Poincare defensive contract tests

"""Defensive contract coverage for the monitor Poincare API."""

from __future__ import annotations

import sys
from collections.abc import Callable
from types import ModuleType
from typing import NoReturn, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor import poincare as poincare_module
from scpn_phase_orchestrator.monitor.poincare import (
    PoincareResult,
    poincare_section,
)

FloatArray = NDArray[np.float64]
SectionBackend = Callable[
    [FloatArray, int, int, FloatArray, float, int],
    tuple[FloatArray, FloatArray, int],
]
PhaseBackend = Callable[
    [FloatArray, int, int, int, float],
    tuple[FloatArray, FloatArray, int],
]
RustSectionKernel = Callable[
    [FloatArray, int, int, FloatArray, float, str],
    tuple[FloatArray, FloatArray, int],
]
RustPhaseKernel = Callable[
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


class _FakeSpoKernel(ModuleType):
    """Typed synthetic ``spo_kernel`` module for Rust adapter tests."""

    poincare_section_rust: RustSectionKernel
    phase_poincare_rust: RustPhaseKernel


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


def test_rust_backend_loader_wraps_and_pads_kernel_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rust backend adapters must preserve output cardinality and padding."""
    kernel = _FakeSpoKernel("spo_kernel")

    def poincare_section_rust(
        traj_flat: FloatArray,
        t: int,
        d: int,
        normal: FloatArray,
        offset: float,
        direction: str,
    ) -> tuple[FloatArray, FloatArray, int]:
        """Return one compact section crossing from a fake Rust kernel."""
        assert traj_flat.shape == (4,)
        assert (t, d, offset, direction) == (2, 2, 0.25, "negative")
        np.testing.assert_allclose(normal, [1.0, 0.0])
        return (
            np.array([3.0, 5.0], dtype=np.float64),
            np.array([0.75], dtype=np.float64),
            1,
        )

    def phase_poincare_rust(
        phases_flat: FloatArray,
        t: int,
        n: int,
        oscillator_idx: int,
        section_phase: float,
    ) -> tuple[FloatArray, FloatArray, int]:
        """Return one compact phase crossing from a fake Rust kernel."""
        assert phases_flat.shape == (4,)
        assert (t, n, oscillator_idx, section_phase) == (2, 2, 1, 0.5)
        return (
            np.array([7.0, 11.0], dtype=np.float64),
            np.array([1.25], dtype=np.float64),
            1,
        )

    kernel.poincare_section_rust = poincare_section_rust
    kernel.phase_poincare_rust = phase_poincare_rust
    monkeypatch.setitem(sys.modules, "spo_kernel", kernel)

    loaded = poincare_module._load_rust_fns()
    section_fn = cast(SectionBackend, loaded["section"])
    phase_fn = cast(PhaseBackend, loaded["phase"])

    section_crossings, section_times, section_count = section_fn(
        np.array([-1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        2,
        2,
        np.array([1.0, 0.0], dtype=np.float64),
        0.25,
        1,
    )
    phase_crossings, phase_times, phase_count = phase_fn(
        np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
        2,
        2,
        1,
        0.5,
    )

    np.testing.assert_allclose(section_crossings, [3.0, 5.0, 0.0, 0.0])
    np.testing.assert_allclose(section_times, [0.75, 0.0])
    assert section_count == 1
    np.testing.assert_allclose(phase_crossings, [7.0, 11.0, 0.0, 0.0])
    np.testing.assert_allclose(phase_times, [1.25, 0.0])
    assert phase_count == 1


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
        assert poincare_module._dispatch("section") is None
    finally:
        poincare_module.ACTIVE_BACKEND = previous_backend
        poincare_module.AVAILABLE_BACKENDS = previous_available
        monkeypatch.setitem(poincare_module._LOADERS, "go", previous_loader)
        poincare_module._BACKEND_CACHE.clear()


def test_public_section_rejects_three_dimensional_history() -> None:
    """The public section API must reject rank-3 trajectories."""
    with pytest.raises(ValueError, match="trajectory must be 1D or 2D"):
        poincare_section(np.zeros((2, 2, 2)), normal=[1.0, 0.0])


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

    with pytest.raises(ValueError, match=match):
        PoincareResult(**payload)


def test_result_assembly_rejects_nonpositive_dimension() -> None:
    """Backend result assembly must reject impossible crossing dimensions."""
    with pytest.raises(ValueError, match="dim must be positive"):
        poincare_module._assemble_result(
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            0,
            0,
        )
