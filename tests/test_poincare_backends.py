# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Per-backend parity for Poincaré kernels

"""Cross-backend parity for :func:`poincare_section` and
:func:`phase_poincare`. All backends must produce the same crossing
count, crossing coordinates (to 1e-9), and times."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _poincare_validation as poincare_validation,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_go import (
    phase_poincare_go,
    poincare_section_go,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_julia import (
    phase_poincare_julia,
    poincare_section_julia,
)
from scpn_phase_orchestrator.experimental.accelerators.monitor._poincare_mojo import (
    phase_poincare_mojo,
    poincare_section_mojo,
)
from scpn_phase_orchestrator.monitor import poincare as p_mod
from scpn_phase_orchestrator.monitor.poincare import (
    AVAILABLE_BACKENDS,
    phase_poincare,
    poincare_section,
)
from tests.typing_contracts import assert_precise_ndarray_hint

SectionBackend = Callable[
    [np.ndarray, object, object, np.ndarray, object, object],
    tuple[np.ndarray, np.ndarray, int],
]
PhaseBackend = Callable[
    [np.ndarray, object, object, object, object],
    tuple[np.ndarray, np.ndarray, int],
]


def test__poincare_validation_helper_is_directly_linked_to_backend_tests() -> None:
    assert callable(poincare_validation.validate_poincare_section_backend_inputs)
    assert callable(poincare_validation.validate_phase_poincare_backend_inputs)



def _force(backend: str) -> str:
    prev = p_mod.ACTIVE_BACKEND
    p_mod.ACTIVE_BACKEND = backend
    return prev


def _reset(prev: str) -> None:
    p_mod.ACTIVE_BACKEND = prev


def _ref_section(traj, normal, offset, direction):
    prev = _force("python")
    try:
        return poincare_section(traj, normal, offset, direction)
    finally:
        _reset(prev)


def _ref_phase(phases, osc, sp):
    prev = _force("python")
    try:
        return phase_poincare(phases, osc, sp)
    finally:
        _reset(prev)


def _sinusoidal_traj(t: int = 200) -> np.ndarray:
    ts = np.linspace(0, 4 * math.pi, t)
    return np.column_stack([np.sin(ts), np.cos(ts), 0.3 * ts])


def _phase_traj(seed: int, t: int = 300, n: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = np.zeros((t, n))
    phases[0] = rng.uniform(0, 2 * math.pi, n)
    omegas = rng.normal(0.5, 0.2, n)
    for i in range(1, t):
        phases[i] = phases[i - 1] + omegas * 0.1
    return phases


def test_backend_array_contracts_are_parameterised() -> None:
    functions = (
        poincare_section_go,
        poincare_section_julia,
        poincare_section_mojo,
        phase_poincare_go,
        phase_poincare_julia,
        phase_poincare_mojo,
    )
    for fn in functions:
        hints = get_type_hints(fn)
        for key in hints:
            if key in {"traj_flat", "normal", "phases_flat", "return"}:
                assert_precise_ndarray_hint(hints[key])
                assert "float64" in str(hints[key])


class TestDirectBackendBoundaryContracts:
    """Direct optional Poincare backends validate before runtime loading."""

    @pytest.mark.parametrize(
        "backend",
        [poincare_section_go, poincare_section_julia, poincare_section_mojo],
    )
    @pytest.mark.parametrize(
        ("traj_flat", "t", "d", "normal", "offset", "direction_id", "match"),
        [
            (np.array([True, False]), 2, 1, np.array([1.0]), 0.0, 0, "traj_flat"),
            (np.array([0.0, np.nan]), 2, 1, np.array([1.0]), 0.0, 0, "traj_flat"),
            (np.array([0.0, 1.0]), True, 1, np.array([1.0]), 0.0, 0, "t"),
            (np.array([0.0, 1.0]), 2, 2, np.array([1.0, 0.0]), 0.0, 0, "t\\*d"),
            (np.array([0.0, 1.0]), 2, 1, np.array([True]), 0.0, 0, "normal"),
            (np.array([0.0, 1.0]), 2, 1, np.array([1.0, 0.0]), 0.0, 0, "normal"),
            (np.array([0.0, 1.0]), 2, 1, np.array([1.0]), math.inf, 0, "offset"),
            (np.array([0.0, 1.0]), 2, 1, np.array([1.0]), 0.0, 3, "direction"),
        ],
    )
    def test_section_validation_precedes_runtime_load(
        self,
        backend: SectionBackend,
        traj_flat: np.ndarray,
        t: object,
        d: object,
        normal: np.ndarray,
        offset: object,
        direction_id: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(traj_flat, t, d, normal, offset, direction_id)

    @pytest.mark.parametrize(
        "backend",
        [phase_poincare_go, phase_poincare_julia, phase_poincare_mojo],
    )
    @pytest.mark.parametrize(
        ("phases_flat", "t", "n", "oscillator_idx", "section_phase", "match"),
        [
            (np.array([True, False]), 2, 1, 0, 0.0, "phases_flat"),
            (np.array([0.0, np.inf]), 2, 1, 0, 0.0, "phases_flat"),
            (np.array([0.0, 1.0]), 0, 1, 0, 0.0, "t"),
            (np.array([0.0, 1.0]), 2, 2, 0, 0.0, "t\\*n"),
            (np.array([0.0, 1.0]), 2, 1, True, 0.0, "oscillator_idx"),
            (np.array([0.0, 1.0]), 2, 1, 1, 0.0, "oscillator_idx"),
            (np.array([0.0, 1.0]), 2, 1, 0, np.nan, "section_phase"),
        ],
    )
    def test_phase_validation_precedes_runtime_load(
        self,
        backend: PhaseBackend,
        phases_flat: np.ndarray,
        t: object,
        n: object,
        oscillator_idx: object,
        section_phase: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            backend(phases_flat, t, n, oscillator_idx, section_phase)


class TestPoincareSectionParity:
    @pytest.mark.parametrize("direction", ["positive", "negative", "both"])
    def test_all_backends_agree(self, direction: str) -> None:
        traj = _sinusoidal_traj()
        normal = np.array([1.0, 0.0, 0.0])
        ref = _ref_section(traj, normal, 0.0, direction)
        for backend in AVAILABLE_BACKENDS:
            if backend == "python":
                continue
            prev = _force(backend)
            try:
                got = poincare_section(traj, normal, 0.0, direction)
            finally:
                _reset(prev)
            assert len(got.crossings) == len(ref.crossings), (
                f"{backend} crossing count differs"
            )
            if len(ref.crossings) > 0:
                np.testing.assert_allclose(
                    got.crossings,
                    ref.crossings,
                    atol=1e-9,
                    err_msg=f"{backend} crossing coords diverged",
                )
                np.testing.assert_allclose(
                    got.crossing_times,
                    ref.crossing_times,
                    atol=1e-9,
                    err_msg=f"{backend} crossing times diverged",
                )


class TestPhasePoincareParity:
    @pytest.mark.parametrize("seed", [0, 42])
    def test_all_backends_agree(self, seed: int) -> None:
        phases = _phase_traj(seed)
        ref = _ref_phase(phases, 0, 0.0)
        for backend in AVAILABLE_BACKENDS:
            if backend == "python":
                continue
            prev = _force(backend)
            try:
                got = phase_poincare(phases, 0, 0.0)
            finally:
                _reset(prev)
            assert len(got.crossings) == len(ref.crossings), (
                f"{backend} phase crossing count differs"
            )
            if len(ref.crossings) > 0:
                np.testing.assert_allclose(
                    got.crossings,
                    ref.crossings,
                    atol=1e-9,
                    err_msg=f"{backend} phase crossings diverged",
                )
                np.testing.assert_allclose(
                    got.crossing_times,
                    ref.crossing_times,
                    atol=1e-9,
                    err_msg=f"{backend} phase times diverged",
                )
