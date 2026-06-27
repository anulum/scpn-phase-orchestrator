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
from types import SimpleNamespace
from typing import NoReturn, get_type_hints

import numpy as np
import pytest

from scpn_phase_orchestrator.experimental.accelerators.monitor import (
    _poincare_mojo as poincare_mojo_mod,
)
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


class _ArrayProtocolFailure:
    def __array__(
        self,
        dtype: object | None = None,
        copy: object | None = None,
    ) -> NoReturn:
        raise ValueError("synthetic array conversion failure")


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

    def test_validation_alias_helpers_fail_closed_on_array_protocol_failure(
        self,
    ) -> None:
        value = _ArrayProtocolFailure()

        assert poincare_validation._contains_boolean_alias(value) is False
        assert poincare_validation._contains_complex_alias(value) is False

    def test_backend_output_contract_accepts_valid_payload(self) -> None:
        crossings = np.array([0.0, 1.0, 0.0, 0.0, 2.0, 0.0])
        times = np.array([0.5, 1.5, 0.0])

        valid_crossings, valid_times, valid_count = (
            poincare_validation.validate_poincare_backend_outputs(
                crossings,
                times,
                2,
                t=3,
                dim=2,
            )
        )

        assert valid_count == 2
        np.testing.assert_allclose(valid_crossings[:4], [0.0, 1.0, 0.0, 0.0])
        np.testing.assert_allclose(valid_times[:2], [0.5, 1.5])

    @pytest.mark.parametrize(
        ("crossings", "times", "n_cr", "match"),
        [
            (np.array([0.0, 1.0, 2.0]), np.zeros(3), 1, "t\\*dim"),
            (np.array([0.0, np.inf, 0.0, 0.0, 0.0, 0.0]), np.zeros(3), 1, "finite"),
            (
                np.array([0.0, True, 0.0, 0.0, 0.0, 0.0], dtype=object),
                np.zeros(3),
                1,
                "crossings_flat",
            ),
            (np.zeros(6), np.array([0.0, np.nan, 0.0]), 1, "finite"),
            (np.zeros(6), np.array([0.0, True, 0.0], dtype=object), 1, "times"),
            (np.zeros(6), np.zeros(2), 1, "times length"),
            (np.zeros(6), np.zeros(3), True, "n_cr"),
            (np.zeros(6), np.zeros(3), 3, "available intervals"),
            (np.zeros(6), np.array([-1.0e-3, 0.0, 0.0]), 1, "sampled intervals"),
            (np.zeros(6), np.array([1.5, 0.5, 0.0]), 2, "strictly increasing"),
        ],
    )
    def test_backend_output_contract_rejects_invalid_payloads(
        self,
        crossings: np.ndarray,
        times: np.ndarray,
        n_cr: object,
        match: str,
    ) -> None:
        with pytest.raises(ValueError, match=match):
            poincare_validation.validate_poincare_backend_outputs(
                crossings,
                times,
                n_cr,
                t=3,
                dim=2,
            )

    @pytest.mark.parametrize(
        ("stdout", "match"),
        [
            ("", "missing crossing count header"),
            ("\n0\n", "crossing count must be an integer"),
            ("0\n\n", "returned 2 lines, expected 1"),
            ("1\n0.0\n1.0\n0.5\n\n", "returned 5 lines, expected 4"),
            (
                "3\n0.0\n0.0\n0.0\n0.0\n0.0\n0.0\n0.5\n1.5\n2.5\n",
                "must not exceed the 2 available intervals",
            ),
        ],
    )
    def test_mojo_runner_preserves_raw_stdout_cardinality(
        self,
        monkeypatch: pytest.MonkeyPatch,
        stdout: str,
        match: str,
    ) -> None:
        monkeypatch.setattr(poincare_mojo_mod, "_ensure_exe", lambda: "poincare_mojo")
        monkeypatch.setattr(
            poincare_mojo_mod.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(
                returncode=0,
                stdout=stdout,
                stderr="",
            ),
        )

        with pytest.raises(ValueError, match=match):
            poincare_mojo_mod._parse(poincare_mojo_mod._run("PHASE\n"), dim=2, t=3)

    @pytest.mark.parametrize(
        "backend",
        [poincare_section_go, poincare_section_julia, poincare_section_mojo],
    )
    @pytest.mark.parametrize(
        ("traj_flat", "t", "d", "normal", "offset", "direction_id", "match"),
        [
            (np.array([True, False]), 2, 1, np.array([1.0]), 0.0, 0, "traj_flat"),
            (
                np.array([0.0, 1.0j], dtype=object),
                2,
                1,
                np.array([1.0]),
                0.0,
                0,
                "traj_flat",
            ),
            (
                np.array(["bad", "1.0"], dtype=object),
                2,
                1,
                np.array([1.0]),
                0.0,
                0,
                "finite one-dimensional",
            ),
            (
                np.array([[0.0], [1.0]]),
                2,
                1,
                np.array([1.0]),
                0.0,
                0,
                "one-dimensional",
            ),
            (np.array([0.0, np.nan]), 2, 1, np.array([1.0]), 0.0, 0, "traj_flat"),
            (np.array([0.0, 1.0]), True, 1, np.array([1.0]), 0.0, 0, "t"),
            (np.array([0.0, 1.0]), 2, 2, np.array([1.0, 0.0]), 0.0, 0, "t\\*d"),
            (np.array([0.0, 1.0]), 2, 1, np.array([True]), 0.0, 0, "normal"),
            (
                np.array([0.0, 1.0]),
                2,
                1,
                np.array([1.0j], dtype=object),
                0.0,
                0,
                "normal",
            ),
            (np.array([0.0, 1.0]), 2, 1, np.array([1.0, 0.0]), 0.0, 0, "normal"),
            (np.array([0.0, 1.0]), 2, 1, np.array([1.0]), "origin", 0, "offset"),
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
            (np.array([0.0, 1.0j], dtype=object), 2, 1, 0, 0.0, "phases_flat"),
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
