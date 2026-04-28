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

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import poincare as p_mod
from scpn_phase_orchestrator.monitor.poincare import (
    AVAILABLE_BACKENDS,
    phase_poincare,
    poincare_section,
)


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
