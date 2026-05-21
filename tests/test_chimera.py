# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor import chimera as chimera_module
from scpn_phase_orchestrator.monitor.chimera import (
    ChimeraState,
    detect_chimera,
    local_order_parameter,
)


def _uniform_knm(n: int, strength: float = 1.0) -> np.ndarray:
    knm = np.full((n, n), strength)
    np.fill_diagonal(knm, 0.0)
    return knm


def test_fully_synchronised_all_coherent():
    phases = np.zeros(20)
    knm = _uniform_knm(20)
    state = detect_chimera(phases, knm)
    assert len(state.coherent_indices) == 20
    assert len(state.incoherent_indices) == 0
    assert state.chimera_index == 0.0


def test_uniform_random_phases_mostly_incoherent():
    rng = np.random.default_rng(7)
    phases = rng.uniform(0, 2 * np.pi, 100)
    knm = _uniform_knm(100)
    state = detect_chimera(phases, knm)
    assert len(state.incoherent_indices) > 50


def test_chimera_state_has_mixed_groups():
    """Half synchronised, half scattered — the hallmark chimera.

    Uses block-diagonal coupling so each group's R_i is computed only
    from its own neighbors, preventing dilution.
    """
    n = 40
    half = n // 2
    phases = np.zeros(n)
    rng = np.random.default_rng(42)
    phases[half:] = rng.uniform(0, 2 * np.pi, half)
    # Block coupling: each half only couples within itself
    knm = np.zeros((n, n))
    knm[:half, :half] = 1.0
    knm[half:, half:] = 1.0
    np.fill_diagonal(knm, 0.0)
    state = detect_chimera(phases, knm)
    assert len(state.coherent_indices) > 0
    assert len(state.incoherent_indices) > 0


def test_chimera_index_between_zero_and_one():
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, 2 * np.pi, 50)
    knm = _uniform_knm(50)
    state = detect_chimera(phases, knm)
    assert 0.0 <= state.chimera_index <= 1.0


def test_empty_phases_returns_empty_state():
    state = detect_chimera(np.array([]), np.zeros((0, 0)))
    assert state == ChimeraState()


def test_two_oscillators_in_phase():
    phases = np.array([0.0, 0.0])
    knm = np.array([[0.0, 1.0], [1.0, 0.0]])
    state = detect_chimera(phases, knm)
    assert len(state.coherent_indices) == 2


def test_no_coupling_gives_zero_r_local():
    """With zero coupling, no oscillator has neighbors — all R_i = 0."""
    phases = np.zeros(10)
    knm = np.zeros((10, 10))
    state = detect_chimera(phases, knm)
    # R_i = 0 < 0.3 → all incoherent
    assert len(state.incoherent_indices) == 10
    assert len(state.coherent_indices) == 0


def test_dataclass_fields():
    state = ChimeraState(
        coherent_indices=[0, 1],
        incoherent_indices=[3],
        chimera_index=0.25,
    )
    assert state.coherent_indices == [0, 1]
    assert state.incoherent_indices == [3]
    assert state.chimera_index == 0.25


def test_local_order_parameter_uses_backend_when_available(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[np.ndarray, np.ndarray, int]] = []

    def _fake_backend(phases: np.ndarray, knm_flat: np.ndarray, n: int) -> np.ndarray:
        calls.append((phases, knm_flat, n))
        return np.ones(n, dtype=np.float64)

    monkeypatch.setattr(chimera_module, "_dispatch", lambda: _fake_backend)
    phases = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    knm = _uniform_knm(3)
    local = local_order_parameter(phases, knm)
    np.testing.assert_allclose(local, np.ones(3), atol=1e-12)
    assert len(calls) == 1


def test_local_order_parameter_falls_back_when_backend_raises(
    monkeypatch: pytest.MonkeyPatch,
):
    def _raising_backend(
        _phases: np.ndarray, _knm_flat: np.ndarray, _n: int
    ) -> np.ndarray:
        raise RuntimeError("boom")

    monkeypatch.setattr(chimera_module, "_dispatch", lambda: _raising_backend)
    phases = np.array([0.0, 0.2, 0.4], dtype=np.float64)
    knm = _uniform_knm(3)
    local = local_order_parameter(phases, knm)
    assert local.shape == (3,)
    assert np.all(np.isfinite(local))
    assert np.all(local >= 0.0)
    assert np.all(local <= 1.0)


class TestChimeraPipelineWiring:
    """Pipeline: engine phases → detect_chimera → chimera_index."""

    def test_engine_phases_to_chimera_detection(self):
        """UPDEEngine → phases → detect_chimera: classifies coherent
        vs incoherent oscillators from engine output."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n = 16
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.normal(1.0, 0.5, n)
        knm = _uniform_knm(n, 0.3)
        alpha = np.zeros((n, n))
        for _ in range(200):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)

        state = detect_chimera(phases, _uniform_knm(n, 0.3))
        assert isinstance(state, ChimeraState)
        assert 0.0 <= state.chimera_index <= 1.0
        total = len(state.coherent_indices) + len(state.incoherent_indices)
        assert total == n
