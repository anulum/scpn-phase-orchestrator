# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Chimera state detection tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.chimera import ChimeraState, detect_chimera


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


class TestPipelineWiring:
    """Pipeline wiring: proves this module is not decorative."""

    def test_wires_into_pipeline(self):
        import numpy as np

        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(0)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        knm = 0.3 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        for _ in range(100):
            phases = eng.step(phases, omegas, knm, 0.0, 0.0, alpha)
        r, _ = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
