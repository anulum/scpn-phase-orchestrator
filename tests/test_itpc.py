# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — ITPC tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.monitor.itpc import compute_itpc, itpc_persistence


def test_perfect_alignment_gives_itpc_one():
    phases = np.zeros((20, 50))
    itpc = compute_itpc(phases)
    np.testing.assert_allclose(itpc, 1.0, atol=1e-12)


def test_uniform_random_phases_give_low_itpc():
    rng = np.random.default_rng(42)
    phases = rng.uniform(0, 2 * np.pi, size=(100, 200))
    itpc = compute_itpc(phases)
    assert np.all(itpc < 0.3)


def test_single_trial_gives_ones():
    phases = np.array([[0.1, 0.5, 1.0]])
    itpc = compute_itpc(phases)
    np.testing.assert_allclose(itpc, 1.0, atol=1e-12)


def test_two_opposite_phases_give_zero():
    phases = np.array([[0.0, 0.0], [np.pi, np.pi]])
    itpc = compute_itpc(phases)
    np.testing.assert_allclose(itpc, 0.0, atol=1e-12)


def test_itpc_shape_matches_timepoints():
    phases = np.zeros((5, 30))
    itpc = compute_itpc(phases)
    assert itpc.shape == (30,)


def test_1d_input_returns_scalar_one():
    itpc = compute_itpc(np.array([0.5, 1.0, 1.5]))
    assert itpc.shape == (1,)
    np.testing.assert_allclose(itpc[0], 1.0, atol=1e-12)


def test_empty_input():
    itpc = compute_itpc(np.zeros((0, 10)))
    assert itpc.size == 0


def test_persistence_high_after_entrained_pause():
    """After entrainment, ITPC stays high even during pause indices."""
    rng = np.random.default_rng(7)
    n_trials, n_time = 50, 100
    # All trials locked to same phase + small noise
    phases = np.tile(np.linspace(0, 4 * np.pi, n_time), (n_trials, 1))
    phases += rng.normal(0, 0.05, phases.shape)
    pause_idx = list(range(80, 100))
    val = itpc_persistence(phases, pause_idx)
    assert val > 0.9


def test_persistence_low_for_random_phases():
    rng = np.random.default_rng(99)
    phases = rng.uniform(0, 2 * np.pi, size=(80, 120))
    pause_idx = list(range(100, 120))
    val = itpc_persistence(phases, pause_idx)
    assert val < 0.3


def test_persistence_empty_pause_returns_zero():
    phases = np.zeros((10, 20))
    assert itpc_persistence(phases, []) == 0.0


def test_persistence_out_of_bounds_indices_ignored():
    phases = np.zeros((10, 20))
    val = itpc_persistence(phases, [100, 200])
    assert val == 0.0


def test_itpc_values_bounded_zero_one():
    rng = np.random.default_rng(123)
    phases = rng.uniform(-np.pi, np.pi, size=(30, 50))
    itpc = compute_itpc(phases)
    assert np.all(itpc >= 0.0)
    assert np.all(itpc <= 1.0 + 1e-12)


class TestITPCPipelineWiring:
    """Pipeline: engine multi-trial phases → ITPC → trial consistency."""

    def test_engine_trials_to_itpc(self):
        """Multiple engine trials → compute_itpc → ITPC∈[0,1].
        Measures inter-trial phase consistency from engine runs."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine

        n_trials, n_time = 10, 50
        n_osc = 4
        eng = UPDEEngine(n_osc, dt=0.01)
        rng = np.random.default_rng(0)
        omegas = np.ones(n_osc)
        knm = 0.5 * np.ones((n_osc, n_osc))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n_osc, n_osc))

        trial_phases = np.zeros((n_trials, n_time))
        for trial in range(n_trials):
            p = rng.uniform(0, 2 * np.pi, n_osc)
            for t in range(n_time):
                p = eng.step(p, omegas, knm, 0.0, 0.0, alpha)
                trial_phases[trial, t] = p[0]

        itpc = compute_itpc(trial_phases)
        assert itpc.shape == (n_time,)
        assert np.all(itpc >= 0.0)
        assert np.all(itpc <= 1.0 + 1e-10)
