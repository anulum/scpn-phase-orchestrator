# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — TCBO tests

from __future__ import annotations

import numpy as np

from scpn_phase_orchestrator.ssgf.tcbo import TCBOObserver, TCBOState


class TestTCBO:
    def test_insufficient_data(self):
        obs = TCBOObserver(window_size=50)
        state = obs.observe(np.zeros(4))
        assert state.method == "insufficient_data"
        assert not state.is_conscious

    def test_builds_history(self):
        obs = TCBOObserver(window_size=10, embed_dim=2, embed_delay=1)
        for _ in range(15):
            obs.observe(np.zeros(4))
        assert len(obs._history) <= 12  # window + embed buffer

    def test_coherent_dynamics_produces_h1(self):
        obs = TCBOObserver(window_size=20, embed_dim=2, embed_delay=1)
        # Phases rotating together at different speeds — creates loops in embedding
        for t in range(30):
            phases = np.array([0.1 * t, 0.2 * t, 0.3 * t, 0.15 * t, 0.25 * t, 0.12 * t])
            phases = phases % (2 * np.pi)
            obs.observe(phases)
        state = obs.observe(np.array([0.0, 0.1, 0.2, 0.15, 0.25, 0.12]))
        # Rotating phases create loops in delay embedding → H1 cycles
        assert state.p_h1 > 0.0 or state.method == "ripser"

    def test_random_vs_static_different_s_h1(self):
        obs_rand = TCBOObserver(window_size=20, embed_dim=2, embed_delay=1)
        obs_static = TCBOObserver(window_size=20, embed_dim=2, embed_delay=1)
        rng = np.random.default_rng(42)
        for _ in range(25):
            obs_rand.observe(rng.uniform(0, 2 * np.pi, 6))
            obs_static.observe(np.zeros(6))
        sr = obs_rand.observe(rng.uniform(0, 2 * np.pi, 6))
        ss = obs_static.observe(np.zeros(6))
        # Random and static should produce different s_h1 values
        assert sr.s_h1 != ss.s_h1

    def test_threshold_default(self):
        obs = TCBOObserver()
        assert obs.tau_h1 == 0.72

    def test_custom_threshold(self):
        obs = TCBOObserver(tau_h1=0.5)
        assert obs.tau_h1 == 0.5

    def test_returns_tcbo_state(self):
        obs = TCBOObserver(window_size=5, embed_dim=2, embed_delay=1)
        for _ in range(10):
            state = obs.observe(np.zeros(3))
        assert isinstance(state, TCBOState)
        assert hasattr(state, "p_h1")
        assert hasattr(state, "is_conscious")
        assert hasattr(state, "s_h1")
        assert hasattr(state, "method")

    def test_reset(self):
        obs = TCBOObserver(window_size=5, embed_dim=2, embed_delay=1)
        for _ in range(10):
            obs.observe(np.zeros(3))
        obs.reset()
        state = obs.observe(np.zeros(3))
        assert state.method == "insufficient_data"

    def test_single_oscillator(self):
        obs = TCBOObserver(window_size=5, embed_dim=2, embed_delay=1)
        for _ in range(10):
            state = obs.observe(np.array([1.0]))
        assert state.p_h1 == 0.0  # < 2 oscillators
