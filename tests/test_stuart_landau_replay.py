# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Stuart-Landau replay contracts

"""Replay determinism contracts for Stuart-Landau amplitude state logs."""

from __future__ import annotations

import logging

import numpy as np
import pytest


class TestStuartLandauReplayDeterminism:
    """Verify that SL replay engine reproduces exact state trajectories,
    including amplitude fields, under both new and legacy log formats."""

    @pytest.fixture()
    def replay_engine(self, tmp_path):
        from scpn_phase_orchestrator.runtime.replay import ReplayEngine

        log = tmp_path / "audit.jsonl"
        log.write_text("")
        return ReplayEngine(log)

    @pytest.fixture()
    def sl_engine(self):
        from scpn_phase_orchestrator.upde.stuart_landau import StuartLandauEngine

        return StuartLandauEngine(n_oscillators=2, dt=0.01)

    def test_amplitude_replay_matches_forward_step(self, replay_engine, sl_engine):
        """Chained SL replay with amplitude fields must reproduce the engine's
        forward integration to machine precision."""
        n = 2
        omegas = [1.0, 2.0]
        mu = [0.5, 0.3]
        knm = np.array([[0.0, 0.1], [0.1, 0.0]])
        knm_r = np.array([[0.0, 0.05], [0.05, 0.0]])
        alpha = np.zeros((n, n))
        phases0 = [0.1, 0.8]
        amps0 = [0.7, 0.5]
        state0 = np.array(phases0 + amps0)

        nxt_state = sl_engine.step(
            state0,
            np.array(omegas),
            np.array(mu),
            knm,
            knm_r,
            0.0,
            0.0,
            alpha,
        )

        entries = [
            {
                "phases": phases0,
                "amplitudes": amps0,
                "omegas": omegas,
                "mu": mu,
                "knm": knm.ravel().tolist(),
                "knm_r": knm_r.ravel().tolist(),
                "alpha": alpha.ravel().tolist(),
            },
            {
                "phases": nxt_state[:n].tolist(),
                "amplitudes": nxt_state[n:].tolist(),
                "omegas": omegas,
                "mu": mu,
                "knm": knm.ravel().tolist(),
                "knm_r": knm_r.ravel().tolist(),
                "alpha": alpha.ravel().tolist(),
            },
        ]

        ok, verified = replay_engine.verify_determinism_sl_chained(sl_engine, entries)
        assert ok, "Replay must match forward integration exactly"
        assert verified == 1, "Exactly one transition should be verified"

        # Cross-validate: amplitudes must evolve toward sqrt(mu) under subcritical
        amps_next = nxt_state[n:]
        for i in range(n):
            equilibrium = np.sqrt(max(mu[i], 0.0))
            dist_before = abs(amps0[i] - equilibrium)
            dist_after = abs(amps_next[i] - equilibrium)
            assert dist_after <= dist_before + 1e-10, (
                f"Osc {i}: should converge toward "
                f"sqrt(mu)={equilibrium:.3f}, "
                f"dist {dist_before:.4f}→{dist_after:.4f}"
            )

    def test_missing_amplitude_fields_warns_and_skips(
        self,
        replay_engine,
        sl_engine,
        caplog,
    ):
        """Entries without amplitude or mu fields must emit a warning and
        skip verification — not silently pass or crash."""
        entries = [
            {
                "phases": [0.1, 0.2],
                "omegas": [1.0, 1.0],
                "knm": [0.0] * 4,
                "alpha": [0.0] * 4,
            },
            {
                "phases": [0.3, 0.4],
                "omegas": [1.0, 1.0],
                "knm": [0.0] * 4,
                "alpha": [0.0] * 4,
            },
        ]
        with caplog.at_level(logging.WARNING):
            ok, verified = replay_engine.verify_determinism_sl_chained(
                sl_engine,
                entries,
            )
        assert ok, "Should pass (nothing to fail on)"
        assert verified == 0, "No steps should be verified when fields are missing"
        assert any("missing" in msg.lower() for msg in caplog.messages), (
            "Must log a warning about missing amplitude fields"
        )

    def test_legacy_format_without_separate_amplitudes(
        self,
        replay_engine,
        sl_engine,
    ):
        """Legacy logs store full SL state [θ; r] in 'phases' with 'mu' present
        but no 'amplitudes' key. Verify replay handles this format correctly."""
        n = 2
        omegas = [1.0, 1.0]
        mu_val = [0.5, 0.5]
        phases0 = [0.1, 0.2]
        amps0 = [0.7, 0.8]
        state0 = np.array(phases0 + amps0)
        nxt_state = sl_engine.step(
            state0,
            np.array(omegas),
            np.array(mu_val),
            np.zeros((n, n)),
            np.zeros((n, n)),
            0.0,
            0.0,
            np.zeros((n, n)),
        )
        entries = [
            {
                "phases": phases0,
                "amplitudes": amps0,
                "omegas": omegas,
                "mu": mu_val,
                "knm": [0.0] * (n * n),
                "knm_r": [0.0] * (n * n),
                "alpha": [0.0] * (n * n),
            },
            {
                # Legacy: no 'amplitudes' key, full state in 'phases'
                "phases": nxt_state.tolist(),
                "omegas": omegas,
                "mu": mu_val,
                "knm": [0.0] * (n * n),
                "alpha": [0.0] * (n * n),
            },
        ]
        ok, verified = replay_engine.verify_determinism_sl_chained(sl_engine, entries)
        assert verified == 1, "Legacy format must be replayed successfully"
        assert ok, "Legacy replay must match forward integration"

    def test_multi_step_chain_accumulates_correctly(self, replay_engine, sl_engine):
        """Chain 5 consecutive steps and verify all are deterministically replayed.
        This catches off-by-one errors in state handoff between steps."""
        n = 2
        omegas = np.array([1.5, 0.8])
        mu = np.array([0.4, 0.6])
        knm = np.array([[0.0, 0.2], [0.2, 0.0]])
        knm_r = np.zeros((n, n))
        alpha = np.zeros((n, n))
        state = np.array([0.0, np.pi, 0.5, 0.5])
        entries = []

        for _ in range(6):
            entries.append(
                {
                    "phases": state[:n].tolist(),
                    "amplitudes": state[n:].tolist(),
                    "omegas": omegas.tolist(),
                    "mu": mu.tolist(),
                    "knm": knm.ravel().tolist(),
                    "knm_r": knm_r.ravel().tolist(),
                    "alpha": alpha.ravel().tolist(),
                }
            )
            state = sl_engine.step(state, omegas, mu, knm, knm_r, 0.0, 0.0, alpha)

        ok, verified = replay_engine.verify_determinism_sl_chained(sl_engine, entries)
        assert ok, "All 5 transitions must replay deterministically"
        assert verified == 5, f"Expected 5 verified steps, got {verified}"
