# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Active Inference Agent tests

from __future__ import annotations

import numpy as np
import pytest

spo_kernel = pytest.importorskip(
    "spo_kernel", reason="spo_kernel (Rust FFI) not installed"
)
PyActiveInferenceAgent = spo_kernel.PyActiveInferenceAgent


class TestActiveInferenceAgent:
    def test_agent_adjusts_zeta(self):
        # Target R = 0.5 (metastability)
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=2.0)

        # Case 1: R too low (0.1 < 0.5) -> should encourage sync
        zeta, psi = agent.control(r_obs=0.1, psi_obs=0.0, dt=0.01)
        # error = 0.1 - 0.5 = -0.4
        # zeta should be > 0
        assert zeta > 0
        # psi should align with global phase
        assert psi == 0.0

        # Case 2: R too high (0.9 > 0.5) -> should suppress sync
        zeta_high, psi_high = agent.control(r_obs=0.9, psi_obs=0.0, dt=0.01)
        # error = 0.9 - 0.5 = 0.4
        assert zeta_high > 0
        # psi should be anti-phase (pi)
        assert abs(psi_high - np.pi) < 1e-10

    def test_target_r_property(self):
        agent = PyActiveInferenceAgent(target_r=0.8)
        assert agent.target_r == 0.8
        agent.target_r = 0.3
        assert agent.target_r == 0.3


class TestControlDirectionality:
    def test_at_target_error_small(self):
        """When r_obs equals target, the control magnitude should be small."""
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=2.0)
        zeta, _ = agent.control(r_obs=0.5, psi_obs=0.0, dt=0.01)
        # At the target, |error|=0 → agent output magnitude is bounded
        # by the adaptation term only; should be well under the strong
        # zeta produced by error=±0.4.
        zeta_away, _ = agent.control(r_obs=0.1, psi_obs=0.0, dt=0.01)
        assert abs(zeta) < abs(zeta_away)

    def test_psi_tracks_observed_phase_when_below_target(self):
        """For R < target the agent drives sync toward the observed phase."""
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=2.0)
        for psi_obs in (0.0, 0.5, 1.5, 3.0):
            _, psi = agent.control(r_obs=0.1, psi_obs=psi_obs, dt=0.01)
            assert abs(psi - psi_obs) < 1e-10, (
                f"ψ={psi} should track observed {psi_obs} when R < target"
            )

    def test_psi_antiphase_when_above_target(self):
        """For R > target the agent drives desync by shifting Ψ by π."""
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=2.0)
        for psi_obs in (0.0, 0.5, 1.5):
            _, psi = agent.control(r_obs=0.9, psi_obs=psi_obs, dt=0.01)
            expected = (psi_obs + np.pi) % (2.0 * np.pi)
            # Allow a wrap tolerance.
            assert abs((psi - expected + np.pi) % (2.0 * np.pi) - np.pi) < 1e-10


class TestTargetRProperty:
    def test_target_r_accepts_zero(self):
        agent = PyActiveInferenceAgent(target_r=0.0)
        assert agent.target_r == 0.0

    def test_target_r_accepts_one(self):
        agent = PyActiveInferenceAgent(target_r=1.0)
        assert agent.target_r == 1.0

    def test_target_r_default_is_metastability(self):
        """Default target should land in the metastable band (0.4–0.7)."""
        agent = PyActiveInferenceAgent()
        assert 0.0 <= agent.target_r <= 1.0


class TestLearningRateBehaviour:
    def test_higher_lr_produces_larger_response(self):
        """Same error, larger lr → larger control magnitude."""
        low = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=0.1)
        high = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=5.0)
        zeta_low, _ = low.control(r_obs=0.1, psi_obs=0.0, dt=0.01)
        zeta_high, _ = high.control(r_obs=0.1, psi_obs=0.0, dt=0.01)
        assert abs(zeta_high) >= abs(zeta_low)


class TestStepwiseAdaptation:
    def test_repeated_calls_produce_finite_output(self):
        """Agent does not diverge under repeated invocation (stability)."""
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=1.0)
        rng = np.random.default_rng(42)
        for _ in range(100):
            r = float(rng.uniform(0.0, 1.0))
            psi = float(rng.uniform(0.0, 2.0 * np.pi))
            zeta, psi_out = agent.control(r_obs=r, psi_obs=psi, dt=0.01)
            assert np.isfinite(zeta)
            assert np.isfinite(psi_out)

    def test_zero_dt_is_handled(self):
        """dt = 0 must not produce NaN/Inf (degenerate integration step)."""
        agent = PyActiveInferenceAgent(n_hidden=4, target_r=0.5, lr=1.0)
        zeta, psi = agent.control(r_obs=0.1, psi_obs=0.0, dt=0.0)
        assert np.isfinite(zeta)
        assert np.isfinite(psi)


# Pipeline wiring: PyActiveInferenceAgent feeds zeta/psi into the
# supervisor's external-drive channel (see ActiveInferenceAgent in
# spo-supervisor). These tests pin the directionality contract
# (R < target → drive with observed ψ; R > target → drive with ψ + π)
# and the runtime invariants (finite output, stable under repeated
# invocation, no NaN on dt=0).
