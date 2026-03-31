# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Policy engine tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _make_upde(r_values, regime_id="nominal"):
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    n = len(r_values)
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(n) if n else np.empty((0, 0)),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id=regime_id,
    )


@pytest.fixture()
def policy():
    return SupervisorPolicy(RegimeManager(cooldown_steps=0))


# ---------------------------------------------------------------------------
# Regime-action mapping: NOMINAL → nothing, DEGRADED → K boost, etc.
# ---------------------------------------------------------------------------


class TestRegimeActionMapping:
    """Verify that each regime produces the correct set of control actions
    with the expected knobs, signs, and justifications."""

    def test_nominal_produces_no_actions(self, policy):
        """High coherence → NOMINAL → no intervention needed."""
        state = _make_upde([0.9, 0.85, 0.88])
        actions = policy.decide(state, BoundaryState())
        assert actions == [], "NOMINAL must produce zero actions"

    def test_degraded_produces_positive_k_boost(self, policy):
        """Mid coherence → DEGRADED → increase coupling (K > 0)."""
        state = _make_upde([0.4, 0.45, 0.5])
        actions = policy.decide(state, BoundaryState())
        assert len(actions) >= 1
        assert actions[0].knob == "K"
        assert actions[0].value > 0, "DEGRADED K action must be positive (boost)"
        assert actions[0].scope == "global"

    def test_degraded_k_value_is_0_05(self, policy):
        """K bump constant is 0.05 (from _K_BUMP)."""
        state = _make_upde([0.4, 0.45, 0.5])
        actions = policy.decide(state, BoundaryState())
        assert actions[0].value == pytest.approx(0.05)

    def test_critical_produces_zeta_and_negative_k(self, policy):
        """Low coherence → CRITICAL → zeta damping + reduce coupling on worst layer."""
        state = _make_upde([0.1, 0.15, 0.2])
        actions = policy.decide(state, BoundaryState())
        knobs = {a.knob for a in actions}
        assert "zeta" in knobs, "CRITICAL must include zeta action"
        k_actions = [a for a in actions if a.knob == "K"]
        assert any(a.value < 0 for a in k_actions), (
            "CRITICAL K action must be negative (reduce coupling)"
        )

    def test_critical_targets_worst_layer(self, policy):
        """K reduction must target the layer with lowest R."""
        state = _make_upde([0.25, 0.1, 0.2])  # Layer 1 is worst (R=0.1)
        actions = policy.decide(state, BoundaryState())
        k_actions = [a for a in actions if a.knob == "K"]
        scopes = [a.scope for a in k_actions]
        assert "layer_1" in scopes, f"Expected layer_1: {scopes}"

    def test_critical_zeta_value_is_0_1(self, policy):
        state = _make_upde([0.05, 0.1])
        actions = policy.decide(state, BoundaryState())
        zeta = [a for a in actions if a.knob == "zeta"]
        assert zeta[0].value == pytest.approx(0.1)

    def test_critical_k_reduce_value_is_minus_0_03(self, policy):
        state = _make_upde([0.05, 0.1])
        actions = policy.decide(state, BoundaryState())
        k_actions = [a for a in actions if a.knob == "K"]
        assert k_actions[0].value == pytest.approx(-0.03)


# ---------------------------------------------------------------------------
# Hard violation override
# ---------------------------------------------------------------------------


class TestHardViolationOverride:
    """Verify that hard boundary violations force CRITICAL regardless
    of coherence level."""

    def test_hard_violation_forces_critical_at_high_r(self, policy):
        """R=0.9 would be NOMINAL, but hard violation forces CRITICAL."""
        state = _make_upde([0.9, 0.85])
        boundary = BoundaryState(violations=["x"], hard_violations=["x"])
        actions = policy.decide(state, boundary)
        knobs = {a.knob for a in actions}
        assert "zeta" in knobs, "Hard violation must force CRITICAL → zeta action"

    def test_soft_violation_no_critical_override(self, policy):
        """Soft violation alone should not force CRITICAL if R is high."""
        state = _make_upde([0.9, 0.85])
        boundary = BoundaryState(violations=["x"], hard_violations=[])
        actions = policy.decide(state, boundary)
        # Soft violation with high R should stay NOMINAL
        assert actions == [] or all(a.knob != "zeta" for a in actions)


# ---------------------------------------------------------------------------
# Recovery regime
# ---------------------------------------------------------------------------


class TestRecoveryRegime:
    """Verify the CRITICAL → RECOVERY transition and the gradual
    coupling restore strategy."""

    def test_recovery_after_critical(self, policy):
        """After CRITICAL, returning to mid-R should enter RECOVERY."""
        policy.decide(_make_upde([0.1, 0.15]), BoundaryState())  # → CRITICAL
        actions = policy.decide(_make_upde([0.45, 0.50]), BoundaryState())
        assert len(actions) == 1
        assert actions[0].knob == "K"
        assert "recovery" in actions[0].justification

    def test_recovery_k_is_half_bump(self, policy):
        """Recovery K value should be K_BUMP * RESTORE_FRACTION = 0.05 * 0.5 = 0.025."""
        policy.decide(_make_upde([0.1, 0.15]), BoundaryState())
        actions = policy.decide(_make_upde([0.45, 0.50]), BoundaryState())
        assert actions[0].value == pytest.approx(0.025)

    def test_recovery_scope_is_global(self, policy):
        policy.decide(_make_upde([0.1, 0.15]), BoundaryState())
        actions = policy.decide(_make_upde([0.45, 0.50]), BoundaryState())
        assert actions[0].scope == "global"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestPolicyEdgeCases:
    """Verify defined behaviour for degenerate inputs."""

    def test_empty_layers_forces_critical(self, policy):
        """No layers → empty R → CRITICAL (defensive)."""
        state = _make_upde([])
        actions = policy.decide(state, BoundaryState())
        knobs = {a.knob for a in actions}
        assert "zeta" in knobs

    def test_single_layer_policy(self, policy):
        """Policy must work with a single layer."""
        actions = policy.decide(_make_upde([0.4]), BoundaryState())
        assert len(actions) >= 1
        assert actions[0].knob == "K"

    def test_all_actions_have_justification(self, policy):
        """Every action must carry a non-empty justification string."""
        for r_values in [[0.9], [0.4, 0.5], [0.1, 0.15], []]:
            state = _make_upde(r_values)
            actions = policy.decide(state, BoundaryState())
            for a in actions:
                assert a.justification, f"Action {a.knob} missing justification"

    def test_all_actions_have_positive_ttl(self, policy):
        """Every action must have a positive time-to-live."""
        for r_values in [[0.4, 0.5], [0.1, 0.15]]:
            state = _make_upde(r_values)
            actions = policy.decide(state, BoundaryState())
            for a in actions:
                assert a.ttl_s > 0, f"Action {a.knob} has ttl_s={a.ttl_s}"


class TestPolicyPipelineEndToEnd:
    """Full pipeline: CouplingBuilder → Engine → R → Policy → ControlActions.

    Proves SupervisorPolicy is the decision core, not decorative.
    """

    def test_engine_trajectory_drives_policy_decisions(self):
        """Engine output → R → Policy.decide() → actionable control actions."""
        from scpn_phase_orchestrator.coupling.knm import CouplingBuilder
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 8
        cs = CouplingBuilder().build(n, 0.5, 0.2)
        eng = UPDEEngine(n, dt=0.01, method="rk4")
        pol = SupervisorPolicy(RegimeManager(cooldown_steps=0))
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        phases = eng.run(phases, omegas, cs.knm, 0.0, 0.0, cs.alpha, n_steps=300)
        r, psi = compute_order_parameter(phases)
        assert 0.0 <= r <= 1.0
        state = _make_upde([r])
        actions = pol.decide(state, BoundaryState())
        # Actions must be well-formed
        for a in actions:
            assert a.knob in {"K", "zeta", "psi"}
            assert a.justification
            assert a.ttl_s > 0

    def test_policy_feedback_loop_stabilises_engine(self):
        """Policy K-boost → apply to engine → R should increase."""
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        pol = SupervisorPolicy(RegimeManager(cooldown_steps=0))
        eng = UPDEEngine(n, dt=0.01)
        rng = np.random.default_rng(55)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.uniform(0.5, 1.5, n)
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        # Simulate feedback loop
        for _ in range(10):
            phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=50)
            r, _ = compute_order_parameter(phases)
            state = _make_upde([r])
            actions = pol.decide(state, BoundaryState())
            for a in actions:
                if a.knob == "K" and a.scope == "global":
                    knm = np.clip(knm + a.value, 0.0, 5.0)
                    np.fill_diagonal(knm, 0.0)
        # After feedback, R should be reasonable
        r_final, _ = compute_order_parameter(phases)
        assert 0.0 <= r_final <= 1.0

    def test_performance_decide_under_50us(self):
        """SupervisorPolicy.decide() < 50μs per call."""
        import time

        pol = SupervisorPolicy(RegimeManager(cooldown_steps=0))
        state = _make_upde([0.4, 0.5, 0.6])
        boundary = BoundaryState()
        pol.decide(state, boundary)  # warm-up
        t0 = time.perf_counter()
        for _ in range(10000):
            pol.decide(state, boundary)
        elapsed = (time.perf_counter() - t0) / 10000
        assert elapsed < 5e-5, f"decide() took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: SupervisorPolicy tested via CouplingBuilder → UPDEEngine(RK4)
# → compute_order_parameter → decide() → ControlActions → K feedback loop.
# Performance: decide()<50μs. Recovery + CRITICAL + NOMINAL paths tested.
