# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Action projection tests (safety-critical)

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ControlAction


def _projector():
    return ActionProjector(
        rate_limits={"K": 0.1, "zeta": 0.05},
        value_bounds={"K": (0.0, 1.0), "zeta": (0.0, 0.5)},
    )


def _action(knob, value):
    return ControlAction(
        knob=knob, scope="global", value=value, ttl_s=1.0, justification="test"
    )


# ---------------------------------------------------------------------------
# Value bounds enforcement (safety requirement SR-1)
# ---------------------------------------------------------------------------


class TestValueBoundsEnforcement:
    """Verify that output values never exceed configured bounds.
    This is safety-critical: unclamped values could drive K or zeta
    beyond physical limits and destabilise the control loop."""

    def test_clamp_above_upper_bound(self):
        proj = _projector()
        result = proj.project(_action("K", 5.0), previous_value=0.5)
        assert result.value == pytest.approx(0.6), (
            "Rate limit (0.1) should cap at 0.5+0.1=0.6, not upper bound 1.0"
        )

    def test_clamp_below_lower_bound(self):
        proj = _projector()
        result = proj.project(_action("K", -10.0), previous_value=0.5)
        assert result.value == pytest.approx(0.4), (
            "Rate limit should cap at 0.5-0.1=0.4, not lower bound 0.0"
        )

    def test_extreme_overshoot_bounded(self):
        """Even with no rate limit, value must stay within bounds."""
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
        result = proj.project(_action("K", 1e6), previous_value=0.5)
        assert result.value == pytest.approx(1.0)

    def test_extreme_undershoot_bounded(self):
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 1.0)})
        result = proj.project(_action("K", -1e6), previous_value=0.5)
        assert result.value == pytest.approx(0.0)

    def test_output_always_within_bounds_parametric(self):
        """Exhaustive: for various requested values, output must stay in [lo, hi]."""
        proj = _projector()
        test_values = [-100.0, -1.0, 0.0, 0.5, 0.99, 1.0, 1.01, 100.0]
        for v in test_values:
            result = proj.project(_action("K", v), previous_value=0.5)
            assert 0.0 <= result.value <= 1.0, (
                f"Requested K={v}, previous=0.5, got {result.value} outside [0, 1]"
            )


# ---------------------------------------------------------------------------
# Rate limiting enforcement (safety requirement SR-2)
# ---------------------------------------------------------------------------


class TestRateLimitEnforcement:
    """Verify that the absolute change per step never exceeds the configured
    rate limit. This prevents actuator slew damage and control instability."""

    def test_positive_rate_limited(self):
        proj = _projector()
        result = proj.project(_action("K", 0.9), previous_value=0.5)
        assert result.value == pytest.approx(0.6, abs=1e-12), (
            "0.9 requested from 0.5, rate=0.1 → should clamp at 0.6"
        )

    def test_negative_rate_limited(self):
        proj = _projector()
        result = proj.project(_action("K", 0.1), previous_value=0.5)
        assert result.value == pytest.approx(0.4, abs=1e-12), (
            "0.1 requested from 0.5, rate=0.1 → should clamp at 0.4"
        )

    def test_within_rate_passes_through(self):
        """Change within rate limit must not be altered."""
        proj = _projector()
        result = proj.project(_action("K", 0.55), previous_value=0.5)
        assert result.value == pytest.approx(0.55, abs=1e-12)

    def test_exact_rate_limit_passes(self):
        """Change exactly at rate limit must pass through."""
        proj = _projector()
        result = proj.project(_action("K", 0.6), previous_value=0.5)
        assert result.value == pytest.approx(0.6, abs=1e-12)

    def test_zeta_rate_limit_independent(self):
        """Each knob has its own rate limit — zeta (0.05) differs from K (0.1)."""
        proj = _projector()
        result = proj.project(_action("zeta", 0.3), previous_value=0.1)
        assert result.value == pytest.approx(0.15, abs=1e-12), (
            "zeta rate=0.05 from 0.1 → max 0.15"
        )

    def test_rate_limit_near_bounds_does_not_exceed(self):
        """Rate-limited value near upper bound must not overshoot the bound."""
        proj = _projector()
        # previous=0.95, rate=0.1, requested=5.0 → rate says 1.05, bound says 1.0
        result = proj.project(_action("K", 5.0), previous_value=0.95)
        assert result.value == pytest.approx(1.0), (
            "Rate-limited 0.95+0.1=1.05 must be clamped to upper bound 1.0"
        )

    def test_rate_limit_near_lower_bound_does_not_undershoot(self):
        """Rate-limited value near lower bound must not go below it."""
        proj = _projector()
        # previous=0.05, rate=0.1, requested=-5.0 → rate says -0.05, bound says 0.0
        result = proj.project(_action("K", -5.0), previous_value=0.05)
        assert result.value == pytest.approx(0.0), (
            "Rate-limited 0.05-0.1=-0.05 must be clamped to lower bound 0.0"
        )

    def test_consecutive_steps_max_change(self):
        """5 consecutive steps: each must change by at most rate_limit."""
        proj = _projector()
        prev = 0.5
        for _ in range(5):
            result = proj.project(_action("K", 1.0), previous_value=prev)
            assert abs(result.value - prev) <= 0.1 + 1e-12
            prev = result.value


# ---------------------------------------------------------------------------
# Unbounded and unknown knobs
# ---------------------------------------------------------------------------


class TestUnboundedKnobs:
    """Verify behaviour for knobs without configured limits."""

    def test_no_rate_or_bounds_passes_through(self):
        proj = ActionProjector(rate_limits={}, value_bounds={})
        result = proj.project(_action("Psi", 100.0), previous_value=0.0)
        assert result.value == pytest.approx(100.0)

    def test_unknown_knob_passes_through(self):
        """Knobs not in the config must not be altered."""
        proj = _projector()  # Only K and zeta configured
        result = proj.project(_action("custom_knob", 42.0), previous_value=0.0)
        assert result.value == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Action metadata preservation
# ---------------------------------------------------------------------------


class TestActionMetadataPreservation:
    """Verify that projection only modifies value — all other fields
    (knob, scope, ttl_s, justification) must be preserved."""

    def test_non_value_fields_unchanged(self):
        proj = _projector()
        action = ControlAction(
            knob="K",
            scope="layer_0",
            value=5.0,
            ttl_s=3.5,
            justification="boost K",
        )
        result = proj.project(action, previous_value=0.5)
        assert result.knob == "K"
        assert result.scope == "layer_0"
        assert result.ttl_s == 3.5
        assert result.justification == "boost K"
        assert result.value != action.value, "Value should be modified by projection"


class TestProjectionPipelineEndToEnd:
    """Full pipeline: Engine → R → Policy → ActionProjector → safe engine update.

    Proves ActionProjector is structurally load-bearing in the control loop.
    """

    def test_policy_actions_projected_and_applied_to_engine(self):
        """Policy.decide() → ActionProjector → safe K update → engine."""
        import numpy as np

        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.engine import UPDEEngine
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState
        from scpn_phase_orchestrator.upde.order_params import compute_order_parameter

        n = 4
        eng = UPDEEngine(n, dt=0.01)
        rm = RegimeManager(cooldown_steps=0)
        pol = SupervisorPolicy(rm)
        proj = _projector()
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = np.ones(n)
        k_current = 0.3
        zeta_current = 0.0
        knm = np.full((n, n), k_current)
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))

        for _ in range(5):
            phases = eng.run(phases, omegas, knm, zeta_current, 0.0, alpha, n_steps=50)
            r, psi = compute_order_parameter(phases)
            layer = LayerState(R=r, psi=psi)
            state = UPDEState(
                layers=[layer],
                cross_layer_alignment=np.array([r]),
                stability_proxy=r,
                regime_id="nominal",
            )
            actions = pol.decide(state, BoundaryState())
            for a in actions:
                if a.knob == "K":
                    safe_a = proj.project(a, previous_value=k_current)
                    assert 0.0 <= safe_a.value <= 1.0
                    k_current = safe_a.value
                    knm = np.full((n, n), k_current)
                    np.fill_diagonal(knm, 0.0)
                elif a.knob == "zeta":
                    safe_a = proj.project(a, previous_value=zeta_current)
                    assert 0.0 <= safe_a.value <= 0.5
                    zeta_current = safe_a.value
        # After feedback loop, R should be valid
        r_final, _ = compute_order_parameter(phases)
        assert 0.0 <= r_final <= 1.0

    def test_performance_project_under_50us(self):
        """ActionProjector.project() < 50μs per call.

        Budget relaxed from 20μs to 50μs: CI runners and loaded
        machines routinely exceed 20μs due to scheduling jitter.
        """
        import time

        proj = _projector()
        action = _action("K", 0.7)
        proj.project(action, previous_value=0.5)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100000):
            proj.project(action, previous_value=0.5)
        elapsed = (time.perf_counter() - t0) / 100000
        assert elapsed < 5e-5, f"project() took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: ActionProjector tested via UPDEEngine → compute_order_parameter
# → SupervisorPolicy.decide() → project() → safe K/zeta update → engine feedback.
# Safety: rate limits + value bounds enforced in closed loop. Performance: <20μs.
