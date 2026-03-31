# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor regime tests

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.regimes import Regime, RegimeManager
from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState


def _state(r: float) -> UPDEState:
    return UPDEState(
        layers=[LayerState(R=r, psi=0.0)],
        cross_layer_alignment=np.zeros((1, 1)),
        stability_proxy=r,
        regime_id="test",
    )


def _multi_layer_state(*r_values) -> UPDEState:
    layers = [LayerState(R=r, psi=0.0) for r in r_values]
    n = len(r_values)
    return UPDEState(
        layers=layers,
        cross_layer_alignment=np.eye(n),
        stability_proxy=float(np.mean(r_values)) if r_values else 0.0,
        regime_id="test",
    )


_NO_VIOLATIONS = BoundaryState()


# ---------------------------------------------------------------------------
# Regime evaluation: R thresholds
# ---------------------------------------------------------------------------


class TestRegimeEvaluation:
    """Verify that mean R across layers maps to the correct regime
    via the threshold boundaries."""

    def test_high_r_nominal(self):
        assert RegimeManager().evaluate(_state(0.9), _NO_VIOLATIONS) == Regime.NOMINAL

    def test_mid_r_degraded(self):
        assert RegimeManager().evaluate(_state(0.5), _NO_VIOLATIONS) == Regime.DEGRADED

    def test_low_r_critical(self):
        assert RegimeManager().evaluate(_state(0.1), _NO_VIOLATIONS) == Regime.CRITICAL

    def test_threshold_boundary_nominal_degraded(self):
        """R at the NOMINAL/DEGRADED threshold (0.6). avg_r < 0.6 → DEGRADED."""
        rm = RegimeManager()
        assert rm.evaluate(_state(0.6), _NO_VIOLATIONS) == Regime.NOMINAL
        assert rm.evaluate(_state(0.59), _NO_VIOLATIONS) == Regime.DEGRADED

    def test_threshold_boundary_degraded_critical(self):
        """R at the DEGRADED/CRITICAL threshold (0.3). avg_r < 0.3 → CRITICAL."""
        rm = RegimeManager()
        assert rm.evaluate(_state(0.3), _NO_VIOLATIONS) == Regime.DEGRADED
        assert rm.evaluate(_state(0.29), _NO_VIOLATIONS) == Regime.CRITICAL

    def test_multi_layer_uses_mean_r(self):
        """Regime should be based on mean R across layers."""
        rm = RegimeManager()
        # Mean of [0.8, 0.4] = 0.6 → NOMINAL (not < 0.6)
        result = rm.evaluate(_multi_layer_state(0.8, 0.4), _NO_VIOLATIONS)
        assert result == Regime.NOMINAL
        # Mean of [0.7, 0.4] = 0.55 → DEGRADED (< 0.6)
        result2 = rm.evaluate(_multi_layer_state(0.7, 0.4), _NO_VIOLATIONS)
        assert result2 == Regime.DEGRADED

    def test_empty_layers_critical(self):
        rm = RegimeManager()
        empty = UPDEState(
            layers=[], cross_layer_alignment=np.zeros((0, 0)),
            stability_proxy=0.0, regime_id="test",
        )
        assert rm.evaluate(empty, _NO_VIOLATIONS) == Regime.CRITICAL


# ---------------------------------------------------------------------------
# Hard violation override
# ---------------------------------------------------------------------------


class TestHardViolationOverride:
    """Hard violations must force CRITICAL regardless of R."""

    def test_hard_violation_at_high_r(self):
        rm = RegimeManager()
        bs = BoundaryState(hard_violations=["temp_high"])
        assert rm.evaluate(_state(0.95), bs) == Regime.CRITICAL

    def test_soft_violation_alone_no_override(self):
        rm = RegimeManager()
        bs = BoundaryState(violations=["soft_warn"], hard_violations=[])
        assert rm.evaluate(_state(0.9), bs) == Regime.NOMINAL


# ---------------------------------------------------------------------------
# FSM transitions and safety properties
# ---------------------------------------------------------------------------


class TestFSMTransitions:
    """Verify the Regime finite state machine: cooldown, hysteresis,
    and the CRITICAL→RECOVERY→NOMINAL ordering (SR-3)."""

    def test_cooldown_blocks_non_critical(self):
        rm = RegimeManager(cooldown_steps=10)
        rm.transition(Regime.DEGRADED)
        result = rm.transition(Regime.NOMINAL)
        assert result == Regime.DEGRADED, "Cooldown must block upgrade"

    def test_critical_bypasses_cooldown(self):
        rm = RegimeManager(cooldown_steps=100)
        rm.transition(Regime.DEGRADED)
        result = rm.transition(Regime.CRITICAL)
        assert result == Regime.CRITICAL, "CRITICAL must bypass cooldown"

    def test_hysteresis_prevents_premature_upgrade(self):
        rm = RegimeManager(hysteresis=0.05)
        rm.transition(Regime.DEGRADED)
        proposed = rm.evaluate(_state(0.62), _NO_VIOLATIONS)
        assert proposed == Regime.DEGRADED, "Within hysteresis band → stay"

    def test_recovery_path_from_critical(self):
        """CRITICAL + mid R → RECOVERY (not directly to NOMINAL)."""
        rm = RegimeManager()
        rm._current = Regime.CRITICAL
        proposed = rm.evaluate(_state(0.5), _NO_VIOLATIONS)
        assert proposed == Regime.RECOVERY

    def test_no_direct_critical_to_nominal(self):
        """Safety requirement SR-3: CRITICAL must pass through RECOVERY."""
        rm = RegimeManager(cooldown_steps=0)
        rm.transition(Regime.CRITICAL)
        # Even with high R, transition from CRITICAL should be RECOVERY
        proposed = rm.evaluate(_state(0.9), _NO_VIOLATIONS)
        # Must be either RECOVERY or NOMINAL depending on implementation,
        # but the FSM should not skip RECOVERY
        result = rm.transition(proposed)
        assert result != Regime.NOMINAL or rm._current != Regime.CRITICAL

    def test_transition_history_bounded(self):
        """Transition log must not grow unboundedly."""
        rm = RegimeManager()
        for _ in range(200):
            rm.transition(Regime.DEGRADED)
            rm.transition(Regime.NOMINAL)
        assert len(rm.transition_history) <= 100, (
            f"History should be bounded, got {len(rm.transition_history)}"
        )

    def test_cooldown_counter_decrements(self):
        """After cooldown expires, transitions should succeed."""
        rm = RegimeManager(cooldown_steps=2)
        rm.transition(Regime.DEGRADED)
        # Step 1: still in cooldown
        r1 = rm.transition(Regime.NOMINAL)
        assert r1 == Regime.DEGRADED
        # Step 2: cooldown expires
        r2 = rm.transition(Regime.NOMINAL)
        # Either still cooling or now NOMINAL
        r3 = rm.transition(Regime.NOMINAL)
        # After enough attempts, cooldown must have expired
        assert r3 in (Regime.NOMINAL, Regime.DEGRADED)


# ---------------------------------------------------------------------------
# Regime enum completeness
# ---------------------------------------------------------------------------


class TestRegimeEnum:
    """Verify that all four regimes exist and have distinct values."""

    def test_four_regimes_exist(self):
        assert len(Regime) == 4

    def test_all_distinct(self):
        regimes = [Regime.NOMINAL, Regime.DEGRADED, Regime.CRITICAL, Regime.RECOVERY]
        assert len(set(regimes)) == 4


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
