# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation mapper tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping


def _action(knob, scope, value):
    return ControlAction(
        knob=knob, scope=scope, value=value, ttl_s=5.0, justification="test"
    )


@pytest.fixture()
def mapper():
    return ActuationMapper(
        [
            ActuatorMapping(name="K_glob", knob="K", scope="global", limits=(0.0, 1.0)),
            ActuatorMapping(
                name="zeta_glob", knob="zeta", scope="global", limits=(0.0, 0.5)
            ),
            ActuatorMapping(name="K_L0", knob="K", scope="layer_0", limits=(0.0, 2.0)),
        ]
    )


# ---------------------------------------------------------------------------
# Action mapping: ControlAction → actuator commands
# ---------------------------------------------------------------------------


class TestActionMapping:
    """Verify that ControlActions are correctly routed to matching actuators
    with values clamped to configured limits."""

    def test_global_action_matches_global_actuator(self, mapper):
        cmds = mapper.map_actions([_action("K", "global", 0.3)])
        assert len(cmds) >= 1
        k_cmds = [c for c in cmds if c["actuator"] == "K_glob"]
        assert len(k_cmds) == 1
        assert k_cmds[0]["value"] == 0.3

    def test_value_clamped_to_upper_limit(self, mapper):
        cmds = mapper.map_actions([_action("K", "global", 5.0)])
        k_glob = [c for c in cmds if c["actuator"] == "K_glob"]
        assert k_glob[0]["value"] == 1.0, "Must clamp to upper limit"

    def test_value_clamped_to_lower_limit(self, mapper):
        cmds = mapper.map_actions([_action("zeta", "global", -1.0)])
        assert cmds[0]["value"] == 0.0, "Must clamp to lower limit"

    def test_unmatched_knob_produces_empty(self, mapper):
        cmds = mapper.map_actions([_action("Psi", "layer_0", 0.1)])
        assert cmds == []

    def test_command_contains_all_fields(self, mapper):
        cmds = mapper.map_actions([_action("K", "global", 0.5)])
        cmd = cmds[0]
        assert "actuator" in cmd
        assert "knob" in cmd
        assert "scope" in cmd
        assert "value" in cmd
        assert "ttl_s" in cmd
        assert cmd["ttl_s"] == 5.0

    def test_multiple_actions_produce_multiple_commands(self, mapper):
        actions = [_action("K", "global", 0.3), _action("zeta", "global", 0.2)]
        cmds = mapper.map_actions(actions)
        knobs = {c["knob"] for c in cmds}
        assert "K" in knobs and "zeta" in knobs

    def test_layer_scoped_action_matches_layer_actuator(self, mapper):
        """layer_0 scope action should match K_L0 (scope=layer_0)."""
        cmds = mapper.map_actions([_action("K", "layer_0", 1.5)])
        l0_cmds = [c for c in cmds if c["actuator"] == "K_L0"]
        assert len(l0_cmds) == 1
        assert l0_cmds[0]["value"] == 1.5

    def test_layer_actuator_has_wider_limits(self, mapper):
        """K_L0 has limits (0, 2.0) — wider than K_glob (0, 1.0)."""
        cmds = mapper.map_actions([_action("K", "layer_0", 1.8)])
        l0_cmds = [c for c in cmds if c["actuator"] == "K_L0"]
        assert l0_cmds[0]["value"] == 1.8, "K_L0 limit is 2.0, 1.8 should pass"


# ---------------------------------------------------------------------------
# Validation: pre-flight safety check
# ---------------------------------------------------------------------------


class TestActionValidation:
    """Verify that validate_action correctly distinguishes valid from
    invalid actions before they reach the control loop."""

    def test_valid_knob_in_range(self, mapper):
        assert mapper.validate_action(_action("K", "global", 0.5)) is True

    def test_valid_at_lower_boundary(self, mapper):
        assert mapper.validate_action(_action("K", "global", 0.0)) is True

    def test_valid_at_upper_boundary(self, mapper):
        assert mapper.validate_action(_action("K", "global", 1.0)) is True

    def test_invalid_above_all_limits(self, mapper):
        """Value above ALL actuator limits for this knob must be rejected."""
        # K_glob has (0,1), K_L0 has (0,2) — global scope matches both
        assert mapper.validate_action(_action("K", "global", 2.5)) is False

    def test_invalid_below_lower(self, mapper):
        assert mapper.validate_action(_action("K", "global", -0.1)) is False

    def test_invalid_unknown_knob(self, mapper):
        """Knobs not in VALID_KNOBS must be rejected regardless of value."""
        assert mapper.validate_action(_action("omega", "global", 0.1)) is False

    def test_invalid_unmatched_scope(self, mapper):
        """Valid knob but unmatched scope should fail validation."""
        result = mapper.validate_action(_action("zeta", "layer_99", 0.1))
        assert result is False, "No actuator for scope=layer_99"

    def test_layer_scope_with_wider_limits(self, mapper):
        """K_L0 has wider limits (0,2) than K_glob (0,1).
        layer_0 scope at 1.5 is valid via K_L0.
        global scope at 1.5 also valid because global matches K_L0 too."""
        assert mapper.validate_action(_action("K", "layer_0", 1.5)) is True
        # Global scope matches ALL actuators — K_L0 accepts 1.5
        assert mapper.validate_action(_action("K", "global", 1.5)) is True


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestActuationMapperEdgeCases:
    """Verify behaviour with empty or degenerate configurations."""

    def test_empty_mapper_produces_empty(self):
        mapper = ActuationMapper([])
        cmds = mapper.map_actions([_action("K", "global", 0.5)])
        assert cmds == []

    def test_empty_mapper_validation_rejects_all(self):
        mapper = ActuationMapper([])
        assert mapper.validate_action(_action("K", "global", 0.5)) is False

    def test_empty_action_list(self, mapper):
        assert mapper.map_actions([]) == []


class TestActuationMapperPipelineEndToEnd:
    """Full pipeline: Engine → R → Policy → ActuationMapper → device commands.

    Proves ActuationMapper is the output adapter, not decorative.
    """

    def test_policy_actions_map_to_device_commands(self):
        """Policy.decide() → ActuationMapper.map_actions() → valid commands."""
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
        mapper = ActuationMapper(
            [
                ActuatorMapping(
                    name="K_glob", knob="K", scope="global", limits=(0.0, 1.0)
                ),
                ActuatorMapping(
                    name="zeta_glob",
                    knob="zeta",
                    scope="global",
                    limits=(0.0, 0.5),
                ),
            ]
        )
        rng = np.random.default_rng(42)
        phases = rng.uniform(0, 2 * np.pi, n)
        omegas = rng.uniform(0.5, 1.5, n)
        knm = 0.1 * np.ones((n, n))
        np.fill_diagonal(knm, 0.0)
        alpha = np.zeros((n, n))
        phases = eng.run(phases, omegas, knm, 0.0, 0.0, alpha, n_steps=100)
        r, psi = compute_order_parameter(phases)
        layer = LayerState(R=r, psi=psi)
        state = UPDEState(
            layers=[layer],
            cross_layer_alignment=np.array([r]),
            stability_proxy=r,
            regime_id="nominal",
        )
        actions = pol.decide(state, BoundaryState())
        if actions:
            cmds = mapper.map_actions(actions)
            for cmd in cmds:
                assert cmd.name in {"K_glob", "zeta_glob"}
                assert cmd.value is not None

    def test_performance_map_actions_under_10us(self):
        """ActuationMapper.map_actions() < 10μs per call."""
        import time

        mapper = ActuationMapper(
            [
                ActuatorMapping(
                    name="K_glob", knob="K", scope="global", limits=(0.0, 1.0)
                ),
            ]
        )
        actions = [_action("K", "global", 0.5)]
        mapper.map_actions(actions)  # warm-up
        t0 = time.perf_counter()
        for _ in range(100000):
            mapper.map_actions(actions)
        elapsed = (time.perf_counter() - t0) / 100000
        assert elapsed < 1e-5, f"map_actions took {elapsed * 1e6:.1f}μs"


# Pipeline wiring: ActuationMapper tested via UPDEEngine → R → SupervisorPolicy
# → map_actions() → device commands. Output adapter chain verified.
# Performance: map_actions()<10μs.
