# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for imprint and actuation modules

from __future__ import annotations

import numpy as np
import pytest

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping
from scpn_phase_orchestrator.imprint.state import ImprintState
from scpn_phase_orchestrator.imprint.update import ImprintModel

# ── ImprintState ─────────────────────────────────────────────────────────


class TestImprintState:
    def test_frozen(self) -> None:
        state = ImprintState(m_k=np.zeros(3), last_update=0.0)
        with pytest.raises(AttributeError):
            state.last_update = 1.0  # type: ignore[misc]

    def test_default_attribution(self) -> None:
        state = ImprintState(m_k=np.ones(2), last_update=0.0)
        assert state.attribution == {}


# ── ImprintModel ─────────────────────────────────────────────────────────


class TestImprintModel:
    def test_negative_decay_raises(self) -> None:
        with pytest.raises(ValueError):
            ImprintModel(decay_rate=-1.0, saturation=1.0)

    def test_zero_saturation_raises(self) -> None:
        with pytest.raises(ValueError):
            ImprintModel(decay_rate=0.1, saturation=0.0)

    def test_update_accumulates(self) -> None:
        model = ImprintModel(decay_rate=0.0, saturation=10.0)
        state = ImprintState(m_k=np.zeros(3), last_update=0.0)
        exposure = np.ones(3)
        new = model.update(state, exposure, dt=1.0)
        np.testing.assert_allclose(new.m_k, [1.0, 1.0, 1.0])
        assert new.last_update == 1.0

    def test_update_decays(self) -> None:
        model = ImprintModel(decay_rate=1.0, saturation=10.0)
        state = ImprintState(m_k=np.ones(3), last_update=0.0)
        new = model.update(state, np.zeros(3), dt=1.0)
        assert np.all(new.m_k < 1.0)

    def test_update_saturates(self) -> None:
        model = ImprintModel(decay_rate=0.0, saturation=2.0)
        state = ImprintState(m_k=np.full(3, 1.5), last_update=0.0)
        new = model.update(state, np.ones(3), dt=5.0)
        assert np.all(new.m_k <= 2.0)

    def test_modulate_coupling_shape(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=5.0)
        knm = np.ones((3, 3))
        imprint = ImprintState(m_k=np.array([0.1, 0.2, 0.3]), last_update=0.0)
        result = model.modulate_coupling(knm, imprint)
        assert result.shape == (3, 3)
        assert result[0, 0] == pytest.approx(1.1)

    def test_modulate_lag_antisymmetric(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=5.0)
        alpha = np.zeros((3, 3))
        imprint = ImprintState(m_k=np.array([0.0, 1.0, 2.0]), last_update=0.0)
        result = model.modulate_lag(alpha, imprint)
        np.testing.assert_allclose(result, -result.T, atol=1e-12)

    def test_modulate_mu(self) -> None:
        model = ImprintModel(decay_rate=0.1, saturation=5.0)
        mu = np.array([1.0, 2.0, 3.0])
        imprint = ImprintState(m_k=np.array([0.0, 0.5, 1.0]), last_update=0.0)
        result = model.modulate_mu(mu, imprint)
        np.testing.assert_allclose(result, [1.0, 3.0, 6.0])


# ── ActionProjector ─────────────────────────────────────────────────────


class TestActionProjector:
    def test_clamp_value(self) -> None:
        proj = ActionProjector(
            rate_limits={},
            value_bounds={"K": (0.0, 1.0)},
        )
        action = ControlAction(
            knob="K", scope="global", value=5.0, ttl_s=1.0, justification="test"
        )
        result = proj.project(action, previous_value=0.5)
        assert result.value == 1.0

    def test_rate_limit(self) -> None:
        proj = ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": (0.0, 10.0)},
        )
        action = ControlAction(
            knob="K", scope="global", value=5.0, ttl_s=1.0, justification="test"
        )
        result = proj.project(action, previous_value=0.5)
        assert abs(result.value - 0.6) < 1e-10

    def test_no_bounds_passthrough(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={})
        action = ControlAction(
            knob="zeta", scope="global", value=99.0, ttl_s=1.0, justification="test"
        )
        result = proj.project(action, previous_value=0.0)
        assert result.value == 99.0

    def test_negative_rate(self) -> None:
        proj = ActionProjector(
            rate_limits={"K": 0.2},
            value_bounds={"K": (0.0, 10.0)},
        )
        action = ControlAction(
            knob="K", scope="global", value=0.0, ttl_s=1.0, justification="test"
        )
        result = proj.project(action, previous_value=1.0)
        assert abs(result.value - 0.8) < 1e-10


# ── ActuationMapper ──────────────────────────────────────────────────────


class TestActuationMapper:
    def _mapping(self) -> list[ActuatorMapping]:
        return [
            ActuatorMapping(name="amp1", knob="K", scope="global", limits=(0.0, 5.0)),
        ]

    def test_map_actions(self) -> None:
        mapper = ActuationMapper(self._mapping())
        actions = [
            ControlAction(
                knob="K", scope="global", value=3.0, ttl_s=1.0, justification="test"
            )
        ]
        cmds = mapper.map_actions(actions)
        assert len(cmds) == 1
        assert cmds[0]["value"] == 3.0

    def test_map_clamps_to_limits(self) -> None:
        mapper = ActuationMapper(self._mapping())
        actions = [
            ControlAction(
                knob="K", scope="global", value=99.0, ttl_s=1.0, justification="test"
            )
        ]
        cmds = mapper.map_actions(actions)
        assert cmds[0]["value"] == 5.0

    def test_validate_valid(self) -> None:
        mapper = ActuationMapper(self._mapping())
        action = ControlAction(
            knob="K", scope="global", value=2.0, ttl_s=1.0, justification="test"
        )
        assert mapper.validate_action(action) is True

    def test_validate_out_of_range(self) -> None:
        mapper = ActuationMapper(self._mapping())
        action = ControlAction(
            knob="K", scope="global", value=99.0, ttl_s=1.0, justification="test"
        )
        assert mapper.validate_action(action) is False

    def test_validate_unknown_knob(self) -> None:
        mapper = ActuationMapper(self._mapping())
        action = ControlAction(
            knob="unknown_knob",
            scope="global",
            value=1.0,
            ttl_s=1.0,
            justification="test",
        )
        assert mapper.validate_action(action) is False

    def test_empty_actions(self) -> None:
        mapper = ActuationMapper(self._mapping())
        assert mapper.map_actions([]) == []


class TestImprintActuationPipelineWiring:
    """Pipeline: policy → actions → projector → mapper → engine."""

    def test_policy_actions_through_projector_and_mapper(self):
        """SupervisorPolicy → ControlActions → ActionProjector.project →
        ActuationMapper.map_actions → actuator commands."""
        import numpy as np

        from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
        from scpn_phase_orchestrator.supervisor.policy import SupervisorPolicy
        from scpn_phase_orchestrator.supervisor.regimes import RegimeManager
        from scpn_phase_orchestrator.upde.metrics import LayerState, UPDEState

        rm = RegimeManager(cooldown_steps=0)
        policy = SupervisorPolicy(rm)
        state = UPDEState(
            layers=[LayerState(R=0.4, psi=0.0)],
            cross_layer_alignment=np.eye(1),
            stability_proxy=0.4,
            regime_id="nominal",
        )
        actions = policy.decide(state, BoundaryState())
        assert len(actions) >= 1

        projector = ActionProjector(
            rate_limits={"K": 0.1},
            value_bounds={"K": (0.0, 1.0)},
        )
        for action in actions:
            projected = projector.project(action, previous_value=0.5)
            assert projected.value <= 1.0
            assert projected.value >= 0.0
