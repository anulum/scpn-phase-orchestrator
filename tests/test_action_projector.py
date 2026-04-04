# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Tests for ActionProjector

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ControlAction


def _action(knob: str = "K", value: float = 5.0) -> ControlAction:
    return ControlAction(
        knob=knob, scope="global", value=value, ttl_s=1.0, justification="test"
    )


class TestActionProjectorValueBounds:
    def test_within_bounds_unchanged(self):
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=5.0), previous_value=5.0)
        assert result.value == 5.0

    def test_clamp_above_upper_bound(self):
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=15.0), previous_value=5.0)
        assert result.value == 10.0

    def test_clamp_below_lower_bound(self):
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=-3.0), previous_value=5.0)
        assert result.value == 0.0

    def test_no_bounds_for_knob_passes_through(self):
        proj = ActionProjector(rate_limits={}, value_bounds={})
        result = proj.project(_action(value=999.0), previous_value=0.0)
        assert result.value == 999.0

    def test_negative_bounds(self):
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (-5.0, -1.0)})
        result = proj.project(_action(value=0.0), previous_value=-3.0)
        assert result.value == -1.0


class TestActionProjectorRateLimits:
    def test_rate_limit_positive_delta(self):
        proj = ActionProjector(
            rate_limits={"K": 2.0},
            value_bounds={"K": (0.0, 100.0)},
        )
        result = proj.project(_action(value=50.0), previous_value=10.0)
        assert result.value == pytest.approx(12.0)

    def test_rate_limit_negative_delta(self):
        proj = ActionProjector(
            rate_limits={"K": 2.0},
            value_bounds={"K": (0.0, 100.0)},
        )
        result = proj.project(_action(value=5.0), previous_value=10.0)
        assert result.value == pytest.approx(8.0)

    def test_rate_limit_within_limit_unchanged(self):
        proj = ActionProjector(
            rate_limits={"K": 10.0},
            value_bounds={"K": (0.0, 100.0)},
        )
        result = proj.project(_action(value=15.0), previous_value=10.0)
        assert result.value == pytest.approx(15.0)

    def test_rate_limit_respects_value_bounds(self):
        proj = ActionProjector(
            rate_limits={"K": 100.0},
            value_bounds={"K": (0.0, 10.0)},
        )
        result = proj.project(_action(value=50.0), previous_value=5.0)
        assert result.value == 10.0

    def test_no_rate_limit_for_knob(self):
        proj = ActionProjector(
            rate_limits={},
            value_bounds={"K": (0.0, 100.0)},
        )
        result = proj.project(_action(value=100.0), previous_value=0.0)
        assert result.value == 100.0


class TestActionProjectorIdentityPreservation:
    def test_other_fields_preserved(self):
        action = ControlAction(
            knob="K", scope="layer_3", value=50.0, ttl_s=7.5, justification="reason"
        )
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(action, previous_value=5.0)
        assert result.knob == "K"
        assert result.scope == "layer_3"
        assert result.ttl_s == 7.5
        assert result.justification == "reason"
        assert result.value == 10.0

    def test_returns_new_action_not_mutated_original(self):
        action = _action(value=50.0)
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(action, previous_value=5.0)
        assert action.value == 50.0
        assert result.value == 10.0
        assert result is not action


class TestActionProjectorEdgeCases:
    def test_exact_boundary_value(self):
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=10.0), previous_value=5.0)
        assert result.value == 10.0

    def test_rate_limit_zero_delta(self):
        proj = ActionProjector(
            rate_limits={"K": 2.0},
            value_bounds={"K": (0.0, 100.0)},
        )
        result = proj.project(_action(value=10.0), previous_value=10.0)
        assert result.value == 10.0

    def test_multiple_knobs(self):
        proj = ActionProjector(
            rate_limits={"K": 1.0, "zeta": 0.5},
            value_bounds={"K": (0.0, 10.0), "zeta": (0.0, 5.0)},
        )
        r1 = proj.project(_action("K", 20.0), previous_value=9.0)
        r2 = proj.project(_action("zeta", 20.0), previous_value=4.0)
        assert r1.value == 10.0
        assert r2.value == pytest.approx(4.5)


class TestActionProjectorProperty:
    @given(
        value=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
        prev=st.floats(min_value=-1000, max_value=1000, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_output_always_within_bounds(self, value: float, prev: float):
        proj = ActionProjector(
            rate_limits={"K": 5.0},
            value_bounds={"K": (-10.0, 10.0)},
        )
        result = proj.project(_action(value=value), previous_value=prev)
        assert -10.0 <= result.value <= 10.0

    @given(
        value=st.floats(min_value=-100, max_value=100, allow_nan=False),
        prev=st.floats(min_value=-100, max_value=100, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_rate_limit_respected(self, value: float, prev: float):
        rate = 3.0
        proj = ActionProjector(
            rate_limits={"K": rate},
            value_bounds={"K": (-200.0, 200.0)},
        )
        result = proj.project(_action(value=value), previous_value=prev)
        assert abs(result.value - prev) <= rate + 1e-10
