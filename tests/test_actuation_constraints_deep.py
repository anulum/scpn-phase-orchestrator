# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Actuation constraints tests

from __future__ import annotations

import pytest

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ControlAction


def _action(
    knob: str = "K", value: float = 5.0, scope: str = "global"
) -> ControlAction:
    return ControlAction(
        knob=knob, scope=scope, value=value, ttl_s=1.0, justification="test"
    )


class TestBoundsEnforcement:
    """Value bounds clamping."""

    def test_within_bounds_unchanged(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=5.0), previous_value=5.0)
        assert result.value == 5.0

    def test_exceeds_upper_bound(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=15.0), previous_value=5.0)
        assert result.value == 10.0

    def test_below_lower_bound(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (2.0, 10.0)})
        result = proj.project(_action(value=0.5), previous_value=5.0)
        assert result.value == 2.0

    def test_unknown_knob_no_bounds(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={})
        result = proj.project(_action(value=999.0), previous_value=0.0)
        assert result.value == 999.0


class TestRateLimits:
    """Rate limit enforcement."""

    def test_within_rate_limit(self) -> None:
        proj = ActionProjector(rate_limits={"K": 2.0}, value_bounds={"K": (0.0, 20.0)})
        result = proj.project(_action(value=6.0), previous_value=5.0)
        assert result.value == 6.0

    def test_exceeds_positive_rate(self) -> None:
        proj = ActionProjector(rate_limits={"K": 1.0}, value_bounds={"K": (0.0, 20.0)})
        result = proj.project(_action(value=10.0), previous_value=5.0)
        assert result.value == 6.0

    def test_exceeds_negative_rate(self) -> None:
        proj = ActionProjector(rate_limits={"K": 1.0}, value_bounds={"K": (0.0, 20.0)})
        result = proj.project(_action(value=2.0), previous_value=5.0)
        assert result.value == 4.0

    def test_rate_limit_then_bounds(self) -> None:
        """Rate-limited value re-clamped to bounds."""
        proj = ActionProjector(rate_limits={"K": 2.0}, value_bounds={"K": (0.0, 6.0)})
        # prev=5, target=20 → rate-limited to 7 → clamped to 6
        result = proj.project(_action(value=20.0), previous_value=5.0)
        assert result.value == 6.0

    def test_no_rate_limit_for_knob(self) -> None:
        proj = ActionProjector(
            rate_limits={"alpha": 0.1}, value_bounds={"K": (0.0, 100.0)}
        )
        result = proj.project(_action(value=50.0), previous_value=0.0)
        assert result.value == 50.0

    def test_zero_rate_limit(self) -> None:
        proj = ActionProjector(rate_limits={"K": 0.0}, value_bounds={"K": (0.0, 20.0)})
        result = proj.project(_action(value=10.0), previous_value=5.0)
        assert result.value == 5.0


class TestImmutability:
    """Original action must not be mutated."""

    def test_original_value_preserved(self) -> None:
        proj = ActionProjector(rate_limits={"K": 1.0}, value_bounds={"K": (0.0, 10.0)})
        action = _action(value=50.0)
        proj.project(action, previous_value=5.0)
        assert action.value == 50.0

    def test_non_value_fields_preserved(self) -> None:
        proj = ActionProjector(rate_limits={"K": 1.0}, value_bounds={"K": (0.0, 10.0)})
        action = ControlAction(
            knob="K",
            scope="layer_7",
            value=999.0,
            ttl_s=42.0,
            justification="physics",
        )
        result = proj.project(action, previous_value=5.0)
        assert result.knob == "K"
        assert result.scope == "layer_7"
        assert result.ttl_s == 42.0
        assert result.justification == "physics"


class TestEdgeCases:
    """Boundary and edge-case behaviour."""

    def test_exact_upper_bound(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=10.0), previous_value=5.0)
        assert result.value == 10.0

    def test_exact_lower_bound(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (2.0, 10.0)})
        result = proj.project(_action(value=2.0), previous_value=5.0)
        assert result.value == 2.0

    def test_negative_value_with_negative_bounds(self) -> None:
        proj = ActionProjector(rate_limits={}, value_bounds={"K": (-10.0, -1.0)})
        result = proj.project(_action(value=-5.0), previous_value=-3.0)
        assert result.value == -5.0

    def test_prev_outside_bounds(self) -> None:
        """Prev outside bounds — result still within bounds."""
        proj = ActionProjector(rate_limits={"K": 1.0}, value_bounds={"K": (0.0, 10.0)})
        result = proj.project(_action(value=20.0), previous_value=15.0)
        assert result.value == 10.0

    def test_multiple_knobs_independent(self) -> None:
        proj = ActionProjector(
            rate_limits={"K": 1.0, "alpha": 0.1},
            value_bounds={"K": (0.0, 10.0), "alpha": (-1.0, 1.0)},
        )
        r1 = proj.project(_action(knob="K", value=20.0), previous_value=5.0)
        r2 = proj.project(_action(knob="alpha", value=5.0), previous_value=0.0)
        assert r1.value == 6.0
        assert r2.value == pytest.approx(0.1)
