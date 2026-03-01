# SCPN Phase Orchestrator
# Copyright concepts (c) 1996-2026 Miroslav Sotek. All rights reserved.
# Copyright code (c) 2026 Miroslav Sotek. All rights reserved.
# ORCID: https://orcid.org/0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# License: GNU AGPL v3 | Commercial licensing available

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


def test_value_clipping():
    proj = _projector()
    result = proj.project(_action("K", 5.0), previous_value=0.5)
    assert result.value <= 1.0


def test_negative_clipping():
    proj = _projector()
    result = proj.project(_action("K", -1.0), previous_value=0.5)
    assert result.value >= 0.0


def test_rate_limiting():
    proj = _projector()
    result = proj.project(_action("K", 0.9), previous_value=0.5)
    assert result.value == pytest.approx(0.6, abs=1e-12)


def test_rate_limiting_negative_direction():
    proj = _projector()
    result = proj.project(_action("K", 0.1), previous_value=0.5)
    assert result.value == pytest.approx(0.4, abs=1e-12)


def test_within_bounds_passes_through():
    proj = _projector()
    result = proj.project(_action("K", 0.55), previous_value=0.5)
    assert result.value == pytest.approx(0.55, abs=1e-12)


def test_unbounded_knob_no_clipping():
    proj = ActionProjector(rate_limits={}, value_bounds={})
    result = proj.project(_action("Psi", 100.0), previous_value=0.0)
    assert result.value == pytest.approx(100.0)
