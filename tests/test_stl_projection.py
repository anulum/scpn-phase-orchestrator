# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL action projection behaviour tests

"""Behaviour tests for STL action-projection templates and projected plans.

Covers the template validation surface (step, ttl, rate-limit, bounds, non-empty
identifiers), the three controller-direction branches (increase, decrease, and the
neutral fallback), template-missing rejection, and the JSON audit record.
"""

from __future__ import annotations

import pytest

import scpn_phase_orchestrator.monitor.stl.projection as _projection
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.stl.controller import (
    STLControllerCandidate,
    STLControllerSynthesis,
)
from scpn_phase_orchestrator.monitor.stl.projection import (
    STLActionProjectionTemplate,
    STLProjectedActionPlan,
    project_stl_controller_candidates,
)

assert _projection is not None

_ACTION = "raise_coupling"


def _template(**overrides: object) -> STLActionProjectionTemplate:
    """Return a valid projection template, with optional field overrides."""
    params: dict[str, object] = {
        "action": _ACTION,
        "knob": "K",
        "scope": "global",
        "base_value": 1.0,
        "step": 0.5,
        "ttl_s": 2.0,
        "previous_value": 1.0,
        "value_bounds": (-5.0, 5.0),
        "rate_limit": None,
    }
    params.update(overrides)
    return STLActionProjectionTemplate(**params)  # type: ignore[arg-type]


def _candidate(
    direction: str,
    *,
    action: str = _ACTION,
    robustness: float = 0.4,
) -> STLControllerCandidate:
    """Return one STL controller candidate with the given direction."""
    return STLControllerCandidate(
        signal="R",
        action=action,
        direction=direction,
        time_index=0,
        robustness=robustness,
        rationale="coverage candidate",
    )


def _synthesis(*candidates: STLControllerCandidate) -> STLControllerSynthesis:
    """Return an STL synthesis result wrapping ``candidates``."""
    return STLControllerSynthesis(
        spec="G(R > 0.9)",
        satisfied=False,
        actuating=False,
        source_backend="admm",
        candidates=tuple(candidates),
    )


def test_template_accepts_a_valid_specification() -> None:
    template = _template(rate_limit=0.25)
    assert template.rate_limit == 0.25
    assert template.value_bounds == (-5.0, 5.0)


@pytest.mark.parametrize("step", [0.0, -0.5])
def test_template_rejects_non_positive_step(step: float) -> None:
    with pytest.raises(ValueError, match="projection step must be positive"):
        _template(step=step)


def test_template_rejects_negative_ttl() -> None:
    with pytest.raises(ValueError, match="projection ttl_s must be non-negative"):
        _template(ttl_s=-1.0)


def test_template_rejects_negative_rate_limit() -> None:
    with pytest.raises(ValueError, match="projection rate_limit must be non-negative"):
        _template(rate_limit=-0.1)


def test_template_rejects_unordered_bounds() -> None:
    with pytest.raises(ValueError, match="value_bounds must be ordered"):
        _template(value_bounds=(2.0, -2.0))


@pytest.mark.parametrize(
    ("field", "label"),
    [
        ("action", "projection action"),
        ("knob", "projection knob"),
        ("scope", "projection scope"),
    ],
)
def test_template_rejects_empty_identifiers(field: str, label: str) -> None:
    with pytest.raises(ValueError, match=label):
        _template(**{field: "  "})


def test_increase_direction_adds_the_scaled_magnitude() -> None:
    plan = project_stl_controller_candidates(
        _synthesis(_candidate("increase", robustness=0.4)),
        [_template()],
    )
    assert plan.actuating is False
    assert len(plan.approved_actions) == 1
    action = plan.approved_actions[0]
    assert action.knob == "K"
    # base 1.0 + |0.4| * step 0.5 = 1.2, inside the (-5, 5) bounds.
    assert action.value == pytest.approx(1.2)


def test_decrease_direction_subtracts_the_scaled_magnitude() -> None:
    plan = project_stl_controller_candidates(
        _synthesis(_candidate("decrease", robustness=0.4)),
        [_template()],
    )
    assert plan.approved_actions[0].value == pytest.approx(0.8)


def test_neutral_direction_keeps_the_base_value() -> None:
    plan = project_stl_controller_candidates(
        _synthesis(_candidate("hold", robustness=0.9)),
        [_template()],
    )
    assert plan.approved_actions[0].value == pytest.approx(1.0)


def test_candidate_without_a_template_is_rejected() -> None:
    plan = project_stl_controller_candidates(
        _synthesis(_candidate("increase", action="unmapped_action")),
        [_template()],
    )
    assert plan.approved_actions == ()
    assert len(plan.rejected_candidates) == 1
    rejection = plan.rejected_candidates[0]
    assert rejection["action"] == "unmapped_action"
    assert rejection["reason"] == "projection_template_missing"


def test_audit_record_is_json_safe_and_non_actuating() -> None:
    plan = project_stl_controller_candidates(
        _synthesis(
            _candidate("increase"),
            _candidate("decrease", action="lower_coupling"),
        ),
        [_template(), _template(action="lower_coupling")],
    )
    record = plan.to_audit_record()
    assert record["actuating"] is False
    assert record["spec"] == "G(R > 0.9)"
    approved = record["approved_actions"]
    assert isinstance(approved, list)
    assert len(approved) == 2
    first = approved[0]
    assert isinstance(first, dict)
    assert set(first) == {"knob", "scope", "value", "ttl_s", "justification"}
    assert isinstance(record["rejected_candidates"], list)


def test_projected_plan_dataclass_is_frozen() -> None:
    plan = STLProjectedActionPlan(
        spec="G(R > 0.9)",
        actuating=False,
        approved_actions=(
            ControlAction(
                knob="K", scope="global", value=1.0, ttl_s=2.0, justification="x"
            ),
        ),
        rejected_candidates=(),
    )
    with pytest.raises((AttributeError, TypeError)):
        plan.actuating = True  # type: ignore[misc]
