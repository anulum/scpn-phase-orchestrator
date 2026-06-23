# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — STL action projection

"""Action projection templates and projected action plans from controllers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ControlAction

from .controller import STLControllerCandidate, STLControllerSynthesis
from .monitor import _require_non_empty


@dataclass(frozen=True)
class STLActionProjectionTemplate:
    """Policy-approved projection template for one STL candidate action."""

    action: str
    knob: str
    scope: str
    base_value: float
    step: float
    ttl_s: float
    previous_value: float
    value_bounds: tuple[float, float]
    rate_limit: float | None = None

    def __post_init__(self) -> None:
        _require_non_empty(self.action, "projection action")
        _require_non_empty(self.knob, "projection knob")
        _require_non_empty(self.scope, "projection scope")
        if self.step <= 0.0:
            raise ValueError("projection step must be positive")
        if self.ttl_s < 0.0:
            raise ValueError("projection ttl_s must be non-negative")
        lo, hi = self.value_bounds
        if lo > hi:
            raise ValueError("projection value_bounds must be ordered")
        if self.rate_limit is not None and self.rate_limit < 0.0:
            raise ValueError("projection rate_limit must be non-negative")


@dataclass(frozen=True)
class STLProjectedActionPlan:
    """Policy-gated, non-actuating projection of STL candidates."""

    spec: str
    actuating: bool
    approved_actions: tuple[ControlAction, ...]
    rejected_candidates: tuple[dict[str, object], ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable projected-action plan.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable projected-action plan.
        """
        return {
            "spec": self.spec,
            "actuating": self.actuating,
            "approved_actions": [
                _control_action_record(action) for action in self.approved_actions
            ],
            "rejected_candidates": list(self.rejected_candidates),
        }


def project_stl_controller_candidates(
    synthesis: STLControllerSynthesis,
    templates: Sequence[STLActionProjectionTemplate],
) -> STLProjectedActionPlan:
    """Project STL candidates into bounded, non-actuating action proposals.

    Only candidates with an explicit policy-approved projection template are
    converted. Projection uses the standard :class:`ActionProjector`; the
    returned plan remains a review artefact with ``actuating=False``.

    Parameters
    ----------
    synthesis : STLControllerSynthesis
        The STL controller synthesis result.
    templates : Sequence[STLActionProjectionTemplate]
        STL action-projection templates.

    Returns
    -------
    STLProjectedActionPlan
        The bounded, non-actuating projected action plan.
    """
    template_by_action = {template.action: template for template in templates}
    approved: list[ControlAction] = []
    rejected: list[dict[str, object]] = []
    for candidate in synthesis.candidates:
        template = template_by_action.get(candidate.action)
        if template is None:
            rejected.append(
                {
                    "action": candidate.action,
                    "signal": candidate.signal,
                    "reason": "projection_template_missing",
                }
            )
            continue
        raw_action = _candidate_to_control_action(candidate, template)
        projector = ActionProjector(
            rate_limits=(
                {template.knob: template.rate_limit}
                if template.rate_limit is not None
                else {}
            ),
            value_bounds={template.knob: template.value_bounds},
        )
        approved.append(
            projector.project(raw_action, previous_value=template.previous_value)
        )
    return STLProjectedActionPlan(
        spec=synthesis.spec,
        actuating=False,
        approved_actions=tuple(approved),
        rejected_candidates=tuple(rejected),
    )


def _candidate_to_control_action(
    candidate: STLControllerCandidate,
    template: STLActionProjectionTemplate,
) -> ControlAction:
    """Convert a controller candidate into a projected control action."""
    magnitude = abs(candidate.robustness) * template.step
    if candidate.direction == "increase":
        value = template.base_value + magnitude
    elif candidate.direction == "decrease":
        value = template.base_value - magnitude
    else:
        value = template.base_value
    return ControlAction(
        knob=template.knob,
        scope=template.scope,
        value=value,
        ttl_s=template.ttl_s,
        justification=f"STL candidate {candidate.action}: {candidate.rationale}",
    )


def _control_action_record(action: ControlAction) -> dict[str, object]:
    """Return the JSON-safe record for a projected control action."""
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": action.value,
        "ttl_s": action.ttl_s,
        "justification": action.justification,
    }
