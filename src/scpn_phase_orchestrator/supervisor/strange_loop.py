# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Strange-loop supervisor monitor

"""Self-monitoring supervisor action-history diagnostics.

``StrangeLoopSupervisor`` embeds recent control-action bundles into native
control-knob space and measures drift, oscillation, coherence, and over-control
from the bounded history. Recommendations are conservative damping proposals
for an outer policy gate to approve. The monitor records diagnostics only and
does not apply actions or alter the underlying supervisor.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from numbers import Integral, Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.mapper import ControlAction

__all__ = [
    "StrangeLoopAssessment",
    "StrangeLoopDriftScenario",
    "StrangeLoopDriftScenarioResult",
    "StrangeLoopSupervisor",
    "build_strange_loop_drift_scenarios",
    "evaluate_strange_loop_drift_scenarios",
]

FloatArray: TypeAlias = NDArray[np.float64]
_KNOB_INDEX = {"K": 0, "alpha": 1, "zeta": 2, "Psi": 3}
_DRIFT_SCENARIO_BOUNDARY = "strange_loop_drift_review_not_live_actuation"
_SUPPORTED_TRIGGERS = frozenset(
    {"stable", "policy_drift", "control_loop_oscillation", "over_control"}
)
_SCENARIO_DRIFT_THRESHOLD = 0.08
_SCENARIO_OSCILLATION_THRESHOLD = 0.35
_SCENARIO_OVERCONTROL_THRESHOLD = 0.35


@dataclass(frozen=True)
class StrangeLoopDriftScenario:
    """Deterministic long-run action-history scenario for strange-loop review."""

    domain: str
    scenario_id: str
    description: str
    expected_trigger: str
    action_schedule: tuple[tuple[ControlAction, ...], ...]
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = _DRIFT_SCENARIO_BOUNDARY

    def scenario_hash(self) -> str:
        """Return a deterministic scenario hash over the full action schedule.

        Returns
        -------
        str
            Return a deterministic scenario hash over the full action schedule.
        """
        return _stable_hash(_scenario_payload(self))

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe long-run scenario record.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe long-run scenario record.
        """
        return {
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "description": self.description,
            "expected_trigger": self.expected_trigger,
            "step_count": len(self.action_schedule),
            "scenario_hash": self.scenario_hash(),
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "claim_boundary": self.claim_boundary,
            "action_schedule": [
                [_action_to_record(action) for action in bundle]
                for bundle in self.action_schedule
            ],
        }


@dataclass(frozen=True)
class StrangeLoopDriftScenarioResult:
    """Audit-ready result for one long-run strange-loop drift scenario."""

    domain: str
    scenario_id: str
    expected_trigger: str
    step_count: int
    max_drift_score: float
    max_oscillation_score: float
    max_overcontrol_score: float
    min_control_coherence: float
    triggered_recommendation_count: int
    final_recommended_knobs: tuple[str, ...]
    passed_expected_trigger: bool
    scenario_hash: str
    result_hash: str
    non_actuating: bool = True
    execution_disabled: bool = True
    claim_boundary: str = _DRIFT_SCENARIO_BOUNDARY

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe drift scenario result.

        Returns
        -------
        dict[str, object]
            Return a JSON-safe drift scenario result.
        """
        return {
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "expected_trigger": self.expected_trigger,
            "step_count": self.step_count,
            "max_drift_score": self.max_drift_score,
            "max_oscillation_score": self.max_oscillation_score,
            "max_overcontrol_score": self.max_overcontrol_score,
            "min_control_coherence": self.min_control_coherence,
            "triggered_recommendation_count": self.triggered_recommendation_count,
            "final_recommended_knobs": list(self.final_recommended_knobs),
            "passed_expected_trigger": self.passed_expected_trigger,
            "scenario_hash": self.scenario_hash,
            "result_hash": self.result_hash,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class StrangeLoopAssessment:
    """Audit-ready metrics for supervisor self-control dynamics."""

    control_phase: float
    control_coherence: float
    drift_score: float
    oscillation_score: float
    overcontrol_score: float
    recommended_actions: tuple[ControlAction, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable record for supervisor audit logs.

        Returns
        -------
        dict[str, object]
            Return a JSON-serialisable record for supervisor audit logs.
        """
        return {
            "control_phase": self.control_phase,
            "control_coherence": self.control_coherence,
            "drift_score": self.drift_score,
            "oscillation_score": self.oscillation_score,
            "overcontrol_score": self.overcontrol_score,
            "recommended_actions": [
                {
                    "knob": action.knob,
                    "scope": action.scope,
                    "value": action.value,
                    "ttl_s": action.ttl_s,
                    "justification": action.justification,
                }
                for action in self.recommended_actions
            ],
        }


def build_strange_loop_drift_scenarios() -> tuple[StrangeLoopDriftScenario, ...]:
    """Build deterministic long-run strange-loop drift review scenarios.

    Returns
    -------
    tuple[StrangeLoopDriftScenario, ...]
        Build deterministic long-run strange-loop drift review scenarios.
    """
    scenarios = (
        StrangeLoopDriftScenario(
            domain="power_grid",
            scenario_id="strange_loop_stable_frequency_trim_v1",
            description=(
                "Stable small frequency-regulation nudges should stay below "
                "drift, oscillation, and over-control review thresholds."
            ),
            expected_trigger="stable",
            action_schedule=tuple(
                (_review_action("K", 0.02, step),) for step in range(40)
            ),
        ),
        StrangeLoopDriftScenario(
            domain="cardiac_rhythm",
            scenario_id="strange_loop_monotone_policy_drift_v1",
            description=(
                "A monotone escalation of coupling proposals should be flagged "
                "as policy drift before it could become live actuation."
            ),
            expected_trigger="policy_drift",
            action_schedule=tuple(
                (_review_action("K", 0.03 + 0.12 * step, step),) for step in range(40)
            ),
        ),
        StrangeLoopDriftScenario(
            domain="traffic_flow",
            scenario_id="strange_loop_alternating_control_v1",
            description=(
                "Alternating sign control proposals should expose control-loop "
                "oscillation in the action-history embedding."
            ),
            expected_trigger="control_loop_oscillation",
            action_schedule=tuple(
                (_review_action("K", 0.22 if step % 2 == 0 else -0.22, step),)
                for step in range(40)
            ),
        ),
        StrangeLoopDriftScenario(
            domain="plasma_control",
            scenario_id="strange_loop_sustained_overcontrol_v1",
            description=(
                "Sustained large but non-oscillatory proposals should trigger "
                "over-control recommendations even when drift is low."
            ),
            expected_trigger="over_control",
            action_schedule=tuple(
                (
                    _review_action("K", 0.52, step),
                    _review_action("zeta", 0.08, step),
                )
                for step in range(40)
            ),
        ),
    )
    for scenario in scenarios:
        _validate_drift_scenario(scenario)
    return scenarios


def evaluate_strange_loop_drift_scenarios(
    scenarios: Sequence[StrangeLoopDriftScenario] | None = None,
) -> tuple[StrangeLoopDriftScenarioResult, ...]:
    """Evaluate long-run drift scenarios through ``StrangeLoopSupervisor``.

    Parameters
    ----------
    scenarios : Sequence[StrangeLoopDriftScenario] | None
        The drift scenarios to evaluate, or ``None`` for the defaults.

    Returns
    -------
    tuple[StrangeLoopDriftScenarioResult, ...]
        The drift-scenario results.
    """
    scenario_tuple = (
        build_strange_loop_drift_scenarios() if scenarios is None else tuple(scenarios)
    )
    results: list[StrangeLoopDriftScenarioResult] = []
    for scenario in scenario_tuple:
        _validate_drift_scenario(scenario)
        supervisor = StrangeLoopSupervisor(
            history_size=16,
            drift_threshold=_SCENARIO_DRIFT_THRESHOLD,
            oscillation_threshold=_SCENARIO_OSCILLATION_THRESHOLD,
            overcontrol_threshold=_SCENARIO_OVERCONTROL_THRESHOLD,
        )
        assessments = [
            supervisor.observe(list(bundle)) for bundle in scenario.action_schedule
        ]
        max_drift = max(assessment.drift_score for assessment in assessments)
        max_oscillation = max(
            assessment.oscillation_score for assessment in assessments
        )
        max_overcontrol = max(
            assessment.overcontrol_score for assessment in assessments
        )
        min_coherence = min(assessment.control_coherence for assessment in assessments)
        triggered_count = sum(
            1 for assessment in assessments if assessment.recommended_actions
        )
        final_knobs = tuple(
            action.knob for action in assessments[-1].recommended_actions
        )
        passed = _passes_expected_trigger(
            expected_trigger=scenario.expected_trigger,
            max_drift_score=max_drift,
            max_oscillation_score=max_oscillation,
            max_overcontrol_score=max_overcontrol,
            triggered_recommendation_count=triggered_count,
        )
        scenario_hash = scenario.scenario_hash()
        result_payload = {
            "domain": scenario.domain,
            "scenario_id": scenario.scenario_id,
            "expected_trigger": scenario.expected_trigger,
            "step_count": len(scenario.action_schedule),
            "max_drift_score": max_drift,
            "max_oscillation_score": max_oscillation,
            "max_overcontrol_score": max_overcontrol,
            "min_control_coherence": min_coherence,
            "triggered_recommendation_count": triggered_count,
            "final_recommended_knobs": list(final_knobs),
            "passed_expected_trigger": passed,
            "scenario_hash": scenario_hash,
            "non_actuating": True,
            "execution_disabled": True,
            "claim_boundary": _DRIFT_SCENARIO_BOUNDARY,
        }
        results.append(
            StrangeLoopDriftScenarioResult(
                domain=scenario.domain,
                scenario_id=scenario.scenario_id,
                expected_trigger=scenario.expected_trigger,
                step_count=len(scenario.action_schedule),
                max_drift_score=float(max_drift),
                max_oscillation_score=float(max_oscillation),
                max_overcontrol_score=float(max_overcontrol),
                min_control_coherence=float(min_coherence),
                triggered_recommendation_count=triggered_count,
                final_recommended_knobs=final_knobs,
                passed_expected_trigger=passed,
                scenario_hash=scenario_hash,
                result_hash=_stable_hash(result_payload),
            )
        )
    return tuple(results)


class StrangeLoopSupervisor:
    """Treat supervisor action history as a self-referential control channel.

    The monitor embeds each recent action bundle into the four native control
    knobs `(K, alpha, zeta, Psi)`. It then measures whether the supervisor is
    drifting, oscillating, or over-actuating and emits conservative damping
    recommendations for an outer policy gate to approve.
    """

    def __init__(
        self,
        *,
        history_size: int = 12,
        drift_threshold: float = 0.25,
        oscillation_threshold: float = 0.5,
        overcontrol_threshold: float = 0.2,
        damping_gain: float = 0.05,
        ttl_s: float = 3.0,
    ) -> None:
        if (
            isinstance(history_size, (bool, np.bool_))
            or not isinstance(history_size, Integral)
            or history_size < 2
        ):
            raise ValueError("history_size must be an integer >= 2")
        self._history: deque[FloatArray] = deque(maxlen=int(history_size))
        self._drift_threshold: float = _require_positive_real(
            drift_threshold, name="drift_threshold"
        )
        self._oscillation_threshold: float = _require_positive_real(
            oscillation_threshold, name="oscillation_threshold"
        )
        self._overcontrol_threshold: float = _require_positive_real(
            overcontrol_threshold, name="overcontrol_threshold"
        )
        self._damping_gain: float = _require_positive_real(
            damping_gain, name="damping_gain"
        )
        self._ttl_s: float = _require_positive_real(ttl_s, name="ttl_s")
        self.last_assessment: StrangeLoopAssessment | None = None

    def observe(self, actions: list[ControlAction]) -> StrangeLoopAssessment:
        """Record one supervisor action bundle and assess self-control state.

        Parameters
        ----------
        actions : list[ControlAction]
            The control actions to apply or assess.

        Returns
        -------
        StrangeLoopAssessment
            The self-control assessment for the action bundle.
        """
        vector = _actions_to_vector(actions)
        self._history.append(vector)
        assessment = self._assess()
        self.last_assessment = assessment
        return assessment

    def reset(self) -> None:
        """Clear action history and the cached assessment."""
        self._history.clear()
        self.last_assessment = None

    def _assess(self) -> StrangeLoopAssessment:
        """Return the drift/oscillation/over-control assessment for a step."""
        matrix = np.vstack(self._history)
        latest = matrix[-1]
        magnitude = float(np.linalg.norm(latest))
        control_phase = float(np.mod(np.arctan2(latest[3], latest[0]), 2.0 * np.pi))
        control_coherence = _control_coherence(matrix)
        drift_score = _drift_score(matrix)
        oscillation_score = _oscillation_score(matrix)
        overcontrol_score = magnitude
        recommendations = self._recommend(
            drift_score=drift_score,
            oscillation_score=oscillation_score,
            overcontrol_score=overcontrol_score,
        )
        return StrangeLoopAssessment(
            control_phase=control_phase,
            control_coherence=control_coherence,
            drift_score=drift_score,
            oscillation_score=oscillation_score,
            overcontrol_score=overcontrol_score,
            recommended_actions=tuple(recommendations),
        )

    def _recommend(
        self,
        *,
        drift_score: float,
        oscillation_score: float,
        overcontrol_score: float,
    ) -> list[ControlAction]:
        """Return the control recommendations for the assessed drift."""
        reasons = []
        if drift_score > self._drift_threshold:
            reasons.append("policy drift")
        if oscillation_score > self._oscillation_threshold:
            reasons.append("control-loop oscillation")
        if overcontrol_score > self._overcontrol_threshold:
            reasons.append("over-control")
        if not reasons:
            return []
        reason = ", ".join(reasons)
        return [
            ControlAction(
                knob="zeta",
                scope="global",
                value=self._damping_gain,
                ttl_s=self._ttl_s,
                justification=f"strange-loop damping: {reason}",
            ),
            ControlAction(
                knob="K",
                scope="global",
                value=-0.5 * self._damping_gain,
                ttl_s=self._ttl_s,
                justification=f"strange-loop coupling trim: {reason}",
            ),
        ]


def _actions_to_vector(actions: list[ControlAction]) -> FloatArray:
    """Return the actions flattened into a numeric vector."""
    vector = np.zeros(4, dtype=np.float64)
    for action in actions:
        if action.knob not in _KNOB_INDEX:
            continue
        if (
            isinstance(action.value, (bool, np.bool_))
            or not isinstance(action.value, Real)
            or not isfinite(action.value)
        ):
            raise ValueError("action values must be finite")
        vector[_KNOB_INDEX[action.knob]] += float(action.value)
    return vector


def _review_action(knob: str, value: float, step: int) -> ControlAction:
    """Return the review-only record for a recommended action."""
    return ControlAction(
        knob=knob,
        scope="global",
        value=value,
        ttl_s=1.0,
        justification=f"strange-loop long-run scenario step {step}",
    )


def _validate_drift_scenario(scenario: StrangeLoopDriftScenario) -> None:
    """Validate a strange-loop drift scenario, else raise."""
    if not isinstance(scenario.domain, str) or not scenario.domain.strip():
        raise ValueError("scenario domain must be a non-empty string")
    if not isinstance(scenario.scenario_id, str) or not scenario.scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")
    if scenario.expected_trigger not in _SUPPORTED_TRIGGERS:
        raise ValueError("expected_trigger is not supported")
    if not isinstance(scenario.action_schedule, tuple) or not scenario.action_schedule:
        raise ValueError("action_schedule must be a non-empty tuple")
    if len(scenario.action_schedule) < 32:
        raise ValueError("action_schedule must contain at least 32 long-run steps")
    if scenario.non_actuating is not True:
        raise ValueError("strange-loop drift scenarios must be non_actuating")
    if scenario.execution_disabled is not True:
        raise ValueError("strange-loop drift scenarios must disable execution")
    if scenario.claim_boundary != _DRIFT_SCENARIO_BOUNDARY:
        raise ValueError("strange-loop drift scenario claim boundary mismatch")
    for bundle in scenario.action_schedule:
        if not isinstance(bundle, tuple):
            raise ValueError("action_schedule entries must be tuples")
        if not bundle:
            raise ValueError("action_schedule entries must not be empty")
        for action in bundle:
            if not isinstance(action, ControlAction):
                raise ValueError("action_schedule entries must contain ControlAction")
            if action.knob not in _KNOB_INDEX:
                raise ValueError("scenario actions must use supported knobs")
            if action.scope != "global":
                raise ValueError("scenario actions must use global scope")
            if (
                isinstance(action.value, (bool, np.bool_))
                or not isinstance(action.value, Real)
                or not isfinite(action.value)
            ):
                raise ValueError("scenario action values must be finite")


def _passes_expected_trigger(
    *,
    expected_trigger: str,
    max_drift_score: float,
    max_oscillation_score: float,
    max_overcontrol_score: float,
    triggered_recommendation_count: int,
) -> bool:
    """Return whether the scenario hits its expected trigger."""
    if expected_trigger == "stable":
        return (
            triggered_recommendation_count == 0
            and max_drift_score < _SCENARIO_DRIFT_THRESHOLD
            and max_oscillation_score < _SCENARIO_OSCILLATION_THRESHOLD
            and max_overcontrol_score < _SCENARIO_OVERCONTROL_THRESHOLD
        )
    if expected_trigger == "policy_drift":
        return (
            max_drift_score > _SCENARIO_DRIFT_THRESHOLD
            and triggered_recommendation_count > 0
        )
    if expected_trigger == "control_loop_oscillation":
        return (
            max_oscillation_score > _SCENARIO_OSCILLATION_THRESHOLD
            and triggered_recommendation_count > 0
        )
    if expected_trigger == "over_control":
        return (
            max_overcontrol_score > _SCENARIO_OVERCONTROL_THRESHOLD
            and triggered_recommendation_count > 0
        )
    raise ValueError("expected_trigger is not supported")


def _action_to_record(action: ControlAction) -> dict[str, object]:
    """Return the JSON-safe record for an action."""
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": float(action.value),
        "ttl_s": float(action.ttl_s),
        "justification": action.justification,
    }


def _scenario_payload(scenario: StrangeLoopDriftScenario) -> dict[str, object]:
    """Return the canonical payload for a scenario."""
    return {
        "domain": scenario.domain,
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "expected_trigger": scenario.expected_trigger,
        "non_actuating": scenario.non_actuating,
        "execution_disabled": scenario.execution_disabled,
        "claim_boundary": scenario.claim_boundary,
        "action_schedule": [
            [_action_to_record(action) for action in bundle]
            for bundle in scenario.action_schedule
        ],
    }


def _stable_hash(payload: object) -> str:
    """Return a stable SHA-256 hash of the inputs."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(blob.encode("utf-8")).hexdigest()


def _require_positive_real(value: object, *, name: str) -> float:
    """Return ``value`` as a strictly positive finite real, else raise."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not isfinite(value)
        or value <= 0.0
    ):
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


def _control_coherence(matrix: FloatArray) -> float:
    """Return the control-coherence score for the trajectory."""
    norms = np.linalg.norm(matrix, axis=1)
    active = norms > 0.0
    if not np.any(active):
        return 1.0
    unit = matrix[active] / norms[active, None]
    mean_vector = np.mean(unit, axis=0)
    return float(np.clip(np.linalg.norm(mean_vector), 0.0, 1.0))


def _drift_score(matrix: FloatArray) -> float:
    """Return the policy-drift score for the trajectory."""
    if matrix.shape[0] < 2:
        return 0.0
    return float(np.mean(np.linalg.norm(np.diff(matrix, axis=0), axis=1)))


def _oscillation_score(matrix: FloatArray) -> float:
    """Return the control-loop oscillation score for the trajectory."""
    if matrix.shape[0] < 3:
        return 0.0
    deltas = np.diff(matrix, axis=0)
    signs = np.sign(deltas)
    flips = signs[1:] * signs[:-1] < 0.0
    active = np.abs(deltas[1:]) + np.abs(deltas[:-1]) > 0.0
    if not np.any(active):
        return 0.0
    return float(np.count_nonzero(flips & active) / np.count_nonzero(active))
