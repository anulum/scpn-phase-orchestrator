# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Runtime STL monitor

"""Signal Temporal Logic monitor backed by rtamt.

rtamt is an optional dependency: ``pip install rtamt``
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.constraints import ActionProjector
from scpn_phase_orchestrator.actuation.mapper import ActuationMapper, ControlAction
from scpn_phase_orchestrator.binding.types import ActuatorMapping

__all__ = [
    "HAS_RTAMT",
    "STLActionProjectionTemplate",
    "STLAutomatonState",
    "STLAutomatonTransition",
    "STLControllerCandidate",
    "STLControllerSynthesis",
    "STLClosedLoopSynthesisPlan",
    "STLMonitor",
    "STLMonitoringAutomaton",
    "STLProjectedActionPlan",
    "STLRuntimeActuationGate",
    "STLTraceResult",
    "project_stl_controller_candidates",
    "synthesise_stl_closed_loop_plan",
    "synthesise_stl_monitoring_automaton",
    "synthesise_stl_controller_candidates",
    "synthesize_stl_closed_loop_plan",
    "synthesize_stl_monitoring_automaton",
    "synthesize_stl_controller_candidates",
    "validate_stl_runtime_actuation_gate",
]

FloatArray: TypeAlias = NDArray[np.float64]

try:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        import rtamt

    HAS_RTAMT = True
except ImportError:
    rtamt = None
    HAS_RTAMT = False


_SIMPLE_SPEC_RE = re.compile(r"^(always|eventually)\s*\((.*)\)\s*$")
_PREDICATE_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(>=|>|<=|<|==)\s*([-+]?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class STLTraceResult:
    """Robustness summary for an STL monitor evaluation."""

    spec: str
    robustness: float
    satisfied: bool
    backend: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable STL audit payload."""
        return {
            "spec": self.spec,
            "robustness": self.robustness,
            "satisfied": self.satisfied,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class STLAutomatonState:
    """State in a synthesized STL monitoring automaton."""

    name: str
    accepting: bool
    violation: bool
    first_hit_index: int | None = None

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable automaton-state payload."""
        return {
            "name": self.name,
            "accepting": self.accepting,
            "violation": self.violation,
            "first_hit_index": self.first_hit_index,
        }


@dataclass(frozen=True)
class STLAutomatonTransition:
    """Trace-indexed transition taken by a runtime STL automaton."""

    source: str
    target: str
    time_index: int
    guard: str
    robustness: float

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable automaton-transition payload."""
        return {
            "source": self.source,
            "target": self.target,
            "time_index": self.time_index,
            "guard": self.guard,
            "robustness": self.robustness,
        }


@dataclass(frozen=True)
class STLMonitoringAutomaton:
    """Audit-ready runtime automaton synthesized from a simple STL monitor."""

    spec: str
    temporal_op: str
    signals: tuple[str, ...]
    states: tuple[STLAutomatonState, ...]
    transitions: tuple[STLAutomatonTransition, ...]
    robustness: float
    satisfied: bool
    backend: str = "builtin"

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable STL automaton audit payload."""
        return {
            "spec": self.spec,
            "temporal_op": self.temporal_op,
            "signals": list(self.signals),
            "states": [state.to_audit_record() for state in self.states],
            "transitions": [
                transition.to_audit_record() for transition in self.transitions
            ],
            "robustness": self.robustness,
            "satisfied": self.satisfied,
            "backend": self.backend,
        }


@dataclass(frozen=True)
class STLControllerCandidate:
    """Non-actuating controller candidate derived from an STL automaton."""

    signal: str
    action: str
    direction: str
    time_index: int
    robustness: float
    rationale: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable controller-candidate payload."""
        return {
            "signal": self.signal,
            "action": self.action,
            "direction": self.direction,
            "time_index": self.time_index,
            "robustness": self.robustness,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class STLControllerSynthesis:
    """Audit-ready, non-actuating controller synthesis proposal."""

    spec: str
    satisfied: bool
    actuating: bool
    source_backend: str
    candidates: tuple[STLControllerCandidate, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable controller-synthesis payload."""
        return {
            "spec": self.spec,
            "satisfied": self.satisfied,
            "actuating": self.actuating,
            "source_backend": self.source_backend,
            "candidates": [
                candidate.to_audit_record() for candidate in self.candidates
            ],
        }


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
        """Return a JSON-serialisable projected-action plan."""
        return {
            "spec": self.spec,
            "actuating": self.actuating,
            "approved_actions": [
                _control_action_record(action) for action in self.approved_actions
            ],
            "rejected_candidates": list(self.rejected_candidates),
        }


@dataclass(frozen=True)
class STLRuntimeActuationGate:
    """Non-actuating runtime-stack validation of projected STL actions.

    The gate verifies projected proposals against the same actuator mapping
    boundary used by runtime actuation, but it never enables execution. This
    makes the closed-loop STL plan auditable through the safety/actuation stack
    without converting a review artefact into a live controller command.
    """

    spec: str
    non_actuating: bool
    execution_disabled: bool
    accepted: bool
    action_count: int
    mapper_valid_action_count: int
    mapped_command_count: int
    commands: tuple[dict[str, object], ...]
    blocked_reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable runtime gate record."""
        return {
            "spec": self.spec,
            "non_actuating": self.non_actuating,
            "execution_disabled": self.execution_disabled,
            "accepted": self.accepted,
            "action_count": self.action_count,
            "mapper_valid_action_count": self.mapper_valid_action_count,
            "mapped_command_count": self.mapped_command_count,
            "commands": [dict(command) for command in self.commands],
            "blocked_reasons": list(self.blocked_reasons),
        }


@dataclass(frozen=True)
class STLClosedLoopSynthesisPlan:
    """Offline closed-loop STL controller plan for operator review.

    The plan binds the current monitor state, signal feedback surface, projected
    action proposals, and next review horizon. It is intentionally non-actuating:
    callers must still pass approved actions through runtime policy, safety, and
    actuation gates before any live controller can use them.
    """

    spec: str
    trace_length: int
    horizon_steps: int
    next_review_start_index: int
    next_review_end_index: int
    feedback_signals: tuple[str, ...]
    satisfied: bool
    actuating: bool
    synthesis: STLControllerSynthesis
    projected_plan: STLProjectedActionPlan
    runtime_gate: STLRuntimeActuationGate
    blocked_reasons: tuple[str, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-serialisable closed-loop synthesis plan."""
        return {
            "spec": self.spec,
            "trace_length": self.trace_length,
            "horizon_steps": self.horizon_steps,
            "next_review_start_index": self.next_review_start_index,
            "next_review_end_index": self.next_review_end_index,
            "feedback_signals": list(self.feedback_signals),
            "satisfied": self.satisfied,
            "actuating": self.actuating,
            "controller_synthesis": self.synthesis.to_audit_record(),
            "projected_action_plan": self.projected_plan.to_audit_record(),
            "runtime_actuation_gate": self.runtime_gate.to_audit_record(),
            "blocked_reasons": list(self.blocked_reasons),
        }


class STLMonitor:
    """Evaluate STL specifications against numeric traces.

    Parameters
    ----------
    spec : str
        An rtamt STL specification string, e.g.
        ``"always (sync_error <= 0.3)"``.
    """

    # IEC 62443 / Kuramoto safety: order-parameter must stay above threshold
    SYNC_THRESHOLD = "always (R >= 0.3)"
    # Coupling gain bounded to prevent instability; Kuramoto 1984
    COUPLING_BOUND = "always (K <= 10.0)"

    def __init__(self, spec: str) -> None:
        self._spec_str = spec
        self._simple = _parse_simple_spec(spec)
        self._stl = rtamt.StlDiscreteTimeSpecification() if rtamt is not None else None
        self._parsed = False

    def evaluate(self, trace: dict[str, list[float]]) -> float:
        """Return the robustness value of *spec* over *trace*.

        A positive value means the specification is satisfied; negative
        means violated.  The magnitude indicates how far from the boundary.
        """
        _validate_trace(trace)
        length = len(next(iter(trace.values())))

        if self._simple is not None:
            return _evaluate_simple(self._simple, trace)

        if self._stl is None:
            raise ImportError(
                "rtamt is required for this STL syntax. Install: pip install rtamt"
            )

        if not self._parsed:
            for name in trace:
                self._stl.declare_var(name, "float")
            self._stl.spec = self._spec_str
            self._stl.parse()
            self._parsed = True

        # rtamt discrete-time offline: flat lists per signal + 'time' key
        datasets: dict[str, list[float]] = {}
        for name in trace:
            datasets[name] = _trace_signal_array(name, trace).tolist()
        if "time" not in datasets:
            datasets["time"] = [float(t) for t in range(length)]

        robustness = self._stl.evaluate(datasets)
        # rtamt returns [[time, robustness], ...]; min is worst-case
        if isinstance(robustness, list) and robustness:
            return float(min(r[1] for r in robustness))
        return float(robustness)

    def evaluate_result(self, trace: dict[str, list[float]]) -> STLTraceResult:
        """Evaluate and return robustness plus audit metadata."""
        robustness = self.evaluate(trace)
        backend = "builtin" if self._simple is not None else "rtamt"
        return STLTraceResult(
            spec=self._spec_str,
            robustness=robustness,
            satisfied=robustness >= 0.0,
            backend=backend,
        )


def synthesise_stl_monitoring_automaton(
    spec: str,
    trace: dict[str, list[float]],
) -> STLMonitoringAutomaton:
    """Synthesize a trace automaton for builtin simple STL safety formulas.

    The synthesized automaton is intentionally conservative and audit-oriented:
    it records the state sequence taken by the monitor over the supplied trace
    for supported ``always (...)`` and ``eventually (...)`` conjunctions. More
    expressive STL remains delegated to ``rtamt`` for robustness evaluation.
    """
    _validate_trace(trace)
    parsed = _parse_simple_spec(spec)
    if parsed is None:
        raise ValueError(
            "monitoring automata synthesis supports builtin simple STL syntax only"
        )

    temporal_op, predicates = parsed
    pointwise = _pointwise_robustness(predicates, trace)
    guard = _format_predicate_guard(predicates)
    signals = tuple(dict.fromkeys(signal for signal, _, _ in predicates))

    if temporal_op == "always":
        robustness = float(np.min(pointwise))
        return _synthesise_always_automaton(
            spec=spec,
            signals=signals,
            pointwise=pointwise,
            guard=guard,
            robustness=robustness,
        )
    if temporal_op == "eventually":
        robustness = float(np.max(pointwise))
        return _synthesise_eventually_automaton(
            spec=spec,
            signals=signals,
            pointwise=pointwise,
            guard=guard,
            robustness=robustness,
        )
    raise ValueError(f"unsupported STL temporal operator {temporal_op!r}")


synthesize_stl_monitoring_automaton = synthesise_stl_monitoring_automaton


def synthesise_stl_controller_candidates(
    automaton: STLMonitoringAutomaton,
    trace: dict[str, list[float]],
    *,
    action_map: dict[str, str] | None = None,
) -> STLControllerSynthesis:
    """Synthesize non-actuating controller candidates from an STL automaton.

    The result is a review artefact, not a controller. It identifies the
    weakest predicate margin and proposes signal-level adjustment directions for
    supported builtin ``always`` and ``eventually`` monitors. Callers must still
    map candidates through policy, projection, safety, and actuation gates.
    """
    _validate_trace(trace)
    parsed = _parse_simple_spec(automaton.spec)
    if parsed is None:
        raise ValueError("controller synthesis supports builtin simple STL syntax only")
    temporal_op, predicates = parsed
    if temporal_op != automaton.temporal_op:
        raise ValueError("automaton temporal operator does not match its STL spec")
    index = _controller_focus_index(automaton, predicates, trace)
    candidates = tuple(
        candidate
        for predicate in predicates
        if (
            candidate := _candidate_for_predicate(
                predicate,
                trace,
                time_index=index,
                action_map=action_map or {},
            )
        )
        is not None
    )
    if automaton.satisfied:
        candidates = ()
    return STLControllerSynthesis(
        spec=automaton.spec,
        satisfied=automaton.satisfied,
        actuating=False,
        source_backend=automaton.backend,
        candidates=candidates,
    )


synthesize_stl_controller_candidates = synthesise_stl_controller_candidates


def project_stl_controller_candidates(
    synthesis: STLControllerSynthesis,
    templates: Sequence[STLActionProjectionTemplate],
) -> STLProjectedActionPlan:
    """Project STL candidates into bounded, non-actuating action proposals.

    Only candidates with an explicit policy-approved projection template are
    converted. Projection uses the standard :class:`ActionProjector`; the
    returned plan remains a review artefact with ``actuating=False``.
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


def validate_stl_runtime_actuation_gate(
    projected_plan: STLProjectedActionPlan,
    templates: Sequence[STLActionProjectionTemplate],
) -> STLRuntimeActuationGate:
    """Validate projected STL actions through runtime actuation mapping.

    This is an audit gate only: returned commands are deterministic evidence
    that proposals can be represented by the configured actuation stack, while
    ``execution_disabled`` and ``non_actuating`` remain true for every outcome.
    Invalid runtime knobs, missing mappings, and empty projected plans fail
    closed with explicit blocker reasons.
    """
    actions = projected_plan.approved_actions
    if not actions:
        return STLRuntimeActuationGate(
            spec=projected_plan.spec,
            non_actuating=True,
            execution_disabled=True,
            accepted=False,
            action_count=0,
            mapper_valid_action_count=0,
            mapped_command_count=0,
            commands=(),
            blocked_reasons=("no_runtime_actions",),
        )

    templates_by_surface = {
        (template.knob, template.scope): template for template in templates
    }
    mappings: list[ActuatorMapping] = []
    blocked_reasons: list[str] = []
    for action in actions:
        template = templates_by_surface.get((action.knob, action.scope))
        if template is None:
            blocked_reasons.append("actuation_template_missing")
            continue
        try:
            mappings.append(
                ActuatorMapping(
                    name=_runtime_actuator_name(template.knob, template.scope),
                    knob=template.knob,
                    scope=template.scope,
                    limits=template.value_bounds,
                    rate_limit_per_step=template.rate_limit,
                )
            )
        except (TypeError, ValueError):
            blocked_reasons.append("actuation_mapper_rejected_template")

    if not mappings:
        return STLRuntimeActuationGate(
            spec=projected_plan.spec,
            non_actuating=True,
            execution_disabled=True,
            accepted=False,
            action_count=len(actions),
            mapper_valid_action_count=0,
            mapped_command_count=0,
            commands=(),
            blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
        )

    try:
        mapper = ActuationMapper(mappings)
    except ValueError:
        return STLRuntimeActuationGate(
            spec=projected_plan.spec,
            non_actuating=True,
            execution_disabled=True,
            accepted=False,
            action_count=len(actions),
            mapper_valid_action_count=0,
            mapped_command_count=0,
            commands=(),
            blocked_reasons=("actuation_mapper_rejected_template",),
        )

    valid_actions = tuple(
        action for action in actions if mapper.validate_action(action)
    )
    if len(valid_actions) != len(actions):
        blocked_reasons.append("runtime_action_validation_failed")
    commands = tuple(
        _normalise_runtime_command(command)
        for command in mapper.map_actions(list(valid_actions))
    )
    if len(commands) != len(valid_actions):
        blocked_reasons.append("actuation_mapping_incomplete")
    accepted = (
        len(valid_actions) == len(actions)
        and len(commands) == len(actions)
        and not blocked_reasons
    )
    return STLRuntimeActuationGate(
        spec=projected_plan.spec,
        non_actuating=True,
        execution_disabled=True,
        accepted=accepted,
        action_count=len(actions),
        mapper_valid_action_count=len(valid_actions),
        mapped_command_count=len(commands),
        commands=commands,
        blocked_reasons=tuple(dict.fromkeys(blocked_reasons)),
    )


def synthesise_stl_closed_loop_plan(
    automaton: STLMonitoringAutomaton,
    trace: dict[str, list[float]],
    templates: Sequence[STLActionProjectionTemplate],
    *,
    horizon_steps: int = 1,
    action_map: dict[str, str] | None = None,
) -> STLClosedLoopSynthesisPlan:
    """Build an offline closed-loop STL controller plan.

    The function synthesizes candidates from the current STL automaton, projects
    them through explicit policy templates, and records the future feedback
    review window. It does not mutate runtime state or permit actuation.
    """
    _validate_horizon_steps(horizon_steps)
    _validate_trace(trace)
    trace_length = len(next(iter(trace.values())))
    synthesis = synthesise_stl_controller_candidates(
        automaton,
        trace,
        action_map=action_map,
    )
    projected_plan = project_stl_controller_candidates(synthesis, templates)
    runtime_gate = validate_stl_runtime_actuation_gate(projected_plan, templates)
    blocked_reasons: list[str] = []
    if synthesis.satisfied:
        blocked_reasons.append("stl_satisfied_no_control_needed")
    if synthesis.candidates and not projected_plan.approved_actions:
        blocked_reasons.append("no_projected_actions")
    if projected_plan.rejected_candidates:
        blocked_reasons.append("unprojected_candidates")
    return STLClosedLoopSynthesisPlan(
        spec=automaton.spec,
        trace_length=trace_length,
        horizon_steps=horizon_steps,
        next_review_start_index=trace_length,
        next_review_end_index=trace_length + horizon_steps - 1,
        feedback_signals=automaton.signals,
        satisfied=synthesis.satisfied,
        actuating=False,
        synthesis=synthesis,
        projected_plan=projected_plan,
        runtime_gate=runtime_gate,
        blocked_reasons=tuple(blocked_reasons),
    )


synthesize_stl_closed_loop_plan = synthesise_stl_closed_loop_plan


def _parse_simple_spec(spec: str) -> tuple[str, list[tuple[str, str, float]]] | None:
    match = _SIMPLE_SPEC_RE.match(spec.strip())
    if match is None:
        return None
    temporal_op = match.group(1)
    predicates: list[tuple[str, str, float]] = []
    for raw_predicate in re.split(r"\s+(?:and|&&)\s+", match.group(2)):
        predicate_match = _PREDICATE_RE.match(raw_predicate)
        if predicate_match is None:
            return None
        signal, op, threshold = predicate_match.groups()
        predicates.append((signal, op, float(threshold)))
    return temporal_op, predicates


def _evaluate_simple(
    parsed: tuple[str, list[tuple[str, str, float]]],
    trace: dict[str, list[float]],
) -> float:
    temporal_op, predicates = parsed
    pointwise = _pointwise_robustness(predicates, trace)
    if temporal_op == "always":
        return float(np.min(pointwise))
    if temporal_op == "eventually":
        return float(np.max(pointwise))
    raise ValueError(f"unsupported STL temporal operator {temporal_op!r}")


def _validate_trace(trace: dict[str, list[float]]) -> None:
    if not trace:
        raise ValueError("trace must contain at least one signal")

    lengths = {len(v) for v in trace.values()}
    if len(lengths) != 1:
        raise ValueError("all signals in trace must have equal length")

    length = lengths.pop()
    if length == 0:
        raise ValueError("trace signals must be non-empty")
    for signal in trace:
        _trace_signal_array(signal, trace)


def _trace_signal_array(signal: str, trace: dict[str, list[float]]) -> FloatArray:
    if signal not in trace:
        raise ValueError(f"trace missing signal {signal!r}")

    raw = np.asarray(trace[signal], dtype=object)
    if raw.ndim != 1:
        raise ValueError(f"trace signal {signal!r} must be a 1-D numeric signal")
    if _contains_boolean_alias(raw):
        raise ValueError(f"trace signal {signal!r} must not contain boolean values")
    if _contains_complex_alias(raw):
        raise ValueError(f"trace signal {signal!r} must contain real-valued samples")

    try:
        values = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"trace signal {signal!r} must be numeric") from exc

    if not np.all(np.isfinite(values)):
        raise ValueError(f"trace signal {signal!r} must contain only finite values")
    return np.ascontiguousarray(values, dtype=np.float64)


def _contains_boolean_alias(raw: NDArray[np.generic]) -> bool:
    if raw.dtype == np.bool_:
        return True
    if raw.dtype == object:
        return any(isinstance(item, (bool, np.bool_)) for item in raw.flat)
    return False


def _contains_complex_alias(raw: NDArray[np.generic]) -> bool:
    if np.iscomplexobj(raw):
        return True
    if raw.dtype == object:
        return any(isinstance(item, (complex, np.complexfloating)) for item in raw.flat)
    return False


def _validate_horizon_steps(horizon_steps: int) -> None:
    if (
        isinstance(horizon_steps, bool)
        or not isinstance(horizon_steps, int)
        or horizon_steps <= 0
    ):
        raise ValueError("closed-loop horizon_steps must be a positive integer")


def _pointwise_robustness(
    predicates: list[tuple[str, str, float]],
    trace: dict[str, list[float]],
) -> FloatArray:
    per_predicate = [
        _predicate_robustness(signal, op, threshold, trace)
        for signal, op, threshold in predicates
    ]
    pointwise: FloatArray = np.asarray(
        np.min(np.vstack(per_predicate), axis=0),
        dtype=np.float64,
    )
    return pointwise


def _predicate_robustness(
    signal: str,
    op: str,
    threshold: float,
    trace: dict[str, list[float]],
) -> FloatArray:
    values = _trace_signal_array(signal, trace)
    if op in {">=", ">"}:
        return values - threshold
    if op in {"<=", "<"}:
        return threshold - values
    if op == "==":
        return -np.abs(values - threshold)
    raise ValueError(f"unsupported STL comparison operator {op!r}")


def _format_predicate_guard(predicates: list[tuple[str, str, float]]) -> str:
    return " and ".join(
        f"{signal} {op} {_format_threshold(threshold)}"
        for signal, op, threshold in predicates
    )


def _format_threshold(threshold: float) -> str:
    if threshold.is_integer():
        return str(int(threshold))
    return f"{threshold:g}"


def _controller_focus_index(
    automaton: STLMonitoringAutomaton,
    predicates: list[tuple[str, str, float]],
    trace: dict[str, list[float]],
) -> int:
    pointwise = _pointwise_robustness(predicates, trace)
    if automaton.temporal_op == "always":
        return int(np.argmin(pointwise))
    if automaton.temporal_op == "eventually":
        return int(np.argmax(pointwise))
    raise ValueError(f"unsupported STL temporal operator {automaton.temporal_op!r}")


def _candidate_for_predicate(
    predicate: tuple[str, str, float],
    trace: dict[str, list[float]],
    *,
    time_index: int,
    action_map: dict[str, str],
) -> STLControllerCandidate | None:
    signal, op, threshold = predicate
    robustness = float(_predicate_robustness(signal, op, threshold, trace)[time_index])
    if robustness >= 0.0:
        return None
    direction = _controller_direction(op)
    action = action_map.get(signal, f"{direction}_{signal}")
    rationale = (
        f"{signal} {op} {_format_threshold(threshold)} violated at "
        f"t={time_index} with robustness {robustness:g}"
    )
    return STLControllerCandidate(
        signal=signal,
        action=action,
        direction=direction,
        time_index=time_index,
        robustness=robustness,
        rationale=rationale,
    )


def _controller_direction(op: str) -> str:
    if op in {">=", ">"}:
        return "increase"
    if op in {"<=", "<"}:
        return "decrease"
    if op == "==":
        return "restore"
    raise ValueError(f"unsupported STL comparison operator {op!r}")


def _candidate_to_control_action(
    candidate: STLControllerCandidate,
    template: STLActionProjectionTemplate,
) -> ControlAction:
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
    return {
        "knob": action.knob,
        "scope": action.scope,
        "value": action.value,
        "ttl_s": action.ttl_s,
        "justification": action.justification,
    }


def _normalise_runtime_command(command: dict[str, object]) -> dict[str, object]:
    return {
        "actuator": command["actuator"],
        "knob": command["knob"],
        "scope": command["scope"],
        "value": command["value"],
        "ttl_s": command["ttl_s"],
    }


def _runtime_actuator_name(knob: str, scope: str) -> str:
    surface = re.sub(r"[^A-Za-z0-9_]+", "_", f"{knob}_{scope}").strip("_")
    return f"stl_runtime_{surface or 'action'}"


def _require_non_empty(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _synthesise_always_automaton(
    *,
    spec: str,
    signals: tuple[str, ...],
    pointwise: FloatArray,
    guard: str,
    robustness: float,
) -> STLMonitoringAutomaton:
    transitions: list[STLAutomatonTransition] = []
    state = "holding"
    first_violation_index: int | None = None
    for time_index, value in enumerate(pointwise):
        target = "violated" if value < 0.0 else state
        if target == "violated" and first_violation_index is None:
            first_violation_index = time_index
        transitions.append(
            STLAutomatonTransition(
                source=state,
                target=target,
                time_index=time_index,
                guard=guard,
                robustness=float(value),
            )
        )
        state = target

    states = (
        STLAutomatonState(
            name="holding",
            accepting=first_violation_index is None,
            violation=False,
        ),
        STLAutomatonState(
            name="violated",
            accepting=False,
            violation=first_violation_index is not None,
            first_hit_index=first_violation_index,
        ),
    )
    return STLMonitoringAutomaton(
        spec=spec,
        temporal_op="always",
        signals=signals,
        states=states,
        transitions=tuple(transitions),
        robustness=robustness,
        satisfied=robustness >= 0.0,
    )


def _synthesise_eventually_automaton(
    *,
    spec: str,
    signals: tuple[str, ...],
    pointwise: FloatArray,
    guard: str,
    robustness: float,
) -> STLMonitoringAutomaton:
    transitions: list[STLAutomatonTransition] = []
    state = "pending"
    first_satisfaction_index: int | None = None
    for time_index, value in enumerate(pointwise):
        target = "satisfied" if value >= 0.0 else state
        if target == "satisfied" and first_satisfaction_index is None:
            first_satisfaction_index = time_index
        transitions.append(
            STLAutomatonTransition(
                source=state,
                target=target,
                time_index=time_index,
                guard=guard,
                robustness=float(value),
            )
        )
        state = target

    states = (
        STLAutomatonState(
            name="pending",
            accepting=False,
            violation=first_satisfaction_index is None,
        ),
        STLAutomatonState(
            name="satisfied",
            accepting=first_satisfaction_index is not None,
            violation=False,
            first_hit_index=first_satisfaction_index,
        ),
    )
    return STLMonitoringAutomaton(
        spec=spec,
        temporal_op="eventually",
        signals=signals,
        states=states,
        transitions=tuple(transitions),
        robustness=robustness,
        satisfied=robustness >= 0.0,
    )
