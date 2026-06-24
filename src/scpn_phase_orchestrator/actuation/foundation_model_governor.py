# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Foundation-model actuation governor

"""Govern an externally-proposed control through SPO's safety envelope.

A foundation model (a Panda-class forecaster, a learned policy, any external
controller) may out-predict SPO's own observer, but it offers no safety guarantee,
no bound on its output, and no audit trail. This module is the harness that makes
such an advisory proposal deployable: :class:`FoundationModelGovernor` takes the
proposal as an *advisory* scalar control and admits only a safe action, by running
it through the trust stack SPO already owns —

1. **actuator bounds** — clamp to ``[control_lo, control_hi]``;
2. **rate limit** — bound the step against the last admitted action
   (``|u − u_prev| ≤ max_rate``);
3. **Control Barrier Function** — project through an optional
   :class:`~scpn_phase_orchestrator.actuation.control_barrier.ControlBarrierFilter`
   so the admitted action keeps the system inside the certified forward-invariant
   safe set, and flag when the state has already left it (``h(x) < 0``);
4. **safety predicates** — veto the action if any supplied predicate (an STL-derived
   check, an operating-envelope rule, …) rejects it.

Every decision is sealed into a content-addressed :class:`GovernorDecision` (the
same canonical-JSON SHA-256 the assurance bundle uses), so the governance record
is tamper-evident and the chain of which envelope stages touched the proposal is
explicit. The governor competes on *governance, not prediction*: it never
forecasts and it is **review-only** — it returns a safe action and a decision; it
never actuates a plant.

References
----------
* EU AI Act 2024/1689 Art. 14 (human oversight) and Art. 12 (logging /
  traceability) — the review-only, audited posture this envelope implements.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.control_barrier import ControlBarrierFilter

FloatArray: TypeAlias = NDArray[np.float64]

#: A named safety check: ``(action, state) -> (ok, reason)``. ``ok=False`` vetoes.
SafetyPredicate: TypeAlias = Callable[[float, FloatArray], "tuple[bool, str]"]

__all__ = [
    "ADMITTED",
    "CONSTRAINED",
    "REJECTED",
    "FoundationModelGovernor",
    "GovernorDecision",
    "SafetyPredicate",
]

#: The proposal passed every envelope stage unchanged.
ADMITTED = "admitted"
#: A safe action is admitted, but the envelope modified the proposal.
CONSTRAINED = "constrained"
#: No safe action; the fallback is held and the violations are recorded.
REJECTED = "rejected"

#: Envelope stage labels recorded in :attr:`GovernorDecision.stages_applied`.
_BOUNDS = "bounds"
_RATE_LIMIT = "rate_limit"
_CBF = "cbf"


@dataclass(frozen=True)
class GovernorDecision:
    """The audited outcome of governing one proposed control.

    Attributes
    ----------
    proposed_action : float
        The advisory action as received from the external source.
    admitted_action : float
        The safe action the governor admits (the reviewed output).
    status : str
        :data:`ADMITTED`, :data:`CONSTRAINED`, or :data:`REJECTED`.
    stages_applied : tuple[str, ...]
        Envelope stages that modified the proposal, in order (``bounds``,
        ``rate_limit``, ``cbf``).
    violations : tuple[str, ...]
        Reasons the action was rejected; empty unless ``status`` is
        :data:`REJECTED`.
    barrier_value : float | None
        The barrier value ``h(state)`` when a Control Barrier Function is
        configured, otherwise ``None``; a negative value means the state has left
        the certified safe set.
    content_hash : str
        SHA-256 of the canonical decision record (excluding this field); computed
        on construction.
    """

    proposed_action: float
    admitted_action: float
    status: str
    stages_applied: tuple[str, ...]
    violations: tuple[str, ...]
    barrier_value: float | None
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "content_hash", _canonical_hash(self._canonical_payload())
        )

    def _canonical_payload(self) -> dict[str, object]:
        """Return the canonical payload for a governed decision."""
        return {
            "proposed_action": self.proposed_action,
            "admitted_action": self.admitted_action,
            "status": self.status,
            "stages_applied": list(self.stages_applied),
            "violations": list(self.violations),
            "barrier_value": self.barrier_value,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe mapping of the decision.

        Returns
        -------
        dict[str, object]
            The canonical payload plus the computed ``content_hash``.
        """
        record = self._canonical_payload()
        record["content_hash"] = self.content_hash
        return record


@dataclass(frozen=True)
class FoundationModelGovernor:
    """Admit an externally-proposed scalar control through the safety envelope.

    Attributes
    ----------
    control_lo : float
        Lower actuator bound.
    control_hi : float
        Upper actuator bound (``> control_lo``).
    max_rate : float
        Maximum admitted change per call, ``|u − u_prev|`` (``> 0``).
    barrier_filter : ControlBarrierFilter | None
        Optional Control Barrier Function gate; ``None`` skips the CBF stage.
    safety_predicates : tuple[tuple[str, SafetyPredicate], ...]
        Named ``(label, predicate)`` safety checks run on the candidate action;
        any predicate returning ``ok=False`` rejects the action.
    hold_on_reject : bool
        On rejection, hold the previous action (``True``) or fall back to the
        bound-clamped neutral action ``0`` (``False``).
    """

    control_lo: float
    control_hi: float
    max_rate: float
    barrier_filter: ControlBarrierFilter | None = None
    safety_predicates: tuple[tuple[str, SafetyPredicate], ...] = ()
    hold_on_reject: bool = True

    def __post_init__(self) -> None:
        lo = _real_scalar(self.control_lo, "control_lo")
        hi = _real_scalar(self.control_hi, "control_hi")
        rate = _real_scalar(self.max_rate, "max_rate")
        if not hi > lo:
            raise ValueError("control_hi must be greater than control_lo")
        if rate <= 0.0:
            raise ValueError("max_rate must be positive")

    def govern(
        self,
        proposed_action: float,
        state: FloatArray,
        drift: FloatArray,
        *,
        previous_action: float = 0.0,
    ) -> GovernorDecision:
        """Govern one advisory control proposal and return an audited decision.

        Parameters
        ----------
        proposed_action : float
            The advisory action from the external source (e.g. a foundation
            model), in actuator units.
        state : FloatArray
            Current system state passed to the Control Barrier Function and the
            safety predicates.
        drift : FloatArray
            Uncontrolled state drift ``f(x)`` passed to the Control Barrier
            Function.
        previous_action : float
            The last admitted action, used for the rate limit and as the
            rejection fallback when ``hold_on_reject`` is set.

        Returns
        -------
        GovernorDecision
            The admitted action, status, applied stages, any violations, the
            barrier value, and a sealing hash.

        Raises
        ------
        ValueError
            If the proposal, previous action, or state arrays are not finite
            reals of the expected shape.
        """
        proposed = _real_scalar(proposed_action, "proposed_action")
        previous = _real_scalar(previous_action, "previous_action")
        state_vec = _finite_vector(state, "state")
        drift_vec = _finite_vector(drift, "drift")

        stages: list[str] = []
        action = proposed
        clamped = min(max(action, self.control_lo), self.control_hi)
        if clamped != action:
            stages.append(_BOUNDS)
            action = clamped
        limited = self._rate_limit(action, previous)
        if limited != action:
            stages.append(_RATE_LIMIT)
            action = limited

        violations: list[str] = []
        barrier_value = self._apply_barrier(action, state_vec, drift_vec, stages)
        if barrier_value is not None:
            action = barrier_value[0]
            if barrier_value[1] < 0.0:
                violations.append("barrier: state outside certified safe set (h<0)")

        violations.extend(self._predicate_violations(action, state_vec))
        barrier = None if barrier_value is None else barrier_value[1]
        return self._decide(proposed, action, previous, stages, violations, barrier)

    def _rate_limit(self, action: float, previous: float) -> float:
        """Apply the per-channel rate limit to a proposed control."""
        delta = action - previous
        if abs(delta) > self.max_rate:
            return previous + math.copysign(self.max_rate, delta)
        return action

    def _apply_barrier(
        self, action: float, state: FloatArray, drift: FloatArray, stages: list[str]
    ) -> tuple[float, float] | None:
        """Apply the control-barrier safety filter to a proposed control."""
        if self.barrier_filter is None:
            return None
        value = self.barrier_filter.barrier.value(state)
        safe_action, modified = self.barrier_filter.filter(action, state, drift)
        if modified:
            stages.append(_CBF)
        return safe_action, value

    def _predicate_violations(self, action: float, state: FloatArray) -> list[str]:
        """Return the safety-predicate violations for a state and control."""
        violations: list[str] = []
        for label, predicate in self.safety_predicates:
            ok, reason = predicate(action, state)
            if not ok:
                violations.append(f"{label}: {reason}")
        return violations

    def _decide(
        self,
        proposed: float,
        action: float,
        previous: float,
        stages: list[str],
        violations: list[str],
        barrier_value: float | None,
    ) -> GovernorDecision:
        """Return the governed admit/modify/reject decision for a proposal."""
        if violations:
            fallback = previous if self.hold_on_reject else 0.0
            admitted = min(max(fallback, self.control_lo), self.control_hi)
            status = REJECTED
        else:
            admitted = action
            status = CONSTRAINED if stages else ADMITTED
        return GovernorDecision(
            proposed_action=proposed,
            admitted_action=admitted,
            status=status,
            stages_applied=tuple(stages),
            violations=tuple(violations),
            barrier_value=barrier_value,
        )


def _canonical_hash(record: dict[str, object]) -> str:
    """Return the SHA-256 of ``record`` under canonical (sorted, compact) JSON."""
    serialised = json.dumps(record, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


def _real_scalar(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar, else raise ``ValueError``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real, got {value!r}")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return scalar


def _finite_vector(value: Sequence[float] | FloatArray, name: str) -> FloatArray:
    """Return ``value`` as a validated finite vector, else raise."""
    raw = np.asarray(value)
    if raw.dtype == np.bool_:
        raise ValueError(f"{name} must not contain boolean values")
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = raw.astype(np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a real float array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array, dtype=np.float64)
