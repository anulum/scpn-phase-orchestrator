# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Supervisor CBF action admission

"""Certificate-bound CBF admission for supervisor ``ControlAction`` proposals.

``SupervisorPolicy`` emits bounded, non-actuating action proposals. This module
adds an optional admission layer for deployments that have a verified neural
Control Barrier Function (CBF): matching actions are passed through the existing
certificate-bound CBF governor, and every decision emits a deterministic SMT-LIB
admission artefact. The artefact captures the exact scalar CBF half-space,
control bounds, selected action, and filter/certificate digests; it does not run
Z3 locally and does not grant actuation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from math import isfinite
from numbers import Real
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.actuation.control_barrier import (
    BarrierCertificate,
    ControlBarrierFilter,
)
from scpn_phase_orchestrator.actuation.foundation_model_governor import (
    FoundationModelGovernor,
)
from scpn_phase_orchestrator.actuation.mapper import ControlAction
from scpn_phase_orchestrator.monitor.boundaries import BoundaryState
from scpn_phase_orchestrator.supervisor.formal_export import FormalTextArtifact
from scpn_phase_orchestrator.upde.metrics import UPDEState

FloatArray: TypeAlias = NDArray[np.float64]

__all__ = [
    "PolicyCBFAdmissionGate",
    "PolicyCBFAdmissionRecord",
    "PolicyCBFAdmissionResult",
    "PolicyCBFChannel",
]


@dataclass(frozen=True)
class PolicyCBFAdmissionRecord:
    """Audit record for one CBF-admitted supervisor action.

    Attributes
    ----------
    knob, scope : str
        Action channel admitted by the CBF gate.
    proposed_value : float
        Original supervisor proposal.
    admitted_value : float
        Value admitted by the CBF governor.
    status : str
        Governor status: ``admitted``, ``constrained``, or ``rejected``.
    stages_applied : tuple[str, ...]
        Envelope stages that modified the proposal.
    violations : tuple[str, ...]
        Rejection reasons, if any.
    barrier_value : float | None
        Current CBF value ``h(x)``.
    filter_digest : str
        Digest of the CBF filter configuration.
    certificate_verification_digest : str
        Digest of the certificate envelope used to validate the filter.
    smt_artifact : FormalTextArtifact
        Deterministic SMT-LIB admission artefact for this decision.
    smt_artifact_hash : str
        SHA-256 hash of :attr:`smt_artifact`.
    content_hash : str
        SHA-256 hash of the audit record excluding the SMT text.
    """

    knob: str
    scope: str
    proposed_value: float
    admitted_value: float
    status: str
    stages_applied: tuple[str, ...]
    violations: tuple[str, ...]
    barrier_value: float | None
    filter_digest: str
    certificate_verification_digest: str
    smt_artifact: FormalTextArtifact
    smt_artifact_hash: str
    content_hash: str = field(default="", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_hash", _sha256_json(self._payload()))

    def _payload(self) -> dict[str, object]:
        """Return the deterministic content payload for hashing."""
        return {
            "knob": self.knob,
            "scope": self.scope,
            "proposed_value": self.proposed_value,
            "admitted_value": self.admitted_value,
            "status": self.status,
            "stages_applied": list(self.stages_applied),
            "violations": list(self.violations),
            "barrier_value": self.barrier_value,
            "filter_digest": self.filter_digest,
            "certificate_verification_digest": self.certificate_verification_digest,
            "smt_artifact_type": self.smt_artifact.artifact_type,
            "smt_artifact_hash": self.smt_artifact_hash,
        }

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe CBF admission audit record.

        Returns
        -------
        dict[str, object]
            Admission decision, barrier/certificate digests, SMT artefact hash,
            and deterministic content hash.
        """
        record = self._payload()
        record["content_hash"] = self.content_hash
        return record


@dataclass(frozen=True)
class PolicyCBFAdmissionResult:
    """CBF admission output for a batch of supervisor actions."""

    actions: tuple[ControlAction, ...]
    records: tuple[PolicyCBFAdmissionRecord, ...]

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe batch admission record.

        Returns
        -------
        dict[str, object]
            Admitted action count plus per-action CBF admission records.
        """
        return {
            "actions": len(self.actions),
            "records": [record.to_audit_record() for record in self.records],
        }


@dataclass(frozen=True)
class PolicyCBFChannel:
    """Certificate-bound CBF admission channel for one action knob/scope.

    Parameters
    ----------
    knob, scope : str
        Action selector. Only exact ``(knob, scope)`` matches are admitted by
        this channel.
    barrier_filter : ControlBarrierFilter
        Verified CBF filter for the scalar action value.
    barrier_certificate : BarrierCertificate
        Certificate that validates :attr:`barrier_filter`.
    state_metrics : tuple[str, ...]
        Names of UPDE/boundary metrics used as the CBF state vector.
    drift_bounds : tuple[float, ...]
        Deterministic drift vector supplied to the CBF filter for admission.
    previous_action : float
        Held fallback and rate-limit reference for rejected decisions.
    max_rate : float | None
        Optional per-call rate limit. ``None`` uses the full control span.
    """

    knob: str
    scope: str
    barrier_filter: ControlBarrierFilter
    barrier_certificate: BarrierCertificate
    state_metrics: tuple[str, ...]
    drift_bounds: tuple[float, ...]
    previous_action: float = 0.0
    max_rate: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.knob, str) or not self.knob.strip():
            raise ValueError("knob must be a non-empty string")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("scope must be a non-empty string")
        if len(self.state_metrics) != self.barrier_filter.barrier.input_dim:
            raise ValueError("state_metrics length must match CBF input dimension")
        if len(self.drift_bounds) != self.barrier_filter.barrier.input_dim:
            raise ValueError("drift_bounds length must match CBF input dimension")
        self.barrier_filter.validate_certificate(self.barrier_certificate)
        _finite_real(self.previous_action, "previous_action")
        if self.max_rate is not None:
            rate = _finite_real(self.max_rate, "max_rate")
            if rate <= 0.0:
                raise ValueError("max_rate must be positive")
        _finite_vector(self.drift_bounds, "drift_bounds")

    def matches(self, action: ControlAction) -> bool:
        """Return whether ``action`` belongs to this CBF channel.

        Parameters
        ----------
        action : ControlAction
            Supervisor action proposal to compare with this channel's selector.

        Returns
        -------
        bool
            ``True`` when both knob and scope match exactly.
        """
        return action.knob == self.knob and action.scope == self.scope

    def admit(
        self,
        action: ControlAction,
        upde_state: UPDEState,
        boundary_state: BoundaryState,
    ) -> tuple[ControlAction, PolicyCBFAdmissionRecord]:
        """Admit one matching action through the verified CBF governor.

        Parameters
        ----------
        action : ControlAction
            Supervisor action proposal. It must match :meth:`matches`.
        upde_state : UPDEState
            Current UPDE metrics used to build the CBF state vector.
        boundary_state : BoundaryState
            Current boundary metrics used to build the CBF state vector.

        Returns
        -------
        tuple[ControlAction, PolicyCBFAdmissionRecord]
            The admitted action and its deterministic audit record.

        Raises
        ------
        ValueError
            If ``action`` does not match this channel.
        """
        if not self.matches(action):
            raise ValueError("action does not match this CBF channel")
        state = _state_vector(self.state_metrics, upde_state, boundary_state)
        drift = _finite_vector(self.drift_bounds, "drift_bounds")
        governor = FoundationModelGovernor(
            control_lo=self.barrier_filter.control_lo,
            control_hi=self.barrier_filter.control_hi,
            max_rate=self._max_rate(),
            barrier_filter=self.barrier_filter,
            barrier_certificate=self.barrier_certificate,
        )
        decision = governor.govern(
            action.value,
            state,
            drift,
            previous_action=self.previous_action,
        )
        smt_artifact = _admission_smt(
            channel=self,
            action=action,
            state=state,
            drift=drift,
            admitted_value=decision.admitted_action,
        )
        smt_hash = _sha256_text(smt_artifact.text)
        record = PolicyCBFAdmissionRecord(
            knob=action.knob,
            scope=action.scope,
            proposed_value=action.value,
            admitted_value=decision.admitted_action,
            status=decision.status,
            stages_applied=decision.stages_applied,
            violations=decision.violations,
            barrier_value=decision.barrier_value,
            filter_digest=self.barrier_filter.filter_digest,
            certificate_verification_digest=(
                self.barrier_certificate.verification_digest
            ),
            smt_artifact=smt_artifact,
            smt_artifact_hash=smt_hash,
        )
        return _action_from_decision(action, record), record

    def _max_rate(self) -> float:
        """Return the configured or full-span rate limit."""
        if self.max_rate is not None:
            return self.max_rate
        return max(
            self.barrier_filter.control_hi - self.barrier_filter.control_lo,
            1.0e-12,
        )


class PolicyCBFAdmissionGate:
    """Apply configured CBF channels to supervisor action proposals."""

    def __init__(self, channels: Sequence[PolicyCBFChannel]) -> None:
        if not channels:
            raise ValueError("channels must contain at least one PolicyCBFChannel")
        if not all(isinstance(channel, PolicyCBFChannel) for channel in channels):
            raise ValueError("channels must contain only PolicyCBFChannel objects")
        keys = [(channel.knob, channel.scope) for channel in channels]
        if len(set(keys)) != len(keys):
            raise ValueError("CBF admission channels must be unique by knob/scope")
        self._channels = tuple(channels)

    def admit_actions(
        self,
        actions: Sequence[ControlAction],
        upde_state: UPDEState,
        boundary_state: BoundaryState,
    ) -> PolicyCBFAdmissionResult:
        """Admit matching actions and return transformed actions plus records.

        Parameters
        ----------
        actions : Sequence[ControlAction]
            Supervisor action proposals.
        upde_state : UPDEState
            Current UPDE metrics.
        boundary_state : BoundaryState
            Current boundary-observer metrics.

        Returns
        -------
        PolicyCBFAdmissionResult
            Admitted action tuple and CBF audit records for matched actions.
        """
        admitted: list[ControlAction] = []
        records: list[PolicyCBFAdmissionRecord] = []
        for action in actions:
            channel = self._matching_channel(action)
            if channel is None:
                admitted.append(action)
                continue
            admitted_action, record = channel.admit(action, upde_state, boundary_state)
            admitted.append(admitted_action)
            records.append(record)
        return PolicyCBFAdmissionResult(tuple(admitted), tuple(records))

    def _matching_channel(self, action: ControlAction) -> PolicyCBFChannel | None:
        """Return the channel matching ``action`` or ``None``."""
        for channel in self._channels:
            if channel.matches(action):
                return channel
        return None


def _action_from_decision(
    action: ControlAction,
    record: PolicyCBFAdmissionRecord,
) -> ControlAction:
    """Return a ``ControlAction`` carrying the CBF-admitted value."""
    prefix = {
        "admitted": "CBF admitted",
        "constrained": "CBF constrained",
        "rejected": "CBF rejected",
    }.get(record.status, f"CBF {record.status}")
    return ControlAction(
        knob=action.knob,
        scope=action.scope,
        value=record.admitted_value,
        ttl_s=action.ttl_s,
        justification=f"{action.justification}; {prefix}: {record.content_hash[:12]}",
    )


def _state_vector(
    metrics: Sequence[str],
    upde_state: UPDEState,
    boundary_state: BoundaryState,
) -> FloatArray:
    """Return the CBF state vector described by ``metrics``."""
    return _finite_vector(
        tuple(_metric_value(metric, upde_state, boundary_state) for metric in metrics),
        "state_metrics",
    )


def _metric_value(
    metric: str,
    upde_state: UPDEState,
    boundary_state: BoundaryState,
) -> float:
    """Return one supported metric value for CBF admission."""
    if metric == "R_min":
        return min((layer.R for layer in upde_state.layers), default=0.0)
    if metric == "R_mean":
        if not upde_state.layers:
            return 0.0
        return float(np.mean([layer.R for layer in upde_state.layers]))
    if metric == "stability_proxy":
        return upde_state.stability_proxy
    if metric == "violation_count":
        return float(len(boundary_state.violations))
    if metric == "hard_violation_count":
        return float(len(boundary_state.hard_violations))
    raise ValueError(f"unsupported CBF admission metric: {metric}")


def _admission_smt(
    *,
    channel: PolicyCBFChannel,
    action: ControlAction,
    state: FloatArray,
    drift: FloatArray,
    admitted_value: float,
) -> FormalTextArtifact:
    """Return SMT-LIB text for one scalar CBF admission decision."""
    barrier = channel.barrier_filter.barrier
    grad = barrier.gradient(state)
    h_value = barrier.value(state)
    lie_g = float(grad @ channel.barrier_filter.control_effect)
    grad_drift = float(grad @ drift)
    rhs = -channel.barrier_filter.gamma * h_value - grad_drift
    lines = [
        "(set-logic QF_LRA)",
        "; Generated from SCPN supervisor CBF admission.",
        f"; channel: {action.knob}/{action.scope}",
        f"; filter_digest: {channel.barrier_filter.filter_digest}",
        f"; certificate_digest: {channel.barrier_certificate.verification_digest}",
        f"; proposed_value: {_smt_real(action.value)}",
        f"; admitted_value: {_smt_real(admitted_value)}",
        f"; barrier_value: {_smt_real(h_value)}",
        "(declare-const u Real)",
        f"(assert (>= u {_smt_real(channel.barrier_filter.control_lo)}))",
        f"(assert (<= u {_smt_real(channel.barrier_filter.control_hi)}))",
        f"(assert (>= (* {_smt_real(lie_g)} u) {_smt_real(rhs)}))",
        f"(assert (= u {_smt_real(admitted_value)}))",
        "(check-sat)",
    ]
    return FormalTextArtifact("smt2", "\n".join(lines) + "\n")


def _finite_vector(value: Sequence[float], name: str) -> FloatArray:
    """Return ``value`` as a finite float vector."""
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return np.ascontiguousarray(array, dtype=np.float64)


def _finite_real(value: object, name: str) -> float:
    """Return ``value`` as a finite real scalar."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real")
    parsed = float(value)
    if not isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _smt_real(value: float) -> str:
    """Return ``value`` as an SMT-LIB real literal."""
    magnitude = f"{abs(value):.17g}"
    if "e" in magnitude.lower():
        magnitude = format(Decimal(magnitude), "f").rstrip("0").rstrip(".")
    if value < 0:
        return f"(- {magnitude})"
    return magnitude


def _sha256_text(text: str) -> str:
    """Return SHA-256 over UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(payload: Mapping[str, object]) -> str:
    """Return SHA-256 over canonical compact JSON."""
    serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialised.encode("utf-8")).hexdigest()
