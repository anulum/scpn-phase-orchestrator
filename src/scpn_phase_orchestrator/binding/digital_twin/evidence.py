# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Digital-twin operator evidence summaries

"""Operator evidence summaries for live and replayed digital-twin sync health."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real

from .contract import DigitalTwinBindingContract
from .envelope import DigitalTwinTransportValidation


@dataclass(frozen=True)
class DigitalTwinOperatorEvidence:
    """Transport-neutral operator summary for live or replayed twin sync."""

    contract_hash: str
    accepted_count: int
    rejected_count: int
    adapter_count: int
    unhealthy_adapter_count: int
    latest_sequence: int | None
    capability_counts: dict[str, int]
    direction_counts: dict[str, int]
    max_abs_twin_residual: float | None
    mismatch_reasons: tuple[str, ...]
    status: str

    def to_audit_record(self) -> dict[str, object]:
        """Return a JSON-safe operator evidence record.

        Returns
        -------
        dict[str, object]
            Deterministic, JSON-safe audit mapping of the DigitalTwinOperatorEvidence
            fields.
        """
        return {
            "contract_hash": self.contract_hash,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "adapter_count": self.adapter_count,
            "unhealthy_adapter_count": self.unhealthy_adapter_count,
            "latest_sequence": self.latest_sequence,
            "capability_counts": dict(sorted(self.capability_counts.items())),
            "direction_counts": dict(sorted(self.direction_counts.items())),
            "max_abs_twin_residual": self.max_abs_twin_residual,
            "mismatch_reasons": list(self.mismatch_reasons),
            "status": self.status,
        }


def build_digital_twin_operator_evidence(
    contract: DigitalTwinBindingContract,
    validations: Sequence[DigitalTwinTransportValidation],
    *,
    rejected: Sequence[Mapping[str, object]] = (),
    adapter_records: Sequence[Mapping[str, object]] = (),
    residual_warning_threshold: float = 0.05,
    residual_critical_threshold: float = 0.2,
) -> DigitalTwinOperatorEvidence:
    """Summarise live or replayed digital-twin sync evidence for operators.

    Accepted validations may come from REST, gRPC, Kafka, hardware, memory, or
    JSONL replay paths. Rejected JSONL lines and adapter audit records are
    folded into the same deterministic summary so dashboards can display live
    and replayed health with the same fields.

    Parameters
    ----------
    contract : DigitalTwinBindingContract
        The binding contract under observation.
    validations : Sequence[DigitalTwinTransportValidation]
        Accepted transport validations from any sync path.
    rejected : Sequence[Mapping[str, object]], optional
        Rejected JSONL lines folded into the summary.
    adapter_records : Sequence[Mapping[str, object]], optional
        Adapter audit records to include.
    residual_warning_threshold : float, optional
        Residual fraction above which a warning status is raised.
    residual_critical_threshold : float, optional
        Residual fraction above which a critical status is raised.

    Returns
    -------
    DigitalTwinOperatorEvidence
        A deterministic operator-facing health summary.

    Raises
    ------
    ValueError
        If the residual warning/critical thresholds are inconsistent.
    """
    warning_threshold = _validated_residual_threshold(
        residual_warning_threshold,
        "residual_warning_threshold",
    )
    critical_threshold = _validated_residual_threshold(
        residual_critical_threshold,
        "residual_critical_threshold",
    )
    if warning_threshold > critical_threshold:
        raise ValueError(
            "residual_warning_threshold must be <= residual_critical_threshold"
        )

    accepted: list[DigitalTwinTransportValidation] = []
    mismatch_reasons: list[str] = []
    capability_counts = {
        capability.name: 0 for capability in contract.sync_capabilities
    }
    direction_counts: dict[str, int] = {}
    latest_sequence: int | None = None
    residuals: list[float] = []

    for validation in validations:
        envelope = validation.envelope
        if envelope.contract_hash != contract.contract_hash:
            mismatch_reasons.append("contract_hash_mismatch")
            continue
        if not validation.accepted:
            mismatch_reasons.append(validation.reason)
            continue
        accepted.append(validation)
        capability_counts[envelope.capability] = (
            capability_counts.get(envelope.capability, 0) + 1
        )
        direction_counts[envelope.direction] = (
            direction_counts.get(
                envelope.direction,
                0,
            )
            + 1
        )
        latest_sequence = (
            envelope.sequence
            if latest_sequence is None
            else max(latest_sequence, envelope.sequence)
        )
        residual = _extract_twin_residual(envelope.payload)
        if residual is not None:
            residuals.append(abs(residual))

    for rejection in rejected:
        reason = rejection.get("reason")
        if isinstance(reason, str) and reason:
            mismatch_reasons.append(reason)
        else:
            mismatch_reasons.append("rejected")

    unhealthy_adapter_count = sum(
        1 for record in adapter_records if record.get("compatible") is False
    )
    max_abs_residual = max(residuals) if residuals else None
    rejected_count = len(validations) - len(accepted) + len(rejected)
    status = _operator_status(
        rejected_count=rejected_count,
        unhealthy_adapter_count=unhealthy_adapter_count,
        max_abs_residual=max_abs_residual,
        warning_threshold=warning_threshold,
        critical_threshold=critical_threshold,
    )
    return DigitalTwinOperatorEvidence(
        contract_hash=contract.contract_hash,
        accepted_count=len(accepted),
        rejected_count=rejected_count,
        adapter_count=len(adapter_records),
        unhealthy_adapter_count=unhealthy_adapter_count,
        latest_sequence=latest_sequence,
        capability_counts=capability_counts,
        direction_counts=direction_counts,
        max_abs_twin_residual=max_abs_residual,
        mismatch_reasons=tuple(sorted(mismatch_reasons)),
        status=status,
    )


def _extract_twin_residual(payload: Mapping[str, object]) -> float | None:
    """Return the digital-twin residual from the evidence."""
    for key in (
        "TwinResidual",
        "twin_residual",
        "twin_residual_norm",
        "residual",
        "residual_norm",
    ):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{key} must be a finite real value")
        result = float(value)
        if not isfinite(result):
            raise ValueError(f"{key} must be a finite real value")
        return result
    return None


def _operator_status(
    *,
    rejected_count: int,
    unhealthy_adapter_count: int,
    max_abs_residual: float | None,
    warning_threshold: float,
    critical_threshold: float,
) -> str:
    """Return the operator status for the twin evidence."""
    if max_abs_residual is not None and max_abs_residual > critical_threshold:
        return "critical"
    if rejected_count or unhealthy_adapter_count:
        return "degraded"
    if max_abs_residual is not None and max_abs_residual > warning_threshold:
        return "warning"
    return "healthy"


def _validated_residual_threshold(value: object, field: str) -> float:
    """Return the validated residual threshold, else raise."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite non-negative real value")
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a finite non-negative real value")
    return result
