# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — Self-model replay examples

"""Deterministic replay-backed self-model reconfiguration examples.

These fixtures remain review-only and serialisable evidence for industrial control
reconfiguration proposals. They intentionally disable execution and require
operator review.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from scpn_phase_orchestrator.monitor.self_model import (
    SelfModelErrorResult,
    compute_self_model_error,
)

FloatArray: TypeAlias = NDArray[np.float64]

SelfModelBoundary: Final[str] = "self_model_reconfiguration_not_live_actuation"
SupportedDomains: Final[tuple[str, ...]] = (
    "power_grid",
    "cardiac_rhythm",
    "cyber_industrial",
    "traffic_flow",
)


def _compute_self_model_error(
    *,
    predicted_phase: FloatArray,
    observed_phase: FloatArray,
    error_threshold: float,
) -> SelfModelErrorResult:
    """Compute the self-model phase-prediction error via the core helper."""
    predicted = _coerce_vector(predicted_phase, label="predicted_phase")
    observed = _coerce_vector(observed_phase, label="observed_phase")
    return compute_self_model_error(
        observed_phases=observed,
        predicted_phases=predicted,
        tolerance=float(error_threshold),
        max_abs_tolerance=float(error_threshold),
        domain="self_model_reconfiguration",
        scenario_id="replay_backed_reconfiguration",
        channel_labels=("phase_trace",),
    )


def _coerce_scalar(value: object, *, label: str) -> float:
    """Return ``value`` as a numeric float, rejecting booleans, else raise."""
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, (np.floating, np.integer)):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{label} must be a numeric value, got {type(value)!r}")


def _coerce_vector(values: object, *, label: str) -> FloatArray:
    """Return ``values`` as a non-empty 1-D finite float64 vector, else raise."""
    try:
        arr = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a float-convertible vector") from exc
    if arr.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if arr.size < 1:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    return np.asarray(arr, dtype=np.float64)


def _coerce_bool(value: object, *, label: str) -> bool:
    """Return ``value`` if it is a real boolean, else raise ``ValueError``."""
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return bool(value)


def _circular_error(predicted: FloatArray, observed: FloatArray) -> FloatArray:
    """Return the wrapped circular phase error between observed and predicted."""
    delta = observed - predicted
    return np.asarray(np.arctan2(np.sin(delta), np.cos(delta)), dtype=np.float64)


def _coerce_error_payload(
    error_result: SelfModelErrorResult | dict[str, object],
    *,
    predicted_phase: FloatArray,
    observed_phase: FloatArray,
    error_threshold: float,
) -> dict[str, Any]:
    """Build a JSON-safe error payload from a self-model error result."""
    diff = np.abs(_circular_error(predicted_phase, observed_phase))
    fallback_norm = float(np.linalg.norm(diff) / math.sqrt(diff.size))
    fallback_max = float(np.max(diff))
    fallback_mean = float(np.mean(diff))

    def _from_obj(
        obj: object, names: tuple[str, ...], default: float | None = None
    ) -> float:
        """Return the first present named scalar field from a dict or object."""
        for name in names:
            if isinstance(obj, dict) and name in obj:
                return _coerce_scalar(obj[name], label=name)
            if hasattr(obj, name):
                return _coerce_scalar(getattr(obj, name), label=name)
        if default is not None:
            return default
        raise ValueError(f"missing error field(s): {', '.join(names)}")

    def _from_obj_bool(
        obj: object, names: tuple[str, ...], default: bool | None = None
    ) -> bool:
        """Return the first present named boolean field from a dict or object."""
        for name in names:
            if isinstance(obj, dict) and name in obj:
                if not isinstance(obj[name], bool):
                    raise ValueError(f"{name} must be boolean")
                return bool(obj[name])
            if hasattr(obj, name):
                value = getattr(obj, name)
                if isinstance(value, bool):
                    return bool(value)
                raise ValueError(f"{name} must be boolean")
        if default is not None:
            return default
        raise ValueError(f"missing error boolean field(s): {', '.join(names)}")

    threshold = _from_obj(
        error_result,
        ("threshold", "error_threshold"),
        default=error_threshold,
    )
    if math.isfinite(threshold) and threshold > 0.0:
        pass
    else:
        threshold = error_threshold

    result: dict[str, Any] = {
        "error_norm": _from_obj(
            error_result,
            ("error_norm", "rms_error", "norm", "overall_rmse"),
            default=fallback_norm,
        ),
        "max_abs_error": _from_obj(
            error_result,
            ("max_abs_error", "max_error", "overall_max_abs_error"),
            default=fallback_max,
        ),
        "mean_abs_error": _from_obj(
            error_result,
            ("mean_abs_error", "mean_error", "overall_mae"),
            default=fallback_mean,
        ),
        "threshold": _from_obj(
            error_result,
            ("threshold", "error_threshold", "tolerance"),
            default=error_threshold,
        ),
        "within_threshold": _from_obj_bool(
            error_result,
            ("within_threshold", "passes_threshold", "safe"),
            default=(
                not bool(getattr(error_result, "breached", fallback_norm > threshold))
            ),
        ),
    }

    metric = "circular_rms_error"
    if isinstance(error_result, dict):
        if "metric" in error_result and isinstance(error_result["metric"], str):
            metric = error_result["metric"]
    elif hasattr(error_result, "metric") and isinstance(error_result.metric, str):
        metric = error_result.metric
    result["metric"] = metric

    if not math.isfinite(result["error_norm"]):
        raise ValueError("error_norm must be finite")
    if not math.isfinite(result["max_abs_error"]):
        raise ValueError("max_abs_error must be finite")
    if not math.isfinite(result["mean_abs_error"]):
        raise ValueError("mean_abs_error must be finite")
    if not math.isfinite(result["threshold"]) or result["threshold"] <= 0.0:
        raise ValueError("threshold must be finite and positive")

    result["within_threshold"] = bool(result["within_threshold"])
    return result


def _error_summary(errors: FloatArray) -> dict[str, float]:
    """Return a JSON-safe summary of the self-model error metrics."""
    return {
        "count": int(errors.size),
        "mean": float(np.mean(errors)),
        "max": float(np.max(errors)),
        "std": float(np.std(errors)),
    }


def _compute_scenario_hash(
    *,
    proposal: SelfModelReconfigurationProposal,
    error_payload: dict[str, Any],
) -> str:
    """Return the canonical-JSON SHA-256 hash of a scenario record."""
    canonical: dict[str, Any] = {
        "domain": proposal.domain,
        "scenario_id": proposal.scenario_id,
        "error_threshold": float(proposal.error_threshold),
        "claim_boundary": proposal.claim_boundary,
        "operator_review_required": proposal.operator_review_required,
        "execution_disabled": proposal.execution_disabled,
        "blocked_live_execution_fields": list(proposal.blocked_live_execution_fields),
        "proposed_reconfiguration_action": proposal.proposed_reconfiguration_action,
        "predicted_phase": [float(v) for v in proposal.predicted_phase.tolist()],
        "observed_phase": [float(v) for v in proposal.observed_phase.tolist()],
        "serialisable_evidence": proposal.serialisable_evidence,
        "self_model_error": error_payload,
    }
    payload = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class SelfModelReconfigurationProposal:
    """Single replay-backed, review-only self-model reconfiguration scenario."""

    domain: str
    scenario_id: str
    predicted_phase: FloatArray
    observed_phase: FloatArray
    error_threshold: float
    self_model_error: SelfModelErrorResult
    proposed_reconfiguration_action: str
    serialisable_evidence: dict[str, Any]
    blocked_live_execution_fields: tuple[str, ...]
    operator_review_required: bool = True
    execution_disabled: bool = True
    claim_boundary: str = SelfModelBoundary
    scenario_hash: str = ""

    def to_audit_record(self) -> dict[str, Any]:
        """Return a deterministic JSON-safe audit record.

        Returns
        -------
        dict[str, Any]
            Return a deterministic JSON-safe audit record.

        Raises
        ------
        ValueError
            If the proposal fields are inconsistent.
        """
        _validate_self_model_reconfiguration_proposal(self)
        error_payload = _coerce_error_payload(
            self.self_model_error,
            predicted_phase=self.predicted_phase,
            observed_phase=self.observed_phase,
            error_threshold=self.error_threshold,
        )
        diff = np.abs(_circular_error(self.predicted_phase, self.observed_phase))
        unsafe = not error_payload["within_threshold"]
        record = {
            "domain": self.domain,
            "scenario_id": self.scenario_id,
            "claim_boundary": self.claim_boundary,
            "error_threshold": float(self.error_threshold),
            "predicted_phase": [float(v) for v in self.predicted_phase.tolist()],
            "observed_phase": [float(v) for v in self.observed_phase.tolist()],
            "proposed_reconfiguration_action": self.proposed_reconfiguration_action,
            "serialisable_evidence": self.serialisable_evidence,
            "blocked_live_execution_fields": list(self.blocked_live_execution_fields),
            "operator_review_required": self.operator_review_required,
            "execution_disabled": self.execution_disabled,
            "unsafe_due_to_threshold": bool(unsafe),
            "self_model_error": error_payload,
            "phase_error_summary": _error_summary(diff),
            "scenario_hash": "",
        }
        record["scenario_hash"] = _compute_scenario_hash(
            proposal=self, error_payload=error_payload
        )
        if self.scenario_hash and self.scenario_hash != record["scenario_hash"]:
            raise ValueError(
                f"scenario {self.scenario_id} has mismatched scenario_hash"
            )
        return record


def _validate_self_model_reconfiguration_proposal(
    scenario: SelfModelReconfigurationProposal,
) -> None:
    """Validate a reconfiguration proposal's fields and review gates."""
    if scenario.domain not in SupportedDomains:
        raise ValueError(f"invalid domain '{scenario.domain}'")
    if not scenario.scenario_id or not scenario.scenario_id.strip():
        raise ValueError("scenario_id must be a non-empty string")

    predicted = _coerce_vector(
        scenario.predicted_phase,
        label=f"{scenario.scenario_id}.predicted_phase",
    )
    observed = _coerce_vector(
        scenario.observed_phase,
        label=f"{scenario.scenario_id}.observed_phase",
    )
    if predicted.shape != observed.shape:
        raise ValueError(
            f"{scenario.scenario_id} predicted and observed phase vectors must match"
        )
    if (
        not math.isfinite(float(scenario.error_threshold))
        or scenario.error_threshold <= 0.0
    ):
        raise ValueError(f"{scenario.scenario_id}.error_threshold must be positive")

    if not scenario.proposed_reconfiguration_action.strip():
        raise ValueError(f"{scenario.scenario_id} needs non-empty proposed action")
    if (
        _coerce_bool(
            scenario.operator_review_required, label="operator_review_required"
        )
        is not True
    ):
        raise ValueError(
            f"{scenario.scenario_id} requires operator_review_required=True"
        )
    if (
        _coerce_bool(scenario.execution_disabled, label="execution_disabled")
        is not True
    ):
        raise ValueError(f"{scenario.scenario_id} requires execution_disabled=True")
    if scenario.claim_boundary != SelfModelBoundary:
        raise ValueError(f"{scenario.scenario_id} has invalid claim boundary")
    if not isinstance(scenario.blocked_live_execution_fields, tuple) or not (
        scenario.blocked_live_execution_fields
    ):
        raise ValueError(f"{scenario.scenario_id} requires blocked fields")
    if not all(
        isinstance(field, str) and field.strip()
        for field in scenario.blocked_live_execution_fields
    ):
        raise ValueError(
            f"{scenario.scenario_id} blocked fields must be non-empty strings"
        )

    if not isinstance(scenario.serialisable_evidence, dict):
        raise ValueError(f"{scenario.scenario_id}.serialisable_evidence must be a dict")

    if scenario.scenario_hash:
        error_payload = _coerce_error_payload(
            scenario.self_model_error,
            predicted_phase=predicted,
            observed_phase=observed,
            error_threshold=scenario.error_threshold,
        )
        expected = _compute_scenario_hash(
            proposal=scenario,
            error_payload=error_payload,
        )
        if scenario.scenario_hash != expected:
            raise ValueError(f"{scenario.scenario_id} has mismatched scenario_hash")
        if len(scenario.scenario_hash) != 64:
            raise ValueError(
                f"{scenario.scenario_id} scenario_hash must be 64 hex chars"
            )


def _validate_scenario_record(record: dict[str, Any]) -> None:
    """Validate a self-model reconfiguration scenario record."""
    required_fields = {
        "domain",
        "scenario_id",
        "claim_boundary",
        "error_threshold",
        "predicted_phase",
        "observed_phase",
        "proposed_reconfiguration_action",
        "serialisable_evidence",
        "blocked_live_execution_fields",
        "operator_review_required",
        "execution_disabled",
        "unsafe_due_to_threshold",
        "self_model_error",
        "phase_error_summary",
        "scenario_hash",
    }
    missing = required_fields - set(record.keys())
    if missing:
        raise ValueError(f"record missing required fields: {sorted(missing)}")

    if not isinstance(record["scenario_hash"], str):
        raise ValueError("record scenario_hash must be a string")

    predicted = _coerce_vector(
        record["predicted_phase"], label="record.predicted_phase"
    )
    observed = _coerce_vector(record["observed_phase"], label="record.observed_phase")
    if predicted.shape != observed.shape:
        raise ValueError("record predicted and observed phase vectors mismatch")
    error_threshold = _coerce_scalar(
        record["error_threshold"],
        label="record.error_threshold",
    )
    preview = SelfModelReconfigurationProposal(
        domain=record["domain"],
        scenario_id=record["scenario_id"],
        predicted_phase=predicted,
        observed_phase=observed,
        error_threshold=error_threshold,
        self_model_error=cast(
            SelfModelErrorResult,
            _coerce_error_payload(
                cast(SelfModelErrorResult, record["self_model_error"]),
                predicted_phase=predicted,
                observed_phase=observed,
                error_threshold=error_threshold,
            ),
        ),
        proposed_reconfiguration_action=record["proposed_reconfiguration_action"],
        serialisable_evidence=record["serialisable_evidence"],
        blocked_live_execution_fields=tuple(record["blocked_live_execution_fields"]),
        operator_review_required=_coerce_bool(
            record["operator_review_required"],
            label="record.operator_review_required",
        ),
        execution_disabled=_coerce_bool(
            record["execution_disabled"],
            label="record.execution_disabled",
        ),
        claim_boundary=record["claim_boundary"],
        scenario_hash=record["scenario_hash"],
    )

    _validate_self_model_reconfiguration_proposal(preview)
    _hash = _compute_scenario_hash(
        proposal=preview,
        error_payload=_coerce_error_payload(
            cast(SelfModelErrorResult, record["self_model_error"]),
            predicted_phase=preview.predicted_phase,
            observed_phase=preview.observed_phase,
            error_threshold=preview.error_threshold,
        ),
    )
    if record["scenario_hash"] != _hash:
        raise ValueError(f"record {record['scenario_id']} has invalid scenario_hash")


def _build_static_proposals() -> tuple[SelfModelReconfigurationProposal, ...]:
    """Build the deterministic static self-model reconfiguration proposals."""
    scenario_specs: tuple[
        tuple[
            str,
            str,
            tuple[float, ...],
            tuple[float, ...],
            float,
            str,
            dict[str, Any],
            tuple[str, ...],
        ],
        ...,
    ] = (
        (
            "power_grid",
            "power_grid_self_model_reconfiguration_v1",
            (0.11, 0.84, 1.73, 2.51, 3.35, 4.20),
            (0.12, 0.86, 1.68, 2.46, 3.32, 4.23),
            0.16,
            "Apply review-only damping rebind to phase-coupling regulators "
            "for islanding contingency containment.",
            {
                "replay_mode": "replay_backed_replay_trace",
                "source": "power_grid_stability_replay_bank",
                "evidence_strength": 0.91,
            },
            ("live_actuation", "binding_write", "spline_update"),
        ),
        (
            "cardiac_rhythm",
            "cardiac_rhythm_self_model_reconfiguration_v1",
            (0.31, 1.04, 2.21, 3.15, 4.04),
            (0.28, 1.01, 2.24, 3.20, 4.02),
            0.12,
            "Request operator review for controller gain re-tuning on "
            "atrial-phase entrainment path.",
            {
                "replay_mode": "replay_backed_replay_trace",
                "source": "cardiac_pacing_guardrail_replay",
                "evidence_strength": 0.83,
            },
            ("runtime_dispatch", "alarm_silencing", "qos_adjust"),
        ),
        (
            "traffic_flow",
            "traffic_flow_self_model_reconfiguration_v1",
            (0.72, 1.61, 2.14, 2.97, 3.44),
            (2.51, 4.03, 0.96, 2.03, 5.21),
            0.18,
            "Hold dynamic lane-balance rebinding and queue adaptive timing "
            "rules for manual adjudication.",
            {
                "replay_mode": "replay_backed_replay_trace",
                "source": "traffic_flow_replay_bank",
                "evidence_strength": 0.47,
            },
            ("traffic_signal_driver", "adaptive_router", "mesh_output"),
        ),
        (
            "cyber_industrial",
            "cyber_industrial_self_model_reconfiguration_v1",
            (0.20, 1.05, 1.92, 2.71),
            (0.23, 1.01, 1.86, 2.75),
            0.20,
            "Block automated patch-assembly and request review for secure "
            "module rebind after replayed anomaly trace.",
            {
                "replay_mode": "replay_backed_replay_trace",
                "source": "cyber_attack_replay_bank",
                "evidence_strength": 0.77,
            },
            ("runtime_code_update", "network_rebind", "policy_update"),
        ),
    )

    proposals: list[SelfModelReconfigurationProposal] = []
    for (
        domain,
        scenario_id,
        predicted_phase,
        observed_phase,
        error_threshold,
        proposed_reconfiguration_action,
        serialisable_evidence,
        blocked_live_execution_fields,
    ) in scenario_specs:
        predicted_array = np.array(predicted_phase, dtype=np.float64)
        observed_array = np.array(observed_phase, dtype=np.float64)
        proposals.append(
            SelfModelReconfigurationProposal(
                domain=domain,
                scenario_id=scenario_id,
                predicted_phase=predicted_array,
                observed_phase=observed_array,
                error_threshold=error_threshold,
                self_model_error=_compute_self_model_error(
                    predicted_phase=predicted_array,
                    observed_phase=observed_array,
                    error_threshold=error_threshold,
                ),
                proposed_reconfiguration_action=proposed_reconfiguration_action,
                serialisable_evidence=serialisable_evidence,
                blocked_live_execution_fields=blocked_live_execution_fields,
            )
        )

    return tuple(proposals)


def build_self_model_reconfiguration_examples() -> tuple[dict[str, Any], ...]:
    """Build deterministic review-only self-model reconfiguration evidence records.

    Returns
    -------
    tuple[dict[str, Any], ...]
        Build deterministic review-only self-model reconfiguration evidence records.
    """
    records: list[dict[str, Any]] = []
    for proposal in _build_static_proposals():
        _validate_self_model_reconfiguration_proposal(proposal)
        record = proposal.to_audit_record()
        _validate_scenario_record(record)
        records.append(record)
    return tuple(records)


def _contains_arrays(value: object) -> bool:
    """Return whether the value contains any nested array payload."""
    if isinstance(value, dict):
        return any(_contains_arrays(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return any(_contains_arrays(item) for item in value)
    return isinstance(value, np.ndarray)


__all__ = [
    "SelfModelBoundary",
    "SelfModelReconfigurationProposal",
    "SupportedDomains",
    "SelfModelErrorResult",
    "build_self_model_reconfiguration_examples",
    "_validate_scenario_record",
    "_contains_arrays",
]
