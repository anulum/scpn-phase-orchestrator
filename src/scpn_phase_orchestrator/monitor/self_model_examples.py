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
from numbers import Real
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
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be numeric, got bool")
    if isinstance(value, (np.floating, np.integer)):
        return float(value.item())
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"{label} must be a numeric value, got {type(value)!r}")


def _coerce_vector(values: object, *, label: str) -> FloatArray:
    """Return ``values`` as a non-empty 1-D finite float64 vector, else raise."""
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError, RuntimeError) as exc:
        raise ValueError(f"{label} must be a real numeric vector") from exc
    if np.issubdtype(raw.dtype, np.bool_) or np.iscomplexobj(raw):
        raise ValueError(f"{label} must be a real numeric vector")
    if raw.dtype == np.dtype("O"):
        if not all(
            isinstance(item, (Real, np.integer, np.floating))
            and not isinstance(item, (bool, np.bool_))
            for item in raw.flat
        ):
            raise ValueError(f"{label} must be a real numeric vector")
    elif raw.dtype.kind not in {"f", "i", "u"}:
        raise ValueError(f"{label} must be a real numeric vector")
    try:
        arr = np.array(raw, dtype=np.float64, copy=True)
    except (  # pragma: no cover - defensive after dtype-kind validation
        TypeError,
        ValueError,
        OverflowError,
    ) as exc:
        raise ValueError(f"{label} must be a real numeric vector") from exc
    if arr.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if arr.size < 1:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(arr).all():
        raise ValueError(f"{label} must contain only finite values")
    result = np.ascontiguousarray(arr, dtype=np.float64)
    result.setflags(write=False)
    return result


def _coerce_canonical_string(value: object, *, label: str) -> str:
    """Return a non-empty, trimmed string suitable for canonical evidence."""
    if type(value) is not str or not value.strip() or value != value.strip():
        raise ValueError(f"{label} must be a non-empty canonical string")
    return value


def _require_string_json_keys(value: object, *, label: str) -> None:
    """Reject mappings with keys that canonical JSON would silently stringify."""
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError(f"{label} must contain only string keys")
        for item in value.values():
            _require_string_json_keys(item, label=label)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _require_string_json_keys(item, label=label)


def _canonicalise_json_evidence(value: object, *, label: str) -> dict[str, Any]:
    """Return a detached strict-JSON evidence mapping with canonical key order."""
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict")
    _require_string_json_keys(value, label=label)
    try:
        payload = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be JSON-serialisable") from exc
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):  # pragma: no cover - guarded before encoding
        raise ValueError(f"{label} must be a dict")
    return cast(dict[str, Any], decoded)


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
    default_within = (
        fallback_norm <= error_threshold and fallback_max <= error_threshold
    )
    if hasattr(error_result, "breached"):
        default_within = not _coerce_bool(
            cast(Any, error_result).breached,
            label="breached",
        )

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
        # every call site passes a non-None default, so this is never reached.
        raise ValueError(
            f"missing error field(s): {', '.join(names)}"
        )  # pragma: no cover

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
        # every call site passes a non-None default, so this is never reached.
        raise ValueError(  # pragma: no cover
            f"missing error boolean field(s): {', '.join(names)}"
        )

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
            default=default_within,
        ),
    }

    metric = "circular_rms_error"
    if isinstance(error_result, dict):
        if "metric" in error_result:
            metric = _coerce_canonical_string(error_result["metric"], label="metric")
    elif hasattr(error_result, "metric"):
        metric = _coerce_canonical_string(error_result.metric, label="metric")
    result["metric"] = metric

    for name in ("error_norm", "max_abs_error", "mean_abs_error"):
        if not math.isfinite(result[name]):
            raise ValueError(f"{name} must be finite")
        if result[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    if not math.isfinite(result["threshold"]) or result["threshold"] <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if result["threshold"] != error_threshold:
        raise ValueError("self-model error threshold contradicts error_threshold")
    if result["mean_abs_error"] > result["error_norm"]:
        raise ValueError("mean_abs_error must not exceed error_norm")
    if result["error_norm"] > result["max_abs_error"]:
        raise ValueError("error_norm must not exceed max_abs_error")

    expected_within = (
        result["error_norm"] <= result["threshold"]
        and result["max_abs_error"] <= result["threshold"]
    )
    if result["within_threshold"] is not expected_within:
        raise ValueError("within_threshold contradicts the error metrics")
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


@dataclass(frozen=True)
class SelfModelReconfigurationProposal:
    """Single replay-backed, review-only self-model reconfiguration scenario."""

    domain: str
    scenario_id: str
    predicted_phase: FloatArray
    observed_phase: FloatArray
    error_threshold: float
    self_model_error: SelfModelErrorResult | dict[str, object]
    proposed_reconfiguration_action: str
    serialisable_evidence: dict[str, Any]
    blocked_live_execution_fields: tuple[str, ...]
    operator_review_required: bool = True
    execution_disabled: bool = True
    claim_boundary: str = SelfModelBoundary
    scenario_hash: str = ""

    def __post_init__(self) -> None:
        """Validate and canonicalise directly constructed proposal evidence."""
        domain = _coerce_canonical_string(self.domain, label="domain")
        if domain not in SupportedDomains:
            raise ValueError(f"invalid domain '{domain}'")
        object.__setattr__(self, "domain", domain)
        object.__setattr__(
            self,
            "scenario_id",
            _coerce_canonical_string(self.scenario_id, label="scenario_id"),
        )

        predicted = _coerce_vector(self.predicted_phase, label="predicted_phase")
        observed = _coerce_vector(self.observed_phase, label="observed_phase")
        if predicted.shape != observed.shape:
            raise ValueError("predicted and observed phase vectors must match")
        object.__setattr__(self, "predicted_phase", predicted)
        object.__setattr__(self, "observed_phase", observed)

        threshold = _coerce_scalar(self.error_threshold, label="error_threshold")
        if not math.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("error_threshold must be finite and positive")
        object.__setattr__(self, "error_threshold", threshold)
        object.__setattr__(
            self,
            "proposed_reconfiguration_action",
            _coerce_canonical_string(
                self.proposed_reconfiguration_action,
                label="proposed_reconfiguration_action",
            ),
        )
        object.__setattr__(
            self,
            "serialisable_evidence",
            _canonicalise_json_evidence(
                self.serialisable_evidence,
                label="serialisable_evidence",
            ),
        )

        if not isinstance(self.blocked_live_execution_fields, tuple) or not (
            self.blocked_live_execution_fields
        ):
            raise ValueError("blocked_live_execution_fields must be a non-empty tuple")
        blocked = tuple(
            _coerce_canonical_string(field, label="blocked_live_execution_fields")
            for field in self.blocked_live_execution_fields
        )
        if len(set(blocked)) != len(blocked):
            raise ValueError("blocked_live_execution_fields must be unique")
        object.__setattr__(self, "blocked_live_execution_fields", blocked)

        if (
            _coerce_bool(
                self.operator_review_required,
                label="operator_review_required",
            )
            is not True
        ):
            raise ValueError("operator_review_required must be true")
        if (
            _coerce_bool(self.execution_disabled, label="execution_disabled")
            is not True
        ):
            raise ValueError("execution_disabled must be true")
        if self.claim_boundary != SelfModelBoundary:
            raise ValueError("claim_boundary must preserve the review-only boundary")
        if type(self.scenario_hash) is not str:
            raise ValueError("scenario_hash must be a string")
        if self.scenario_hash and (
            len(self.scenario_hash) != 64
            or self.scenario_hash != self.scenario_hash.lower()
            or any(char not in "0123456789abcdef" for char in self.scenario_hash)
        ):
            raise ValueError(
                "scenario_hash must be 64 lowercase hexadecimal characters"
            )

        _coerce_error_payload(
            self.self_model_error,
            predicted_phase=predicted,
            observed_phase=observed,
            error_threshold=threshold,
        )
        if self.scenario_hash:
            _validate_self_model_reconfiguration_proposal(self)

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
        return record


def _validate_self_model_reconfiguration_proposal(
    scenario: SelfModelReconfigurationProposal,
) -> None:
    """Validate a frozen proposal's optional stored hash against current evidence."""
    if scenario.scenario_hash:
        error_payload = _coerce_error_payload(
            scenario.self_model_error,
            predicted_phase=scenario.predicted_phase,
            observed_phase=scenario.observed_phase,
            error_threshold=scenario.error_threshold,
        )
        expected = _compute_scenario_hash(
            proposal=scenario,
            error_payload=error_payload,
        )
        if scenario.scenario_hash != expected:
            raise ValueError(f"{scenario.scenario_id} has mismatched scenario_hash")


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
    unexpected = set(record.keys()) - required_fields
    if unexpected:
        raise ValueError(f"record has unexpected fields: {sorted(unexpected)}")

    if not isinstance(record["scenario_hash"], str):
        raise ValueError("record scenario_hash must be a string")
    if not isinstance(record["predicted_phase"], list) or not isinstance(
        record["observed_phase"], list
    ):
        raise ValueError("record phase vectors must be JSON arrays")
    if not isinstance(record["self_model_error"], dict):
        raise ValueError("record self_model_error must be a JSON object")
    if not isinstance(record["phase_error_summary"], dict):
        raise ValueError("record phase_error_summary must be a JSON object")

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
    if not math.isfinite(error_threshold) or error_threshold <= 0.0:
        raise ValueError("record.error_threshold must be finite and positive")
    if not isinstance(record["blocked_live_execution_fields"], list):
        raise ValueError("record blocked_live_execution_fields must be a list")

    preview = SelfModelReconfigurationProposal(
        domain=record["domain"],
        scenario_id=record["scenario_id"],
        predicted_phase=predicted,
        observed_phase=observed,
        error_threshold=error_threshold,
        self_model_error=cast(
            dict[str, object],
            _coerce_error_payload(
                cast(dict[str, object], record["self_model_error"]),
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
    error_payload = _coerce_error_payload(
        cast(dict[str, object], record["self_model_error"]),
        predicted_phase=preview.predicted_phase,
        observed_phase=preview.observed_phase,
        error_threshold=preview.error_threshold,
    )
    expected_unsafe = not error_payload["within_threshold"]
    if type(record["unsafe_due_to_threshold"]) is not bool or (
        record["unsafe_due_to_threshold"] is not expected_unsafe
    ):
        raise ValueError("record unsafe_due_to_threshold contradicts error evidence")
    expected_summary = _error_summary(
        np.abs(_circular_error(preview.predicted_phase, preview.observed_phase))
    )
    if record["phase_error_summary"] != expected_summary:
        raise ValueError("record phase_error_summary contradicts phase evidence")
    _hash = _compute_scenario_hash(
        proposal=preview,
        error_payload=error_payload,
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
