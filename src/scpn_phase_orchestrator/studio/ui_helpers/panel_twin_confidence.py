# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio twin-confidence review panel

"""Digital-twin confidence review panel builder."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import cast

from ._shared import (
    _non_negative_float,
    _non_negative_int,
    _positive_int,
    _require_non_empty_text,
    _require_sha256_hex,
    _required_bool,
    _unit_interval_number,
)

_TWIN_CONFIDENCE_CLAIM_BOUNDARY = "digital_twin_confidence_observability_not_actuation"
_STATUS_LEVELS = {"healthy": 0, "warning": 1, "critical": 2}


def build_twin_confidence_studio_panel(
    score_records: Sequence[Mapping[str, object]],
    summary_record: Mapping[str, object],
) -> dict[str, object]:
    """Return a Studio panel for digital-twin confidence audit evidence.

    The panel consumes the real audit records emitted by
    ``monitor.twin_confidence``. It validates score and summary hashes,
    cross-checks summary aggregates against the supplied tick records, preserves
    calibrated confidence/status evidence, and keeps the operator surface
    observability-only: no actuation, hot patch, live merge, or execution gate is
    enabled by this payload.

    Parameters
    ----------
    score_records : Sequence[Mapping[str, object]]
        Chronological ``TwinConfidenceScore.to_audit_record()`` mappings.
    summary_record : Mapping[str, object]
        ``TwinConfidenceSummary.to_audit_record()`` mapping for the same score
        sequence.

    Returns
    -------
    dict[str, object]
        A review-only Studio panel payload for twin-confidence evidence.

    Raises
    ------
    ValueError
        If a record is malformed, a hash does not match, or the summary does not
        describe the supplied score sequence.
    """
    scores = _normalise_twin_confidence_scores(score_records)
    summary = _normalise_twin_confidence_summary(summary_record)
    _require_matching_summary(scores, summary)

    latest = scores[-1]
    worst = max(
        scores,
        key=lambda score: (
            _STATUS_LEVELS[cast("str", score["status"])],
            cast("float", score["composite_z"]),
        ),
    )
    backends = tuple(sorted({cast("str", score["backend"]) for score in scores}))
    status_counts = {
        "healthy": summary["healthy_count"],
        "warning": summary["warning_count"],
        "critical": summary["critical_count"],
    }
    return {
        "panel_kind": "studio_twin_confidence_panel",
        "monitor": "digital_twin_confidence",
        "claim_boundary": _TWIN_CONFIDENCE_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "score_count": len(scores),
        "status_counts": status_counts,
        "worst_status_level": _STATUS_LEVELS[cast("str", summary["worst_status"])],
        "latest": latest,
        "worst": worst,
        "summary": summary,
        "series": scores,
        "backends": backends,
        "operator_summary": (
            "twin-confidence review: "
            f"{summary['tick_count']} scored ticks, worst status "
            f"{summary['worst_status']}, latest confidence "
            f"{cast('float', summary['latest_confidence']):.6g}"
        ),
        "operator_action": (
            "render as digital-twin observability evidence only; investigate "
            "warning or critical confidence drift before considering any "
            "separate reviewed control action"
        ),
    }


def _normalise_twin_confidence_scores(
    score_records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate and normalise chronological twin-confidence score records."""
    if (
        isinstance(score_records, Mapping | str | bytes)
        or not isinstance(score_records, Sequence)
        or not score_records
    ):
        raise ValueError("score_records must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, record in enumerate(score_records):
        if not isinstance(record, Mapping):
            raise ValueError("score_records entries must be mappings")
        normalised.append(_normalise_twin_confidence_score(record, index))
    return tuple(normalised)


def _normalise_twin_confidence_score(
    record: Mapping[str, object],
    index: int,
) -> dict[str, object]:
    """Validate one ``TwinConfidenceScore`` audit mapping and its hash."""
    label = f"twin-confidence score {index}"
    normalised = {
        "confidence": _unit_interval_number(record.get("confidence"), "confidence"),
        "status": _normalise_status(record.get("status"), f"{label} status"),
        "phase_js_divergence": _non_negative_float(
            record.get("phase_js_divergence"),
            f"{label} phase_js_divergence",
        ),
        "order_wasserstein": _non_negative_float(
            record.get("order_wasserstein"),
            f"{label} order_wasserstein",
        ),
        "phase_js_z": _non_negative_float(
            record.get("phase_js_z"),
            f"{label} phase_js_z",
        ),
        "order_w1_z": _non_negative_float(
            record.get("order_w1_z"),
            f"{label} order_w1_z",
        ),
        "composite_z": _non_negative_float(
            record.get("composite_z"),
            f"{label} composite_z",
        ),
        "phase_js_within_band": _required_bool(
            record.get("phase_js_within_band"),
            f"{label} phase_js_within_band",
        ),
        "order_w1_within_band": _required_bool(
            record.get("order_w1_within_band"),
            f"{label} order_w1_within_band",
        ),
        "backend": _require_non_empty_text(record.get("backend"), f"{label} backend"),
        "score_hash": _require_sha256_hex(
            record.get("score_hash"),
            f"{label} score_hash",
        ),
    }
    _require_matching_hash(
        normalised,
        hash_field="score_hash",
        label=f"{label} score_hash",
    )
    return normalised


def _normalise_twin_confidence_summary(
    summary_record: Mapping[str, object],
) -> dict[str, object]:
    """Validate one ``TwinConfidenceSummary`` audit mapping and its hash."""
    if not isinstance(summary_record, Mapping):
        raise ValueError("summary_record must be a mapping")
    summary = {
        "tick_count": _positive_int(
            summary_record.get("tick_count"),
            "tick_count",
            minimum=1,
        ),
        "healthy_count": _non_negative_int(
            summary_record.get("healthy_count"),
            "healthy_count",
        ),
        "warning_count": _non_negative_int(
            summary_record.get("warning_count"),
            "warning_count",
        ),
        "critical_count": _non_negative_int(
            summary_record.get("critical_count"),
            "critical_count",
        ),
        "min_confidence": _unit_interval_number(
            summary_record.get("min_confidence"),
            "min_confidence",
        ),
        "mean_confidence": _unit_interval_number(
            summary_record.get("mean_confidence"),
            "mean_confidence",
        ),
        "latest_confidence": _unit_interval_number(
            summary_record.get("latest_confidence"),
            "latest_confidence",
        ),
        "worst_status": _normalise_status(
            summary_record.get("worst_status"),
            "worst_status",
        ),
        "latest_status": _normalise_status(
            summary_record.get("latest_status"),
            "latest_status",
        ),
        "summary_hash": _require_sha256_hex(
            summary_record.get("summary_hash"),
            "summary_hash",
        ),
    }
    if (
        cast("int", summary["healthy_count"])
        + cast("int", summary["warning_count"])
        + cast("int", summary["critical_count"])
        != summary["tick_count"]
    ):
        raise ValueError("summary status counts must sum to tick_count")
    _require_matching_hash(
        summary,
        hash_field="summary_hash",
        label="summary_hash",
    )
    return summary


def _normalise_status(value: object, name: str) -> str:
    """Return a known twin-confidence status string, else raise."""
    status = _require_non_empty_text(value, name)
    if status not in _STATUS_LEVELS:
        raise ValueError(f"{name} must be healthy, warning, or critical")
    return status


def _require_matching_hash(
    record: Mapping[str, object],
    *,
    hash_field: str,
    label: str,
) -> None:
    """Validate a monitor audit hash against canonical JSON without the hash."""
    expected = cast("str", record[hash_field])
    payload = {key: value for key, value in record.items() if key != hash_field}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    actual = sha256(canonical.encode("utf-8")).hexdigest()
    if actual != expected:
        raise ValueError(f"{label} does not match record payload")


def _require_matching_summary(
    scores: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
) -> None:
    """Cross-check the supplied summary against the score sequence."""
    if summary["tick_count"] != len(scores):
        raise ValueError("summary tick_count must match score_records length")
    counts = {
        "healthy": sum(1 for score in scores if score["status"] == "healthy"),
        "warning": sum(1 for score in scores if score["status"] == "warning"),
        "critical": sum(1 for score in scores if score["status"] == "critical"),
    }
    if (
        summary["healthy_count"] != counts["healthy"]
        or summary["warning_count"] != counts["warning"]
        or summary["critical_count"] != counts["critical"]
    ):
        raise ValueError("summary status counts must match score_records")
    confidences = [cast("float", score["confidence"]) for score in scores]
    latest = cast("float", scores[-1]["confidence"])
    if not _close(cast("float", summary["min_confidence"]), min(confidences)):
        raise ValueError("summary min_confidence must match score_records")
    if not _close(
        cast("float", summary["mean_confidence"]),
        sum(confidences) / len(confidences),
    ):
        raise ValueError("summary mean_confidence must match score_records")
    if not _close(cast("float", summary["latest_confidence"]), latest):
        raise ValueError("summary latest_confidence must match score_records")
    if summary["latest_status"] != scores[-1]["status"]:
        raise ValueError("summary latest_status must match score_records")
    if summary["worst_status"] != _worst_status(scores):
        raise ValueError("summary worst_status must match score_records")


def _worst_status(scores: Sequence[Mapping[str, object]]) -> str:
    """Return the worst status present in a score sequence."""
    if any(score["status"] == "critical" for score in scores):
        return "critical"
    if any(score["status"] == "warning" for score in scores):
        return "warning"
    return "healthy"


def _close(left: float, right: float) -> bool:
    """Return whether two summary floats match to audit-record tolerance."""
    return abs(left - right) <= 1e-12
