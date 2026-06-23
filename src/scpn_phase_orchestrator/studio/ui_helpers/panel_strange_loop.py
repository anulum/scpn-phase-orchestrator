# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio strange-loop review panel

"""Strange-loop drift review panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._shared import (
    _non_negative_float,
    _non_negative_int,
    _normalise_text_sequence,
    _positive_int,
    _require_non_empty_text,
    _require_sha256_hex,
    _unit_interval_number,
)

_STRANGE_LOOP_CLAIM_BOUNDARY = "strange_loop_drift_review_not_live_actuation"


_STRANGE_LOOP_TRIGGERS = frozenset(
    {"stable", "policy_drift", "control_loop_oscillation", "over_control"}
)


def build_strange_loop_studio_panel(
    records: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Return a Studio panel payload for strange-loop drift scenario records.

    The panel renders precomputed ``StrangeLoopSupervisor`` review evidence.
    It does not observe live actions, execute recommendations, or apply control
    changes. All scenario records must keep the supervisor's non-actuating
    claim boundary and disabled-execution flags intact.

    Parameters
    ----------
    records : Sequence[Mapping[str, object]]
        The records to summarise.

    Returns
    -------
    dict[str, object]
        A Studio panel payload for strange-loop drift scenario records.
    """
    normalised_records = _normalise_strange_loop_records(records)
    drift_scores = [
        cast("float", record["max_drift_score"]) for record in normalised_records
    ]
    oscillation_scores = [
        cast("float", record["max_oscillation_score"]) for record in normalised_records
    ]
    overcontrol_scores = [
        cast("float", record["max_overcontrol_score"]) for record in normalised_records
    ]
    coherence_scores = [
        cast("float", record["min_control_coherence"]) for record in normalised_records
    ]
    failed_ids = [
        cast("str", record["scenario_id"])
        for record in normalised_records
        if record["passed_expected_trigger"] is not True
    ]
    triggered_modes = tuple(
        sorted(
            {cast("str", record["expected_trigger"]) for record in normalised_records}
        )
    )
    return {
        "panel_kind": "studio_strange_loop_panel",
        "supervisor": "strange_loop",
        "scenario_count": len(normalised_records),
        "passed_count": len(normalised_records) - len(failed_ids),
        "failed_scenario_ids": failed_ids,
        "triggered_modes": triggered_modes,
        "claim_boundary": _STRANGE_LOOP_CLAIM_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "actuation_permitted": False,
        "series": normalised_records,
        "maxima": {
            "drift_score": max(drift_scores),
            "oscillation_score": max(oscillation_scores),
            "overcontrol_score": max(overcontrol_scores),
        },
        "minima": {
            "control_coherence": min(coherence_scores),
        },
        "operator_summary": (
            "strange-loop scenario review: "
            f"{len(normalised_records) - len(failed_ids)}/"
            f"{len(normalised_records)} expected triggers passed"
        ),
        "operator_action": (
            "render as offline supervisor self-control evidence; keep all "
            "recommendations behind the normal review and safety gate"
        ),
    }


def _normalise_strange_loop_records(
    records: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Validate and normalise strange-loop scenario records for the panel.

    Each record must carry the non-actuating claim boundary and disabled-execution
    flags and a supported expected trigger; scores, counts, hashes, and knob names
    are coerced to their checked types. Raises ``ValueError`` on any malformed,
    empty, or boundary-violating record.
    """
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise ValueError("strange-loop records must be a sequence")
    if not records:
        raise ValueError("strange-loop records must not be empty")
    normalised: list[dict[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("strange-loop records must be mappings")
        expected_trigger = _require_non_empty_text(
            record.get("expected_trigger"),
            "expected_trigger",
        )
        if expected_trigger not in _STRANGE_LOOP_TRIGGERS:
            raise ValueError("expected_trigger is not supported")
        claim_boundary = _require_non_empty_text(
            record.get("claim_boundary"),
            "claim_boundary",
        )
        if claim_boundary != _STRANGE_LOOP_CLAIM_BOUNDARY:
            raise ValueError("strange-loop claim boundary is required")
        if record.get("non_actuating") is not True:
            raise ValueError("non_actuating must be true")
        if record.get("execution_disabled") is not True:
            raise ValueError("execution_disabled must be true")
        passed = record.get("passed_expected_trigger")
        if not isinstance(passed, bool):
            raise ValueError("passed_expected_trigger must be boolean")
        final_knobs = _normalise_text_sequence(
            record.get("final_recommended_knobs", ()),
            "final_recommended_knobs",
        )
        normalised.append(
            {
                "domain": _require_non_empty_text(record.get("domain"), "domain"),
                "scenario_id": _require_non_empty_text(
                    record.get("scenario_id"),
                    "scenario_id",
                ),
                "expected_trigger": expected_trigger,
                "step_count": _positive_int(
                    record.get("step_count"),
                    "step_count",
                    minimum=1,
                ),
                "max_drift_score": _non_negative_float(
                    record.get("max_drift_score"),
                    "max_drift_score",
                ),
                "max_oscillation_score": _non_negative_float(
                    record.get("max_oscillation_score"),
                    "max_oscillation_score",
                ),
                "max_overcontrol_score": _non_negative_float(
                    record.get("max_overcontrol_score"),
                    "max_overcontrol_score",
                ),
                "min_control_coherence": _unit_interval_number(
                    record.get("min_control_coherence"),
                    "min_control_coherence",
                ),
                "triggered_recommendation_count": _non_negative_int(
                    record.get("triggered_recommendation_count"),
                    "triggered_recommendation_count",
                ),
                "final_recommended_knobs": final_knobs,
                "passed_expected_trigger": passed,
                "scenario_hash": _require_sha256_hex(
                    record.get("scenario_hash"),
                    "scenario_hash",
                ),
                "result_hash": _require_sha256_hex(
                    record.get("result_hash"),
                    "result_hash",
                ),
                "non_actuating": True,
                "execution_disabled": True,
                "claim_boundary": claim_boundary,
            }
        )
    return tuple(normalised)
