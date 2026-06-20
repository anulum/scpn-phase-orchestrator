# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio evolutionary policy-search panel

"""Evolutionary supervisor policy-search review panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._shared import (
    _finite_number,
    _non_negative_int,
    _normalise_information_geometry_gradient,
    _normalise_optional_text_sequence,
    _optional_non_negative_int,
    _optional_sha256_hex,
    _positive_float,
    _positive_int,
    _require_non_empty_text,
    _require_sha256_hex,
    _required_bool,
)

_EVOLUTIONARY_SEARCH_BOUNDARY = (
    "offline_evolutionary_supervisor_review_not_live_actuation"
)


_EVOLUTIONARY_EXAMPLE_BOUNDARY = "evolutionary_supervisor_search_not_live_actuation"


_EVOLUTIONARY_SEARCH_SCHEMA = "evolutionary_supervisor_policy_search"


_EVOLUTIONARY_DSL_SCHEMA = "policy_dsl_evolution"


def build_evolutionary_supervisor_policy_search_studio_panel(
    reports: Sequence[Mapping[str, object]],
    *,
    examples: Sequence[Mapping[str, object]] = (),
    dsl_reports: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for offline evolutionary policy-search evidence.

    The panel renders deterministic replay search reports, optional enriched
    domain examples, and optional policy-DSL mutation reports for operator
    review. It validates hashes, candidate counts, replay/STL summaries, DSL
    mutation records, and all disabled execution gates before exposing data to
    Studio. No returned field permits live merge, hot patching, or actuation.

    Parameters
    ----------
    reports : Sequence[Mapping[str, object]]
        The report records.
    examples : Sequence[Mapping[str, object]]
        Example records.
    dsl_reports : Sequence[Mapping[str, object]]
        Policy-DSL search reports.

    Returns
    -------
    dict[str, object]
        A Studio panel for offline evolutionary policy-search evidence.
    """
    normalised_reports = _normalise_evolutionary_search_reports(reports)
    normalised_examples, example_rows = _normalise_evolutionary_examples(examples)
    normalised_dsl_reports = _normalise_evolutionary_dsl_reports(dsl_reports)
    candidate_counts = [
        cast("int", report["candidate_count"]) for report in normalised_reports
    ] + [cast("int", report["candidate_count"]) for report in normalised_dsl_reports]
    accepted_total = sum(
        cast("int", report["accepted_count"]) for report in normalised_reports
    ) + sum(cast("int", report["accepted_count"]) for report in normalised_dsl_reports)
    rejected_total = sum(
        cast("int", report["rejected_count"]) for report in normalised_reports
    ) + sum(cast("int", report["rejected_count"]) for report in normalised_dsl_reports)
    best_rows = [
        report["best_candidate"]
        for report in normalised_reports
        if report["best_candidate"] is not None
    ]
    replay_reward_values = [
        cast(
            "float",
            cast("Mapping[str, object]", report["replay_summary"])["mean_reward"],
        )
        for report in normalised_reports
    ]
    return {
        "panel_kind": "studio_evolutionary_supervisor_policy_search_panel",
        "supervisor": "evolutionary_policy_search",
        "search_report_count": len(normalised_reports),
        "dsl_report_count": len(normalised_dsl_reports),
        "example_count": len(normalised_examples),
        "claim_boundary": _EVOLUTIONARY_SEARCH_BOUNDARY,
        "non_actuating": True,
        "execution_disabled": True,
        "operator_review_required": True,
        "hot_patch_permitted": False,
        "live_merge_permitted": False,
        "actuation_permitted": False,
        "search_reports": normalised_reports,
        "dsl_reports": normalised_dsl_reports,
        "example_rows": example_rows,
        "example_domains": tuple(
            sorted({cast("str", example["domain"]) for example in normalised_examples})
        ),
        "best_candidate_rows": tuple(best_rows),
        "candidate_count_range": {
            "minimum": min(candidate_counts),
            "maximum": max(candidate_counts),
        },
        "accepted_candidate_total": accepted_total,
        "rejected_candidate_total": rejected_total,
        "replay_reward_range": {
            "minimum": min(replay_reward_values),
            "maximum": max(replay_reward_values),
        },
        "operator_summary": (
            "evolutionary policy-search review: "
            f"{len(normalised_reports)} replay report(s), "
            f"{len(normalised_dsl_reports)} DSL report(s), "
            f"{len(normalised_examples)} domain example(s), "
            f"{accepted_total} accepted candidate(s)"
        ),
        "operator_action": (
            "render as offline evolutionary review evidence only; compare replay "
            "reward, STL robustness, candidate rejection reasons, and DSL mutation "
            "rows before any separately reviewed policy merge workflow"
        ),
    }


def _normalise_evolutionary_search_reports(
    reports: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(reports, Mapping) or not isinstance(reports, Sequence) or not reports:
        raise ValueError("evolutionary search reports must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError("evolutionary search report must be a mapping")
        label = f"evolutionary search report {index}"
        if report.get("schema_name") != _EVOLUTIONARY_SEARCH_SCHEMA:
            raise ValueError(f"{label} schema_name is not supported")
        if report.get("claim_boundary") != _EVOLUTIONARY_SEARCH_BOUNDARY:
            raise ValueError(f"{label} claim_boundary is not review-safe")
        _require_evolutionary_review_gates(report, label, require_actuation=False)
        candidate_count = _positive_int(
            report.get("candidate_count"), f"{label} candidate_count", minimum=1
        )
        accepted_count = _non_negative_int(
            report.get("accepted_count"), f"{label} accepted_count"
        )
        rejected_count = _non_negative_int(
            report.get("rejected_count"), f"{label} rejected_count"
        )
        if accepted_count + rejected_count != candidate_count:
            raise ValueError(
                f"{label} accepted_count and rejected_count must sum to candidate_count"
            )
        candidates = _normalise_evolutionary_candidates(
            report.get("candidates"), f"{label} candidates"
        )
        if len(candidates) != candidate_count:
            raise ValueError(f"{label} candidate_count must match candidates length")
        if (
            sum(1 for candidate in candidates if candidate["accepted"])
            != accepted_count
        ):
            raise ValueError(f"{label} accepted_count must match candidate statuses")
        if (
            sum(1 for candidate in candidates if not candidate["accepted"])
            != rejected_count
        ):
            raise ValueError(f"{label} rejected_count must match candidate statuses")
        best_candidate = _normalise_optional_evolutionary_candidate(
            report.get("best_candidate"), f"{label} best_candidate"
        )
        normalised.append(
            {
                "schema_name": _EVOLUTIONARY_SEARCH_SCHEMA,
                "schema_version": _require_non_empty_text(
                    report.get("schema_version"), f"{label} schema_version"
                ),
                "generation_count": _positive_int(
                    report.get("generation_count"),
                    f"{label} generation_count",
                    minimum=1,
                ),
                "population_size": _positive_int(
                    report.get("population_size"), f"{label} population_size", minimum=1
                ),
                "mutation_step": _positive_float(
                    report.get("mutation_step"), f"{label} mutation_step"
                ),
                "minimum_replay_reward": _finite_number(
                    report.get("minimum_replay_reward"),
                    f"{label} minimum_replay_reward",
                ),
                "minimum_safety_margin": _finite_number(
                    report.get("minimum_safety_margin"),
                    f"{label} minimum_safety_margin",
                ),
                "parent_policy_hash": _require_sha256_hex(
                    report.get("parent_policy_hash"), f"{label} parent_policy_hash"
                ),
                "replay_summary": _normalise_evolutionary_replay_summary(
                    report.get("replay_summary"), f"{label} replay_summary"
                ),
                "stl_spec": _require_non_empty_text(
                    report.get("stl_spec"), f"{label} stl_spec"
                ),
                "stl_monitoring": _normalise_evolutionary_stl_monitoring(
                    report.get("stl_monitoring"), f"{label} stl_monitoring"
                ),
                "candidate_count": candidate_count,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "best_candidate": best_candidate,
                "candidates": candidates,
                "claim_boundary": _EVOLUTIONARY_SEARCH_BOUNDARY,
                "non_actuating": True,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "operator_review_required": True,
                "report_hash": _require_sha256_hex(
                    report.get("report_hash"), f"{label} report_hash"
                ),
            }
        )
    return tuple(normalised)


def _normalise_evolutionary_candidates(
    value: object, name: str
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    candidates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for candidate in value:
        candidates.append(_normalise_evolutionary_candidate(candidate, name, seen_ids))
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    return tuple(candidates)


def _normalise_optional_evolutionary_candidate(
    value: object, name: str
) -> dict[str, object] | None:
    if value is None:
        return None
    return _normalise_evolutionary_candidate(value, name, set())


def _normalise_evolutionary_candidate(
    value: object, name: str, seen_ids: set[str]
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} entries must be mappings")
    candidate_id = _require_non_empty_text(value.get("candidate_id"), "candidate_id")
    if candidate_id in seen_ids:
        raise ValueError(f"{name} candidate_id values must be unique")
    seen_ids.add(candidate_id)
    for field, expected in (
        ("review_required", True),
        ("live_merge_permitted", False),
        ("hot_patch_permitted", False),
        ("actuation_permitted", False),
    ):
        if _required_bool(value.get(field), f"{name} {field}") is not expected:
            raise ValueError(f"{name} {field} must be {expected}")
    status = _require_non_empty_text(value.get("status"), f"{name} status")
    if status not in {"accepted_for_review", "rejected"}:
        raise ValueError(f"{name} status is not supported")
    blocked_reasons = _normalise_optional_text_sequence(
        value.get("blocked_reasons"), f"{name} blocked_reasons"
    )
    accepted = status == "accepted_for_review"
    if accepted and blocked_reasons:
        raise ValueError(f"{name} accepted candidates must not have blocked_reasons")
    if not accepted and not blocked_reasons:
        raise ValueError(f"{name} rejected candidates require blocked_reasons")
    return {
        "candidate_id": candidate_id,
        "generation": _positive_int(
            value.get("generation"), f"{name} generation", minimum=1
        ),
        "knob": _require_non_empty_text(value.get("knob"), f"{name} knob"),
        "parent_value": _finite_number(
            value.get("parent_value"), f"{name} parent_value"
        ),
        "candidate_value": _finite_number(
            value.get("candidate_value"), f"{name} candidate_value"
        ),
        "mutation_delta": _finite_number(
            value.get("mutation_delta"), f"{name} mutation_delta"
        ),
        "genome": _normalise_information_geometry_gradient(
            value.get("genome"), f"{name} genome"
        ),
        "replay_fitness": _finite_number(
            value.get("replay_fitness"), f"{name} replay_fitness"
        ),
        "stl_robustness": _finite_number(
            value.get("stl_robustness"), f"{name} stl_robustness"
        ),
        "stl_satisfied": _required_bool(
            value.get("stl_satisfied"), f"{name} stl_satisfied"
        ),
        "replay_violation_count": _non_negative_int(
            value.get("replay_violation_count"), f"{name} replay_violation_count"
        ),
        "blocked_reasons": tuple(blocked_reasons),
        "status": status,
        "accepted": accepted,
        "candidate_hash": _require_sha256_hex(
            value.get("candidate_hash"), f"{name} candidate_hash"
        ),
        "review_required": True,
        "live_merge_permitted": False,
        "hot_patch_permitted": False,
        "actuation_permitted": False,
    }


def _normalise_evolutionary_replay_summary(
    value: object, name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {
        "replay_count": _positive_int(
            value.get("replay_count"), f"{name} replay_count", minimum=1
        ),
        "mean_reward": _finite_number(value.get("mean_reward"), f"{name} mean_reward"),
        "min_reward": _finite_number(value.get("min_reward"), f"{name} min_reward"),
        "mean_safety_margin": _finite_number(
            value.get("mean_safety_margin"), f"{name} mean_safety_margin"
        ),
        "min_safety_margin": _finite_number(
            value.get("min_safety_margin"), f"{name} min_safety_margin"
        ),
        "violation_count": _non_negative_int(
            value.get("violation_count"), f"{name} violation_count"
        ),
    }


def _normalise_evolutionary_stl_monitoring(
    value: object, name: str
) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    result: dict[str, object] = {}
    for key, raw_value in value.items():
        key_text = _require_non_empty_text(key, f"{name} key")
        if isinstance(raw_value, bool):
            result[key_text] = bool(raw_value)
        elif isinstance(raw_value, int | float):
            result[key_text] = _finite_number(raw_value, f"{name} {key_text}")
        elif isinstance(raw_value, str):
            result[key_text] = _require_non_empty_text(raw_value, f"{name} {key_text}")
        elif isinstance(raw_value, Sequence) and not isinstance(raw_value, str | bytes):
            result[key_text] = list(
                _normalise_evolutionary_json_sequence(raw_value, f"{name} {key_text}")
            )
        else:
            raise ValueError(f"{name} values must be JSON-safe scalars or sequences")
    return result


def _normalise_evolutionary_json_sequence(
    values: Sequence[object], name: str
) -> tuple[object, ...]:
    normalised: list[object] = []
    for item in values:
        if isinstance(item, bool):
            normalised.append(bool(item))
        elif isinstance(item, int | float):
            normalised.append(_finite_number(item, name))
        elif isinstance(item, str):
            normalised.append(_require_non_empty_text(item, name))
        else:
            raise ValueError(f"{name} sequence values must be JSON-safe scalars")
    return tuple(normalised)


def _normalise_evolutionary_examples(
    examples: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if isinstance(examples, Mapping) or not isinstance(examples, Sequence):
        raise ValueError("evolutionary examples must be a sequence")
    normalised: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("evolutionary example must be a mapping")
        label = f"evolutionary example {index}"
        if example.get("claim_boundary") != _EVOLUTIONARY_EXAMPLE_BOUNDARY:
            raise ValueError(f"{label} claim_boundary is not review-safe")
        for field, expected in (
            ("operator_review_required", True),
            ("execution_disabled", True),
            ("hot_patch_permitted", False),
            ("live_merge_permitted", False),
            ("actuation_permitted", False),
        ):
            if _required_bool(example.get(field), f"{label} {field}") is not expected:
                raise ValueError(f"{label} {field} must be {expected}")
        scenario_hash = _require_sha256_hex(
            example.get("scenario_hash"), f"{label} scenario_hash"
        )
        domain = _require_non_empty_text(example.get("domain"), f"{label} domain")
        candidate_count = _optional_non_negative_int(
            example.get("candidate_count"), f"{label} candidate_count"
        )
        accepted_count = _optional_non_negative_int(
            example.get("accepted_candidate_count"), f"{label} accepted_candidate_count"
        )
        rejected_count = _optional_non_negative_int(
            example.get("rejected_candidate_count"), f"{label} rejected_candidate_count"
        )
        report_hash = _optional_sha256_hex(
            example.get("report_hash"), f"{label} report_hash"
        )
        if candidate_count is not None:
            if accepted_count is None or rejected_count is None or report_hash is None:
                raise ValueError(
                    f"{label} enriched candidate counts require report_hash"
                )
            if accepted_count + rejected_count != candidate_count:
                raise ValueError(f"{label} accepted/rejected counts must sum")
        normalised.append(
            {
                "domain": domain,
                "scenario_id": _require_non_empty_text(
                    example.get("scenario_id"), f"{label} scenario_id"
                ),
                "scenario_hash": scenario_hash,
                "claim_boundary": _EVOLUTIONARY_EXAMPLE_BOUNDARY,
                "operator_review_required": True,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "actuation_permitted": False,
                "candidate_count": candidate_count,
                "accepted_candidate_count": accepted_count,
                "rejected_candidate_count": rejected_count,
                "report_hash": report_hash,
            }
        )
        rows.append(
            {
                "domain": domain,
                "scenario_id": example["scenario_id"],
                "scenario_hash": scenario_hash,
                "candidate_count": candidate_count,
                "accepted_candidate_count": accepted_count,
                "rejected_candidate_count": rejected_count,
            }
        )
    return (tuple(normalised), tuple(rows))


def _normalise_evolutionary_dsl_reports(
    reports: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    if isinstance(reports, Mapping) or not isinstance(reports, Sequence):
        raise ValueError("evolutionary DSL reports must be a sequence")
    normalised: list[dict[str, object]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError("evolutionary DSL report must be a mapping")
        label = f"evolutionary DSL report {index}"
        if report.get("schema_name") != _EVOLUTIONARY_DSL_SCHEMA:
            raise ValueError(f"{label} schema_name is not supported")
        _require_evolutionary_review_gates(report, label, require_actuation=True)
        candidate_count = _positive_int(
            report.get("candidate_count"), f"{label} candidate_count", minimum=1
        )
        accepted_count = _non_negative_int(
            report.get("accepted_count"), f"{label} accepted_count"
        )
        rejected_count = _non_negative_int(
            report.get("rejected_count"), f"{label} rejected_count"
        )
        if accepted_count + rejected_count != candidate_count:
            raise ValueError(f"{label} accepted/rejected counts must sum")
        candidates = _normalise_evolutionary_dsl_candidates(
            report.get("candidates"), f"{label} candidates"
        )
        if len(candidates) != candidate_count:
            raise ValueError(f"{label} candidate_count must match candidates length")
        normalised.append(
            {
                "schema_name": _EVOLUTIONARY_DSL_SCHEMA,
                "schema_version": _require_non_empty_text(
                    report.get("schema_version"), f"{label} schema_version"
                ),
                "generation_count": _positive_int(
                    report.get("generation_count"),
                    f"{label} generation_count",
                    minimum=1,
                ),
                "population_size": _positive_int(
                    report.get("population_size"), f"{label} population_size", minimum=1
                ),
                "mutation_step": _positive_float(
                    report.get("mutation_step"), f"{label} mutation_step"
                ),
                "source_policy_hash": _require_sha256_hex(
                    report.get("source_policy_hash"), f"{label} source_policy_hash"
                ),
                "candidate_count": candidate_count,
                "accepted_count": accepted_count,
                "rejected_count": rejected_count,
                "candidates": candidates,
                "execution_disabled": True,
                "hot_patch_permitted": False,
                "live_merge_permitted": False,
                "actuation_permitted": False,
                "operator_review_required": True,
                "non_actuating": True,
                "report_hash": _require_sha256_hex(
                    report.get("report_hash"), f"{label} report_hash"
                ),
            }
        )
    return tuple(normalised)


def _normalise_evolutionary_dsl_candidates(
    value: object, name: str
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    candidates: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for candidate in value:
        if not isinstance(candidate, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        candidate_id = _require_non_empty_text(
            candidate.get("candidate_id"), f"{name} candidate_id"
        )
        if candidate_id in seen_ids:
            raise ValueError(f"{name} candidate_id values must be unique")
        seen_ids.add(candidate_id)
        for field, expected in (
            ("operator_review_required", True),
            ("execution_disabled", True),
            ("live_merge_permitted", False),
            ("hot_patch_permitted", False),
            ("actuation_permitted", False),
        ):
            if _required_bool(candidate.get(field), f"{name} {field}") is not expected:
                raise ValueError(f"{name} {field} must be {expected}")
        status = _require_non_empty_text(candidate.get("status"), f"{name} status")
        if status not in {"accepted", "rejected"}:
            raise ValueError(f"{name} status is not supported")
        candidates.append(
            {
                "candidate_id": candidate_id,
                "generation": _positive_int(
                    candidate.get("generation"), f"{name} generation", minimum=1
                ),
                "mutation_index": _non_negative_int(
                    candidate.get("mutation_index"), f"{name} mutation_index"
                ),
                "source_rule_name": _require_non_empty_text(
                    candidate.get("source_rule_name"), f"{name} source_rule_name"
                ),
                "blocked_reasons": _normalise_optional_text_sequence(
                    candidate.get("blocked_reasons"), f"{name} blocked_reasons"
                ),
                "status": status,
                "accepted": status == "accepted",
                "candidate_hash": _require_sha256_hex(
                    candidate.get("candidate_hash"), f"{name} candidate_hash"
                ),
                "operator_review_required": True,
                "execution_disabled": True,
                "live_merge_permitted": False,
                "hot_patch_permitted": False,
                "actuation_permitted": False,
            }
        )
    if not candidates:
        raise ValueError(f"{name} must not be empty")
    return tuple(candidates)


def _require_evolutionary_review_gates(
    record: Mapping[str, object], label: str, *, require_actuation: bool
) -> None:
    fields = [
        ("non_actuating", True),
        ("execution_disabled", True),
        ("hot_patch_permitted", False),
        ("live_merge_permitted", False),
        ("operator_review_required", True),
    ]
    if require_actuation:
        fields.append(("actuation_permitted", False))
    for field, expected in fields:
        if _required_bool(record.get(field), f"{label} {field}") is not expected:
            raise ValueError(f"{label} {field} must be {expected}")
