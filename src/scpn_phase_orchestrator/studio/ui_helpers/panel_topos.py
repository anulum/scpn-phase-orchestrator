# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Phase Orchestrator — SPO Studio topos binding-validation panel

"""Topos semantic-binding validation panel builder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from ._shared import (
    _normalise_text_sequence,
    _positive_int,
    _require_non_empty_text,
    _require_sha256_hex,
    _required_bool,
)

_TOPOS_PROOF_BOUNDARY = "categorical_validation_prototype_not_formal_topos_proof"


_TOPOS_REPORT_SCHEMAS = frozenset(
    {
        "symbolic_binding_functor",
        "policy_composition_category",
    }
)


def build_topos_semantic_binding_studio_panel(
    symbolic_reports: Sequence[Mapping[str, object]],
    policy_reports: Sequence[Mapping[str, object]],
    *,
    examples: Sequence[Mapping[str, object]] = (),
) -> dict[str, object]:
    """Return a Studio panel for Topos semantic-binding evidence.

    The helper renders categorical validation reports and deterministic domain
    obligations as review evidence only. It validates schema names, proof
    boundaries, report hashes, obligation/object/morphism counts, non-actuation
    flags, and example hashes before exposing a compact payload to Studio. The
    payload intentionally makes no formal Topos proof claim and emits no
    executable policy actions.

    Parameters
    ----------
    symbolic_reports : Sequence[Mapping[str, object]]
        Symbolic-binding reports.
    policy_reports : Sequence[Mapping[str, object]]
        Policy-composition reports.
    examples : Sequence[Mapping[str, object]]
        Example records.

    Returns
    -------
    dict[str, object]
        A Studio panel for Topos semantic-binding evidence.
    """
    normalised_symbolic = _normalise_topos_validation_reports(
        symbolic_reports,
        schema_name="symbolic_binding_functor",
        label="symbolic binding report",
    )
    normalised_policy = _normalise_topos_validation_reports(
        policy_reports,
        schema_name="policy_composition_category",
        label="policy composition report",
    )
    normalised_examples, example_rows = _normalise_topos_domain_examples(examples)
    all_reports = (*normalised_symbolic, *normalised_policy)
    object_counts = [cast("int", report["object_count"]) for report in all_reports]
    morphism_counts = [cast("int", report["morphism_count"]) for report in all_reports]
    failed_symbolic = [
        cast("str", report["report_hash"])
        for report in normalised_symbolic
        if report["passed"] is not True
    ]
    failed_policy = [
        cast("str", report["report_hash"])
        for report in normalised_policy
        if report["passed"] is not True
    ]
    return {
        "panel_kind": "studio_topos_semantic_binding_panel",
        "proof_surface": "topos_semantic_binding",
        "symbolic_report_count": len(normalised_symbolic),
        "policy_report_count": len(normalised_policy),
        "example_count": len(normalised_examples),
        "passed_symbolic_report_count": len(normalised_symbolic) - len(failed_symbolic),
        "passed_policy_report_count": len(normalised_policy) - len(failed_policy),
        "failed_symbolic_report_hashes": failed_symbolic,
        "failed_policy_report_hashes": failed_policy,
        "proof_boundary": _TOPOS_PROOF_BOUNDARY,
        "non_actuating": True,
        "actuation_permitted": False,
        "formal_proof_claim_permitted": False,
        "symbolic_reports": normalised_symbolic,
        "policy_reports": normalised_policy,
        "example_rows": example_rows,
        "example_domains": tuple(
            sorted({cast("str", example["domain"]) for example in normalised_examples})
        ),
        "object_count_range": {
            "minimum": min(object_counts),
            "maximum": max(object_counts),
        },
        "morphism_count_range": {
            "minimum": min(morphism_counts),
            "maximum": max(morphism_counts),
        },
        "operator_summary": (
            "Topos semantic-binding review: "
            f"{len(normalised_symbolic)} symbolic report(s), "
            f"{len(normalised_policy)} policy report(s), "
            f"{len(normalised_examples)} domain example(s)"
        ),
        "operator_action": (
            "render as categorical validation prototype evidence only; preserve "
            "the proof boundary and require a separate formal-methods gate "
            "before claiming machine-checked Topos proofs or applying policy"
        ),
    }


def _normalise_topos_validation_reports(
    reports: Sequence[Mapping[str, object]],
    *,
    schema_name: str,
    label: str,
) -> tuple[dict[str, object], ...]:
    if schema_name not in _TOPOS_REPORT_SCHEMAS:
        raise ValueError("Topos report schema is not supported")
    if isinstance(reports, Mapping) or not isinstance(reports, Sequence) or not reports:
        raise ValueError(f"{label}s must be a non-empty sequence")
    normalised: list[dict[str, object]] = []
    for index, report in enumerate(reports):
        if not isinstance(report, Mapping):
            raise ValueError(f"{label} must be a mapping")
        item_label = f"{label} {index}"
        if report.get("schema_name") != schema_name:
            raise ValueError(f"{item_label} schema_name must be {schema_name}")
        proof_boundary = _require_non_empty_text(
            report.get("proof_boundary"),
            f"{item_label} proof_boundary",
        )
        if proof_boundary != _TOPOS_PROOF_BOUNDARY:
            raise ValueError(f"{item_label} proof boundary is not review-safe")
        if report.get("non_actuating") is not True:
            raise ValueError(f"{item_label} non_actuating must be true")
        object_count = _positive_int(
            report.get("object_count"),
            f"{item_label} object_count",
            minimum=0,
        )
        morphism_count = _positive_int(
            report.get("morphism_count"),
            f"{item_label} morphism_count",
            minimum=0,
        )
        obligations = _normalise_topos_obligations(
            report.get("obligation_records"),
            f"{item_label} obligation_records",
        )
        objects = _normalise_topos_named_records(
            report.get("objects"),
            f"{item_label} objects",
        )
        morphisms = _normalise_topos_morphisms(
            report.get("morphisms"),
            f"{item_label} morphisms",
        )
        if len(objects) != object_count:
            raise ValueError(f"{item_label} object_count must match objects length")
        if len(morphisms) != morphism_count:
            raise ValueError(f"{item_label} morphism_count must match morphisms length")
        normalised.append(
            {
                "schema_name": schema_name,
                "schema_version": _require_non_empty_text(
                    report.get("schema_version"),
                    f"{item_label} schema_version",
                ),
                "object_count": object_count,
                "morphism_count": morphism_count,
                "obligation_records": obligations,
                "objects": objects,
                "morphisms": morphisms,
                "passed": _required_bool(report.get("passed"), f"{item_label} passed"),
                "report_hash": _require_sha256_hex(
                    report.get("report_hash"),
                    f"{item_label} report_hash",
                ),
                "proof_boundary": _TOPOS_PROOF_BOUNDARY,
                "non_actuating": True,
            }
        )
    return tuple(normalised)


def _normalise_topos_obligations(
    value: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    if not value:
        raise ValueError(f"{name} must not be empty")
    obligations: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        status = _require_non_empty_text(item.get("status"), f"{name} status")
        if status not in {"passed", "failed"}:
            raise ValueError(f"{name} status must be passed or failed")
        obligations.append(
            {
                "name": _require_non_empty_text(item.get("name"), f"{name} name"),
                "status": status,
                "evidence": _require_non_empty_text(
                    item.get("evidence"),
                    f"{name} evidence",
                ),
            }
        )
    return tuple(obligations)


def _normalise_topos_named_records(
    value: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    records: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        record: dict[str, object] = {
            "name": _require_non_empty_text(item.get("name"), f"{name} name"),
        }
        if "kind" in item:
            record["kind"] = _require_non_empty_text(item.get("kind"), f"{name} kind")
        if "detail" in item:
            record["detail"] = _require_non_empty_text(
                item.get("detail"),
                f"{name} detail",
            )
        if "regimes" in item:
            record["regimes"] = list(
                _normalise_text_sequence(item.get("regimes"), f"{name} regimes")
            )
        if "action_labels" in item:
            record["action_labels"] = list(
                _normalise_text_sequence(
                    item.get("action_labels"),
                    f"{name} action_labels",
                )
            )
        records.append(record)
    return tuple(records)


def _normalise_topos_morphisms(
    value: object,
    name: str,
) -> tuple[dict[str, object], ...]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence")
    morphisms: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError(f"{name} entries must be mappings")
        deterministic = _required_bool(
            item.get("deterministic"),
            f"{name} deterministic",
        )
        if deterministic is not True:
            raise ValueError(f"{name} deterministic must be true")
        morphisms.append(
            {
                "source": _require_non_empty_text(
                    item.get("source"),
                    f"{name} source",
                ),
                "target": _require_non_empty_text(
                    item.get("target"),
                    f"{name} target",
                ),
                "label": _require_non_empty_text(item.get("label"), f"{name} label"),
                "deterministic": True,
            }
        )
    return tuple(morphisms)


def _normalise_topos_domain_examples(
    examples: Sequence[Mapping[str, object]],
) -> tuple[tuple[dict[str, object], ...], tuple[dict[str, object], ...]]:
    if isinstance(examples, Mapping) or not isinstance(examples, Sequence):
        raise ValueError("Topos examples must be a sequence")
    normalised: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        if not isinstance(example, Mapping):
            raise ValueError("Topos example must be a mapping")
        label = f"Topos example {index}"
        proof_boundary = _require_non_empty_text(
            example.get("proof_boundary"),
            f"{label} proof_boundary",
        )
        if proof_boundary != _TOPOS_PROOF_BOUNDARY:
            raise ValueError(f"{label} proof boundary is not review-safe")
        if example.get("non_actuating") is not True:
            raise ValueError(f"{label} non_actuating must be true")
        passed = _required_bool(example.get("passed"), f"{label} passed")
        if passed is not True:
            raise ValueError(f"{label} must be passed")
        obligation_names = _normalise_text_sequence(
            example.get("obligation_names"),
            f"{label} obligation_names",
        )
        domain = _require_non_empty_text(example.get("domain"), f"{label} domain")
        example_hash = _require_sha256_hex(
            example.get("example_hash"),
            f"{label} example_hash",
        )
        binding_count = _positive_int(
            example.get("binding_object_count"),
            f"{label} binding_object_count",
            minimum=1,
        )
        policy_count = _positive_int(
            example.get("policy_object_count"),
            f"{label} policy_object_count",
            minimum=1,
        )
        record = {
            "domain": domain,
            "symbolic_prompt": _require_non_empty_text(
                example.get("symbolic_prompt"),
                f"{label} symbolic_prompt",
            ),
            "binding_object_count": binding_count,
            "policy_object_count": policy_count,
            "obligation_names": list(obligation_names),
            "passed": True,
            "non_actuating": True,
            "proof_boundary": _TOPOS_PROOF_BOUNDARY,
            "example_hash": example_hash,
        }
        normalised.append(record)
        rows.append(
            {
                "domain": domain,
                "example_hash": example_hash,
                "binding_object_count": binding_count,
                "policy_object_count": policy_count,
                "obligation_count": len(obligation_names),
            }
        )
    return (tuple(normalised), tuple(rows))
